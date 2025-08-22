# AWS GPU Training Plan for MLLM Baseline

## Overview
This document provides a complete guide for running our neural two-tower experiments on AWS GPU instances, enabling 5-20x speedup over local Apple Silicon (MPS) training.

## Performance & Cost Comparison

### Current Setup (Apple Silicon MPS)
- **Hardware**: M1/M2 Mac with Metal Performance Shaders
- **Runtime**: 7-8 hours for 876K examples (10-fold CV)
- **Cost**: $0 (local hardware)
- **Limitations**: Single device, memory constraints, no CUDA optimizations

### AWS GPU Options

| Instance Type | GPUs | Memory | Speedup | On-Demand | Spot | Runtime (Config O) |
|--------------|------|---------|---------|-----------|------|-------------------|
| p3.2xlarge | 1x V100 | 16GB | 3-5x | $3.06/hr | ~$0.92/hr | 1.5-2.5 hrs |
| p3.8xlarge | 4x V100 | 64GB | 12-15x | $12.24/hr | ~$3.67/hr | 0.5-1 hr |
| p4d.24xlarge | 8x A100 | 320GB | 25-35x | $32.77/hr | ~$9.83/hr | 15-30 min |

## Prerequisites Setup

```bash
# Install AWS CLI
pip install awscli boto3 paramiko scp

# Configure AWS credentials
aws configure
# Enter: 
#   AWS Access Key ID
#   Secret Access Key
#   Default region: us-east-1
#   Output format: json

# Create key pair for SSH access
aws ec2 create-key-pair --key-name mllm-gpu-key --query 'KeyMaterial' --output text > mllm-gpu-key.pem
chmod 400 mllm-gpu-key.pem

# Verify setup
aws ec2 describe-instances
```

## Quick Start: Single Experiment

### Step 1: Launch GPU Instance

```bash
# Launch p3.2xlarge instance (1x V100 GPU)
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id ami-0c55b159cbfafe1f0 \
    --instance-type p3.2xlarge \
    --key-name mllm-gpu-key \
    --security-groups default \
    --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":100,\"VolumeType\":\"gp3\"}}]" \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=mllm-config-o}]' \
    --query 'Instances[0].InstanceId' \
    --output text)

echo "Launched instance: $INSTANCE_ID"

# Wait for instance to be running
aws ec2 wait instance-running --instance-ids $INSTANCE_ID

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids $INSTANCE_ID \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo "Instance running at: $PUBLIC_IP"
```

### Step 2: Setup and Run Training

```bash
# Wait for SSH to be ready
sleep 60

# Copy code to instance
scp -r -i mllm-gpu-key.pem models data shared ubuntu@$PUBLIC_IP:~/

# SSH into instance and run training
ssh -i mllm-gpu-key.pem ubuntu@$PUBLIC_IP << 'EOF'
    # Install dependencies
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip3 install pandas numpy scikit-learn sentence-transformers
    
    # Modify code to use CUDA instead of MPS
    cd models/neural_two_tower
    sed -i "s/device = 'mps'/device = 'cuda'/g" evaluate_tier2_config_o.py
    sed -i "s/torch.backends.mps.is_available()/torch.cuda.is_available()/g" evaluate_tier2_config_o.py
    
    # Run training
    nohup python3 -u evaluate_tier2_config_o.py > config_o_gpu.log 2>&1 &
    echo "Training started with PID: $!"
EOF
```

### Step 3: Monitor Progress

```bash
# Check training progress every 5 minutes
while true; do
    ssh -i mllm-gpu-key.pem ubuntu@$PUBLIC_IP "tail -20 models/neural_two_tower/config_o_gpu.log"
    sleep 300
done
```

### Step 4: Download Results and Terminate

```bash
# Download results
scp -i mllm-gpu-key.pem ubuntu@$PUBLIC_IP:~/data/results/*.json ./data/results/
scp -i mllm-gpu-key.pem ubuntu@$PUBLIC_IP:~/models/neural_two_tower/config_o_gpu.log ./

# Terminate instance (IMPORTANT: stops billing)
aws ec2 terminate-instances --instance-ids $INSTANCE_ID
```

## Automated Training Pipeline

### Python Script for Automation

Create `aws_gpu_trainer.py`:

```python
import boto3
import time
import paramiko
from scp import SCPClient
import json

class AWSGPUTrainer:
    def __init__(self, key_path='mllm-gpu-key.pem'):
        self.ec2 = boto3.client('ec2')
        self.key_path = key_path
        self.instance_id = None
        self.public_ip = None
        
    def launch_instance(self, instance_type='p3.2xlarge', use_spot=False):
        """Launch GPU instance (on-demand or spot)"""
        
        if use_spot:
            # Request spot instance (70% cheaper)
            response = self.ec2.request_spot_instances(
                SpotPrice="1.00",  # Max price willing to pay
                InstanceCount=1,
                Type="one-time",
                LaunchSpecification={
                    'ImageId': 'ami-0c55b159cbfafe1f0',
                    'InstanceType': instance_type,
                    'KeyName': 'mllm-gpu-key',
                    'SecurityGroups': ['default'],
                    'BlockDeviceMappings': [{
                        'DeviceName': '/dev/sda1',
                        'Ebs': {'VolumeSize': 100, 'VolumeType': 'gp3'}
                    }]
                }
            )
            # Wait for spot request to be fulfilled
            time.sleep(60)
            spot_request_id = response['SpotInstanceRequests'][0]['SpotInstanceRequestId']
            response = self.ec2.describe_spot_instance_requests(
                SpotInstanceRequestIds=[spot_request_id]
            )
            self.instance_id = response['SpotInstanceRequests'][0]['InstanceId']
        else:
            # Launch on-demand instance
            response = self.ec2.run_instances(
                ImageId='ami-0c55b159cbfafe1f0',  # Deep Learning AMI
                InstanceType=instance_type,
                MinCount=1,
                MaxCount=1,
                KeyName='mllm-gpu-key',
                SecurityGroups=['default'],
                BlockDeviceMappings=[{
                    'DeviceName': '/dev/sda1',
                    'Ebs': {'VolumeSize': 100, 'VolumeType': 'gp3'}
                }]
            )
            self.instance_id = response['Instances'][0]['InstanceId']
        
        print(f"Launched {'spot' if use_spot else 'on-demand'} instance: {self.instance_id}")
        
        # Wait for instance to be running
        waiter = self.ec2.get_waiter('instance_running')
        waiter.wait(InstanceIds=[self.instance_id])
        
        # Get public IP
        response = self.ec2.describe_instances(InstanceIds=[self.instance_id])
        self.public_ip = response['Reservations'][0]['Instances'][0]['PublicIpAddress']
        print(f"Instance running at: {self.public_ip}")
        
        # Wait for SSH to be ready
        time.sleep(60)
        
        return self.instance_id, self.public_ip
    
    def setup_instance(self):
        """Install dependencies on instance"""
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(self.public_ip, username='ubuntu', key_filename=self.key_path)
        
        commands = [
            "pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
            "pip3 install pandas numpy scikit-learn sentence-transformers",
            "pip3 install tqdm matplotlib seaborn"
        ]
        
        for cmd in commands:
            print(f"Running: {cmd}")
            stdin, stdout, stderr = ssh.exec_command(cmd)
            stdout.read()  # Wait for completion
            
        ssh.close()
        print("Instance setup complete")
    
    def upload_code(self):
        """Upload code and data to instance"""
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(self.public_ip, username='ubuntu', key_filename=self.key_path)
        
        print("Uploading code and data...")
        with SCPClient(ssh.get_transport()) as scp:
            scp.put('models', recursive=True, remote_path='~/')
            scp.put('data', recursive=True, remote_path='~/')
            scp.put('shared', recursive=True, remote_path='~/')
        
        ssh.close()
        print("Upload complete")
    
    def run_training(self, config='o', batch_size=256):
        """Run training on GPU instance"""
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(self.public_ip, username='ubuntu', key_filename=self.key_path)
        
        # Modify code for CUDA and larger batch size
        commands = [
            "cd ~/models/neural_two_tower",
            f"sed -i \"s/device = 'mps'/device = 'cuda'/g\" evaluate_tier2_config_{config}.py",
            f"sed -i \"s/torch.backends.mps.is_available()/torch.cuda.is_available()/g\" evaluate_tier2_config_{config}.py",
            f"sed -i \"s/batch_size = 96/batch_size = {batch_size}/g\" evaluate_tier2_config_{config}.py",
            f"nohup python3 -u evaluate_tier2_config_{config}.py > config_{config}_gpu.log 2>&1 &",
            "echo $! > training.pid"
        ]
        
        for cmd in commands:
            stdin, stdout, stderr = ssh.exec_command(cmd)
            print(stdout.read().decode())
            
        ssh.close()
        print(f"Training started for Config {config.upper()} with batch size {batch_size}")
    
    def monitor_training(self, check_interval=300):
        """Monitor training progress"""
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        start_time = time.time()
        while True:
            try:
                ssh.connect(self.public_ip, username='ubuntu', key_filename=self.key_path)
                
                # Check if process is still running
                stdin, stdout, stderr = ssh.exec_command("ps -p $(cat ~/models/neural_two_tower/training.pid 2>/dev/null) 2>/dev/null")
                is_running = len(stdout.read().decode()) > 0
                
                # Get last lines of log
                stdin, stdout, stderr = ssh.exec_command("tail -30 ~/models/neural_two_tower/config_*_gpu.log")
                log_output = stdout.read().decode()
                
                ssh.close()
                
                print(f"\n[{time.time() - start_time:.0f}s] Training status:")
                print(log_output)
                
                if not is_running or "Results saved to" in log_output:
                    print("Training completed!")
                    break
                    
            except Exception as e:
                print(f"Monitoring error: {e}")
            
            time.sleep(check_interval)
    
    def download_results(self):
        """Download results from instance"""
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(self.public_ip, username='ubuntu', key_filename=self.key_path)
        
        print("Downloading results...")
        with SCPClient(ssh.get_transport()) as scp:
            scp.get('~/data/results/*.json', 'data/results/')
            scp.get('~/models/neural_two_tower/config_*_gpu.log', '.')
        
        ssh.close()
        print("Results downloaded")
    
    def terminate_instance(self):
        """Terminate the instance to stop billing"""
        if self.instance_id:
            self.ec2.terminate_instances(InstanceIds=[self.instance_id])
            print(f"Terminated instance: {self.instance_id}")
    
    def run_complete_experiment(self, config='o', instance_type='p3.2xlarge', use_spot=True):
        """Run complete experiment pipeline"""
        try:
            # Launch instance
            self.launch_instance(instance_type, use_spot)
            
            # Setup and upload
            self.setup_instance()
            self.upload_code()
            
            # Run training
            batch_size = 256 if 'p3.2x' in instance_type else 512
            self.run_training(config, batch_size)
            
            # Monitor until completion
            self.monitor_training()
            
            # Download results
            self.download_results()
            
        finally:
            # Always terminate instance
            self.terminate_instance()

# Usage
if __name__ == "__main__":
    trainer = AWSGPUTrainer()
    
    # Run Config O on spot instance
    trainer.run_complete_experiment(
        config='o',
        instance_type='p3.2xlarge',
        use_spot=True
    )
```

## Batch Processing Multiple Configurations

### Shell Script for Parallel Training

Create `batch_aws_training.sh`:

```bash
#!/bin/bash

# Function to run training for a config
run_config() {
    CONFIG=$1
    INSTANCE_TYPE=$2
    
    echo "Starting Config $CONFIG on $INSTANCE_TYPE"
    
    python3 - << EOF
from aws_gpu_trainer import AWSGPUTrainer
trainer = AWSGPUTrainer()
trainer.run_complete_experiment('$CONFIG', '$INSTANCE_TYPE', use_spot=True)
EOF
}

# Run multiple configs in parallel
run_config "o" "p3.2xlarge" &
run_config "p" "p3.2xlarge" &
run_config "q" "p3.2xlarge" &

# Wait for all to complete
wait
echo "All training completed"
```

## Large-Scale Weak Labeling on GPU

### Modified Training for 5M+ Examples

```python
# In evaluate_tier2_config_o.py, add GPU optimizations:

# 1. Mixed precision training
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

# 2. Gradient accumulation for large datasets
accumulation_steps = 4

# 3. Larger batch size
batch_size = 512  # GPU can handle much larger batches

# 4. Multi-GPU if available
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    batch_size = batch_size * torch.cuda.device_count()

# Training loop with optimizations
for epoch in range(epochs):
    for i, batch in enumerate(train_loader):
        with autocast():
            outputs = model(batch)
            loss = criterion(outputs, targets) / accumulation_steps
        
        scaler.scale(loss).backward()
        
        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
```

## S3 Integration for Large Datasets

```bash
# Upload large weak-labeled dataset to S3
aws s3 mb s3://mllm-training-data
aws s3 cp data/supervised_training_5m_weak.csv s3://mllm-training-data/

# In training script, download from S3
aws s3 cp s3://mllm-training-data/supervised_training_5m_weak.csv ~/data/

# Upload results back to S3
aws s3 cp ~/data/results/ s3://mllm-training-data/results/ --recursive
```

## Cost Optimization Strategies

### 1. Use Spot Instances
- 70% discount vs on-demand
- Good for training (can handle interruptions)
- Set max price at 50% of on-demand

### 2. Use Appropriate Instance Size
- p3.2xlarge for single experiments
- p3.8xlarge for parallel configs
- p4d only for 10M+ examples

### 3. Auto-termination
```bash
# Add auto-termination after 4 hours (safety measure)
ssh -i mllm-gpu-key.pem ubuntu@$PUBLIC_IP << 'EOF'
    echo "sudo shutdown -h +240" | at now
EOF
```

### 4. Use EBS Snapshots
```bash
# Create snapshot of trained model
aws ec2 create-snapshot --volume-id vol-xxxxx --description "Config O trained model"

# Restore in future instances
aws ec2 create-volume --snapshot-id snap-xxxxx --availability-zone us-east-1a
```

## Expected Results with GPU Acceleration

### Performance Improvements
- **Config O (876K examples)**: 7-8 hours → 1.5 hours
- **Config R (5M examples)**: 40 hours → 5-7 hours  
- **Full discovery (16M)**: Impossible → 8-12 hours

### Cost Estimates
| Configuration | Examples | Instance | Time | On-Demand | Spot |
|--------------|----------|----------|------|-----------|------|
| Config O | 876K | p3.2xlarge | 1.5h | $4.59 | $1.38 |
| Config R (5M) | 5M | p3.8xlarge | 2h | $24.48 | $7.34 |
| Full Scale | 16M | p4d.24xlarge | 4h | $131.08 | $39.32 |

### Iteration Speed
- **Local (MPS)**: 1-2 experiments per day
- **AWS GPU**: 10-15 experiments per day
- **Potential**: Test all weak labeling variations in 2-3 days

## Troubleshooting

### SSH Connection Issues
```bash
# Check security group allows SSH (port 22)
aws ec2 authorize-security-group-ingress \
    --group-name default \
    --protocol tcp \
    --port 22 \
    --cidr 0.0.0.0/0
```

### CUDA Out of Memory
```python
# Reduce batch size
batch_size = 128  # or 64

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Clear cache periodically
torch.cuda.empty_cache()
```

### Spot Instance Interruption
```bash
# Check spot price history
aws ec2 describe-spot-price-history \
    --instance-types p3.2xlarge \
    --product-descriptions "Linux/UNIX" \
    --max-results 10
```

## Next Steps

1. **Immediate**: Run Config O on p3.2xlarge spot (~$2)
2. **This Week**: Test 2M weak labels on p3.8xlarge (~$10)
3. **If Successful**: Scale to 5M examples (~$25)
4. **Final Push**: Full 16M with p4d.24xlarge (~$40)

Total estimated cost for complete experimental sweep: **$75-100**
Expected outcome: **0.48-0.50 nDCG@10** (from current 0.4482)