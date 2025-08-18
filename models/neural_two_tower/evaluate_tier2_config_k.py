#!/usr/bin/env python3
"""
Tier 2 Enhanced Two-Tower - CONFIG K

CONFIG J + MASSIVE SYNTHETIC DATASET:
- Config J's proven 256D LLM embedding architecture
- Extended training (35 epochs, patience=15)
- MASSIVE synthetic dataset: 1.4M examples (73% synthetic)
- Ultimate test: Best architecture + largest high-quality dataset

Hypothesis: State-of-the-art architecture + massive scale = peak performance
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import time
import json
from datetime import datetime
import warnings
import os
import sys

# Add project root directory to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from model import TwoTowerModel, ContrastiveLossWithDiversity, create_model
from data_loader import create_data_loaders, load_data, LLMEvaluationDataset, HardNegativeMiner
from shared.utils.evaluation import calculate_metrics, save_standardized_results
from collaborative_pretraining import CollaborativePretrainingModel

warnings.filterwarnings('ignore')

# Force CPU usage with optimized thread count
torch.set_num_threads(8)


class PretrainedTwoTowerModel(nn.Module):
    """Two-Tower model with pre-trained query encoder from collaborative filtering"""
    
    def __init__(self, num_llms=1131, embedding_dim=128, num_heads=4, dropout=0.2,
                 pretrained_model_path=None):
        super(PretrainedTwoTowerModel, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        
        # Load pre-trained collaborative filtering model
        if pretrained_model_path and os.path.exists(pretrained_model_path):
            print(f"Loading pre-trained query encoder from: {pretrained_model_path}")
            self._load_pretrained_query_encoder(pretrained_model_path)
        else:
            print("No pre-trained model found, using fresh initialization")
            self._initialize_fresh_model(num_llms, embedding_dim, num_heads, dropout)
        
        # LLM tower (CONFIG K: Same as Config J - 256D embeddings)
        self.llm_embedding = nn.Embedding(num_llms, 256)
        self.llm_tower = nn.Sequential(
            nn.Linear(256, 512),  # 256D embedding input
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, embedding_dim)
        )
        
    def _load_pretrained_query_encoder(self, pretrained_model_path):
        """Load and adapt pre-trained query encoder"""
        # Load checkpoint
        checkpoint = torch.load(pretrained_model_path, map_location='cpu', weights_only=False)
        
        # Create temporary collaborative model to extract weights
        temp_model = CollaborativePretrainingModel(
            num_llms=1131,
            embedding_dim=128,
            num_heads=4,
            dropout=0.2
        )
        temp_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Extract pre-trained components
        self.sentence_transformer = temp_model.sentence_transformer
        self.multi_head_attention = temp_model.multi_head_attention
        self.fusion = temp_model.fusion
        self.query_tower = temp_model.query_tower
        
        # Freeze sentence transformer (keep pre-trained representations)
        for param in self.sentence_transformer.parameters():
            param.requires_grad = False
            
        print("✓ Pre-trained query encoder loaded and sentence transformer frozen")
        print(f"✓ Pre-training validation loss was: {checkpoint['val_loss']:.4f}")
        
    def _initialize_fresh_model(self, num_llms, embedding_dim, num_heads, dropout):
        """Initialize fresh model if no pre-training available"""
        from sentence_transformers import SentenceTransformer
        
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        query_input_dim = self.sentence_transformer.get_sentence_embedding_dimension()
        
        # Multi-head attention (same as Config F)
        self.multi_head_attention = nn.ModuleList([
            nn.Sequential(
                nn.Linear(query_input_dim, 96),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(96, 96)
            ) for _ in range(num_heads)
        ])
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(num_heads * 96, query_input_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Query tower
        self.query_tower = nn.Sequential(
            nn.Linear(query_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 192),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(192, embedding_dim)
        )
    
    def encode_queries(self, query_texts):
        """Encode queries using pre-trained pipeline"""
        # Get sentence embeddings (always frozen for pre-trained)
        with torch.no_grad():
            embeddings = self.sentence_transformer.encode(
                query_texts, convert_to_tensor=True, device=next(self.parameters()).device
            )
        
        # Clone for gradient computation
        embeddings = embeddings.clone().detach().requires_grad_(True)
        
        # Multi-head attention (trainable)
        head_outputs = [head(embeddings) for head in self.multi_head_attention]
        concatenated = torch.cat(head_outputs, dim=1)
        fused = self.fusion(concatenated)
        
        # Query tower (trainable)
        query_repr = self.query_tower(fused)
        
        # Store head outputs for diversity loss
        if hasattr(self, '_store_head_outputs') and self._store_head_outputs:
            self._last_head_outputs = head_outputs
        
        return query_repr
    
    def encode_llms(self, llm_ids):
        """Encode LLM IDs (same as Config J)"""
        embedded = self.llm_embedding(llm_ids)
        return self.llm_tower(embedded)
    
    def predict_batch(self, query_texts, llm_encoded):
        """Batch prediction for evaluation"""
        query_embeddings = self.encode_queries(query_texts)
        
        # Check if llm_encoded needs to be processed through LLM tower
        if llm_encoded.shape[-1] != query_embeddings.shape[-1]:
            # llm_encoded contains raw LLM IDs, need to process them
            llm_embeddings = self.encode_llms(llm_encoded.long())
        else:
            # llm_encoded already processed
            llm_embeddings = llm_encoded
        
        # Compute similarities - element-wise for batch evaluation
        similarities = torch.cosine_similarity(
            query_embeddings, 
            llm_embeddings, 
            dim=1
        )
        
        return similarities.cpu().numpy()


def train_epoch_config_k(model, train_loader, optimizer, criterion, device, hard_miner=None):
    """Training epoch for Config K (Config J + massive synthetic)"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    # Enable head output storage for diversity loss
    model._store_head_outputs = True
    
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx == 0:
            print(f"    Starting training...") 
            
        optimizer.zero_grad()
        
        # Move data to device
        query_texts = batch['query_texts']
        positive_llms = batch['positive_llms'].to(device)
        negative_llms = batch['negative_llms'].to(device)
        
        # Update hard negative miner
        if hard_miner is not None and num_batches % 20 == 0:
            hard_miner.model = model
        
        # Get embeddings with pre-trained query encoder
        query_embeddings = model.encode_queries(query_texts)
        positive_embeddings = model.encode_llms(positive_llms)
        
        # Reshape negative LLMs for batch processing
        batch_size = positive_llms.size(0)
        neg_per_pos = negative_llms.size(0) // batch_size
        negative_llms_reshaped = negative_llms.view(batch_size, neg_per_pos)
        
        # Get negative embeddings
        negative_embeddings = []
        for i in range(batch_size):
            neg_embs = model.encode_llms(negative_llms_reshaped[i])
            negative_embeddings.append(neg_embs)
        negative_embeddings = torch.stack(negative_embeddings)
        
        # Get head outputs for diversity regularization
        head_outputs = getattr(model, '_last_head_outputs', None)
        
        # Compute contrastive loss with diversity regularization
        loss = criterion(query_embeddings, positive_embeddings, negative_embeddings, head_outputs)
        
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Progress indicator
        if batch_idx == 0:
            print(f"    First batch processed successfully")
        elif (batch_idx + 1) % 50 == 0:  # More frequent updates for massive dataset
            print(f"    Processed {batch_idx + 1}/{len(train_loader)} batches...")
    
    # Disable head output storage
    model._store_head_outputs = False
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def evaluate_model_config_k(model, val_loader, device):
    """Fast evaluation for Config K"""
    model.eval()
    
    print(f"  Evaluating on {len(val_loader)} batches...", end='', flush=True)
    
    # Collect predictions by query
    y_true_by_query = {}
    y_pred_by_query = {}
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            query_texts = batch['query_text']
            query_ids = batch['query_id']
            llm_encoded = batch['llm_encoded'].to(device)
            relevances = batch['relevance'].numpy()
            
            # Get predictions with pre-trained query encoder
            scores = model.predict_batch(query_texts, llm_encoded)
            
            # Group by query
            for i, query_id in enumerate(query_ids):
                if hasattr(query_id, 'item'):
                    query_id = query_id.item()
                if isinstance(query_id, (int, float)):
                    query_id = str(query_id)
                    
                if query_id not in y_true_by_query:
                    y_true_by_query[query_id] = []
                    y_pred_by_query[query_id] = []
                
                y_true_by_query[query_id].append(relevances[i])
                y_pred_by_query[query_id].append(scores[i])
            
            # Progress indicator
            if (batch_idx + 1) % 20 == 0:  # More frequent for massive dataset
                print(".", end='', flush=True)
    
    # Convert to numpy arrays
    for query_id in y_true_by_query.keys():
        y_true_by_query[query_id] = np.array(y_true_by_query[query_id])
        y_pred_by_query[query_id] = np.array(y_pred_by_query[query_id])
    
    # Calculate metrics using shared utilities
    metrics = calculate_metrics(y_true_by_query, y_pred_by_query)
    ndcg_10 = metrics['ndcg_10']
    ndcg_5 = metrics['ndcg_5']  
    mrr = metrics['mrr']
    
    print(f" Done! nDCG@10={ndcg_10:.4f}, MRR={mrr:.4f}")
    
    return ndcg_10, ndcg_5, mrr


def run_tier2_config_k():
    """Run Tier 2 evaluation with Config K (Config J + massive synthetic)"""
    print("=" * 80)
    print("TIER 2 ENHANCED TWO-TOWER - CONFIG K")
    print("=" * 80)
    print("Config K - ULTIMATE SCALE TEST:")
    print("  - Base: Config J architecture (256D LLM embeddings)")
    print("  - Extended training: 35 epochs with patience=15")
    print("  - MASSIVE synthetic dataset: 1.4M examples (73% synthetic)")
    print("  - 4 attention heads + batch size 96 (proven optimal)")
    print("  - Learning rate: 0.0008 with cosine annealing")
    print("  - Discovery data knowledge integrated")
    print()
    print("Hypothesis: State-of-the-art architecture + massive high-quality data = PEAK performance")
    print()
    
    # Use GPU if available for Config K peak performance
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cpu':
        print(f"CPU threads: {torch.get_num_threads()}")
    else:
        print("GPU acceleration enabled for ultimate performance")
    
    # Load MASSIVE synthetic dataset
    df = load_data('../../data/supervised_training_improved_synthetic.csv')
    
    print(f"MASSIVE DATASET LOADED:")
    print(f"  Total examples: {len(df):,}")
    print(f"  Unique queries: {df['query_id'].nunique():,}")
    print(f"  Unique LLMs: {df['llm_id'].nunique()}")
    
    # Show source breakdown
    if 'source' in df.columns:
        source_counts = df['source'].value_counts()
        for source, count in source_counts.items():
            percentage = count / len(df) * 100
            print(f"  {source}: {count:,} ({percentage:.1f}%)")
    
    # CONFIG K HYPERPARAMETERS (Same as Config J - proven optimal)
    n_folds = 10
    epochs = 35  # Extended training for massive dataset
    batch_size = 96  # Proven optimal
    base_lr = 0.0008  # Proven optimal
    num_heads = 4  # Proven optimal
    negative_samples = 4  # Proven optimal
    hard_ratio = 0.7  # Proven optimal
    temperature = 0.09  # Proven optimal
    diversity_weight = 0.015  # Proven optimal
    weight_decay = 5e-5  # Proven optimal
    dropout = 0.18  # Proven optimal
    pretrained_model_path = '../../data/collaborative_pretrained_model.pth'
    
    print(f"\nConfiguration: {epochs} epochs, batch_size={batch_size}, lr={base_lr}")
    print(f"Architecture: {num_heads} attention heads, {negative_samples} negative samples")
    print(f"Pre-training: {pretrained_model_path}")
    print(f"Expected: Config J architecture + massive scale = ULTIMATE performance")
    
    # Cross-validation setup - use original queries only for fair comparison
    original_queries = df[df['source'] == 'original']['query_id'].unique() if 'source' in df.columns else df['query_id'].unique()
    num_llms = df['llm_id'].nunique()
    
    print(f"\nValidation queries: {len(original_queries)} (fair comparison with baselines)")
    
    np.random.seed(42)
    np.random.shuffle(original_queries)
    
    kf = KFold(n_splits=n_folds, shuffle=False)
    fold_results = []
    total_start_time = time.time()
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(original_queries)):
        print(f"\n=== FOLD {fold+1}/{n_folds} ===")
        
        # Create datasets - training includes ALL data (original + synthetic)
        # Validation uses ONLY original queries for fair comparison
        train_queries = original_queries[train_idx]
        val_queries = original_queries[val_idx]
        
        # Training: original train queries + ALL synthetic data
        if 'source' in df.columns:
            original_train_mask = df['query_id'].isin(train_queries) & (df['source'] == 'original')
            synthetic_mask = df['source'] == 'query_aware_synthetic'
            train_mask = original_train_mask | synthetic_mask
            train_df = df[train_mask].copy()
        else:
            train_df = df[df['query_id'].isin(train_queries)].copy()
        
        # Validation: only original validation queries
        val_df = df[df['query_id'].isin(val_queries) & (df['source'] == 'original' if 'source' in df.columns else True)].copy()
        
        print(f"Train: {len(train_df):,} examples (including massive synthetic data)")
        print(f"Val: {len(val_df):,} examples (original queries only)")
        if 'source' in df.columns:
            train_sources = train_df['source'].value_counts()
            print(f"  Training sources: {train_sources.to_dict()}")
        
        fold_start = time.time()
        
        # Create optimized components
        hard_miner = HardNegativeMiner(model=None, hard_ratio=hard_ratio, temperature=temperature)
        
        train_loader, val_loader, _ = create_data_loaders(
            train_df, val_df, batch_size=batch_size, negative_samples=negative_samples
        )
        
        print(f"  Training batches: {len(train_loader)} (massive scale)")
        print(f"  Validation batches: {len(val_loader)}")
        
        # Create model with Config K architecture (Config J + massive data)
        model = PretrainedTwoTowerModel(
            num_llms=num_llms,
            embedding_dim=128,
            num_heads=num_heads,
            dropout=dropout,
            pretrained_model_path=pretrained_model_path
        ).to(device)
        
        hard_miner.model = model
        
        # Same successful loss and optimizer as Config J
        criterion = ContrastiveLossWithDiversity(temperature=temperature, diversity_weight=diversity_weight)
        optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
        
        # Learning rate scheduler - cosine annealing
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=base_lr/10)
        
        best_ndcg = -1
        best_metrics = {'ndcg_10': 0.0, 'ndcg_5': 0.0, 'mrr': 0.0, 'epoch': 0}
        patience = 15  # Extended patience for massive dataset
        patience_counter = 0
        
        # Training loop
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Activate hard mining after epoch 1
            active_miner = hard_miner if epoch >= 1 else None
            
            train_loss = train_epoch_config_k(model, train_loader, optimizer, criterion, device, active_miner)
            ndcg_10, ndcg_5, mrr = evaluate_model_config_k(model, val_loader, device)
            
            # Step the scheduler
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            
            if ndcg_10 > best_ndcg:
                best_ndcg = ndcg_10
                best_metrics = {'ndcg_10': ndcg_10, 'ndcg_5': ndcg_5, 'mrr': mrr, 'epoch': epoch}
                patience_counter = 0
            else:
                patience_counter += 1
            
            epoch_time = time.time() - epoch_start
            remaining = (epochs - epoch - 1) * epoch_time
            
            print(f"  Epoch {epoch+1}/{epochs}: Loss={train_loss:.4f}, nDCG@10={ndcg_10:.4f}, "
                  f"MRR={mrr:.4f}, LR={current_lr:.6f} ({epoch_time:.1f}s, ~{remaining/60:.1f}min left)")
            
            if ndcg_10 == best_ndcg:
                print(f"    *** New best: {best_ndcg:.4f} ***")
            
            # Early stopping
            if patience_counter >= patience and epoch >= 15:
                print(f"  Early stopping triggered after {epoch+1} epochs")
                break
        
        fold_time = time.time() - fold_start
        
        # Store results
        fold_results.append({
            'fold': fold + 1,
            'ndcg_10': best_metrics['ndcg_10'],
            'ndcg_5': best_metrics['ndcg_5'],
            'mrr': best_metrics['mrr'],
            'best_epoch': best_metrics['epoch'] + 1,
            'train_time': fold_time,
            'n_queries': len(val_queries),
            'train_examples': len(train_df),
            'val_examples': len(val_df)
        })
        
        print(f"\nFold {fold+1} Results:")
        print(f"  Best nDCG@10: {best_metrics['ndcg_10']:.4f} (epoch {best_metrics['epoch']+1})")
        print(f"  Best nDCG@5:  {best_metrics['ndcg_5']:.4f}")
        print(f"  Best MRR:     {best_metrics['mrr']:.4f}")
        print(f"  Time: {fold_time:.1f}s ({fold_time/60:.1f} minutes)")
        print(f"  Training scale: {len(train_df):,} examples")
    
    total_time = time.time() - total_start_time
    
    # Aggregate results
    ndcg_10_scores = [r['ndcg_10'] for r in fold_results]
    ndcg_5_scores = [r['ndcg_5'] for r in fold_results]
    mrr_scores = [r['mrr'] for r in fold_results]
    
    print("\n" + "=" * 80)
    print("FINAL RESULTS - CONFIG K (MASSIVE SYNTHETIC SCALE)")
    print("=" * 80)
    print(f"nDCG@10: {np.mean(ndcg_10_scores):.4f} ± {np.std(ndcg_10_scores):.4f}")
    print(f"  Min: {np.min(ndcg_10_scores):.4f}, Max: {np.max(ndcg_10_scores):.4f}")
    print(f"nDCG@5:  {np.mean(ndcg_5_scores):.4f} ± {np.std(ndcg_5_scores):.4f}")
    print(f"  Min: {np.min(ndcg_5_scores):.4f}, Max: {np.max(ndcg_5_scores):.4f}")
    print(f"MRR:     {np.mean(mrr_scores):.4f} ± {np.std(mrr_scores):.4f}")
    print(f"  Min: {np.min(mrr_scores):.4f}, Max: {np.max(mrr_scores):.4f}")
    print(f"\nTotal runtime: {total_time:.1f}s ({total_time/3600:.2f} hours)")
    print(f"Average per fold: {total_time/n_folds:.1f}s ({total_time/n_folds/60:.1f} minutes)")
    
    # Comparison with Config J
    config_j_ndcg = 0.4417  # Config J baseline result
    current_ndcg = np.mean(ndcg_10_scores)
    improvement = current_ndcg - config_j_ndcg
    
    print(f"\nMassive Scale Analysis:")
    print(f"  Config J (256D, normal scale):     {config_j_ndcg:.4f} nDCG@10")
    print(f"  Config K (256D, massive scale):    {current_ndcg:.4f} nDCG@10")
    print(f"  Massive scale benefit:             {improvement:+.4f}")
    
    if improvement > 0.01:
        print(f"  ✅ SUCCESS: Massive synthetic data significantly improves neural architecture!")
        scale_benefit_pct = improvement / config_j_ndcg * 100
        print(f"     {scale_benefit_pct:.1f}% improvement from massive scale")
    elif improvement > 0.005:
        print(f"  📈 MODEST GAIN: Massive scale provides small but meaningful improvement")
    else:
        print(f"  📊 PLATEAU: Architecture may have reached capacity limits")
    
    # Prepare standardized results
    results = {
        'evaluation_info': {
            'timestamp': datetime.now().isoformat(),
            'pipeline': 'Tier2 Config K (Massive Synthetic Scale)',
            'description': 'Config J (256D LLM embeddings) + massive synthetic dataset (1.4M examples)',
            'dataset': 'TREC 2025 Million LLMs Track - Massive Synthetic Augmented',
            'total_examples': len(df),
            'unique_queries': df['query_id'].nunique(),
            'unique_llms': num_llms,
            'synthetic_examples': len(df[df['source'] == 'query_aware_synthetic']) if 'source' in df.columns else 0,
            'original_examples': len(df[df['source'] == 'original']) if 'source' in df.columns else len(df),
            'synthetic_ratio': len(df[df['source'] == 'query_aware_synthetic']) / len(df) if 'source' in df.columns else 0.0,
            'hyperparameters': {
                'config': 'K',
                'base_config': 'J',
                'pretrained': True,
                'pretrained_model_path': pretrained_model_path,
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': base_lr,
                'num_heads': num_heads,
                'negative_samples': negative_samples,
                'hard_ratio': hard_ratio,
                'temperature': temperature,
                'diversity_weight': diversity_weight,
                'weight_decay': weight_decay,
                'dropout': dropout,
                'scheduler': 'CosineAnnealingLR'
            },
            'total_runtime_seconds': total_time,
            'total_runtime_hours': total_time / 3600
        },
        'performance_metrics': {
            'ndcg_10': {
                'mean': float(np.mean(ndcg_10_scores)),
                'std': float(np.std(ndcg_10_scores)),
                'min': float(np.min(ndcg_10_scores)),
                'max': float(np.max(ndcg_10_scores)),
                'confidence_interval_95': [
                    float(np.mean(ndcg_10_scores) - 1.96 * np.std(ndcg_10_scores) / np.sqrt(10)),
                    float(np.mean(ndcg_10_scores) + 1.96 * np.std(ndcg_10_scores) / np.sqrt(10))
                ]
            },
            'ndcg_5': {
                'mean': float(np.mean(ndcg_5_scores)),
                'std': float(np.std(ndcg_5_scores)),
                'min': float(np.min(ndcg_5_scores)),
                'max': float(np.max(ndcg_5_scores)),
                'confidence_interval_95': [
                    float(np.mean(ndcg_5_scores) - 1.96 * np.std(ndcg_5_scores) / np.sqrt(10)),
                    float(np.mean(ndcg_5_scores) + 1.96 * np.std(ndcg_5_scores) / np.sqrt(10))
                ]
            },
            'mrr': {
                'mean': float(np.mean(mrr_scores)),
                'std': float(np.std(mrr_scores)),
                'min': float(np.min(mrr_scores)),
                'max': float(np.max(mrr_scores)),
                'confidence_interval_95': [
                    float(np.mean(mrr_scores) - 1.96 * np.std(mrr_scores) / np.sqrt(10)),
                    float(np.mean(mrr_scores) + 1.96 * np.std(mrr_scores) / np.sqrt(10))
                ]
            }
        },
        'fold_by_fold_results': fold_results,
        'massive_scale_analysis': {
            'config_j_baseline': config_j_ndcg,
            'config_k_massive': float(current_ndcg),
            'massive_scale_benefit': float(improvement),
            'improvement_percentage': float(improvement / config_j_ndcg * 100),
            'dataset_scale_multiplier': len(df) / 386801  # vs original dataset
        }
    }
    
    # Save results
    save_standardized_results(results, 'tier2_config_k', '../../data/results/')
    print("\n✓ Results saved to data/results/tier2_config_k_results.json")
    
    print("\n" + "=" * 80)
    print("Config K (massive synthetic scale) evaluation complete!")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    results = run_tier2_config_k()