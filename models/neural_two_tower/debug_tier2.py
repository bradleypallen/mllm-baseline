#!/usr/bin/env python3
"""Debug script to find where Tier 2 evaluation hangs"""

import os
import sys

print("DEBUG: Starting script...")

# Add project root directory to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
print(f"DEBUG: Added {project_root} to path")

try:
    print("DEBUG: Importing basic modules...")
    import pandas as pd
    import numpy as np
    import torch
    print(f"DEBUG: PyTorch version: {torch.__version__}")
    
    print("DEBUG: Checking device availability...")
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"DEBUG: Using CUDA: {device}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"DEBUG: Using MPS: {device}")
    else:
        device = torch.device('cpu')
        print(f"DEBUG: Using CPU: {device}")
        
    print("DEBUG: Importing custom modules...")
    from model import create_model, ContrastiveLossWithDiversity
    print("DEBUG: Model imports successful")
    
    from data_loader import load_data, HardNegativeMiner
    print("DEBUG: Data loader imports successful")
    
    from shared.utils.evaluation import calculate_metrics
    print("DEBUG: Evaluation imports successful")
    
    print("DEBUG: Loading data...")
    df = load_data('../../data/supervised_training_full.csv')
    print(f"DEBUG: Data loaded: {len(df)} examples")
    
    print("DEBUG: Creating small test model...")
    test_model = create_model(100, device, use_multi_head=True, num_heads=4)
    print("DEBUG: Test model created successfully")
    
    print("DEBUG: Testing model forward pass...")
    test_queries = ["test query"]
    test_llm_ids = torch.tensor([1], device=device)
    scores = test_model.predict_batch(test_queries, test_llm_ids)
    print(f"DEBUG: Forward pass successful: {scores}")
    
    print("DEBUG: All components working! The issue might be in the training loop setup.")
    
except Exception as e:
    print(f"DEBUG: Error at import/setup stage: {e}")
    import traceback
    traceback.print_exc()