#!/usr/bin/env python3
"""
Tier 2 Enhanced Two-Tower - CPU OPTIMIZED VERSION

Temporarily forces CPU usage to avoid MPS compatibility issues,
but includes optimizations to run faster on CPU.
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
from shared.utils.evaluation import calculate_metrics

warnings.filterwarnings('ignore')

# Force CPU usage to avoid MPS hanging
torch.set_num_threads(4)  # Limit CPU threads for better performance


def train_epoch_optimized(model, train_loader, optimizer, criterion, device, hard_miner=None):
    """Optimized training epoch for CPU"""
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
        
        # Update hard negative miner less frequently for CPU efficiency
        if hard_miner is not None and num_batches % 30 == 0:
            hard_miner.model = model
        
        # Get embeddings with multi-head attention
        query_embeddings = model.encode_queries(query_texts)
        positive_embeddings = model.encode_llms(positive_llms)
        
        # Reshape negative LLMs for batch processing
        batch_size = positive_llms.size(0)
        neg_per_pos = negative_llms.size(0) // batch_size
        negative_llms_reshaped = negative_llms.view(batch_size, neg_per_pos)
        
        # Get negative embeddings [batch_size, num_negatives, embed_dim]
        negative_embeddings = []
        for i in range(batch_size):
            neg_embs = model.encode_llms(negative_llms_reshaped[i])
            negative_embeddings.append(neg_embs)
        negative_embeddings = torch.stack(negative_embeddings)
        
        # Get head outputs for diversity regularization
        head_outputs = getattr(model, '_last_head_outputs', None)
        
        # Compute contrastive loss with diversity regularization (reduced weight for CPU)
        loss = criterion(query_embeddings, positive_embeddings, negative_embeddings, head_outputs)
        
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Progress indicator for first epoch
        if batch_idx == 0:
            print(f"    First batch processed successfully")
        elif (batch_idx + 1) % 50 == 0:
            print(f"    Processed {batch_idx + 1}/{len(train_loader)} batches...")
    
    # Disable head output storage
    model._store_head_outputs = False
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def evaluate_model_fast(model, val_loader, device):
    """Fast evaluation for CPU"""
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
            
            # Get predictions with multi-head attention
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
            if (batch_idx + 1) % 25 == 0:
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


def run_tier2_cpu():
    """Run Tier 2 evaluation optimized for CPU"""
    print("="*80)
    print("TIER 2 ENHANCED TWIN TOWERS - CPU OPTIMIZED")
    print("="*80)
    print("Features: Multi-head attention + Active hard negative mining + Optimizations")
    print()
    
    # Force CPU device
    device = torch.device('cpu')
    print(f"Using device: {device} (CPU optimized)")
    print(f"CPU threads: {torch.get_num_threads()}")
    
    # Load data
    df = load_data('../../data/supervised_training_full.csv')
    
    # Parameters for fair comparison with Enhanced Tier 1 baseline
    n_folds = 10
    epochs = 20  # Match Enhanced Tier 1 for fair comparison
    batch_size = 32  # Smaller batch size for CPU
    learning_rate = 0.001
    
    print(f"Configuration: {epochs} epochs, batch_size={batch_size} (matching Tier 1 for fair comparison)")
    
    # Cross-validation setup
    unique_queries = df['query_id'].unique()
    num_llms = df['llm_id'].nunique()
    
    np.random.seed(42)
    np.random.shuffle(unique_queries)
    
    kf = KFold(n_splits=n_folds, shuffle=False)
    fold_results = []
    total_start_time = time.time()
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(unique_queries)):
        print(f"\n=== FOLD {fold+1}/{n_folds} ===")
        
        # Create datasets
        train_queries = unique_queries[train_idx]
        val_queries = unique_queries[val_idx]
        
        train_df = df[df['query_id'].isin(train_queries)].copy()
        val_df = df[df['query_id'].isin(val_queries)].copy()
        
        print(f"Train: {len(train_df)} examples, Val: {len(val_df)} examples")
        
        fold_start = time.time()
        
        # Create optimized components
        hard_miner = HardNegativeMiner(model=None, hard_ratio=0.6, temperature=0.1)  # Less aggressive
        
        train_loader, val_loader, _ = create_data_loaders(
            train_df, val_df, batch_size=batch_size, negative_samples=3  # Fewer negatives
        )
        
        # Create model with fewer heads for CPU efficiency
        model = create_model(num_llms, device, use_multi_head=True, num_heads=3)  # Reduced from 4
        hard_miner.model = model
        
        # Reduced diversity weight for CPU
        criterion = ContrastiveLossWithDiversity(temperature=0.1, diversity_weight=0.01)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        best_ndcg = -1
        best_metrics = {'ndcg_10': 0.0, 'ndcg_5': 0.0, 'mrr': 0.0, 'epoch': 0}
        
        # Training loop
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Activate hard mining after epoch 2
            active_miner = hard_miner if epoch >= 2 else None
            
            train_loss = train_epoch_optimized(model, train_loader, optimizer, criterion, device, active_miner)
            ndcg_10, ndcg_5, mrr = evaluate_model_fast(model, val_loader, device)
            
            if ndcg_10 > best_ndcg:
                best_ndcg = ndcg_10
                best_metrics = {'ndcg_10': ndcg_10, 'ndcg_5': ndcg_5, 'mrr': mrr, 'epoch': epoch}
            
            epoch_time = time.time() - epoch_start
            remaining = (epochs - epoch - 1) * epoch_time
            
            print(f"  Epoch {epoch+1}/{epochs}: Loss={train_loss:.4f}, nDCG@10={ndcg_10:.4f}, "
                  f"MRR={mrr:.4f} ({epoch_time:.1f}s, ~{remaining/60:.1f}min left)")
            
            if ndcg_10 == best_ndcg:
                print(f"    *** New best: {best_ndcg:.4f} ***")
        
        fold_time = time.time() - fold_start
        fold_results.append({
            'fold': fold + 1,
            'ndcg_10': best_metrics['ndcg_10'],
            'ndcg_5': best_metrics['ndcg_5'],
            'mrr': best_metrics['mrr'],
            'train_time': fold_time,
            'n_queries': len(val_queries),
            'best_epoch': best_metrics['epoch']
        })
        
        print(f"Fold {fold+1} complete ({fold_time/60:.1f}min): nDCG@10={best_metrics['ndcg_10']:.4f}")
        
        # Progress update
        current_ndcg = [r['ndcg_10'] for r in fold_results]
        current_mrr = [r['mrr'] for r in fold_results]
        elapsed = time.time() - total_start_time
        remaining_folds = n_folds - len(fold_results)
        est_remaining = (elapsed / len(fold_results)) * remaining_folds
        
        print(f"Progress: {len(fold_results)}/{n_folds} folds ({est_remaining/60:.1f}min remaining)")
        print(f"Current: nDCG@10={np.mean(current_ndcg):.4f}±{np.std(current_ndcg):.4f}")
        print("-" * 60)
    
    # Final results
    total_time = time.time() - total_start_time
    ndcg_10_scores = [r['ndcg_10'] for r in fold_results]
    ndcg_5_scores = [r['ndcg_5'] for r in fold_results]
    mrr_scores = [r['mrr'] for r in fold_results]
    
    results = {
        'evaluation_info': {
            'timestamp': datetime.now().isoformat(),
            'pipeline': 'Tier 2 CPU Optimized (Multi-head + Hard Negatives)',
            'dataset': 'TREC 2025 Million LLMs Track - Complete Dataset',
            'total_examples': len(df),
            'unique_queries': len(unique_queries),
            'unique_llms': num_llms,
            'epochs_per_fold': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'device': 'cpu_optimized',
            'optimizations': ['Reduced_heads', 'Smaller_batches', 'Less_frequent_hard_mining'],
            'total_runtime_seconds': round(total_time, 1),
            'total_runtime_hours': round(total_time / 3600, 2)
        },
        'performance_metrics': {
            'ndcg_10': {
                'mean': round(np.mean(ndcg_10_scores), 4),
                'std': round(np.std(ndcg_10_scores), 4),
                'min': round(np.min(ndcg_10_scores), 4),
                'max': round(np.max(ndcg_10_scores), 4),
                'confidence_interval_95': [
                    round(np.mean(ndcg_10_scores) - 1.96 * np.std(ndcg_10_scores), 4),
                    round(np.mean(ndcg_10_scores) + 1.96 * np.std(ndcg_10_scores), 4)
                ]
            },
            'ndcg_5': {
                'mean': round(np.mean(ndcg_5_scores), 4),
                'std': round(np.std(ndcg_5_scores), 4),
                'min': round(np.min(ndcg_5_scores), 4),
                'max': round(np.max(ndcg_5_scores), 4),
                'confidence_interval_95': [
                    round(np.mean(ndcg_5_scores) - 1.96 * np.std(ndcg_5_scores), 4),
                    round(np.mean(ndcg_5_scores) + 1.96 * np.std(ndcg_5_scores), 4)
                ]
            },
            'mrr': {
                'mean': round(np.mean(mrr_scores), 4),
                'std': round(np.std(mrr_scores), 4),
                'min': round(np.min(mrr_scores), 4),
                'max': round(np.max(mrr_scores), 4),
                'confidence_interval_95': [
                    round(np.mean(mrr_scores) - 1.96 * np.std(mrr_scores), 4),
                    round(np.mean(mrr_scores) + 1.96 * np.std(mrr_scores), 4)
                ]
            }
        },
        'fold_by_fold_results': fold_results
    }
    
    # Save results
    output_file = '../../data/results/tier2_cpu_optimized_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print("TIER 2 CPU OPTIMIZED EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"Final Results:")
    print(f"  nDCG@10: {np.mean(ndcg_10_scores):.4f} ± {np.std(ndcg_10_scores):.4f}")
    print(f"  MRR: {np.mean(mrr_scores):.4f} ± {np.std(mrr_scores):.4f}")
    print(f"  Runtime: {total_time/3600:.2f} hours")
    print(f"  Results: {output_file}")


if __name__ == "__main__":
    run_tier2_cpu()