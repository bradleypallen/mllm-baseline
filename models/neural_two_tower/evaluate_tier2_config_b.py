#!/usr/bin/env python3
"""
Tier 2 Enhanced Two-Tower - CONFIG B

Testing 8 attention heads:
- 8 attention heads (vs 6 in Config A)
- Batch size 64 (proven effective)
- Learning rate 0.0008 with cosine annealing
- All other successful tunings maintained

Hypothesis: More attention heads = better query representation = higher performance
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

warnings.filterwarnings('ignore')

# Force CPU usage with optimized thread count
torch.set_num_threads(8)


def train_epoch_config_b(model, train_loader, optimizer, criterion, device, hard_miner=None):
    """Training epoch with Config B optimizations (8 heads)"""
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
        
        # Update hard negative miner more frequently
        if hard_miner is not None and num_batches % 20 == 0:
            hard_miner.model = model
        
        # Get embeddings with 8-head attention
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
        elif (batch_idx + 1) % 30 == 0:
            print(f"    Processed {batch_idx + 1}/{len(train_loader)} batches...")
    
    # Disable head output storage
    model._store_head_outputs = False
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def evaluate_model_config_b(model, val_loader, device):
    """Fast evaluation"""
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
            
            # Get predictions with 8-head attention
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
            if (batch_idx + 1) % 15 == 0:
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


def run_tier2_config_b():
    """Run Tier 2 evaluation with Config B hyperparameters (8 heads)"""
    print("="*80)
    print("TIER 2 ENHANCED TWO-TOWER - CONFIG B")
    print("="*80)
    print("Config B Optimizations:")
    print("  - 8 attention heads (vs 6 in Config A)")
    print("  - Batch size: 64 (proven effective)")
    print("  - Learning rate: 0.0008 with cosine annealing")
    print("  - Hard negative mining: 0.7")
    print("  - Diversity weight: 0.015")
    print("  - 25 epochs with early stopping")
    print()
    print("Hypothesis: More attention heads → Better query representation → Higher performance")
    print()
    
    # Force CPU device
    device = torch.device('cpu')
    print(f"Using device: {device}")
    print(f"CPU threads: {torch.get_num_threads()}")
    
    # Load data
    df = load_data('../../data/supervised_training_full.csv')
    
    # CONFIG B HYPERPARAMETERS
    n_folds = 10
    epochs = 25
    batch_size = 64  # Keep successful setting
    base_lr = 0.0008  # Keep successful setting
    num_heads = 8  # INCREASED from 6
    negative_samples = 4  # Keep successful setting
    hard_ratio = 0.7  # Keep successful setting
    temperature = 0.09  # Keep successful setting
    diversity_weight = 0.015  # Keep successful setting
    weight_decay = 5e-5  # Keep successful setting
    dropout = 0.18  # Keep successful setting
    
    print(f"Configuration: {epochs} epochs, batch_size={batch_size}, lr={base_lr}")
    print(f"Architecture: {num_heads} attention heads, {negative_samples} negative samples")
    print(f"Expected: Slightly more compute due to 8 heads vs 6")
    
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
        hard_miner = HardNegativeMiner(model=None, hard_ratio=hard_ratio, temperature=temperature)
        
        train_loader, val_loader, _ = create_data_loaders(
            train_df, val_df, batch_size=batch_size, negative_samples=negative_samples
        )
        
        # Create model with Config B architecture (8 heads)
        model = TwoTowerModel(
            num_llms=num_llms,
            query_embedding_model='all-MiniLM-L6-v2',
            embedding_dim=128,
            dropout=dropout,
            use_multi_head=True,
            num_heads=num_heads  # 8 heads
        ).to(device)
        
        hard_miner.model = model
        
        # Same successful loss and optimizer
        criterion = ContrastiveLossWithDiversity(temperature=temperature, diversity_weight=diversity_weight)
        optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
        
        # Learning rate scheduler - cosine annealing
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=base_lr/10)
        
        best_ndcg = -1
        best_metrics = {'ndcg_10': 0.0, 'ndcg_5': 0.0, 'mrr': 0.0, 'epoch': 0}
        patience = 5
        patience_counter = 0
        
        # Training loop
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Activate hard mining after epoch 1
            active_miner = hard_miner if epoch >= 1 else None
            
            train_loss = train_epoch_config_b(model, train_loader, optimizer, criterion, device, active_miner)
            ndcg_10, ndcg_5, mrr = evaluate_model_config_b(model, val_loader, device)
            
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
            'n_queries': len(val_queries)
        })
        
        print(f"\nFold {fold+1} Results:")
        print(f"  Best nDCG@10: {best_metrics['ndcg_10']:.4f} (epoch {best_metrics['epoch']+1})")
        print(f"  Best nDCG@5:  {best_metrics['ndcg_5']:.4f}")
        print(f"  Best MRR:     {best_metrics['mrr']:.4f}")
        print(f"  Time: {fold_time:.1f}s ({fold_time/60:.1f} minutes)")
    
    total_time = time.time() - total_start_time
    
    # Aggregate results
    ndcg_10_scores = [r['ndcg_10'] for r in fold_results]
    ndcg_5_scores = [r['ndcg_5'] for r in fold_results]
    mrr_scores = [r['mrr'] for r in fold_results]
    
    print("\n" + "="*80)
    print("FINAL RESULTS - CONFIG B (8 HEADS)")
    print("="*80)
    print(f"nDCG@10: {np.mean(ndcg_10_scores):.4f} ± {np.std(ndcg_10_scores):.4f}")
    print(f"  Min: {np.min(ndcg_10_scores):.4f}, Max: {np.max(ndcg_10_scores):.4f}")
    print(f"nDCG@5:  {np.mean(ndcg_5_scores):.4f} ± {np.std(ndcg_5_scores):.4f}")
    print(f"  Min: {np.min(ndcg_5_scores):.4f}, Max: {np.max(ndcg_5_scores):.4f}")
    print(f"MRR:     {np.mean(mrr_scores):.4f} ± {np.std(mrr_scores):.4f}")
    print(f"  Min: {np.min(mrr_scores):.4f}, Max: {np.max(mrr_scores):.4f}")
    print(f"\nTotal runtime: {total_time:.1f}s ({total_time/3600:.2f} hours)")
    print(f"Average per fold: {total_time/n_folds:.1f}s ({total_time/n_folds/60:.1f} minutes)")
    
    # Comparison with previous configs
    config_a_best = 0.4542  # From what we've seen so far
    improvement = np.mean(ndcg_10_scores) - config_a_best
    print(f"\nComparison to Config A (6 heads):")
    print(f"  Config A peak: ~{config_a_best:.4f} nDCG@10")
    print(f"  Config B avg:  {np.mean(ndcg_10_scores):.4f} nDCG@10")
    print(f"  Change:        {improvement:+.4f} ({'improvement' if improvement > 0 else 'regression'})")
    
    # Prepare standardized results
    results = {
        'evaluation_info': {
            'timestamp': datetime.now().isoformat(),
            'pipeline': 'Tier2 CPU Optimized (Config B)',
            'description': 'Enhanced Two-Tower with Config B: 8 attention heads, batch size 64',
            'dataset': 'TREC 2025 Million LLMs Track - Complete Dataset',
            'total_examples': len(df),
            'unique_queries': len(unique_queries),
            'unique_llms': num_llms,
            'hyperparameters': {
                'config': 'B',
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
        'fold_by_fold_results': fold_results
    }
    
    # Save results
    save_standardized_results(results, 'tier2_cpu_optimized_config_b', '../../data/results/')
    print("\n✓ Results saved to data/results/tier2_cpu_optimized_config_b_results.json")
    
    print("\n" + "="*80)
    print("Config B (8 heads) evaluation complete!")
    print("="*80)
    
    return results


if __name__ == "__main__":
    results = run_tier2_config_b()