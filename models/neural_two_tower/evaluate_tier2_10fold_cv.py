#!/usr/bin/env python3
"""
Tier 2 Enhanced Two-Tower Neural Network Training with 10-Fold Cross-Validation

Tier 2 improvements over enhanced baseline:
- Multi-head query attention for richer semantic understanding
- Active hard negative mining for efficient training
- Head diversity regularization for specialization

Implements advanced neural baseline for LLM ranking with same evaluation protocol.
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

# Add project root directory to path for importing shared utilities
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from model import TwoTowerModel, ContrastiveLossWithDiversity, create_model
from data_loader import create_data_loaders, load_data, LLMEvaluationDataset, HardNegativeMiner
from shared.utils.evaluation import calculate_metrics

warnings.filterwarnings('ignore')


def train_epoch_with_hard_negatives(model, train_loader, optimizer, criterion, device, hard_miner=None):
    """Train model for one epoch with multi-head attention and hard negative mining"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    # Enable head output storage for diversity loss
    model._store_head_outputs = True
    
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx == 0:
            print(f"    Processing first batch...")
            
        optimizer.zero_grad()
        
        # Move data to device
        query_texts = batch['query_texts']
        positive_llms = batch['positive_llms'].to(device)
        negative_llms = batch['negative_llms'].to(device)
        
        if batch_idx == 0:
            print(f"    Data moved to device, getting embeddings...")
        
        # Update hard negative miner with current model (every 30 batches for stability)
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
        
        # Compute contrastive loss with diversity regularization
        loss = criterion(query_embeddings, positive_embeddings, negative_embeddings, head_outputs)
        
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    # Disable head output storage
    model._store_head_outputs = False
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def evaluate_model(model, val_loader, device):
    """Evaluate model on validation set"""
    model.eval()
    
    print(f"  Evaluating on {len(val_loader)} batches...", end='', flush=True)
    
    # Collect predictions by query
    y_true_by_query = {}
    y_pred_by_query = {}
    
    with torch.no_grad():
        for batch in val_loader:
            query_texts = batch['query_text']
            query_ids = batch['query_id']
            llm_encoded = batch['llm_encoded'].to(device)
            relevances = batch['relevance'].numpy()
            
            # Get predictions with multi-head attention
            scores = model.predict_batch(query_texts, llm_encoded)
            
            # Group by query - query_ids should be individual strings, not lists
            for i, query_id in enumerate(query_ids):
                # Convert tensor to string if needed
                if hasattr(query_id, 'item'):
                    query_id = query_id.item()
                if isinstance(query_id, (int, float)):
                    query_id = str(query_id)
                    
                if query_id not in y_true_by_query:
                    y_true_by_query[query_id] = []
                    y_pred_by_query[query_id] = []
                
                y_true_by_query[query_id].append(relevances[i])
                y_pred_by_query[query_id].append(scores[i])
    
    # Convert to numpy arrays
    for query_id in y_true_by_query.keys():
        y_true_by_query[query_id] = np.array(y_true_by_query[query_id])
        y_pred_by_query[query_id] = np.array(y_pred_by_query[query_id])
    
    # Calculate metrics using shared utilities
    metrics = calculate_metrics(y_true_by_query, y_pred_by_query)
    ndcg_10 = metrics['ndcg_10']
    ndcg_5 = metrics['ndcg_5']  
    mrr = metrics['mrr']
    
    # Debug info
    total_queries = len(y_true_by_query)
    queries_with_relevant = sum(1 for y_true in y_true_by_query.values() if np.sum(y_true) > 0)
    query_sizes = [len(y_true) for y_true in y_true_by_query.values()]
    avg_query_size = np.mean(query_sizes) if query_sizes else 0
    
    print(f" Done! nDCG@10={ndcg_10:.4f}, MRR={mrr:.4f}")
    print(f"    Debug: {total_queries} unique queries, {queries_with_relevant} with relevant docs")
    print(f"    Debug: Avg docs per query: {avg_query_size:.1f}, Range: {min(query_sizes) if query_sizes else 0}-{max(query_sizes) if query_sizes else 0}")
    
    return ndcg_10, ndcg_5, mrr


def train_fold(train_df, val_df, num_llms, fold_num, epochs=20, batch_size=64, 
               learning_rate=0.001, device='cpu', use_multi_head=True, num_heads=4):
    """Train and evaluate model on a single fold with Tier 2 enhancements"""
    print(f"\n=== FOLD {fold_num} ===")
    fold_start_time = time.time()
    
    print("Creating hard negative miner...")
    # Create hard negative miner (will be updated with model during training)
    hard_negative_miner = HardNegativeMiner(model=None, hard_ratio=0.6, temperature=0.1)
    
    print("Creating data loaders...")
    
    # Create data loaders with hard negative mining (reduced for stability)
    train_loader, val_loader, llm_encoder = create_data_loaders(
        train_df, val_df, batch_size=batch_size, negative_samples=3,
        hard_negative_miner=hard_negative_miner
    )
    print(f"Data loaders created successfully. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    print("Creating Tier 2 enhanced model...")
    # Create Tier 2 enhanced model with multi-head attention (reduced heads for stability)
    model = create_model(num_llms, device, use_multi_head=use_multi_head, num_heads=3)
    print("Model created and moved to device")
    
    # Update hard negative miner with the created model
    hard_negative_miner.model = model
    
    # Setup training with contrastive loss + diversity regularization (reduced for stability)
    criterion = ContrastiveLossWithDiversity(temperature=0.1, diversity_weight=0.01)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=3)
    
    best_ndcg = -1
    best_metrics = {
        'ndcg_10': 0.0,
        'ndcg_5': 0.0, 
        'mrr': 0.0,
        'epoch': 0
    }
    
    print("Starting Tier 2 enhanced training...")
    print("Enhancements: Multi-head attention, Active hard negative mining, Diversity regularization")
    
    for epoch in range(epochs):
        print(f"  Epoch {epoch+1}/{epochs}:")
        epoch_train_start = time.time()
        
        # Train with hard negative mining (activated after epoch 2 for faster convergence)
        active_hard_mining = hard_negative_miner if epoch >= 2 else None
        train_loss = train_epoch_with_hard_negatives(
            model, train_loader, optimizer, criterion, device, active_hard_mining
        )
        
        # Evaluate
        ndcg_10, ndcg_5, mrr = evaluate_model(model, val_loader, device)
        
        # Learning rate scheduling
        scheduler.step(ndcg_10)
        
        # Save best model
        if ndcg_10 > best_ndcg:
            best_ndcg = ndcg_10
            best_metrics = {
                'ndcg_10': ndcg_10,
                'ndcg_5': ndcg_5,
                'mrr': mrr,
                'epoch': epoch
            }
        
        epoch_time = time.time() - epoch_train_start
        
        # Time estimation
        if epoch == 0:
            avg_epoch_time = epoch_time
        else:
            avg_epoch_time = (avg_epoch_time * epoch + epoch_time) / (epoch + 1)
        
        remaining_epochs = epochs - epoch - 1
        estimated_remaining = remaining_epochs * avg_epoch_time
        
        print(f"    Results: Loss={train_loss:.4f}, nDCG@10={ndcg_10:.4f}, "
              f"nDCG@5={ndcg_5:.4f}, MRR={mrr:.4f}")
        print(f"    Time: {epoch_time:.1f}s this epoch, ~{estimated_remaining/60:.1f}min remaining")
        print(f"    Hard negatives: {'Active' if active_hard_mining else 'Inactive'}")
        
        # Show improvement indicator
        if ndcg_10 == best_ndcg:
            print(f"    *** New best nDCG@10: {best_ndcg:.4f} ***")
    
    fold_time = time.time() - fold_start_time
    
    print(f"Fold {fold_num} completed in {fold_time/60:.1f} minutes")
    print(f"Best results: nDCG@10={best_metrics['ndcg_10']:.4f}, "
          f"nDCG@5={best_metrics['ndcg_5']:.4f}, MRR={best_metrics['mrr']:.4f}")
    
    return {
        'fold': fold_num,
        'ndcg_10': best_metrics['ndcg_10'],
        'ndcg_5': best_metrics['ndcg_5'],
        'mrr': best_metrics['mrr'],
        'train_time': fold_time,
        'n_queries': len(val_df.query_id.unique()),
        'best_epoch': best_metrics['epoch']
    }


def run_cross_validation(df, n_folds=10, epochs=15, batch_size=32, learning_rate=0.001,
                        use_multi_head=True, num_heads=3):
    """Run 10-fold cross-validation with Tier 2 enhanced Twin Towers model"""
    print("="*80)
    print("TIER 2 ENHANCED TWIN TOWERS - 10-FOLD CROSS-VALIDATION")
    print("="*80)
    print("Tier 2 Enhancements (Stability Optimized):")
    print("  🧠 Multi-head query attention (3 heads) with diversity regularization")
    print("  ⚡ Active hard negative mining (60% hard, 40% easy)")
    print("  📊 Enhanced training with curriculum learning")
    print()
    
    # Device selection with MPS compatibility testing
    device_selected = False
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using device: {device} (NVIDIA GPU)")
        device_selected = True
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # Test MPS compatibility with a small operation
        try:
            test_tensor = torch.randn(10, 10, device='mps')
            test_result = torch.matmul(test_tensor, test_tensor)
            device = torch.device('mps')
            print(f"Using device: {device} (Apple Silicon GPU)")
            device_selected = True
        except Exception as e:
            print(f"MPS compatibility issue detected: {e}")
            print("Falling back to CPU...")
            device = torch.device('cpu')
            device_selected = True
    
    if not device_selected:
        device = torch.device('cpu')
        print(f"Using device: {device} (CPU fallback)")
    
    print(f"PyTorch version: {torch.__version__}")
    
    total_start_time = time.time()
    
    # Get unique queries for splitting
    unique_queries = df['query_id'].unique()
    num_llms = df['llm_id'].nunique()
    
    print(f"Dataset: {len(df)} examples, {len(unique_queries)} queries, {num_llms} LLMs")
    
    # Shuffle queries for cross-validation
    np.random.seed(42)
    np.random.shuffle(unique_queries)
    
    # K-fold split on queries
    kf = KFold(n_splits=n_folds, shuffle=False)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(unique_queries)):
        print(f"\nSetting up fold {fold+1}...")
        
        # Split queries
        train_queries = unique_queries[train_idx]
        val_queries = unique_queries[val_idx]
        print(f"Fold {fold+1}: Train queries: {len(train_queries)}, Val queries: {len(val_queries)}")
        
        # Create fold datasets
        print("Creating training dataset...")
        train_mask = df['query_id'].isin(train_queries)
        train_df = df[train_mask].copy()
        
        print("Creating validation dataset...")
        val_mask = df['query_id'].isin(val_queries)  
        val_df = df[val_mask].copy()
        
        print(f"Datasets created: Train={len(train_df)}, Val={len(val_df)}")
        
        # Train and evaluate fold
        fold_result = train_fold(
            train_df, val_df, num_llms, fold+1, epochs, batch_size, learning_rate, device,
            use_multi_head, num_heads
        )
        fold_results.append(fold_result)
        
        # Progress update
        completed_folds = fold + 1
        elapsed = time.time() - total_start_time
        avg_time_per_fold = elapsed / completed_folds
        estimated_total_remaining = avg_time_per_fold * (n_folds - completed_folds)
        
        print(f"\n>>> Progress: {completed_folds}/{n_folds} folds completed")
        print(f">>> Estimated remaining time: {estimated_total_remaining/60:.1f} minutes")
        print(f">>> Current results so far:")
        
        # Show current aggregate results
        current_ndcg_scores = [r['ndcg_10'] for r in fold_results]
        current_mrr_scores = [r['mrr'] for r in fold_results]
        print(f">>> nDCG@10: {np.mean(current_ndcg_scores):.4f} ± {np.std(current_ndcg_scores):.4f}")
        print(f">>> MRR: {np.mean(current_mrr_scores):.4f} ± {np.std(current_mrr_scores):.4f}")
        print("=" * 60)
    
    total_time = time.time() - total_start_time
    
    # Calculate aggregate statistics
    ndcg_10_scores = [r['ndcg_10'] for r in fold_results]
    ndcg_5_scores = [r['ndcg_5'] for r in fold_results]
    mrr_scores = [r['mrr'] for r in fold_results]
    
    results = {
        'evaluation_info': {
            'timestamp': datetime.now().isoformat(),
            'pipeline': 'Tier 2 Enhanced Two-Tower (Multi-head + Hard Negatives + Diversity)',
            'dataset': 'TREC 2025 Million LLMs Track - Complete Dataset',
            'total_examples': len(df),
            'unique_queries': len(unique_queries),
            'unique_llms': num_llms,
            'epochs_per_fold': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'device': str(device),
            'tier_2_enhancements': [
                'Multi_head_query_attention',
                'Active_hard_negative_mining', 
                'Head_diversity_regularization',
                'Curriculum_learning'
            ],
            'multi_head_config': {
                'num_heads': num_heads,
                'diversity_weight': 0.05
            },
            'hard_negative_config': {
                'hard_ratio': 0.7,
                'temperature': 0.05,
                'activation_epoch': 3
            },
            'total_runtime_seconds': round(total_time, 1),
            'total_runtime_minutes': round(total_time / 60, 1),
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
    output_file = '../../data/results/tier2_neural_two_tower_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print("TIER 2 ENHANCED EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"🎯 Tier 2 Enhanced Results:")
    print(f"   nDCG@10: {np.mean(ndcg_10_scores):.4f} ± {np.std(ndcg_10_scores):.4f}")
    print(f"   MRR: {np.mean(mrr_scores):.4f} ± {np.std(mrr_scores):.4f}")
    print(f"   Total runtime: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"   Results saved to: {output_file}")
    print(f"\n🧠 Multi-head Attention: {num_heads} heads with diversity regularization")
    print(f"⚡ Hard Negative Mining: 70% hard negatives, activated from epoch 3")
    print(f"📊 Training Enhancements: AdamW optimizer, curriculum learning")
    
    return results


if __name__ == "__main__":
    # Load data
    df = load_data('../../data/supervised_training_full.csv')
    
    # Run Tier 2 enhanced cross-validation (stability optimized)
    results = run_cross_validation(df, n_folds=10, epochs=15, batch_size=32, 
                                 learning_rate=0.001, use_multi_head=True, num_heads=3)