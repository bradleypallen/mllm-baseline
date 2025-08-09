#!/usr/bin/env python3
"""
Tier 3 Cross-Encoder Enhanced LLM Ranking with 10-Fold Cross-Validation

Tier 3 innovations over Tier 2:
- Cross-encoder with joint query-LLM encoding using transformer attention
- Direct query-LLM interaction modeling vs. dot-product similarity
- DistilBERT backbone for efficient yet powerful cross-attention
- Direct ranking optimization without two-stage retrieval for baseline comparison

This implements a pure cross-encoder approach to establish upper bound performance
before implementing the full two-stage retrieval+rerank pipeline.
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

from model import CrossEncoder, create_cross_encoder
from data_loader import create_data_loaders, load_data, LLMEvaluationDataset
from shared.utils.evaluation import calculate_metrics

warnings.filterwarnings('ignore')


def train_epoch_cross_encoder(model, train_loader, optimizer, criterion, device, epoch_num):
    """Train cross-encoder model for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx == 0 and epoch_num == 1:
            print(f"    Processing first batch of epoch {epoch_num}...")
            
        optimizer.zero_grad()
        
        # Extract data from batch
        query_texts = batch['query_texts']
        positive_llms = batch['positive_llms'].to(device)
        negative_llms = batch['negative_llms'].to(device)
        
        batch_size = len(query_texts)
        neg_per_pos = negative_llms.size(0) // batch_size
        
        # Create training pairs: each query with its positive and negative LLMs
        all_queries = []
        all_llm_ids = []
        all_labels = []
        
        for i in range(batch_size):
            query_text = query_texts[i]
            pos_llm = positive_llms[i]
            
            # Add positive pair
            all_queries.append(query_text)
            all_llm_ids.append(pos_llm.item())
            all_labels.append(1.0)  # Positive label
            
            # Add negative pairs
            neg_start = i * neg_per_pos
            neg_end = (i + 1) * neg_per_pos
            neg_llms = negative_llms[neg_start:neg_end]
            
            for neg_llm in neg_llms:
                all_queries.append(query_text)
                all_llm_ids.append(neg_llm.item())
                all_labels.append(0.0)  # Negative label
        
        # Convert to tensors
        all_llm_ids = torch.tensor(all_llm_ids, device=device)
        all_labels = torch.tensor(all_labels, device=device)
        
        # Get model predictions
        scores = model(all_queries, all_llm_ids)
        
        # Binary classification loss (BCE with logits)
        loss = criterion(scores, all_labels)
        
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Progress indicator for first epoch
        if batch_idx == 0 and epoch_num == 1:
            print(f"    First batch processed successfully")
        elif (batch_idx + 1) % 30 == 0:
            print(f"    Processed {batch_idx + 1}/{len(train_loader)} batches...")
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def evaluate_cross_encoder(model, val_loader, device):
    """Evaluate cross-encoder model on validation set"""
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
            
            # Get cross-encoder predictions
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
            if (batch_idx + 1) % 20 == 0:
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


def train_fold_cross_encoder(train_df, val_df, num_llms, fold_num, epochs=15, batch_size=48, 
                            learning_rate=2e-5, device='cpu'):
    """Train and evaluate cross-encoder model on a single fold"""
    print(f"\n=== FOLD {fold_num} (Cross-Encoder) ===")
    fold_start_time = time.time()
    
    print("Creating data loaders...")
    
    # Create data loaders (no hard negative mining for cross-encoder baseline)
    train_loader, val_loader, llm_encoder = create_data_loaders(
        train_df, val_df, batch_size=batch_size, negative_samples=2  # Fewer negatives for cross-encoder
    )
    print(f"Data loaders created. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    print("Creating Tier 3 cross-encoder model...")
    # Create cross-encoder model
    model = create_cross_encoder(num_llms, device, model_name='distilbert-base-uncased')
    print("Cross-encoder model created and moved to device")
    
    # Setup training with BCE loss for binary relevance prediction
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=2)
    
    best_ndcg = -1
    best_metrics = {
        'ndcg_10': 0.0,
        'ndcg_5': 0.0, 
        'mrr': 0.0,
        'epoch': 0
    }
    
    print("Starting Tier 3 cross-encoder training...")
    print("Features: Joint query-LLM encoding with transformer attention")
    
    for epoch in range(epochs):
        print(f"  Epoch {epoch+1}/{epochs}:")
        epoch_train_start = time.time()
        
        # Train cross-encoder
        train_loss = train_epoch_cross_encoder(
            model, train_loader, optimizer, criterion, device, epoch+1
        )
        
        # Evaluate
        ndcg_10, ndcg_5, mrr = evaluate_cross_encoder(model, val_loader, device)
        
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


def run_cross_validation_tier3(df, n_folds=10, epochs=15, batch_size=48, learning_rate=2e-5):
    """Run 10-fold cross-validation with Tier 3 Cross-Encoder model"""
    print("="*80)
    print("TIER 3 CROSS-ENCODER - 10-FOLD CROSS-VALIDATION")
    print("="*80)
    print("Tier 3 Cross-Encoder Features:")
    print("  🔗 Joint query-LLM encoding with transformer attention")
    print("  🧠 DistilBERT backbone for efficient cross-attention")
    print("  📊 Direct relevance prediction with BCE loss")
    print("  ⚡ Memory-optimized: batch_size=48 for 24GB unified memory")
    print("  🚀 Estimated runtime: ~6-7 hours (3x faster than batch_size=16)")
    print()
    
    # Device selection with optimization for transformer models
    device_selected = False
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using device: {device} (NVIDIA GPU - optimal for transformers)")
        device_selected = True
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # Test MPS compatibility
        try:
            test_tensor = torch.randn(10, 10, device='mps')
            test_result = torch.matmul(test_tensor, test_tensor)
            device = torch.device('mps')
            print(f"Using device: {device} (Apple Silicon GPU)")
            device_selected = True
        except Exception as e:
            print(f"MPS compatibility issue: {e}")
            print("Falling back to CPU...")
            device = torch.device('cpu')
            device_selected = True
    
    if not device_selected:
        device = torch.device('cpu')
        print(f"Using device: {device} (CPU)")
    
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
        train_mask = df['query_id'].isin(train_queries)
        train_df = df[train_mask].copy()
        
        val_mask = df['query_id'].isin(val_queries)  
        val_df = df[val_mask].copy()
        
        print(f"Datasets created: Train={len(train_df)}, Val={len(val_df)}")
        
        # Train and evaluate fold
        fold_result = train_fold_cross_encoder(
            train_df, val_df, num_llms, fold+1, epochs, batch_size, learning_rate, device
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
            'pipeline': 'Tier 3 Cross-Encoder (Joint Query-LLM Encoding)',
            'dataset': 'TREC 2025 Million LLMs Track - Complete Dataset',
            'total_examples': len(df),
            'unique_queries': len(unique_queries),
            'unique_llms': num_llms,
            'epochs_per_fold': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'device': str(device),
            'tier_3_features': [
                'Cross_encoder_transformer',
                'Joint_query_LLM_encoding', 
                'DistilBERT_backbone',
                'Direct_relevance_prediction',
                'Fine_tuned_attention'
            ],
            'model_config': {
                'transformer_model': 'distilbert-base-uncased',
                'max_sequence_length': 512,
                'dropout': 0.1,
                'loss_function': 'BCEWithLogitsLoss'
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
    output_file = '../../data/results/tier3_cross_encoder_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print("TIER 3 CROSS-ENCODER EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"🎯 Tier 3 Cross-Encoder Results:")
    print(f"   nDCG@10: {np.mean(ndcg_10_scores):.4f} ± {np.std(ndcg_10_scores):.4f}")
    print(f"   MRR: {np.mean(mrr_scores):.4f} ± {np.std(mrr_scores):.4f}")
    print(f"   Total runtime: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"   Results saved to: {output_file}")
    print(f"\n🔗 Cross-Encoder: Joint query-LLM encoding with DistilBERT")
    print(f"📊 Direct Optimization: BCE loss for relevance prediction") 
    print(f"⚡ Memory-Optimized: batch_size={batch_size} for 24GB unified memory")
    print(f"🚀 Performance: ~3x faster than batch_size=16, better gradient estimates")
    
    return results


if __name__ == "__main__":
    # Load data
    df = load_data('../../data/supervised_training_full.csv')
    
    # Run Tier 3 cross-encoder cross-validation  
    results = run_cross_validation_tier3(df, n_folds=10, epochs=15, batch_size=48, 
                                       learning_rate=2e-5)