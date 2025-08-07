#!/usr/bin/env python3
"""
Two-Tower Neural Network Training with 10-Fold Cross-Validation

Implements the neural baseline for LLM ranking using a Two-Tower architecture
with the same evaluation protocol as the Random Forest baseline.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import ndcg_score
import time
import json
from datetime import datetime
import warnings
import os
import sys

# Add parent directory to path for importing data_loader and model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import TwoTowerModel, RankingLoss, create_model
from data_loader import create_data_loaders, load_data, LLMEvaluationDataset

warnings.filterwarnings('ignore')

def mean_reciprocal_rank(y_true_by_query, y_pred_by_query):
    """Calculate Mean Reciprocal Rank"""
    reciprocal_ranks = []
    
    for query_id in y_true_by_query.keys():
        y_true = np.array(y_true_by_query[query_id])
        y_pred = np.array(y_pred_by_query[query_id])
        
        # Sort by predicted scores (descending)
        sorted_indices = np.argsort(y_pred)[::-1]
        sorted_true = y_true[sorted_indices]
        
        # Find first relevant item (relevance > 0)
        relevant_indices = np.where(sorted_true > 0)[0]
        if len(relevant_indices) > 0:
            first_relevant_rank = relevant_indices[0] + 1  # 1-indexed
            reciprocal_ranks.append(1.0 / first_relevant_rank)
        else:
            reciprocal_ranks.append(0.0)
    
    return np.mean(reciprocal_ranks)


def evaluate_ndcg_at_k(y_true_by_query, y_pred_by_query, k=10):
    """Calculate nDCG@k across all queries"""
    ndcg_scores = []
    skipped_queries = 0
    
    for query_id in y_true_by_query.keys():
        y_true = y_true_by_query[query_id]
        y_pred = y_pred_by_query[query_id]
        
        # Need at least 2 documents for meaningful nDCG, but handle edge cases
        if len(y_true) < 2:
            skipped_queries += 1
            continue
        
        # Check if there's any relevance signal (not all zeros)
        if np.sum(y_true) == 0:
            # No relevant documents, nDCG is 0
            ndcg_scores.append(0.0)
        else:
            try:
                ndcg = ndcg_score([y_true], [y_pred], k=k)
                ndcg_scores.append(ndcg)
            except ValueError as e:
                # Handle any other sklearn nDCG issues
                skipped_queries += 1
                continue
    
    if len(ndcg_scores) == 0:
        print(f"    WARNING: No valid queries for nDCG calculation (skipped {skipped_queries})")
        return 0.0
    
    return np.mean(ndcg_scores)


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train model for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    print(f"    Training on {len(train_loader)} batches...", end='', flush=True)
    
    for batch_idx, batch in enumerate(train_loader):
        # Move data to device
        positive_llms = batch['positive_llms'].to(device)
        negative_llms = batch['negative_llms'].to(device)
        
        # Get query texts (list of strings)
        query_texts = batch['query_texts']
        
        # Repeat query texts for negative samples
        batch_size = len(query_texts)
        neg_samples_per_query = len(negative_llms) // batch_size
        repeated_queries = []
        for query in query_texts:
            repeated_queries.extend([query] * neg_samples_per_query)
        
        # Forward pass
        positive_scores = model(query_texts, positive_llms)
        negative_scores = model(repeated_queries, negative_llms)
        
        # Reshape negative scores to match positive scores
        negative_scores = negative_scores.view(batch_size, neg_samples_per_query)
        
        # Calculate loss for each positive-negative pair
        loss = 0
        for i in range(batch_size):
            pos_score = positive_scores[i]
            neg_scores = negative_scores[i]
            
            # Pairwise ranking loss
            for neg_score in neg_scores:
                loss += criterion(pos_score.unsqueeze(0), neg_score.unsqueeze(0))
        
        loss = loss / (batch_size * neg_samples_per_query)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Progress indicator every 100 batches
        if (batch_idx + 1) % 100 == 0:
            print(f".", end='', flush=True)
    
    print(f" Done! Avg loss: {total_loss/num_batches:.4f}")
    return total_loss / num_batches


def evaluate_model(model, val_loader, device):
    """Evaluate model and return nDCG and MRR scores"""
    model.eval()
    
    print(f"    Evaluating on {len(val_loader)} batches...", end='', flush=True)
    
    # Collect predictions by query
    y_true_by_query = {}
    y_pred_by_query = {}
    
    with torch.no_grad():
        for batch in val_loader:
            query_texts = batch['query_text']
            query_ids = batch['query_id']
            llm_encoded = batch['llm_encoded'].to(device)
            relevances = batch['relevance'].numpy()
            
            # Get predictions
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
    
    # Calculate metrics
    ndcg_10 = evaluate_ndcg_at_k(y_true_by_query, y_pred_by_query, k=10)
    ndcg_5 = evaluate_ndcg_at_k(y_true_by_query, y_pred_by_query, k=5)
    mrr = mean_reciprocal_rank(y_true_by_query, y_pred_by_query)
    
    # Debug info for first few epochs
    total_queries = len(y_true_by_query)
    queries_with_relevant = sum(1 for y_true in y_true_by_query.values() if np.sum(y_true) > 0)
    
    # Show query size distribution for debugging
    query_sizes = [len(y_true) for y_true in y_true_by_query.values()]
    avg_query_size = np.mean(query_sizes) if query_sizes else 0
    
    print(f" Done! nDCG@10={ndcg_10:.4f}, MRR={mrr:.4f}")
    print(f"    Debug: {total_queries} unique queries, {queries_with_relevant} with relevant docs")
    print(f"    Debug: Avg docs per query: {avg_query_size:.1f}, Range: {min(query_sizes) if query_sizes else 0}-{max(query_sizes) if query_sizes else 0}")
    
    return ndcg_10, ndcg_5, mrr


def train_fold(fold_num, train_df, val_df, num_llms, device, 
               epochs=50, batch_size=32, learning_rate=0.001):
    """Train and evaluate model for one fold"""
    print(f"\n=== FOLD {fold_num} ===")
    print(f"Train: {len(train_df)} examples, Val: {len(val_df)} examples")
    
    fold_start_time = time.time()
    
    # Create data loaders
    train_loader, val_loader, llm_encoder = create_data_loaders(
        train_df, val_df, batch_size=batch_size, negative_samples=4
    )
    
    # Create model
    model = create_model(num_llms, device)
    
    # Setup training
    criterion = RankingLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=5)
    
    best_ndcg = -1
    best_metrics = {
        'ndcg_10': 0.0,
        'ndcg_5': 0.0, 
        'mrr': 0.0,
        'epoch': 0
    }
    
    print("Starting training...")
    epoch_start_time = time.time()
    
    for epoch in range(epochs):
        print(f"  Epoch {epoch+1}/{epochs}:")
        epoch_train_start = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
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
        
        # Calculate time estimates
        epoch_time = time.time() - epoch_train_start
        if epoch == 0:
            avg_epoch_time = epoch_time
        else:
            avg_epoch_time = (time.time() - epoch_start_time) / (epoch + 1)
        
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
        'n_queries': val_df.query_id.nunique(),
        'best_epoch': best_metrics['epoch']
    }


def run_cross_validation(df, n_folds=10, epochs=50, batch_size=32, learning_rate=0.001):
    """Run 10-fold cross-validation"""
    print("=== TWO-TOWER NEURAL NETWORK - 10-FOLD CROSS-VALIDATION ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get unique queries for CV splitting
    unique_queries = df.query_id.unique()
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(unique_queries)
    
    num_llms = df.llm_id.nunique()
    print(f"Dataset: {len(df)} examples, {len(unique_queries)} queries, {num_llms} LLMs")
    print(f"Configuration: {n_folds} folds, {epochs} epochs/fold, batch_size={batch_size}")
    
    # Cross-validation setup
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results = []
    
    total_start_time = time.time()
    
    print(f"\nStarting {n_folds}-fold cross-validation...")
    
    for fold_num, (train_idx, val_idx) in enumerate(kf.split(unique_queries), 1):
        # Split queries
        train_queries = unique_queries[train_idx]
        val_queries = unique_queries[val_idx]
        
        # Filter dataframe by queries
        train_df = df[df.query_id.isin(train_queries)]
        val_df = df[df.query_id.isin(val_queries)]
        
        # Train and evaluate fold
        fold_result = train_fold(
            fold_num, train_df, val_df, num_llms, device,
            epochs=epochs, batch_size=batch_size, learning_rate=learning_rate
        )
        
        fold_results.append(fold_result)
        
        # Overall progress update
        completed_folds = len(fold_results)
        avg_fold_time = (time.time() - total_start_time) / completed_folds
        remaining_folds = n_folds - completed_folds
        estimated_total_remaining = remaining_folds * avg_fold_time
        
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
            'pipeline': '10-fold Cross-Validation - Two-Tower Neural Network',
            'dataset': 'TREC 2025 Million LLMs Track - Complete Dataset',
            'total_examples': len(df),
            'unique_queries': len(unique_queries),
            'unique_llms': num_llms,
            'epochs_per_fold': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'device': str(device),
            'total_runtime_seconds': total_time,
            'total_runtime_minutes': total_time / 60,
            'total_runtime_hours': total_time / 3600
        },
        'performance_metrics': {
            'ndcg_10': {
                'mean': np.mean(ndcg_10_scores),
                'std': np.std(ndcg_10_scores),
                'min': np.min(ndcg_10_scores),
                'max': np.max(ndcg_10_scores),
                'confidence_interval_95': [
                    np.mean(ndcg_10_scores) - 1.96 * np.std(ndcg_10_scores),
                    np.mean(ndcg_10_scores) + 1.96 * np.std(ndcg_10_scores)
                ]
            },
            'ndcg_5': {
                'mean': np.mean(ndcg_5_scores),
                'std': np.std(ndcg_5_scores),
                'min': np.min(ndcg_5_scores),
                'max': np.max(ndcg_5_scores),
                'confidence_interval_95': [
                    np.mean(ndcg_5_scores) - 1.96 * np.std(ndcg_5_scores),
                    np.mean(ndcg_5_scores) + 1.96 * np.std(ndcg_5_scores)
                ]
            },
            'mrr': {
                'mean': np.mean(mrr_scores),
                'std': np.std(mrr_scores),
                'min': np.min(mrr_scores),
                'max': np.max(mrr_scores),
                'confidence_interval_95': [
                    np.mean(mrr_scores) - 1.96 * np.std(mrr_scores),
                    np.mean(mrr_scores) + 1.96 * np.std(mrr_scores)
                ]
            }
        },
        'fold_by_fold_results': fold_results
    }
    
    # Print summary
    print(f"\n=== NEURAL BASELINE RESULTS ===")
    print(f"nDCG@10: {np.mean(ndcg_10_scores):.4f} ± {np.std(ndcg_10_scores):.4f}")
    print(f"nDCG@5:  {np.mean(ndcg_5_scores):.4f} ± {np.std(ndcg_5_scores):.4f}")
    print(f"MRR:     {np.mean(mrr_scores):.4f} ± {np.std(mrr_scores):.4f}")
    print(f"Total Runtime: {total_time/3600:.2f} hours ({total_time/60:.1f} minutes)")
    
    # Save results
    with open('../../data/results/neural_two_tower_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to ../../data/results/neural_two_tower_results.json")
    
    return results


def main():
    """Main training function"""
    # Load data
    df = load_data('../../data/supervised_training_full.csv')
    
    # Run cross-validation
    results = run_cross_validation(
        df, 
        n_folds=10,  # Full 10-fold CV
        epochs=20,   # Reasonable number of epochs
        batch_size=64, 
        learning_rate=0.001
    )
    
    return results


if __name__ == "__main__":
    results = main()