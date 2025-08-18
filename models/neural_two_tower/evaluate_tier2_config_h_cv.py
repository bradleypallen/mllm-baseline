#!/usr/bin/env python3
"""
Config H: GPU + Pre-trained Query Encoder + Epistemic Data Augmentation - 10-FOLD CV

PROPER EVALUATION with standard 10-fold cross-validation methodology:
- Config G's proven foundation (GPU + pre-training + optimal hyperparameters)
- Epistemic data augmentation (r=0.96 correlation with human qrels) 
- Standard 10-fold CV with consistent progress reporting
- Standardized results format matching all other model evaluations

Target: Beat Config G's 0.4303±0.0394 baseline with statistical robustness
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import json
import time
from datetime import datetime
import logging
from typing import Dict, List, Tuple
import warnings

# Add shared utilities to path
sys.path.append('../../shared')
from utils.evaluation import calculate_metrics, save_standardized_results

# Import model components
from model import create_model, ContrastiveLossWithDiversity
from data_loader import create_data_loaders, load_data, HardNegativeMiner

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

def load_augmented_training_data():
    """Load the epistemic-augmented training dataset"""
    print("Loading epistemic-augmented training data...")
    
    # Use the clean augmented dataset
    train_df = pd.read_csv('../../data/supervised_training_augmented_clean.csv')
    
    print(f"Loaded {len(train_df):,} total examples")
    print(f"  Unique queries: {train_df['query_id'].nunique()}")
    print(f"  Unique LLMs: {train_df['llm_id'].nunique()}")
    print(f"  Qrel distribution: {train_df['qrel'].value_counts().to_dict()}")
    
    return train_df

def train_fold_config_h(train_data: pd.DataFrame, val_data: pd.DataFrame, 
                       device: torch.device, fold_num: int) -> Dict:
    """Train single fold with epistemic augmentation and Config G architecture"""
    
    print(f"\n=== FOLD {fold_num} TRAINING ===")
    print(f"Train: {len(train_data)} examples, Val: {len(val_data)} examples")
    
    # Prepare data loaders with hard negative mining
    hard_miner = HardNegativeMiner(model=None, hard_ratio=0.3, temperature=0.1)
    
    train_loader, val_loader, _ = create_data_loaders(
        train_data, val_data, batch_size=96, negative_samples=4
    )
    
    # Create model with optimal hyperparameters from Config G  
    model = create_model(
        num_llms=1131,
        device=device,
        use_multi_head=True,
        num_heads=4
    )
    
    print(f"Dataset created with {len(train_loader)} training batches")
    
    # Load pre-trained query encoder (from Config G)
    pretrained_path = "../../data/collaborative_pretrained_model.pth"
    if os.path.exists(pretrained_path):
        print(f"Loading pre-trained query encoder from: {pretrained_path}")
        try:
            checkpoint = torch.load(pretrained_path, map_location=device, weights_only=False)
            
            # Load the query encoder state
            if 'query_tower_state' in checkpoint:
                model.query_tower.load_state_dict(checkpoint['query_tower_state'])
                print("✓ Pre-trained query encoder loaded successfully")
                
                # Freeze sentence transformer for stability
                for param in model.sentence_transformer.parameters():
                    param.requires_grad = False
                print("✓ Sentence transformer frozen for fine-tuning")
                
                if 'validation_loss' in checkpoint:
                    print(f"✓ Pre-training validation loss was: {checkpoint['validation_loss']:.4f}")
            else:
                print("⚠ No query tower state found in checkpoint")
                
        except Exception as e:
            print(f"⚠ Error loading pre-trained model: {e}")
            print("Continuing with random initialization...")
    else:
        print("⚠ No pre-trained model found, using random initialization")
    
    # Optimizer and loss with Config G's proven settings
    optimizer = optim.Adam(model.parameters(), lr=0.0008, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-5)
    criterion = ContrastiveLossWithDiversity(temperature=0.1, diversity_weight=0.02)
    
    # Training loop with early stopping (same as Config G)
    best_ndcg = 0.0
    best_metrics = {}
    patience = 5
    patience_counter = 0
    training_start_time = time.time()
    
    for epoch in range(25):
        # Training epoch using Config G's exact approach
        model.train()
        total_loss = 0
        num_batches = 0
        epoch_start = time.time()
        
        # Enable head output storage for diversity loss
        model._store_head_outputs = True
        
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx == 0:
                print(f"    Starting training epoch {epoch+1}...")
                
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
            elif (batch_idx + 1) % 20 == 0:
                print(f"    Processed {batch_idx + 1}/{len(train_loader)} batches...")
        
        avg_loss = total_loss / max(num_batches, 1)
        scheduler.step()
        
        # Validation
        model.eval()
        val_predictions = {}
        val_true_labels = {}
        
        print(f"  Evaluating on {len(val_loader)} batches", end="")
        with torch.no_grad():
            for batch in val_loader:
                query_texts = batch['query_text']
                llm_encoded = batch['llm_encoded'].to(device)
                relevances = batch['relevance']
                
                scores = model.predict_batch(query_texts, llm_encoded)
                
                # Group by query for ranking evaluation
                for i, (qtext, label) in enumerate(zip(query_texts, relevances)):
                    if qtext not in val_predictions:
                        val_predictions[qtext] = []
                        val_true_labels[qtext] = []
                    
                    val_predictions[qtext].append(scores[i])
                    val_true_labels[qtext].append(label.item())
                
                print(".", end="", flush=True)
        
        print(" Done!", end="")
        
        # Calculate metrics
        metrics = calculate_metrics(val_true_labels, val_predictions)
        
        epoch_time = time.time() - epoch_start
        remaining_epochs = 25 - epoch - 1
        estimated_remaining = remaining_epochs * epoch_time
        
        print(f" nDCG@10={metrics['ndcg_10']:.4f}, MRR={metrics['mrr']:.4f}")
        print(f"  Epoch {epoch+1}/25: Loss={avg_loss:.4f}, nDCG@10={metrics['ndcg_10']:.4f}, MRR={metrics['mrr']:.4f}, LR={scheduler.get_last_lr()[0]:.6f} ({epoch_time:.1f}s, ~{estimated_remaining/60:.1f}min left)")
        
        # Early stopping logic
        if metrics['ndcg_10'] > best_ndcg:
            best_ndcg = metrics['ndcg_10']
            best_metrics = metrics.copy()
            patience_counter = 0
            print(f"    *** New best: {best_ndcg:.4f} ***")
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"  Early stopping triggered after {epoch+1} epochs")
            break
    
    training_time = time.time() - training_start_time
    
    print(f"\nFold {fold_num} Training Results:")
    print(f"  Best nDCG@10: {best_metrics['ndcg_10']:.4f} (epoch {epoch+1-patience_counter})")
    print(f"  Best nDCG@5:  {best_metrics['ndcg_5']:.4f}")
    print(f"  Best MRR:     {best_metrics['mrr']:.4f}")
    print(f"  Time: {training_time:.1f}s ({training_time/60:.1f} minutes)")
    
    return {
        'fold': fold_num,
        'ndcg_10': best_metrics['ndcg_10'],
        'ndcg_5': best_metrics['ndcg_5'],
        'mrr': best_metrics['mrr'],
        'train_time': training_time,
        'n_queries': len(val_true_labels)
    }

def run_config_h_cv():
    """Run Config H with proper 10-fold cross-validation"""
    print("=" * 80)
    print("CONFIG H: GPU + PRE-TRAINED + EPISTEMIC AUGMENTATION - 10-FOLD CV")
    print("=" * 80)
    print()
    print("Configuration:")
    print("  - Base: Config G (4 heads, batch 96, GPU, pre-trained)")
    print("  - Data: Epistemic augmentation (r=0.96 correlation)")
    print("  - Evaluation: Standard 10-fold cross-validation")
    print("  - Validation: Held-out original dev data per fold")
    print("  - Target: Beat Config G 0.4303±0.0394 with statistical robustness")
    print()
    
    # Device setup
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    print()
    
    # Load original dev data for consistent CV splitting
    print("Loading original development data for CV splits...")
    original_df = pd.read_csv('../../data/supervised_training_full.csv')
    print(f"Original dataset: {len(original_df):,} examples")
    
    # Load augmented data
    augmented_df = load_augmented_training_data()
    
    # Identify synthetic examples by qrel values (synthetic uses 0.7 for qrel=2)
    synthetic_data = augmented_df[augmented_df['qrel'] == 0.7].copy()
    
    print(f"Synthetic augmentation: {len(synthetic_data):,} examples")
    print(f"Augmentation ratio: {len(synthetic_data) / len(original_df) * 100:.1f}%")
    print()
    
    # Cross-validation setup (same as all other models)
    unique_queries = original_df['query_id'].unique()
    np.random.seed(42)
    np.random.shuffle(unique_queries)
    
    kf = KFold(n_splits=10, shuffle=False)
    fold_results = []
    total_start_time = time.time()
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(unique_queries)):
        print(f"\n=== FOLD {fold+1}/10 ===")
        
        # Create original data splits
        train_queries = unique_queries[train_idx]
        val_queries = unique_queries[val_idx]
        
        original_train = original_df[original_df['query_id'].isin(train_queries)].copy()
        val_data = original_df[original_df['query_id'].isin(val_queries)].copy()
        
        # Augment training data with synthetic examples
        train_data = pd.concat([original_train, synthetic_data]).reset_index(drop=True)
        
        print(f"Original train: {len(original_train):,} examples from {len(train_queries)} queries")
        print(f"Synthetic augmentation: {len(synthetic_data):,} examples")
        print(f"Combined train: {len(train_data):,} examples")
        print(f"Validation: {len(val_data):,} examples from {len(val_queries)} queries")
        
        fold_start = time.time()
        
        # Train fold
        result = train_fold_config_h(train_data, val_data, device, fold+1)
        
        fold_time = time.time() - fold_start
        result['train_time'] = fold_time
        
        fold_results.append(result)
        
        print(f"\nFold {fold+1} Final Results:")
        print(f"  Best nDCG@10: {result['ndcg_10']:.4f}")
        print(f"  Best nDCG@5:  {result['ndcg_5']:.4f}")
        print(f"  Best MRR:     {result['mrr']:.4f}")
        print(f"  Time: {fold_time:.1f}s ({fold_time/60:.1f} minutes)")
    
    total_time = time.time() - total_start_time
    
    # Aggregate results (STANDARD FORMAT)
    ndcg_10_scores = [r['ndcg_10'] for r in fold_results]
    ndcg_5_scores = [r['ndcg_5'] for r in fold_results]
    mrr_scores = [r['mrr'] for r in fold_results]
    
    print("\n" + "=" * 80)
    print("FINAL RESULTS - CONFIG H (GPU + PRE-TRAINED + EPISTEMIC) - 10-FOLD CV")
    print("=" * 80)
    print(f"nDCG@10: {np.mean(ndcg_10_scores):.4f} ± {np.std(ndcg_10_scores):.4f}")
    print(f"  Min: {np.min(ndcg_10_scores):.4f}, Max: {np.max(ndcg_10_scores):.4f}")
    print(f"nDCG@5:  {np.mean(ndcg_5_scores):.4f} ± {np.std(ndcg_5_scores):.4f}")
    print(f"  Min: {np.min(ndcg_5_scores):.4f}, Max: {np.max(ndcg_5_scores):.4f}")
    print(f"MRR:     {np.mean(mrr_scores):.4f} ± {np.std(mrr_scores):.4f}")
    print(f"  Min: {np.min(mrr_scores):.4f}, Max: {np.max(mrr_scores):.4f}")
    print(f"\nTotal runtime: {total_time:.1f}s ({total_time/3600:.2f} hours)")
    print(f"Average per fold: {total_time/10:.1f}s ({total_time/10/60:.1f} minutes)")
    
    # Comparison with Config G
    config_g_ndcg = 0.4303
    improvement = np.mean(ndcg_10_scores) - config_g_ndcg
    print(f"\nComparison Analysis:")
    print(f"  Config G (pre-trained):          {config_g_ndcg:.4f} ± 0.0394 nDCG@10")
    print(f"  Config H (epistemic):            {np.mean(ndcg_10_scores):.4f} ± {np.std(ndcg_10_scores):.4f} nDCG@10")
    print(f"  Epistemic augmentation benefit:  {improvement:+.4f}")
    
    # Prepare standardized results (EXACTLY like other models)
    results = {
        'evaluation_info': {
            'timestamp': datetime.now().isoformat(),
            'pipeline': 'Config H: GPU + Pre-trained + Epistemic (10-fold CV)',
            'dataset': 'TREC 2025 Million LLMs Track - Epistemic Augmented',
            'total_examples': len(original_df) + len(synthetic_data),
            'original_examples': len(original_df),
            'synthetic_examples': len(synthetic_data),
            'augmentation_ratio': len(synthetic_data) / len(original_df),
            'unique_queries': len(unique_queries),
            'unique_llms': original_df['llm_id'].nunique(),
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
    
    # Save results with standard naming
    save_standardized_results(results, 'tier2_config_h_cv_epistemic', '../../data/results/')
    print("\n✓ Results saved to data/results/tier2_config_h_cv_epistemic_results.json")
    print()
    print("=" * 80)
    print("Config H (10-fold CV) evaluation complete!")
    print("=" * 80)
    
    return results

if __name__ == "__main__":
    results = run_config_h_cv()