#!/usr/bin/env python3
"""
Tier 2 Evaluation - Configuration R
Config O architecture + Original + Weak Labels + Pseudo-Labels

CONFIG R: Combines three sources of supervision:
1. Original ground truth labels (386K)
2. Weak labels from Config L (490K) 
3. Pseudo-labels from discovery data (4K)
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from datetime import datetime
import json
import os
import sys
from sklearn.model_selection import KFold
import warnings

# Add project root directory to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from model import TwoTowerModel, ContrastiveLossWithDiversity, create_model
from data_loader import create_data_loaders, load_data, LLMEvaluationDataset, HardNegativeMiner
from shared.utils.evaluation import calculate_metrics
from sentence_transformers import SentenceTransformer
from collaborative_pretraining import CollaborativePretrainingModel

warnings.filterwarnings('ignore')

# Force CPU usage with optimized thread count
torch.set_num_threads(8)


class PretrainedTwoTowerModel(nn.Module):
    """CONFIG R: Same as Config O architecture but with pseudo-labels added"""
    
    def __init__(self, num_llms=1131, embedding_dim=128, num_heads=4, dropout=0.2,
                 pretrained_model_path=None):
        super(PretrainedTwoTowerModel, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        
        # Load pre-trained components if available
        if pretrained_model_path and os.path.exists(pretrained_model_path):
            print(f"Loading pre-trained query encoder from: {pretrained_model_path}")
            checkpoint = torch.load(pretrained_model_path, map_location='cpu', weights_only=False)
            
            # Initialize sentence transformer
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            query_input_dim = self.sentence_transformer.get_sentence_embedding_dimension()
            
            # Freeze sentence transformer
            for param in self.sentence_transformer.parameters():
                param.requires_grad = False
            
            # Multi-head attention (trainable, loaded from checkpoint)
            self.multi_head_attention = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(query_input_dim, 96),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(96, 96)
                ) for _ in range(num_heads)
            ])
            
            # Fusion layer (trainable, loaded from checkpoint)
            self.fusion = nn.Sequential(
                nn.Linear(num_heads * 96, query_input_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            
            # Query tower (trainable, loaded from checkpoint)
            self.query_tower = nn.Sequential(
                nn.Linear(query_input_dim, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 192),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(192, embedding_dim)
            )
            
            # Load pre-trained weights for the trainable parts
            pretrained_model = CollaborativePretrainingModel()
            pretrained_model.load_state_dict(checkpoint['model_state_dict'])
            
            # Copy weights from pre-trained model
            self.multi_head_attention.load_state_dict(pretrained_model.multi_head_attention.state_dict())
            self.fusion.load_state_dict(pretrained_model.fusion.state_dict())
            self.query_tower.load_state_dict(pretrained_model.query_tower.state_dict())
            
            print(f"✓ Pre-trained query encoder loaded and sentence transformer frozen")
            print(f"✓ Pre-training validation loss was: {checkpoint.get('best_val_loss', checkpoint.get('val_loss', 'N/A'))}")
        else:
            raise ValueError(f"Pre-trained model not found at {pretrained_model_path}")
        
        # Enhanced attention mechanism with 4 heads (up from 3)
        self.attention_dim = 64
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,  # 128D from query tower
            num_heads=num_heads,  # 4 heads
            kdim=embedding_dim,
            vdim=embedding_dim,
            dropout=dropout,
            batch_first=True
        )
        
        # Query projection for final output
        self.query_projection = nn.Sequential(
            nn.Linear(embedding_dim, 192),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(192, embedding_dim)
        )
        
        # CONFIG O ENHANCEMENT: Deeper LLM tower (4 layers instead of 3)
        self.llm_embedding = nn.Embedding(num_llms, 256)  # Keep 256D as optimal
        self.llm_tower = nn.Sequential(
            nn.Linear(256, 384),      # Layer 1: Expand
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(384, 256),      # Layer 2: Contract
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 192),      # Layer 3: Further contract
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(192, embedding_dim)  # Layer 4: Final projection
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier/Kaiming initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.1)
    
    def forward(self, query_texts, positive_llms, negative_llms):
        """
        Forward pass for contrastive learning
        
        Args:
            query_texts: List of query strings
            positive_llms: Tensor of positive LLM IDs [batch_size]
            negative_llms: Tensor of negative LLM IDs [batch_size, num_negatives]
        """
        # Encode queries using pre-trained encoder
        with torch.no_grad():
            query_features = self.sentence_transformer.encode(
                query_texts, 
                convert_to_tensor=True,
                show_progress_bar=False
            )
        
        # Clone for gradient computation
        query_features = query_features.clone().detach().requires_grad_(True)
        
        # Multi-head attention
        head_outputs = []
        for head in self.multi_head_attention:
            head_output = head(query_features)
            head_outputs.append(head_output)
        
        concatenated = torch.cat(head_outputs, dim=1)
        fused = self.fusion(concatenated)
        
        # Query tower
        query_repr = self.query_tower(fused)
        
        # Self-attention for Config O enhancement
        query_repr_unsqueezed = query_repr.unsqueeze(1)
        attended_features, _ = self.attention(
            query_repr_unsqueezed,
            query_repr_unsqueezed,
            query_repr_unsqueezed
        )
        attended_features = attended_features.squeeze(1)
        
        # Combine original and attended features
        query_combined = query_repr + 0.5 * attended_features
        
        # Project to final embedding space
        query_embeddings = self.query_projection(query_combined)
        
        # Encode positive LLMs through deeper tower
        positive_embeddings = self.llm_embedding(positive_llms)
        positive_embeddings = self.llm_tower(positive_embeddings)
        
        # Encode negative LLMs
        batch_size, num_negatives = negative_llms.shape
        negative_llms_flat = negative_llms.view(-1)
        negative_embeddings_flat = self.llm_embedding(negative_llms_flat)
        negative_embeddings_flat = self.llm_tower(negative_embeddings_flat)
        negative_embeddings = negative_embeddings_flat.view(batch_size, num_negatives, -1)
        
        # L2 normalize
        query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
        positive_embeddings = torch.nn.functional.normalize(positive_embeddings, p=2, dim=1)
        negative_embeddings = torch.nn.functional.normalize(negative_embeddings, p=2, dim=2)
        
        return query_embeddings, positive_embeddings, negative_embeddings, head_outputs


def run_10fold_cv():
    """Run 10-fold cross-validation with Config R (including pseudo-labels)"""
    
    print("="*80)
    print("TIER 2 EVALUATION - CONFIG R")
    print("Config O + Pseudo-Labels from Discovery Data")
    print("="*80)
    print()
    
    # Load original training data
    original_path = '../../data/supervised_training_full.csv'
    print(f"Loading original training data from {original_path}...")
    original_df = pd.read_csv(original_path)
    print(f"  Loaded {len(original_df):,} original examples")
    
    # Load Config L's weak labeled data
    weak_labeled_path = '../../data/supervised_training_config_n_weak_labeled.csv'
    if os.path.exists(weak_labeled_path):
        print(f"Loading Config L weak labeled data...")
        weak_df = pd.read_csv(weak_labeled_path)
        print(f"  Loaded {len(weak_df):,} weak labeled examples")
    else:
        print("Warning: Config L weak labeled data not found")
        weak_df = pd.DataFrame()
    
    # Load pseudo-labels
    pseudo_path = '../../data/pseudo_labels_from_pretrained.csv'
    if os.path.exists(pseudo_path):
        print(f"Loading pseudo-labels...")
        pseudo_df = pd.read_csv(pseudo_path)
        # Convert to match training format - need query_id, query_text, llm_id, qrel
        # Add unique INTEGER query_ids for pseudo-labels (starting from a high number to avoid conflicts)
        unique_queries = pseudo_df['query_text'].unique()
        # Use integers starting from 900000 to avoid conflicts with existing query IDs
        query_id_map = {q: 900000 + i for i, q in enumerate(unique_queries)}
        pseudo_df['query_id'] = pseudo_df['query_text'].map(query_id_map)
        # Reorder columns to match original format
        pseudo_df = pseudo_df[['query_id', 'llm_id', 'qrel', 'query_text']]
        print(f"  Loaded {len(pseudo_df):,} pseudo-labels")
    else:
        print("Warning: Pseudo-labels not found")
        pseudo_df = pd.DataFrame()
    
    # Combine all datasets
    print("\nCombining datasets...")
    datasets_to_combine = [original_df]
    
    if not weak_df.empty:
        datasets_to_combine.append(weak_df)
    
    if not pseudo_df.empty:
        datasets_to_combine.append(pseudo_df)
    
    combined_df = pd.concat(datasets_to_combine, ignore_index=True)
    print(f"Combined dataset: {len(combined_df):,} total examples")
    print(f"  Original: {len(original_df):,}")
    print(f"  Weak labels: {len(weak_df):,}")
    print(f"  Pseudo-labels: {len(pseudo_df):,}")
    
    # Training parameters
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Configuration R parameters
    epochs = 25
    batch_size = 96
    base_lr = 0.0008
    patience = 10
    k_folds = 10
    negative_samples = 7
    
    print(f"\nTraining Configuration R:")
    print(f"  Architecture: Config O (4-layer LLM tower)")
    print(f"  LLM embedding dim: 256")
    print(f"  Final embedding dim: 128")
    print(f"  Attention heads: 4")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {base_lr}")
    print(f"  Negative samples: {negative_samples}")
    print(f"  Early stopping patience: {patience}")
    print(f"  Total training examples: {len(combined_df):,}")
    
    # Group by query for proper CV splitting - ONLY use original queries for splitting
    # This prevents data leakage from pseudo-labels
    original_queries = original_df['query_text'].unique()
    print(f"\nUnique queries for CV splitting: {len(original_queries)} (original queries only)")
    print(f"Total unique queries in combined dataset: {combined_df['query_text'].nunique()}")
    
    # Prepare for K-fold CV
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_results = []
    
    print(f"\nStarting {k_folds}-fold cross-validation...")
    print("NOTE: Pseudo-labels are used for training only, not validation")
    print("="*80)
    
    start_time_total = time.time()
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(original_queries), 1):
        print(f"\nFOLD {fold}/{k_folds}")
        print("-"*40)
        
        fold_start_time = time.time()
        
        # Split queries (only from original dataset)
        train_queries = original_queries[train_idx]
        val_queries = original_queries[val_idx]
        
        # Create validation dataframe - ONLY from original + weak labeled data, NO pseudo-labels
        val_df = combined_df[
            (combined_df['query_text'].isin(val_queries)) & 
            (combined_df['query_id'] < 900000)  # Exclude pseudo-labels (which have query_id >= 900000)
        ].copy()
        
        # Create training dataframe - includes all data for train queries + ALL pseudo-labels
        train_df = combined_df[
            (combined_df['query_text'].isin(train_queries)) |  # Original/weak data for train queries
            (combined_df['query_id'] >= 900000)  # Include ALL pseudo-labels in training
        ].copy()
        
        print(f"Train: {len(train_df):,} examples ({len(train_queries)} queries)")
        print(f"Val: {len(val_df):,} examples ({len(val_queries)} queries)")
        
        # Create model
        model = PretrainedTwoTowerModel(
            num_llms=1131,
            embedding_dim=128,
            num_heads=4,
            dropout=0.2,
            pretrained_model_path='../../data/collaborative_pretrained_model.pth'
        ).to(device)
        
        # Data loaders
        train_loader, val_loader, llm_encoder = create_data_loaders(
            train_df, val_df, 
            batch_size=batch_size, 
            negative_samples=negative_samples
        )
        
        # Optimizer and scheduler
        optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=base_lr/10)
        criterion = ContrastiveLossWithDiversity(temperature=0.05, diversity_weight=0.1)
        
        # Training loop
        best_ndcg = 0
        patience_counter = 0
        
        for epoch in range(1, epochs + 1):
            # Training
            model.train()
            train_loss = 0
            train_batches = 0
            
            epoch_start = time.time()
            
            for batch in train_loader:
                query_texts = batch['query_texts']
                positive_llms = batch['positive_llms'].to(device)
                negative_llms = batch['negative_llms'].to(device)
                
                # Reshape negative_llms for the model
                batch_size_actual = len(query_texts)
                negative_llms = negative_llms.view(batch_size_actual, negative_samples)
                
                optimizer.zero_grad()
                
                # Forward pass
                query_embeddings, positive_embeddings, negative_embeddings, head_outputs = model(
                    query_texts, positive_llms, negative_llms
                )
                
                # Compute contrastive loss with diversity regularization
                loss = criterion(query_embeddings, positive_embeddings, negative_embeddings, head_outputs)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                train_batches += 1
            
            avg_train_loss = train_loss / train_batches if train_batches > 0 else 0
            
            # Validation
            model.eval()
            all_scores = []
            all_labels = []
            
            with torch.no_grad():
                for batch in val_loader:
                    # Handle both single items and batches
                    if isinstance(batch['query_text'], str):
                        query_texts = [batch['query_text']]
                        llm_ids = torch.tensor([batch['llm_encoded']], dtype=torch.long).to(device)
                        labels = [batch['relevance']]
                    else:
                        query_texts = batch['query_text']
                        llm_ids = batch['llm_encoded'].to(device)
                        labels = batch['relevance'].tolist()
                    
                    # Create dummy tensors for the forward pass
                    dummy_negatives = torch.zeros((len(query_texts), 1), dtype=torch.long).to(device)
                    
                    # Get embeddings
                    query_emb, llm_emb, _, _ = model(query_texts, llm_ids, dummy_negatives)
                    
                    # Calculate similarities
                    similarities = torch.sum(query_emb * llm_emb, dim=1)
                    
                    all_scores.extend(similarities.cpu().numpy())
                    all_labels.extend(labels)
            
            # Calculate metrics
            val_df_eval = val_df.copy()
            val_df_eval['predictions'] = all_scores
            
            # Group by query for metric calculation
            y_true_by_query = {}
            y_pred_by_query = {}
            
            for query in val_df_eval['query_text'].unique():
                query_data = val_df_eval[val_df_eval['query_text'] == query]
                y_true_by_query[query] = query_data['qrel'].values
                y_pred_by_query[query] = query_data['predictions'].values
            
            metrics = calculate_metrics(y_true_by_query, y_pred_by_query)
            
            epoch_time = time.time() - epoch_start
            current_lr = optimizer.param_groups[0]['lr']
            
            print(f"  Epoch {epoch:2d}: Loss={avg_train_loss:.4f}, nDCG@10={metrics['ndcg_10']:.4f}, "
                  f"MRR={metrics['mrr']:.4f}, LR={current_lr:.6f} ({epoch_time:.1f}s)")
            
            # Early stopping
            if metrics['ndcg_10'] > best_ndcg:
                best_ndcg = metrics['ndcg_10']
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping triggered at epoch {epoch}")
                    break
            
            scheduler.step()
        
        # Record fold results
        fold_time = time.time() - fold_start_time
        fold_results.append({
            'fold': fold,
            'ndcg_10': best_ndcg,
            'ndcg_5': metrics['ndcg_5'],
            'mrr': metrics['mrr'],
            'train_time': fold_time,
            'n_queries': len(val_queries)
        })
        
        print(f"  Fold {fold} best nDCG@10: {best_ndcg:.4f} (Time: {fold_time/60:.1f} min)")
    
    total_time = time.time() - start_time_total
    
    # Calculate aggregate metrics
    ndcg_10_scores = [r['ndcg_10'] for r in fold_results]
    ndcg_5_scores = [r['ndcg_5'] for r in fold_results]
    mrr_scores = [r['mrr'] for r in fold_results]
    
    # Print final results
    print("\n" + "="*80)
    print("10-FOLD CROSS-VALIDATION RESULTS - CONFIG R")
    print("="*80)
    print(f"\nnDCG@10: {np.mean(ndcg_10_scores):.4f} ± {np.std(ndcg_10_scores):.4f}")
    print(f"nDCG@5:  {np.mean(ndcg_5_scores):.4f} ± {np.std(ndcg_5_scores):.4f}")
    print(f"MRR:     {np.mean(mrr_scores):.4f} ± {np.std(mrr_scores):.4f}")
    
    print(f"\nTotal runtime: {total_time/3600:.2f} hours")
    print(f"Average per fold: {total_time/k_folds/60:.1f} minutes")
    
    # Print per-fold results
    print("\nPer-fold performance:")
    print("-"*40)
    for result in fold_results:
        print(f"Fold {result['fold']:2d}: nDCG@10={result['ndcg_10']:.4f}, "
              f"nDCG@5={result['ndcg_5']:.4f}, MRR={result['mrr']:.4f}")
    
    # Save results
    results = {
        'evaluation_info': {
            'timestamp': datetime.now().isoformat(),
            'pipeline': 'Tier 2 Config R - Config O + Pseudo-Labels',
            'model': 'PretrainedTwoTowerModel with Pseudo-Labels',
            'dataset': 'TREC 2025 Million LLMs Track - Combined with Pseudo-Labels',
            'total_examples': len(combined_df),
            'unique_queries': len(unique_queries),
            'unique_llms': 1131,
            'total_runtime_seconds': total_time,
            'total_runtime_hours': total_time / 3600,
            'dataset_composition': {
                'original': len(original_df),
                'weak_labels': len(weak_df),
                'pseudo_labels': len(pseudo_df)
            }
        },
        'hyperparameters': {
            'embedding_dim': 128,
            'llm_embedding_dim': 256,
            'num_heads': 4,
            'dropout': 0.2,
            'batch_size': batch_size,
            'learning_rate': base_lr,
            'epochs': epochs,
            'patience': patience,
            'negative_samples': negative_samples,
            'optimizer': 'AdamW',
            'scheduler': 'CosineAnnealingLR',
            'loss': 'ContrastiveLossWithDiversity',
            'temperature': 0.05,
            'diversity_weight': 0.1
        },
        'performance_metrics': {
            'ndcg_10': {
                'mean': np.mean(ndcg_10_scores),
                'std': np.std(ndcg_10_scores),
                'min': np.min(ndcg_10_scores),
                'max': np.max(ndcg_10_scores),
                'confidence_interval_95': [
                    np.mean(ndcg_10_scores) - 1.96 * np.std(ndcg_10_scores) / np.sqrt(k_folds),
                    np.mean(ndcg_10_scores) + 1.96 * np.std(ndcg_10_scores) / np.sqrt(k_folds)
                ]
            },
            'ndcg_5': {
                'mean': np.mean(ndcg_5_scores),
                'std': np.std(ndcg_5_scores),
                'min': np.min(ndcg_5_scores),
                'max': np.max(ndcg_5_scores),
                'confidence_interval_95': [
                    np.mean(ndcg_5_scores) - 1.96 * np.std(ndcg_5_scores) / np.sqrt(k_folds),
                    np.mean(ndcg_5_scores) + 1.96 * np.std(ndcg_5_scores) / np.sqrt(k_folds)
                ]
            },
            'mrr': {
                'mean': np.mean(mrr_scores),
                'std': np.std(mrr_scores),
                'min': np.min(mrr_scores),
                'max': np.max(mrr_scores),
                'confidence_interval_95': [
                    np.mean(mrr_scores) - 1.96 * np.std(mrr_scores) / np.sqrt(k_folds),
                    np.mean(mrr_scores) + 1.96 * np.std(mrr_scores) / np.sqrt(k_folds)
                ]
            }
        },
        'fold_by_fold_results': fold_results
    }
    
    # Save to JSON
    output_path = '../../data/results/tier2_config_r_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    
    return results


if __name__ == "__main__":
    results = run_10fold_cv()