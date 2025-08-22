#!/usr/bin/env python3
"""
Collaborative Filtering Pre-training with Sparse Discovery Data

Two-stage approach:
1. Stage 1: Pre-train on filtered discovery interactions (sparse matrix)
2. Stage 2: Fine-tune on development relevance labels

This leverages the 43x more interaction data with meaningful sparsity patterns.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import json
import time
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
import warnings
import os
import sys

# Add project root directory to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from discovery_response_filter import filter_discovery_responses
from model import TwoTowerModel

warnings.filterwarnings('ignore')


class CollaborativeFilteringDataset(Dataset):
    """Dataset for collaborative filtering pre-training"""
    
    def __init__(self, queries, interaction_matrix, num_negatives=4):
        """
        Args:
            queries: List of query texts
            interaction_matrix: [num_queries, num_llms] sparse binary matrix
            num_negatives: Number of negative samples per positive
        """
        self.queries = queries
        self.interaction_matrix = interaction_matrix
        self.num_negatives = num_negatives
        self.num_llms = interaction_matrix.shape[1]
        
        # Pre-compute positive pairs for efficiency
        print("Creating positive interaction pairs...")
        self.positive_pairs = []
        for query_idx in range(len(queries)):
            positive_llms = np.where(interaction_matrix[query_idx] == 1)[0]
            for llm_idx in positive_llms:
                self.positive_pairs.append((query_idx, llm_idx))
            
            # Progress logging for large datasets
            if query_idx % 500 == 0 and query_idx > 0:
                print(f"  Processed {query_idx}/{len(queries)} queries, {len(self.positive_pairs)} pairs so far...")
        
        print(f"Created dataset with {len(self.positive_pairs)} positive interactions")
        
    def __len__(self):
        return len(self.positive_pairs)
    
    def __getitem__(self, idx):
        query_idx, positive_llm = self.positive_pairs[idx]
        
        # Sample negative LLMs (those that didn't respond to this query)
        negative_candidates = np.where(self.interaction_matrix[query_idx] == 0)[0]
        if len(negative_candidates) >= self.num_negatives:
            negative_llms = np.random.choice(negative_candidates, self.num_negatives, replace=False)
        else:
            # If not enough negatives, sample with replacement
            negative_llms = np.random.choice(negative_candidates, self.num_negatives, replace=True)
        
        return {
            'query_text': self.queries[query_idx],
            'positive_llm': torch.tensor(positive_llm, dtype=torch.long),
            'negative_llms': torch.tensor(negative_llms, dtype=torch.long)
        }


class CollaborativePretrainingModel(nn.Module):
    """Two-tower model for collaborative filtering pre-training"""
    
    def __init__(self, num_llms=1131, query_embedding_model='all-MiniLM-L6-v2', 
                 embedding_dim=128, num_heads=4, dropout=0.2):
        super(CollaborativePretrainingModel, self).__init__()
        
        # Pre-trained sentence transformer
        self.sentence_transformer = SentenceTransformer(query_embedding_model)
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
        
        # Query tower (same architecture as main model)
        self.query_tower = nn.Sequential(
            nn.Linear(query_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 192),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(192, embedding_dim)
        )
        
        # LLM tower (same architecture as main model)
        self.llm_embedding = nn.Embedding(num_llms, 64)
        self.llm_tower = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, embedding_dim)
        )
        
    def encode_query(self, query_texts):
        """Encode queries through full pipeline"""
        # Get sentence embeddings
        with torch.no_grad():
            embeddings = self.sentence_transformer.encode(
                query_texts, convert_to_tensor=True, device=next(self.parameters()).device
            )
        
        # Clone for gradient computation
        embeddings = embeddings.clone().detach().requires_grad_(True)
        
        # Multi-head attention
        head_outputs = [head(embeddings) for head in self.multi_head_attention]
        concatenated = torch.cat(head_outputs, dim=1)
        fused = self.fusion(concatenated)
        
        # Query tower
        query_repr = self.query_tower(fused)
        
        return query_repr
    
    def encode_llms(self, llm_ids):
        """Encode LLM IDs through embedding and tower"""
        embedded = self.llm_embedding(llm_ids)
        return self.llm_tower(embedded)
    
    def forward(self, query_texts, positive_llms, negative_llms):
        """Forward pass for contrastive learning"""
        # Encode queries
        query_embeddings = self.encode_query(query_texts)  # [batch_size, embedding_dim]
        
        # Encode positive LLMs
        positive_embeddings = self.encode_llms(positive_llms)  # [batch_size, embedding_dim]
        
        # Encode negative LLMs
        batch_size, num_negatives = negative_llms.shape
        negative_llms_flat = negative_llms.view(-1)  # [batch_size * num_negatives]
        negative_embeddings_flat = self.encode_llms(negative_llms_flat)
        negative_embeddings = negative_embeddings_flat.view(batch_size, num_negatives, -1)
        
        return query_embeddings, positive_embeddings, negative_embeddings


def contrastive_loss(query_embeddings, positive_embeddings, negative_embeddings, temperature=0.1):
    """
    Contrastive loss for collaborative filtering
    
    Args:
        query_embeddings: [batch_size, embedding_dim]
        positive_embeddings: [batch_size, embedding_dim]
        negative_embeddings: [batch_size, num_negatives, embedding_dim]
        temperature: Temperature for softmax
    """
    batch_size = query_embeddings.size(0)
    
    # Compute similarities
    positive_sim = torch.cosine_similarity(query_embeddings, positive_embeddings, dim=1)  # [batch_size]
    negative_sim = torch.cosine_similarity(
        query_embeddings.unsqueeze(1), negative_embeddings, dim=2
    )  # [batch_size, num_negatives]
    
    # Apply temperature
    positive_sim = positive_sim / temperature
    negative_sim = negative_sim / temperature
    
    # Combine positive and negative similarities
    all_sim = torch.cat([positive_sim.unsqueeze(1), negative_sim], dim=1)  # [batch_size, 1 + num_negatives]
    
    # Labels (positive is always index 0)
    labels = torch.zeros(batch_size, dtype=torch.long, device=query_embeddings.device)
    
    # Cross-entropy loss
    loss = nn.CrossEntropyLoss()(all_sim, labels)
    
    return loss


def pretrain_collaborative_filtering(discovery_file_path, output_model_path,
                                   num_epochs=20, batch_size=64, learning_rate=0.001,
                                   max_queries=5000, temperature=0.1):
    """
    Pre-train collaborative filtering model on discovery data
    """
    print("=" * 80)
    print("COLLABORATIVE FILTERING PRE-TRAINING")
    print("=" * 80)
    
    device = torch.device('cpu')  # Start with CPU
    print(f"Using device: {device}")
    
    # Load and filter discovery data
    print(f"Loading discovery data from: {discovery_file_path}")
    print(f"Processing up to {max_queries} queries for testing...")
    print("Filtering discovery data to create sparse interaction matrix...")
    print("This may take several minutes for large datasets...")
    
    import time
    start_time = time.time()
    queries, interaction_matrix, query_ids, stats = filter_discovery_responses(
        discovery_file_path, max_queries=max_queries
    )
    filtering_time = time.time() - start_time
    print(f"Data filtering completed in {filtering_time:.1f} seconds")
    
    print(f"\nFiltered data stats:")
    print(f"  Sparsity: {stats['sparsity']:.1%}")
    print(f"  Response rate: {stats['response_rate']:.1%}")
    print(f"  Avg LLMs per query: {stats['avg_llms_per_query']:.1f}")
    
    # Split into train/validation
    train_queries, val_queries, train_matrix, val_matrix = train_test_split(
        queries, interaction_matrix, test_size=0.2, random_state=42
    )
    
    print(f"\nTraining queries: {len(train_queries)}")
    print(f"Validation queries: {len(val_queries)}")
    
    # Create datasets
    train_dataset = CollaborativeFilteringDataset(train_queries, train_matrix, num_negatives=4)
    val_dataset = CollaborativeFilteringDataset(val_queries, val_matrix, num_negatives=4)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model (same architecture as Config F)
    model = CollaborativePretrainingModel(
        num_llms=1131,
        embedding_dim=128,
        num_heads=4,
        dropout=0.2
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=learning_rate/10)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training configuration: {num_epochs} epochs, batch_size={batch_size}, lr={learning_rate}")
    
    # Training loop
    best_val_loss = float('inf')
    training_history = []
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            query_texts = batch['query_text']
            positive_llms = batch['positive_llm'].to(device)
            negative_llms = batch['negative_llms'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            query_emb, positive_emb, negative_emb = model(query_texts, positive_llms, negative_llms)
            
            # Contrastive loss
            loss = contrastive_loss(query_emb, positive_emb, negative_emb, temperature)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
            
            if batch_idx == 0:
                print(f"  Epoch {epoch+1} training started...")
            elif (batch_idx + 1) % 50 == 0:
                print(f"    Processed {batch_idx + 1}/{len(train_loader)} batches...")
        
        avg_train_loss = train_loss / num_batches
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                query_texts = batch['query_text']
                positive_llms = batch['positive_llm'].to(device)
                negative_llms = batch['negative_llms'].to(device)
                
                query_emb, positive_emb, negative_emb = model(query_texts, positive_llms, negative_llms)
                loss = contrastive_loss(query_emb, positive_emb, negative_emb, temperature)
                
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        epoch_time = time.time() - epoch_start
        remaining = (num_epochs - epoch - 1) * epoch_time
        
        print(f"  Epoch {epoch+1}/{num_epochs}: Train Loss={avg_train_loss:.4f}, "
              f"Val Loss={avg_val_loss:.4f}, LR={current_lr:.6f} "
              f"({epoch_time:.1f}s, ~{remaining/60:.1f}min left)")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_loss': avg_val_loss,
                'train_loss': avg_train_loss,
                'config': {
                    'num_heads': 4,
                    'embedding_dim': 128,
                    'num_llms': 1131,
                    'temperature': temperature
                },
                'discovery_stats': stats
            }, output_model_path)
            print(f"    *** New best model saved: {avg_val_loss:.4f} ***")
        
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'lr': current_lr
        })
    
    print(f"\nPre-training complete! Best validation loss: {best_val_loss:.4f}")
    print(f"Pre-trained model saved to: {output_model_path}")
    
    return training_history


if __name__ == "__main__":
    # Start collaborative filtering pre-training
    discovery_file = "../../data/llm_discovery_data_1.json"
    output_model = "../../data/collaborative_pretrained_model.pth"
    
    print("Starting collaborative filtering pre-training...")
    print("Using filtered discovery data with sparse interaction patterns")
    
    history = pretrain_collaborative_filtering(
        discovery_file_path=discovery_file,
        output_model_path=output_model,
        num_epochs=15,
        batch_size=32,  # Start smaller
        learning_rate=0.001,
        max_queries=2000,  # Test with 2K queries first
        temperature=0.1
    )
    
    print("\nCollaborative filtering pre-training complete!")
    print("Next step: Integrate with main two-tower model for fine-tuning")