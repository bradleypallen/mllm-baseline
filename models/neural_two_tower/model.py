#!/usr/bin/env python3
"""
Two-Tower Neural Network Architecture for LLM Ranking

Query Tower: Processes query text into dense embeddings
LLM Tower: Processes LLM identifiers and features into embeddings
Similarity: Cosine similarity between query and LLM embeddings for ranking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer


class QueryTower(nn.Module):
    """Neural network tower for processing query text"""
    
    def __init__(self, input_dim=384, hidden_dims=[256, 128], output_dim=64, dropout=0.2):
        super(QueryTower, self).__init__()
        
        # Build layers dynamically
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer (no activation, no dropout)
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


class LLMTower(nn.Module):
    """Neural network tower for processing LLM identifiers and features"""
    
    def __init__(self, num_llms, embedding_dim=64, hidden_dims=[128], 
                 output_dim=64, dropout=0.2):
        super(LLMTower, self).__init__()
        
        # LLM embedding layer
        self.llm_embedding = nn.Embedding(num_llms, embedding_dim)
        
        # Build dense layers
        layers = []
        prev_dim = embedding_dim  # Start with embedding dimension
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, llm_ids):
        # Convert LLM IDs to embeddings
        embedded = self.llm_embedding(llm_ids)
        return self.network(embedded)


class TwoTowerModel(nn.Module):
    """Two-Tower architecture for query-LLM ranking"""
    
    def __init__(self, num_llms, query_embedding_model='all-MiniLM-L6-v2', 
                 embedding_dim=64, dropout=0.2):
        super(TwoTowerModel, self).__init__()
        
        # Pre-trained sentence transformer for query embeddings
        self.sentence_transformer = SentenceTransformer(query_embedding_model)
        
        # Get embedding dimension from sentence transformer
        query_input_dim = self.sentence_transformer.get_sentence_embedding_dimension()
        
        # Query and LLM towers
        self.query_tower = QueryTower(
            input_dim=query_input_dim,
            hidden_dims=[256, 128],
            output_dim=embedding_dim,
            dropout=dropout
        )
        
        self.llm_tower = LLMTower(
            num_llms=num_llms,
            embedding_dim=embedding_dim,
            hidden_dims=[128],
            output_dim=embedding_dim,
            dropout=dropout
        )
        
    def encode_queries(self, query_texts):
        """Encode query texts using sentence transformer + query tower"""
        # Get sentence transformer embeddings
        query_embeddings = self.sentence_transformer.encode(
            query_texts, convert_to_tensor=True, device=next(self.parameters()).device
        )
        
        # Clone to create a proper autograd tensor
        query_embeddings = query_embeddings.clone().requires_grad_(False)
        
        # Pass through query tower
        return self.query_tower(query_embeddings)
    
    def encode_llms(self, llm_ids):
        """Encode LLM IDs using LLM tower"""
        return self.llm_tower(llm_ids)
    
    def forward(self, query_texts, llm_ids):
        """Forward pass: compute similarity scores"""
        query_embeddings = self.encode_queries(query_texts)
        llm_embeddings = self.encode_llms(llm_ids)
        
        # Cosine similarity
        query_normalized = F.normalize(query_embeddings, p=2, dim=1)
        llm_normalized = F.normalize(llm_embeddings, p=2, dim=1)
        
        # Compute similarity scores
        similarity_scores = torch.sum(query_normalized * llm_normalized, dim=1)
        
        return similarity_scores
    
    def predict_batch(self, query_texts, llm_ids):
        """Prediction for evaluation (no gradients)"""
        self.eval()
        with torch.no_grad():
            # Get sentence transformer embeddings
            query_embeddings = self.sentence_transformer.encode(
                query_texts, convert_to_tensor=True, device=next(self.parameters()).device
            )
            llm_embeddings = self.encode_llms(llm_ids)
            
            # Pass through towers
            query_out = self.query_tower(query_embeddings)
            
            # Cosine similarity
            query_normalized = F.normalize(query_out, p=2, dim=1)
            llm_normalized = F.normalize(llm_embeddings, p=2, dim=1)
            
            # Compute similarity scores
            similarity_scores = torch.sum(query_normalized * llm_normalized, dim=1)
            
        return similarity_scores.cpu().numpy()


class RankingLoss(nn.Module):
    """Margin-based pairwise ranking loss"""
    
    def __init__(self, margin=1.0):
        super(RankingLoss, self).__init__()
        self.margin = margin
        
    def forward(self, positive_scores, negative_scores):
        """
        Args:
            positive_scores: Scores for relevant query-LLM pairs
            negative_scores: Scores for non-relevant query-LLM pairs
        """
        # Margin loss: max(0, margin - (pos_score - neg_score))
        loss = torch.clamp(self.margin - (positive_scores - negative_scores), min=0.0)
        return loss.mean()


class TripletLoss(nn.Module):
    """Triplet loss for ranking"""
    
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def forward(self, anchor_scores, positive_scores, negative_scores):
        """
        Args:
            anchor_scores: Anchor query embeddings
            positive_scores: Positive (relevant) LLM scores
            negative_scores: Negative (non-relevant) LLM scores
        """
        loss = torch.clamp(self.margin + negative_scores - positive_scores, min=0.0)
        return loss.mean()


def create_model(num_llms, device='cpu'):
    """Create and initialize Two-Tower model"""
    model = TwoTowerModel(num_llms=num_llms)
    model = model.to(device)
    
    # Initialize weights
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0, std=0.1)
    
    model.apply(init_weights)
    return model


if __name__ == "__main__":
    # Test model creation
    print("Testing Two-Tower Model Creation...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model with sample parameters
    num_llms = 1131  # From TREC dataset
    model = create_model(num_llms, device)
    
    print(f"Model created successfully with {num_llms} LLMs")
    print(f"Query tower: {model.query_tower}")
    print(f"LLM tower: {model.llm_tower}")
    
    # Test forward pass
    sample_queries = ["What is machine learning?", "How does neural networks work?"]
    sample_llm_ids = torch.tensor([0, 1], device=device)
    
    try:
        scores = model.predict_batch(sample_queries, sample_llm_ids)
        print(f"Forward pass successful. Output shape: {scores.shape}")
        print(f"Sample scores: {scores}")
    except Exception as e:
        print(f"Forward pass failed: {e}")
        import traceback
        traceback.print_exc()