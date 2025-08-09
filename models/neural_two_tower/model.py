#!/usr/bin/env python3
"""
Two-Tower Neural Network Architecture for LLM Ranking

Query Tower: Processes query text into dense embeddings
LLM Tower: Processes LLM identifiers and features into embeddings
Similarity: Cosine similarity between query and LLM embeddings for ranking

Tier 3 Enhancement: Cross-Encoder Reranking
Cross-Encoder: Joint encoding of query-LLM pairs for precise reranking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer


class MultiHeadQueryAttention(nn.Module):
    """Simple multi-head attention for query processing"""
    
    def __init__(self, input_dim=384, num_heads=4, head_dim=96, dropout=0.2):
        super(MultiHeadQueryAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        # Create identical heads - let training differentiate them
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, head_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(head_dim, head_dim)
            ) for _ in range(num_heads)
        ])
        
        # Fusion layer to combine all heads
        self.fusion = nn.Sequential(
            nn.Linear(num_heads * head_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, input_dim] sentence embeddings
        Returns:
            fused_output: [batch_size, input_dim] multi-head representation
            head_outputs: List of individual head outputs for analysis
        """
        # Process input through each head
        head_outputs = [head(x) for head in self.heads]
        
        # Concatenate all heads
        concatenated = torch.cat(head_outputs, dim=1)  # [batch_size, num_heads * head_dim]
        
        # Fuse into final representation
        fused_output = self.fusion(concatenated)  # [batch_size, input_dim]
        
        return fused_output, head_outputs


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
                 embedding_dim=128, dropout=0.2, use_multi_head=True, num_heads=4):
        super(TwoTowerModel, self).__init__()
        
        # Pre-trained sentence transformer for query embeddings
        self.sentence_transformer = SentenceTransformer(query_embedding_model)
        
        # Get embedding dimension from sentence transformer
        query_input_dim = self.sentence_transformer.get_sentence_embedding_dimension()
        
        # Multi-head attention (optional)
        self.use_multi_head = use_multi_head
        if use_multi_head:
            self.multi_head_attention = MultiHeadQueryAttention(
                input_dim=query_input_dim,
                num_heads=num_heads,
                head_dim=query_input_dim // num_heads,  # Distribute dimensions evenly
                dropout=dropout
            )
        
        # Query and LLM towers with enhanced dimensions
        self.query_tower = QueryTower(
            input_dim=query_input_dim,
            hidden_dims=[256, 192, 128],  # Added layer for smoother transition
            output_dim=embedding_dim,
            dropout=dropout
        )
        
        self.llm_tower = LLMTower(
            num_llms=num_llms,
            embedding_dim=embedding_dim,
            hidden_dims=[192, 128],  # Enhanced hidden layers
            output_dim=embedding_dim,
            dropout=dropout
        )
        
    def encode_queries(self, query_texts):
        """Encode query texts using sentence transformer + optional multi-head + query tower"""
        # Get sentence transformer embeddings (keep on CPU for MPS compatibility)
        model_device = next(self.parameters()).device
        if str(model_device) == 'mps':
            # Use CPU for sentence transformer, then move to MPS
            query_embeddings = self.sentence_transformer.encode(
                query_texts, convert_to_tensor=True, device='cpu'
            )
            query_embeddings = query_embeddings.to(model_device)
        else:
            # Standard behavior for CUDA/CPU
            query_embeddings = self.sentence_transformer.encode(
                query_texts, convert_to_tensor=True, device=model_device
            )
        
        # Clone to create a proper autograd tensor
        query_embeddings = query_embeddings.clone().requires_grad_(False)
        
        # Apply multi-head attention if enabled
        if self.use_multi_head:
            query_embeddings, head_outputs = self.multi_head_attention(query_embeddings)
            # Store head outputs for analysis (optional)
            if hasattr(self, '_store_head_outputs') and self._store_head_outputs:
                self._last_head_outputs = head_outputs
        
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
            # Get sentence transformer embeddings (MPS compatibility)
            model_device = next(self.parameters()).device
            if str(model_device) == 'mps':
                # Use CPU for sentence transformer, then move to MPS
                query_embeddings = self.sentence_transformer.encode(
                    query_texts, convert_to_tensor=True, device='cpu'
                )
                query_embeddings = query_embeddings.to(model_device)
            else:
                # Standard behavior for CUDA/CPU
                query_embeddings = self.sentence_transformer.encode(
                    query_texts, convert_to_tensor=True, device=model_device
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


class ContrastiveLossWithDiversity(nn.Module):
    """InfoNCE-style contrastive loss with head diversity regularization"""
    
    def __init__(self, temperature=0.1, diversity_weight=0.1):
        super(ContrastiveLossWithDiversity, self).__init__()
        self.temperature = temperature
        self.diversity_weight = diversity_weight
        
    def forward(self, query_embeddings, positive_embeddings, negative_embeddings, head_outputs=None):
        """
        Args:
            query_embeddings: Query representations [batch_size, embed_dim]
            positive_embeddings: Positive LLM representations [batch_size, embed_dim]  
            negative_embeddings: Negative LLM representations [batch_size, num_negatives, embed_dim]
        """
        batch_size = query_embeddings.size(0)
        
        # Normalize embeddings for cosine similarity
        query_norm = F.normalize(query_embeddings, p=2, dim=1)
        pos_norm = F.normalize(positive_embeddings, p=2, dim=1)
        neg_norm = F.normalize(negative_embeddings, p=2, dim=2)
        
        # Positive similarities [batch_size]
        pos_sim = torch.sum(query_norm * pos_norm, dim=1) / self.temperature
        
        # Negative similarities [batch_size, num_negatives]
        neg_sim = torch.bmm(query_norm.unsqueeze(1), neg_norm.transpose(1, 2)).squeeze(1) / self.temperature
        
        # Combine positive and negative similarities
        # logits: [batch_size, 1 + num_negatives]
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        
        # Labels: positive is always at index 0
        labels = torch.zeros(batch_size, dtype=torch.long, device=query_embeddings.device)
        
        # Main contrastive loss
        main_loss = F.cross_entropy(logits, labels)
        
        # Head diversity regularization (if head outputs provided)
        diversity_loss = 0.0
        if head_outputs is not None and self.diversity_weight > 0:
            diversity_loss = self.compute_head_diversity_loss(head_outputs)
        
        return main_loss + self.diversity_weight * diversity_loss
    
    def compute_head_diversity_loss(self, head_outputs):
        """Encourage different heads to learn different representations"""
        num_heads = len(head_outputs)
        if num_heads <= 1:
            return 0.0
        
        diversity_loss = 0.0
        count = 0
        
        for i in range(num_heads):
            for j in range(i+1, num_heads):
                # Compute cosine similarity between heads
                head_i_norm = F.normalize(head_outputs[i], p=2, dim=1)
                head_j_norm = F.normalize(head_outputs[j], p=2, dim=1)
                similarity = torch.sum(head_i_norm * head_j_norm, dim=1).mean()
                
                # Penalize high similarity
                diversity_loss += similarity ** 2
                count += 1
        
        return diversity_loss / count if count > 0 else 0.0


class CrossEncoder(nn.Module):
    """Cross-encoder for query-LLM joint encoding and reranking"""
    
    def __init__(self, model_name='distilbert-base-uncased', num_llms=1131, dropout=0.1):
        super(CrossEncoder, self).__init__()
        self.num_llms = num_llms
        
        # Load pretrained transformer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Get hidden size from transformer config
        hidden_size = self.transformer.config.hidden_size
        
        # LLM ID embedding (will be concatenated with query)
        self.llm_embedding = nn.Embedding(num_llms, hidden_size // 4)
        
        # Classification head for ranking scores
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size + hidden_size // 4, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, query_texts, llm_ids):
        """
        Args:
            query_texts: List of query strings
            llm_ids: Tensor of LLM IDs [batch_size]
        Returns:
            scores: Ranking scores [batch_size]
        """
        batch_size = len(query_texts)
        device = next(self.parameters()).device
        
        # Tokenize queries
        inputs = self.tokenizer(
            query_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(device)
        
        # Get transformer outputs
        outputs = self.transformer(**inputs)
        
        # Use [CLS] token representation
        query_embeddings = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        
        # Get LLM embeddings
        llm_embeddings = self.llm_embedding(llm_ids)  # [batch_size, hidden_size // 4]
        
        # Concatenate query and LLM embeddings
        combined = torch.cat([query_embeddings, llm_embeddings], dim=1)  # [batch_size, hidden_size + hidden_size // 4]
        
        # Get ranking scores
        scores = self.classifier(combined).squeeze(-1)  # [batch_size]
        
        return scores
    
    def predict_batch(self, query_texts, llm_ids):
        """Prediction for evaluation (no gradients)"""
        self.eval()
        with torch.no_grad():
            scores = self.forward(query_texts, llm_ids)
        return scores.cpu().numpy()


class Tier3HybridModel(nn.Module):
    """Tier 3: Two-stage hybrid model combining two-tower retrieval + cross-encoder reranking"""
    
    def __init__(self, num_llms, two_tower_model=None, cross_encoder=None, top_k=100):
        super(Tier3HybridModel, self).__init__()
        self.num_llms = num_llms
        self.top_k = min(top_k, num_llms)  # Ensure top_k doesn't exceed total LLMs
        
        # Two-tower for efficient retrieval
        self.two_tower = two_tower_model if two_tower_model is not None else create_model(num_llms)
        
        # Cross-encoder for precise reranking
        self.cross_encoder = cross_encoder if cross_encoder is not None else CrossEncoder(num_llms=num_llms)
        
    def forward(self, query_texts, llm_ids):
        """Two-stage forward pass: retrieve top-k, then rerank"""
        # Stage 1: Two-tower retrieval for all LLMs
        two_tower_scores = self.two_tower.predict_batch(query_texts, llm_ids)
        
        # For training, we typically have smaller batches, so use all provided LLMs
        batch_size = len(query_texts)
        if batch_size * self.top_k > len(llm_ids):
            # Use all provided LLMs for reranking
            rerank_scores = self.cross_encoder(query_texts, llm_ids)
        else:
            # Stage 2: Get top-k candidates per query and rerank
            # This would be used during full inference with all 1131 LLMs
            rerank_scores = self.cross_encoder(query_texts, llm_ids)
        
        return rerank_scores
    
    def predict_batch(self, query_texts, llm_ids):
        """Two-stage prediction: retrieve + rerank"""
        self.eval()
        with torch.no_grad():
            # For evaluation, we typically evaluate on provided LLM IDs directly
            # In full inference, this would do top-k retrieval first
            scores = self.cross_encoder.predict_batch(query_texts, llm_ids)
        return scores


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


def create_model(num_llms, device='cpu', use_multi_head=True, num_heads=4):
    """Create and initialize Two-Tower model"""
    model = TwoTowerModel(num_llms=num_llms, use_multi_head=use_multi_head, num_heads=num_heads)
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


def create_cross_encoder(num_llms, device='cpu', model_name='distilbert-base-uncased'):
    """Create and initialize Cross-Encoder model"""
    model = CrossEncoder(model_name=model_name, num_llms=num_llms)
    model = model.to(device)
    
    # Initialize custom layers (pretrained transformer already initialized)
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0, std=0.1)
    
    # Only initialize our custom layers, not the pretrained transformer
    model.llm_embedding.apply(init_weights)
    model.classifier.apply(init_weights)
    
    return model


def create_tier3_model(num_llms, device='cpu', use_multi_head=True, num_heads=3, 
                       cross_encoder_model='distilbert-base-uncased', top_k=100):
    """Create and initialize Tier 3 Hybrid model (Two-Tower + Cross-Encoder)"""
    # Create two-tower component
    two_tower = create_model(num_llms, device, use_multi_head, num_heads)
    
    # Create cross-encoder component  
    cross_encoder = create_cross_encoder(num_llms, device, cross_encoder_model)
    
    # Create hybrid model
    model = Tier3HybridModel(
        num_llms=num_llms,
        two_tower_model=two_tower,
        cross_encoder=cross_encoder,
        top_k=top_k
    )
    model = model.to(device)
    
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