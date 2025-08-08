#!/usr/bin/env python3
"""
Neural Data Loader for Two-Tower LLM Ranking

Handles batch creation, positive/negative sampling, and data preprocessing
for the neural ranking model.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import random


class HardNegativeMiner:
    """Hard negative mining for improved training efficiency"""
    
    def __init__(self, model=None, hard_ratio=0.5, temperature=0.1):
        """
        Args:
            model: Current model for scoring negatives
            hard_ratio: Fraction of negatives to select as "hard" (high-scoring)
            temperature: Temperature for sampling (lower = more deterministic)
        """
        self.model = model
        self.hard_ratio = hard_ratio
        self.temperature = temperature
        
    def mine_hard_negatives(self, query_text, negative_candidates, num_negatives):
        """
        Select hard negative examples based on current model predictions
        
        Args:
            query_text: Query text string
            negative_candidates: DataFrame of negative examples
            num_negatives: Number of negatives to select
            
        Returns:
            Selected negative examples (DataFrame)
        """
        if self.model is None or len(negative_candidates) <= num_negatives:
            # Fallback to random sampling
            return negative_candidates.sample(n=min(num_negatives, len(negative_candidates)), replace=True)
        
        try:
            # Get model predictions for all negative candidates
            self.model.eval()
            with torch.no_grad():
                device = next(self.model.parameters()).device
                
                # Prepare batch data
                query_texts = [query_text] * len(negative_candidates)
                llm_ids = torch.tensor(negative_candidates['llm_encoded'].values, 
                                     dtype=torch.long, device=device)
                
                # Get similarity scores
                scores = self.model.forward(query_texts, llm_ids).cpu().numpy()
            
            # Split into hard and easy negatives
            num_hard = int(num_negatives * self.hard_ratio)
            num_easy = num_negatives - num_hard
            
            # Select hard negatives (highest scoring)
            if num_hard > 0:
                top_indices = np.argpartition(scores, -min(num_hard, len(scores)))[-min(num_hard, len(scores)):]
                hard_negatives = negative_candidates.iloc[top_indices]
            else:
                hard_negatives = pd.DataFrame()
            
            # Select easy negatives (random from remaining)
            if num_easy > 0:
                remaining_indices = np.setdiff1d(np.arange(len(negative_candidates)), top_indices if num_hard > 0 else [])
                if len(remaining_indices) > 0:
                    easy_indices = np.random.choice(remaining_indices, size=min(num_easy, len(remaining_indices)), replace=True)
                    easy_negatives = negative_candidates.iloc[easy_indices]
                else:
                    easy_negatives = pd.DataFrame()
            else:
                easy_negatives = pd.DataFrame()
            
            # Combine hard and easy negatives
            if len(hard_negatives) > 0 and len(easy_negatives) > 0:
                selected = pd.concat([hard_negatives, easy_negatives])
            elif len(hard_negatives) > 0:
                selected = hard_negatives
            elif len(easy_negatives) > 0:
                selected = easy_negatives
            else:
                selected = negative_candidates.sample(n=min(num_negatives, len(negative_candidates)), replace=True)
                
            # Ensure we have exactly num_negatives samples
            if len(selected) < num_negatives:
                additional_needed = num_negatives - len(selected)
                additional = negative_candidates.sample(n=additional_needed, replace=True)
                selected = pd.concat([selected, additional])
            elif len(selected) > num_negatives:
                selected = selected.sample(n=num_negatives, replace=False)
                
            return selected
            
        except Exception as e:
            # Fallback to random sampling on any error
            print(f"Hard negative mining failed, using random sampling: {e}")
            return negative_candidates.sample(n=min(num_negatives, len(negative_candidates)), replace=True)


class LLMRankingDataset(Dataset):
    """Dataset for LLM ranking with positive/negative sampling"""
    
    def __init__(self, df, llm_encoder=None, negative_samples_per_positive=4, fit_encoder=True, hard_negative_miner=None):
        """
        Args:
            df: DataFrame with columns [query_id, query_text, llm_id, qrel]
            llm_encoder: LabelEncoder for LLM IDs (if None, creates new one)
            negative_samples_per_positive: Number of negative samples per positive
            fit_encoder: Whether to fit the encoder (False for test data)
        """
        self.df = df.copy()
        self.negative_samples_per_positive = negative_samples_per_positive
        self.hard_negative_miner = hard_negative_miner
        
        # Encode LLM IDs
        if llm_encoder is None:
            self.llm_encoder = LabelEncoder()
        else:
            self.llm_encoder = llm_encoder
            
        if fit_encoder:
            self.df['llm_encoded'] = self.llm_encoder.fit_transform(self.df['llm_id'])
        else:
            self.df['llm_encoded'] = self.llm_encoder.transform(self.df['llm_id'])
        
        # Map qrel scores to relevance (same as Random Forest baseline)
        qrel_mapping = {0: 0.0, 1: 1.0, 2: 0.7}
        self.df['relevance'] = self.df['qrel'].map(qrel_mapping)
        
        # Separate positive and negative examples
        self.positive_examples = self.df[self.df['relevance'] > 0].copy()
        self.negative_examples = self.df[self.df['relevance'] == 0].copy()
        
        # Group by query for efficient sampling
        self.positive_by_query = self.positive_examples.groupby('query_id')
        self.negative_by_query = self.negative_examples.groupby('query_id')
        
        # Get all query IDs that have both positive and negative examples
        pos_queries = set(self.positive_by_query.groups.keys())
        neg_queries = set(self.negative_by_query.groups.keys())
        self.valid_queries = list(pos_queries.intersection(neg_queries))
        
        print(f"Dataset created with {len(self.positive_examples)} positive and {len(self.negative_examples)} negative examples")
        print(f"Valid queries for training: {len(self.valid_queries)}")
        
    def __len__(self):
        # Length based on positive examples (each generates multiple neg samples)
        return len(self.positive_examples)
    
    def __getitem__(self, idx):
        """
        Returns a batch of positive and negative examples for a query
        """
        # Get positive example
        pos_example = self.positive_examples.iloc[idx]
        query_id = pos_example['query_id']
        
        # Skip if query doesn't have negative examples
        if query_id not in self.valid_queries:
            # Fallback to a random valid query
            query_id = random.choice(self.valid_queries)
            pos_examples = self.positive_by_query.get_group(query_id)
            pos_example = pos_examples.iloc[random.randint(0, len(pos_examples) - 1)]
        
        # Sample negative examples for this query
        neg_examples = self.negative_by_query.get_group(query_id)
        
        # Use hard negative mining if available, otherwise random sampling
        if self.hard_negative_miner is not None:
            try:
                neg_examples_sampled = self.hard_negative_miner.mine_hard_negatives(
                    pos_example['query_text'], 
                    neg_examples, 
                    self.negative_samples_per_positive
                )
            except:
                # Fallback to random sampling on any error
                neg_examples_sampled = self._random_sample_negatives(neg_examples)
        else:
            neg_examples_sampled = self._random_sample_negatives(neg_examples)
        
        # Return batch data
        return {
            'query_text': pos_example['query_text'],
            'query_id': pos_example['query_id'],
            'positive_llm': pos_example['llm_encoded'],
            'positive_relevance': pos_example['relevance'],
            'negative_llms': neg_examples_sampled['llm_encoded'].values,
            'negative_relevances': neg_examples_sampled['relevance'].values
        }
    
    def _random_sample_negatives(self, neg_examples):
        """Helper method for random negative sampling"""
        # Sample negatives (with replacement if needed)
        n_negatives = min(self.negative_samples_per_positive, len(neg_examples))
        if n_negatives < self.negative_samples_per_positive:
            # Sample with replacement
            neg_indices = np.random.choice(len(neg_examples), 
                                         size=self.negative_samples_per_positive, 
                                         replace=True)
        else:
            # Sample without replacement
            neg_indices = np.random.choice(len(neg_examples), 
                                         size=self.negative_samples_per_positive, 
                                         replace=False)
        
        return neg_examples.iloc[neg_indices]


def collate_fn(batch):
    """Custom collate function to handle batched positive/negative sampling"""
    query_texts = []
    query_ids = []
    positive_llms = []
    positive_relevances = []
    negative_llms = []
    negative_relevances = []
    
    for item in batch:
        query_texts.append(item['query_text'])
        query_ids.append(item['query_id'])
        positive_llms.append(item['positive_llm'])
        positive_relevances.append(item['positive_relevance'])
        negative_llms.extend(item['negative_llms'])
        negative_relevances.extend(item['negative_relevances'])
    
    return {
        'query_texts': query_texts,
        'query_ids': query_ids,
        'positive_llms': torch.tensor(positive_llms, dtype=torch.long),
        'positive_relevances': torch.tensor(positive_relevances, dtype=torch.float),
        'negative_llms': torch.tensor(negative_llms, dtype=torch.long),
        'negative_relevances': torch.tensor(negative_relevances, dtype=torch.float)
    }


class LLMEvaluationDataset(Dataset):
    """Simple dataset for evaluation (no negative sampling)"""
    
    def __init__(self, df, llm_encoder):
        self.df = df.copy()
        self.llm_encoder = llm_encoder
        
        # Encode LLMs
        self.df['llm_encoded'] = self.llm_encoder.transform(self.df['llm_id'])
        
        # Map qrel scores to relevance
        qrel_mapping = {0: 0.0, 1: 1.0, 2: 0.7}
        self.df['relevance'] = self.df['qrel'].map(qrel_mapping)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            'query_text': row['query_text'],
            'query_id': row['query_id'],
            'llm_encoded': row['llm_encoded'],
            'relevance': row['relevance'],
            'llm_id': row['llm_id']
        }


def create_data_loaders(train_df, val_df, batch_size=32, negative_samples=4):
    """Create training and validation data loaders"""
    
    # Create training dataset with negative sampling
    train_dataset = LLMRankingDataset(
        train_df, 
        negative_samples_per_positive=negative_samples,
        fit_encoder=True
    )
    
    # Create validation dataset (no negative sampling for evaluation)
    val_dataset = LLMEvaluationDataset(
        val_df, 
        llm_encoder=train_dataset.llm_encoder
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Use 0 for debugging, increase for performance
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 4,  # Larger batch size for evaluation
        shuffle=False
    )
    
    return train_loader, val_loader, train_dataset.llm_encoder


def load_data(data_path='../../data/supervised_training_full.csv'):
    """Load and preprocess the training data"""
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    print(f"Loaded {len(df)} examples")
    print(f"Unique queries: {df.query_id.nunique()}")
    print(f"Unique LLMs: {df.llm_id.nunique()}")
    
    # Print relevance distribution
    print("\nRelevance distribution:")
    rel_dist = df.qrel.value_counts().sort_index()
    for qrel, count in rel_dist.items():
        pct = count / len(df) * 100
        print(f"  qrel={qrel}: {count:,} ({pct:.1f}%)")
    
    return df


if __name__ == "__main__":
    # Test data loading
    print("Testing Neural Data Loader...")
    
    try:
        # Load data
        df = load_data()
        
        # Create simple train/val split for testing
        unique_queries = df.query_id.unique()
        np.random.shuffle(unique_queries)
        
        split_idx = int(0.8 * len(unique_queries))
        train_queries = unique_queries[:split_idx]
        val_queries = unique_queries[split_idx:]
        
        train_df = df[df.query_id.isin(train_queries)]
        val_df = df[df.query_id.isin(val_queries)]
        
        print(f"\nTrain set: {len(train_df)} examples from {len(train_queries)} queries")
        print(f"Val set: {len(val_df)} examples from {len(val_queries)} queries")
        
        # Create data loaders
        train_loader, val_loader, llm_encoder = create_data_loaders(
            train_df, val_df, batch_size=16, negative_samples=4
        )
        
        print(f"\nData loaders created successfully")
        print(f"LLM encoder vocabulary size: {len(llm_encoder.classes_)}")
        
        # Test a batch
        batch = next(iter(train_loader))
        print(f"\nSample training batch:")
        print(f"  Query texts: {len(batch['query_texts'])}")
        print(f"  Positive LLMs shape: {batch['positive_llms'].shape}")
        print(f"  Negative LLMs shape: {batch['negative_llms'].shape}")
        print(f"  First query: '{batch['query_texts'][0][:100]}...'")
        
        val_batch = next(iter(val_loader))
        print(f"\nSample validation batch:")
        print(f"  Query texts: {len(val_batch['query_text'])}")
        print(f"  LLM encoded shape: {val_batch['llm_encoded'].shape}")
        
        print("\nData loader test completed successfully!")
        
    except Exception as e:
        print(f"Data loader test failed: {e}")
        import traceback
        traceback.print_exc()