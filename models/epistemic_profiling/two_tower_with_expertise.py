#!/usr/bin/env python3
"""
Neural Two-Tower model with expertise profile features.
Integrates bilateral profiles and expertise matching into the two-tower architecture.
"""

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import KFold
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from tqdm import tqdm
import time
import sys
sys.path.append('../..')
from shared.utils.evaluation import calculate_metrics, save_standardized_results

class ExpertiseDataset(Dataset):
    """Dataset with expertise features for two-tower model"""
    
    def __init__(self, queries, llm_ids, labels, query_encoder, bilateral_profiles, 
                 expertise_profiles, cluster_assignments):
        self.queries = queries
        self.llm_ids = llm_ids
        self.labels = labels
        self.query_encoder = query_encoder
        self.bilateral_profiles = bilateral_profiles
        self.expertise_profiles = expertise_profiles
        self.cluster_assignments = cluster_assignments
        
        # Pre-encode all unique queries for efficiency
        unique_queries = list(set(queries))
        print(f"Encoding {len(unique_queries)} unique queries...")
        self.query_embeddings = {}
        for q in tqdm(unique_queries, desc="Encoding queries"):
            self.query_embeddings[q] = query_encoder.encode([q], show_progress_bar=False)[0]
        
        # Create LLM ID to index mapping
        unique_llms = sorted(list(set(llm_ids)))
        self.llm_to_idx = {llm: idx for idx, llm in enumerate(unique_llms)}
        self.n_llms = len(unique_llms)
        print(f"Found {self.n_llms} unique LLMs")
    
    def __len__(self):
        return len(self.queries)
    
    def __getitem__(self, idx):
        query = self.queries[idx]
        llm_id = self.llm_ids[idx]
        label = self.labels[idx]
        
        # Query embedding
        query_emb = self.query_embeddings[query]
        
        # LLM index
        llm_idx = self.llm_to_idx.get(llm_id, 0)
        
        # Bilateral profile features (5 features)
        if llm_id in self.bilateral_profiles:
            bp = self.bilateral_profiles[llm_id]
            bilateral_features = np.array([
                bp['confident_correct'],
                bp['overconfident_wrong'],
                bp['uncertain'],
                bp['inconsistent'],
                bp['reliability']
            ], dtype=np.float32)
        else:
            bilateral_features = np.array([0.5, 0.5, 0.0, 0.0, 0.0], dtype=np.float32)
        
        # Cluster features (6 features: 5 one-hot + 1 is_elite)
        cluster_features = np.zeros(6, dtype=np.float32)
        if llm_id in self.cluster_assignments:
            cluster = self.cluster_assignments[llm_id]
            if 0 <= cluster < 5:
                cluster_features[cluster] = 1.0
            cluster_features[5] = 1.0 if cluster in [1, 2] else 0.0
        
        # Expertise matching features (9 features)
        expertise_features = self.compute_expertise_features(query_emb, llm_id)
        
        return {
            'query_emb': torch.FloatTensor(query_emb),
            'llm_idx': torch.LongTensor([llm_idx]),
            'bilateral': torch.FloatTensor(bilateral_features),
            'cluster': torch.FloatTensor(cluster_features),
            'expertise': torch.FloatTensor(expertise_features),
            'label': torch.FloatTensor([label])
        }
    
    def compute_expertise_features(self, query_emb, llm_id):
        """Compute expertise matching features for query-LLM pair"""
        
        if llm_id not in self.expertise_profiles:
            return np.zeros(9, dtype=np.float32)
        
        profile = self.expertise_profiles[llm_id]
        if 'expertise' not in profile:
            return np.zeros(9, dtype=np.float32)
        
        best_similarity = 0
        best_quality = 0
        best_features = None
        second_best_similarity = 0
        
        for cluster_id, cluster_info in profile['expertise'].items():
            centroid = np.array(cluster_info['centroid'])
            similarity = cosine_similarity([query_emb], [centroid])[0][0]
            
            if similarity > best_similarity:
                second_best_similarity = best_similarity
                best_similarity = similarity
                best_quality = cluster_info['quality']
                best_features = cluster_info['features']
        
        if best_features:
            return np.array([
                best_similarity,
                best_quality,
                best_similarity * best_quality,
                best_similarity - second_best_similarity,  # confidence
                best_features.get('avg_answer_length', 0) / 1000.0,
                best_features.get('non_answer_rate', 0),
                best_features.get('code_rate', 0),
                best_features.get('list_rate', 0),
                best_features.get('cluster_size', 100) / 100.0  # Use cluster_size, not size
            ], dtype=np.float32)
        
        return np.zeros(9, dtype=np.float32)


class TwoTowerWithExpertise(nn.Module):
    """Two-tower architecture augmented with expertise features"""
    
    def __init__(self, query_dim=384, n_llms=1131, llm_emb_dim=64):
        super().__init__()
        
        # Query tower: processes query embeddings + potentially expertise context
        self.query_tower = nn.Sequential(
            nn.Linear(query_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # LLM tower: processes LLM ID + bilateral + cluster + expertise features
        self.llm_embeddings = nn.Embedding(n_llms, llm_emb_dim)
        
        # Combine LLM embedding with profile features
        # Input: 64 (embedding) + 5 (bilateral) + 6 (cluster) + 9 (expertise) = 84
        self.llm_tower = nn.Sequential(
            nn.Linear(llm_emb_dim + 5 + 6 + 9, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Final similarity computation
        self.output = nn.Sequential(
            nn.Linear(64 + 64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
            # No sigmoid - using regression
        )
    
    def forward(self, query_emb, llm_idx, bilateral, cluster, expertise):
        # Process query
        query_features = self.query_tower(query_emb)
        
        # Process LLM
        llm_emb = self.llm_embeddings(llm_idx).squeeze(1)
        llm_input = torch.cat([llm_emb, bilateral, cluster, expertise], dim=1)
        llm_features = self.llm_tower(llm_input)
        
        # Combine and predict
        combined = torch.cat([query_features, llm_features], dim=1)
        score = self.output(combined)
        
        return score.squeeze()


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training", leave=False):
        query_emb = batch['query_emb'].to(device)
        llm_idx = batch['llm_idx'].to(device)
        bilateral = batch['bilateral'].to(device)
        cluster = batch['cluster'].to(device)
        expertise = batch['expertise'].to(device)
        labels = batch['label'].squeeze().to(device)
        
        optimizer.zero_grad()
        outputs = model(query_emb, llm_idx, bilateral, cluster, expertise)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """Evaluate model"""
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            query_emb = batch['query_emb'].to(device)
            llm_idx = batch['llm_idx'].to(device)
            bilateral = batch['bilateral'].to(device)
            cluster = batch['cluster'].to(device)
            expertise = batch['expertise'].to(device)
            labels = batch['label'].squeeze()
            
            outputs = model(query_emb, llm_idx, bilateral, cluster, expertise)
            
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(labels.numpy())
    
    return predictions, actuals


def main():
    """Run two-tower model with expertise features"""
    print("="*80)
    print("TWO-TOWER MODEL WITH EXPERTISE PROFILES")
    print("="*80)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load profiles
    print("\nLoading profile data...")
    
    with open('complete_profiles.json', 'r') as f:
        profiles = json.load(f)
        bilateral_profiles = {}
        for llm_id, profile in profiles.items():
            bilateral_profiles[llm_id] = {
                'inconsistent': float(profile[0]),
                'confident_correct': float(profile[2]),
                'overconfident_wrong': float(profile[6]),
                'uncertain': float(profile[8]),
                'reliability': float(profile[2]) - float(profile[6])
            }
    
    with open('all_llm_expertise_profiles.json', 'r') as f:
        expertise_profiles = json.load(f)
    
    with open('llm_clusters.json', 'r') as f:
        cluster_assignments = json.load(f)
    
    print(f"  Loaded {len(bilateral_profiles)} bilateral profiles")
    print(f"  Loaded {len(expertise_profiles)} expertise profiles")
    print(f"  Loaded {len(cluster_assignments)} cluster assignments")
    
    # Load training data
    print("\nLoading training data...")
    train_df = pd.read_csv('../../data/supervised_training_full.csv')
    print(f"Loaded {len(train_df)} training examples")
    
    # Initialize query encoder
    query_encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Prepare data
    queries = train_df['query_text'].tolist()
    llm_ids = train_df['llm_id'].tolist()
    labels = train_df['qrel'].tolist()
    query_ids = train_df['query_id'].tolist()
    
    # Create dataset
    dataset = ExpertiseDataset(
        queries, llm_ids, labels, query_encoder,
        bilateral_profiles, expertise_profiles, cluster_assignments
    )
    
    # 10-fold cross-validation
    unique_queries = sorted(set(query_ids))
    query_to_indices = defaultdict(list)
    for i, q in enumerate(query_ids):
        query_to_indices[q].append(i)
    
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    
    fold_results = []
    start_time = time.time()
    
    print("\n" + "="*80)
    print("RUNNING 10-FOLD CROSS-VALIDATION")
    print("="*80)
    
    for fold, (train_queries_idx, val_queries_idx) in enumerate(kf.split(unique_queries), 1):
        print(f"\nFold {fold}/10")
        
        # Get train/val queries
        train_queries = [unique_queries[i] for i in train_queries_idx]
        val_queries = [unique_queries[i] for i in val_queries_idx]
        
        # Get indices
        train_idx = []
        val_idx = []
        for q in train_queries:
            train_idx.extend(query_to_indices[q])
        for q in val_queries:
            val_idx.extend(query_to_indices[q])
        
        print(f"  Training: {len(train_idx)} examples from {len(train_queries)} queries")
        print(f"  Validation: {len(val_idx)} examples from {len(val_queries)} queries")
        
        # Create data loaders
        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        val_dataset = torch.utils.data.Subset(dataset, val_idx)
        
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
        
        # Initialize model
        model = TwoTowerWithExpertise(
            query_dim=384,
            n_llms=dataset.n_llms,
            llm_emb_dim=64
        ).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()  # Use MSE instead of BCE for regression targets
        
        # Train
        fold_start = time.time()
        best_loss = float('inf')
        patience = 3
        patience_counter = 0
        
        for epoch in range(20):  # Max 20 epochs
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            
            if train_loss < best_loss:
                best_loss = train_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break
        
        train_time = time.time() - fold_start
        
        # Evaluate
        predictions, actuals = evaluate(model, val_loader, device)
        
        # Group by query for metrics
        val_queries_in_fold = [query_ids[i] for i in val_idx]
        y_true_by_query = defaultdict(list)
        y_pred_by_query = defaultdict(list)
        
        for i, q in enumerate(val_queries_in_fold):
            y_true_by_query[q].append(actuals[i])
            y_pred_by_query[q].append(predictions[i])
        
        # Calculate metrics
        metrics = calculate_metrics(y_true_by_query, y_pred_by_query)
        
        print(f"  nDCG@10: {metrics['ndcg_10']:.4f}")
        print(f"  nDCG@5: {metrics['ndcg_5']:.4f}")
        print(f"  MRR: {metrics['mrr']:.4f}")
        print(f"  Training time: {train_time:.2f}s")
        
        fold_results.append({
            'fold': fold,
            'ndcg_10': metrics['ndcg_10'],
            'ndcg_5': metrics['ndcg_5'],
            'mrr': metrics['mrr'],
            'train_time': train_time,
            'n_queries': len(val_queries)
        })
    
    total_time = time.time() - start_time
    
    # Aggregate results
    ndcg_10_scores = [r['ndcg_10'] for r in fold_results]
    ndcg_5_scores = [r['ndcg_5'] for r in fold_results]
    mrr_scores = [r['mrr'] for r in fold_results]
    
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    print(f"\nnDCG@10: {np.mean(ndcg_10_scores):.4f} ± {np.std(ndcg_10_scores):.4f}")
    print(f"nDCG@5:  {np.mean(ndcg_5_scores):.4f} ± {np.std(ndcg_5_scores):.4f}")
    print(f"MRR:     {np.mean(mrr_scores):.4f} ± {np.std(mrr_scores):.4f}")
    print(f"\nTotal runtime: {total_time/60:.2f} minutes")
    
    # Save results
    results = {
        'evaluation_info': {
            'pipeline': 'Two-Tower with Expertise Profiles',
            'features': 'Query embeddings + LLM embeddings + Bilateral + Clusters + Expertise',
            'dataset': 'TREC 2025 Million LLMs Track',
            'total_examples': len(train_df),
            'unique_queries': len(unique_queries),
            'total_runtime_seconds': total_time,
            'total_runtime_minutes': total_time / 60
        },
        'performance_metrics': {
            'ndcg_10': {
                'mean': np.mean(ndcg_10_scores),
                'std': np.std(ndcg_10_scores),
                'min': np.min(ndcg_10_scores),
                'max': np.max(ndcg_10_scores)
            },
            'ndcg_5': {
                'mean': np.mean(ndcg_5_scores),
                'std': np.std(ndcg_5_scores),
                'min': np.min(ndcg_5_scores),
                'max': np.max(ndcg_5_scores)
            },
            'mrr': {
                'mean': np.mean(mrr_scores),
                'std': np.std(mrr_scores),
                'min': np.min(mrr_scores),
                'max': np.max(mrr_scores)
            }
        },
        'fold_results': fold_results
    }
    
    with open('two_tower_expertise_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to two_tower_expertise_results.json")
    
    # Compare to baseline
    print("\n" + "="*80)
    print("COMPARISON TO BASELINES")
    print("="*80)
    
    print("\nOriginal Two-Tower:          nDCG@10 = 0.4022")
    print(f"Two-Tower with expertise:    nDCG@10 = {results['performance_metrics']['ndcg_10']['mean']:.4f}")
    
    improvement = (results['performance_metrics']['ndcg_10']['mean'] - 0.4022) / 0.4022 * 100
    print(f"\nImprovement: {improvement:+.1f}%")


if __name__ == "__main__":
    main()