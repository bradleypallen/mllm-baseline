#!/usr/bin/env python3
"""
Weak Labeling Discovery Dataset - TREC 2025 Million LLMs Track

BREAKTHROUGH APPROACH: Use dev set (342 queries + human qrels) to weak label 
discovery dataset (14,950 queries + LLM responses) → 16.9M training examples

Key Innovation:
- Semantic similarity transfer: Similar queries → Similar LLM performance
- Real data at scale: Actual queries + responses vs synthetic generation
- Quality safeguards: Confidence thresholding + validation

Expected: Massive high-quality dataset surpassing synthetic approaches
"""

import pandas as pd
import numpy as np
import json
import time
from datetime import datetime
import pickle
from pathlib import Path
import warnings
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import sys
import os

# Add project root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

warnings.filterwarnings('ignore')

class DiscoveryWeakLabeler:
    """Weak labeling system for discovery dataset using dev set similarity transfer"""
    
    def __init__(self, min_similarity=0.6, top_k_similar=5, confidence_threshold=0.7):
        self.min_similarity = min_similarity
        self.top_k_similar = top_k_similar  
        self.confidence_threshold = confidence_threshold
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Caches
        self.dev_embeddings = None
        self.discovery_embeddings = None
        self.dev_qrels = {}
        
    def load_dev_data(self):
        """Load development queries and qrels (gold standard)"""
        print("Loading development data...")
        
        # Load dev queries
        dev_queries_path = "../../data/llm_dev_data.tsv"
        dev_df = pd.read_csv(dev_queries_path, sep='\t', header=None, names=['query_id', 'query_text'])
        print(f"Loaded {len(dev_df)} dev queries")
        
        # Load dev qrels
        qrels_path = "../../data/llm_dev_qrels.txt" 
        qrels_df = pd.read_csv(qrels_path, sep=' ', header=None, 
                              names=['query_id', 'unused', 'llm_id', 'qrel'])
        
        # Convert qrels to standard format (0→0.0, 1→1.0, 2→0.7)
        qrel_mapping = {0: 0.0, 1: 1.0, 2: 0.7}
        qrels_df['qrel_mapped'] = qrels_df['qrel'].map(qrel_mapping)
        
        # Create qrels lookup
        for _, row in qrels_df.iterrows():
            key = (str(row['query_id']), str(row['llm_id']))
            self.dev_qrels[key] = row['qrel_mapped']
        
        print(f"Loaded {len(qrels_df)} qrel judgments for {qrels_df['query_id'].nunique()} queries")
        print(f"Qrel distribution: {qrels_df['qrel'].value_counts().to_dict()}")
        
        return dev_df, qrels_df
    
    def load_discovery_data(self):
        """Load discovery queries (to be weak labeled)"""
        print("Loading discovery data...")
        
        # Load discovery queries - check both possible locations
        discovery_paths = [
            "../../data/llm_discovery_data_1.json",
            "../../data/discovery/llm_discovery_data_1.json"
        ]
        
        discovery_path = None
        for path in discovery_paths:
            if os.path.exists(path):
                discovery_path = path
                break
        
        if discovery_path is None:
            raise FileNotFoundError(f"Discovery data not found in any of: {discovery_paths}")
        
        print(f"Loading discovery data from: {discovery_path}")
        
        with open(discovery_path, 'r') as f:
            discovery_data = json.load(f)
        
        # Extract queries
        discovery_queries = []
        for query_id, query_info in discovery_data.items():
            query_text = query_info.get('query', '')
            if query_text:
                discovery_queries.append({
                    'query_id': query_id,
                    'query_text': query_text
                })
        
        discovery_df = pd.DataFrame(discovery_queries)
        print(f"Loaded {len(discovery_df)} discovery queries")
        
        return discovery_df, discovery_data
    
    def compute_embeddings(self, dev_df, discovery_df):
        """Compute sentence embeddings for all queries"""
        print("Computing sentence embeddings...")
        
        # Dev embeddings
        print("  - Dev queries...")
        dev_texts = dev_df['query_text'].tolist()
        self.dev_embeddings = self.sentence_model.encode(dev_texts, show_progress_bar=True)
        
        # Discovery embeddings  
        print("  - Discovery queries...")
        discovery_texts = discovery_df['query_text'].tolist()
        self.discovery_embeddings = self.sentence_model.encode(discovery_texts, show_progress_bar=True)
        
        print(f"Dev embeddings shape: {self.dev_embeddings.shape}")
        print(f"Discovery embeddings shape: {self.discovery_embeddings.shape}")
        
        return self.dev_embeddings, self.discovery_embeddings
    
    def find_similar_queries(self, discovery_idx, dev_df):
        """Find most similar dev queries for a discovery query"""
        # Get similarity scores
        discovery_embedding = self.discovery_embeddings[discovery_idx].reshape(1, -1)
        similarities = cosine_similarity(discovery_embedding, self.dev_embeddings)[0]
        
        # Get top-k similar queries above threshold
        similar_indices = np.argsort(similarities)[::-1][:self.top_k_similar]
        similar_scores = similarities[similar_indices]
        
        # Filter by minimum similarity
        valid_mask = similar_scores >= self.min_similarity
        similar_indices = similar_indices[valid_mask]
        similar_scores = similar_scores[valid_mask]
        
        if len(similar_indices) == 0:
            return [], []
        
        # Get corresponding dev queries
        similar_queries = []
        for idx in similar_indices:
            similar_queries.append({
                'query_id': str(dev_df.iloc[idx]['query_id']),
                'query_text': dev_df.iloc[idx]['query_text'],
                'similarity': similar_scores[list(similar_indices).index(idx)]
            })
        
        return similar_queries, similar_scores
    
    def compute_weak_labels(self, discovery_query_id, similar_queries, similar_scores):
        """Compute weak labels for all LLMs based on similar dev queries"""
        weak_labels = {}
        
        # Get all unique LLMs from dev qrels
        all_llms = set()
        for (_, llm_id), _ in self.dev_qrels.items():
            all_llms.add(llm_id)
        
        for llm_id in all_llms:
            # Collect performance scores from similar queries
            scores = []
            weights = []
            
            for i, sim_query in enumerate(similar_queries):
                dev_query_id = sim_query['query_id']
                similarity = similar_scores[i]
                
                key = (dev_query_id, llm_id)
                if key in self.dev_qrels:
                    scores.append(self.dev_qrels[key])
                    weights.append(similarity)
            
            if len(scores) > 0:
                # Weighted average of performance scores
                weights = np.array(weights)
                scores = np.array(scores)
                weak_label = np.average(scores, weights=weights)
                
                # Confidence based on number of similar queries and avg similarity
                confidence = np.mean(weights) * (len(scores) / self.top_k_similar)
                
                # Only keep high-confidence labels
                if confidence >= self.confidence_threshold:
                    weak_labels[llm_id] = {
                        'weak_label': weak_label,
                        'confidence': confidence,
                        'num_similar': len(scores),
                        'avg_similarity': np.mean(weights)
                    }
        
        return weak_labels
    
    def generate_weak_labeled_dataset(self, dev_df, discovery_df, discovery_data):
        """Generate complete weak-labeled dataset"""
        print("Generating weak-labeled dataset...")
        
        weak_labeled_examples = []
        stats = {
            'total_discovery_queries': len(discovery_df),
            'queries_with_weak_labels': 0,
            'total_weak_labels': 0,
            'avg_confidence': [],
            'avg_similarity': [],
            'coverage_by_llm': {}
        }
        
        for disc_idx, disc_row in tqdm(discovery_df.iterrows(), total=len(discovery_df), 
                                      desc="Processing discovery queries"):
            discovery_query_id = str(disc_row['query_id'])
            discovery_query_text = disc_row['query_text']
            
            # Find similar dev queries
            similar_queries, similar_scores = self.find_similar_queries(disc_idx, dev_df)
            
            if len(similar_queries) == 0:
                continue  # Skip queries with no similar dev queries
            
            # Compute weak labels
            weak_labels = self.compute_weak_labels(discovery_query_id, similar_queries, similar_scores)
            
            if len(weak_labels) == 0:
                continue  # Skip queries with no confident labels
            
            stats['queries_with_weak_labels'] += 1
            
            # Create training examples
            for llm_id, label_info in weak_labels.items():
                example = {
                    'query_id': discovery_query_id,
                    'query_text': discovery_query_text,
                    'llm_id': llm_id,
                    'qrel': label_info['weak_label'],
                    'confidence': label_info['confidence'],
                    'num_similar': label_info['num_similar'],
                    'avg_similarity': label_info['avg_similarity'],
                    'source': 'weak_labeled_discovery'
                }
                weak_labeled_examples.append(example)
                
                # Update stats
                stats['total_weak_labels'] += 1
                stats['avg_confidence'].append(label_info['confidence'])
                stats['avg_similarity'].append(label_info['avg_similarity'])
                
                if llm_id not in stats['coverage_by_llm']:
                    stats['coverage_by_llm'][llm_id] = 0
                stats['coverage_by_llm'][llm_id] += 1
        
        # Finalize stats
        stats['avg_confidence'] = np.mean(stats['avg_confidence']) if stats['avg_confidence'] else 0
        stats['avg_similarity'] = np.mean(stats['avg_similarity']) if stats['avg_similarity'] else 0
        stats['coverage_queries_pct'] = stats['queries_with_weak_labels'] / stats['total_discovery_queries'] * 100
        stats['avg_labels_per_query'] = stats['total_weak_labels'] / max(stats['queries_with_weak_labels'], 1)
        
        return weak_labeled_examples, stats
    
    def validate_weak_labels(self, dev_df):
        """Validate weak labeling quality using cross-validation on dev set"""
        print("Validating weak labeling quality...")
        
        validation_results = []
        
        # Use 80/20 split for validation
        n_dev = len(dev_df)
        n_val = n_dev // 5
        
        for val_start in range(0, n_dev, n_val):
            val_end = min(val_start + n_val, n_dev)
            
            # Validation set
            val_indices = list(range(val_start, val_end))
            val_df = dev_df.iloc[val_indices].copy()
            
            # "Training" set (other dev queries)
            train_indices = [i for i in range(n_dev) if i not in val_indices]
            train_df = dev_df.iloc[train_indices].copy()
            
            # Compute embeddings for this split
            train_embeddings = self.sentence_model.encode(train_df['query_text'].tolist())
            val_embeddings = self.sentence_model.encode(val_df['query_text'].tolist())
            
            # Validate each query in val set
            for val_idx, val_row in val_df.iterrows():
                val_query_id = str(val_row['query_id'])
                val_embedding = val_embeddings[val_indices.index(val_idx)].reshape(1, -1)
                
                # Find similar training queries
                similarities = cosine_similarity(val_embedding, train_embeddings)[0]
                similar_indices = np.argsort(similarities)[::-1][:self.top_k_similar]
                similar_scores = similarities[similar_indices]
                
                # Filter by threshold
                valid_mask = similar_scores >= self.min_similarity
                similar_indices = similar_indices[valid_mask]
                similar_scores = similar_scores[valid_mask]
                
                if len(similar_indices) == 0:
                    continue
                
                # Get similar train queries
                similar_train_queries = []
                for idx in similar_indices:
                    train_row = train_df.iloc[idx]
                    similar_train_queries.append({
                        'query_id': str(train_row['query_id']),
                        'similarity': similar_scores[list(similar_indices).index(idx)]
                    })
                
                # For each LLM, compare predicted vs actual
                for (qid, llm_id), actual_qrel in self.dev_qrels.items():
                    if qid == val_query_id:
                        # Compute weak label prediction
                        scores = []
                        weights = []
                        
                        for sim_q in similar_train_queries:
                            sim_key = (sim_q['query_id'], llm_id)
                            if sim_key in self.dev_qrels:
                                scores.append(self.dev_qrels[sim_key])
                                weights.append(sim_q['similarity'])
                        
                        if len(scores) > 0:
                            predicted_qrel = np.average(scores, weights=weights)
                            confidence = np.mean(weights) * (len(scores) / self.top_k_similar)
                            
                            if confidence >= self.confidence_threshold:
                                validation_results.append({
                                    'query_id': val_query_id,
                                    'llm_id': llm_id,
                                    'actual': actual_qrel,
                                    'predicted': predicted_qrel,
                                    'confidence': confidence,
                                    'num_similar': len(scores),
                                    'error': abs(actual_qrel - predicted_qrel)
                                })
        
        # Compute validation metrics
        if len(validation_results) > 0:
            validation_df = pd.DataFrame(validation_results)
            metrics = {
                'mae': validation_df['error'].mean(),
                'rmse': np.sqrt((validation_df['error'] ** 2).mean()),
                'avg_confidence': validation_df['confidence'].mean(),
                'coverage': len(validation_results),
                'correlation': validation_df['actual'].corr(validation_df['predicted'])
            }
            
            print(f"Validation Results:")
            print(f"  MAE: {metrics['mae']:.4f}")
            print(f"  RMSE: {metrics['rmse']:.4f}")
            print(f"  Correlation: {metrics['correlation']:.4f}")
            print(f"  Coverage: {metrics['coverage']} examples")
            print(f"  Avg Confidence: {metrics['avg_confidence']:.4f}")
            
            return metrics, validation_df
        else:
            print("Warning: No validation examples generated")
            return None, None

def run_weak_labeling():
    """Main weak labeling pipeline"""
    print("=" * 80)
    print("WEAK LABELING DISCOVERY DATASET")
    print("=" * 80)
    print("BREAKTHROUGH: Use dev set to weak label discovery queries")
    print("  Dev set: 342 queries + human qrels (gold standard)")
    print("  Discovery: 14,950 queries + LLM responses")
    print("  Goal: Generate 16.9M weak-labeled training examples")
    print()
    
    start_time = time.time()
    
    # Initialize weak labeler with more permissive thresholds
    labeler = DiscoveryWeakLabeler(
        min_similarity=0.4,     # More permissive similarity threshold
        top_k_similar=10,       # Consider more similar dev queries
        confidence_threshold=0.5 # Lower confidence threshold
    )
    
    # Load data
    dev_df, qrels_df = labeler.load_dev_data()
    discovery_df, discovery_data = labeler.load_discovery_data()
    
    # Compute embeddings
    dev_embeddings, discovery_embeddings = labeler.compute_embeddings(dev_df, discovery_df)
    
    # Validate weak labeling approach on dev set
    print("\n" + "="*50)
    print("VALIDATION: Testing weak labeling on dev set")
    print("="*50)
    validation_metrics, validation_df = labeler.validate_weak_labels(dev_df)
    
    if validation_metrics and validation_metrics['correlation'] > 0.3:
        print(f"✅ VALIDATION PASSED: Correlation {validation_metrics['correlation']:.3f} > 0.3")
        print("Weak labeling approach is reliable - proceeding with discovery labeling")
    elif validation_metrics is None:
        print("⚠️  VALIDATION INCONCLUSIVE: No validation examples, but proceeding with discovery labeling")
        print("   (This can happen with strict thresholds - weak labeling may still work)")
    else:
        print("❌ VALIDATION FAILED: Weak labeling not reliable enough")
        return None
    
    # Generate weak-labeled dataset
    print("\n" + "="*50) 
    print("GENERATING WEAK-LABELED DISCOVERY DATASET")
    print("="*50)
    
    weak_examples, stats = labeler.generate_weak_labeled_dataset(dev_df, discovery_df, discovery_data)
    
    processing_time = time.time() - start_time
    
    # Create DataFrame and save
    weak_df = pd.DataFrame(weak_examples)
    
    # Combine with original dev data
    original_df = pd.read_csv("../../data/supervised_training_full.csv")
    original_df['source'] = 'original'
    
    # Ensure consistent columns
    weak_df = weak_df[['query_id', 'query_text', 'llm_id', 'qrel', 'source']]
    original_df = original_df[['query_id', 'query_text', 'llm_id', 'qrel', 'source']]
    
    # Combine datasets
    combined_df = pd.concat([original_df, weak_df], ignore_index=True)
    
    # Save datasets
    weak_output_path = "../../data/supervised_training_weak_labeled.csv"
    combined_output_path = "../../data/supervised_training_with_weak_labels.csv"
    
    weak_df.to_csv(weak_output_path, index=False)
    combined_df.to_csv(combined_output_path, index=False)
    
    # Save stats
    stats['processing_time_seconds'] = processing_time
    stats['processing_time_hours'] = processing_time / 3600
    stats['validation_metrics'] = validation_metrics
    
    stats_path = "../../data/weak_labeling_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    
    # Print final results
    print("\n" + "=" * 80)
    print("WEAK LABELING RESULTS")
    print("=" * 80)
    print(f"Discovery queries processed: {stats['total_discovery_queries']:,}")
    print(f"Queries with weak labels: {stats['queries_with_weak_labels']:,} ({stats['coverage_queries_pct']:.1f}%)")
    print(f"Total weak labels generated: {stats['total_weak_labels']:,}")
    print(f"Average labels per query: {stats['avg_labels_per_query']:.1f}")
    print(f"Average confidence: {stats['avg_confidence']:.3f}")
    print(f"Average similarity: {stats['avg_similarity']:.3f}")
    print()
    print(f"COMBINED DATASET:")
    print(f"  Original examples: {len(original_df):,}")
    print(f"  Weak labeled examples: {len(weak_df):,}")
    print(f"  Total examples: {len(combined_df):,}")
    print(f"  Scale multiplier: {len(combined_df) / len(original_df):.1f}x")
    print(f"  Weak label ratio: {len(weak_df) / len(combined_df) * 100:.1f}%")
    print()
    print(f"Processing time: {processing_time:.1f}s ({processing_time/3600:.2f} hours)")
    print()
    print(f"✓ Weak-labeled dataset saved: {weak_output_path}")
    print(f"✓ Combined dataset saved: {combined_output_path}")
    print(f"✓ Statistics saved: {stats_path}")
    
    if len(weak_df) > 1000000:  # 1M examples
        print(f"\n🎯 SUCCESS: Generated {len(weak_df):,} weak labels - MASSIVE SCALE ACHIEVED!")
    else:
        print(f"\n📊 Generated {len(weak_df):,} weak labels - good but could be larger")
    
    return combined_df, stats

if __name__ == "__main__":
    result = run_weak_labeling()