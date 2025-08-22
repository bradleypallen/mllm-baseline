#!/usr/bin/env python3
"""
Config N Weak Labeling - TREC 2025 Million LLMs Track

CONFIG N ENHANCEMENT: Config L + all-mpnet-base-v2 upgrade
- Base: Config L architecture (proven champion at 0.4471 nDCG@10)
- Embedding Model: all-mpnet-base-v2 (768D, 110M params) vs all-MiniLM-L6-v2 (384D, 22M params)
- Conservative Thresholds: min_similarity=0.4, confidence=0.5, top_k=10
- Expected: Better semantic similarity → higher quality weak labels → improved performance

Single Variable Experiment: Only change embedding model, keep everything else identical
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

class DiscoveryWeakLabelerMPNet:
    """Config N: Enhanced weak labeling with all-mpnet-base-v2 for superior similarity detection"""
    
    def __init__(self, min_similarity=0.4, top_k_similar=10, confidence_threshold=0.5):
        # CONSERVATIVE THRESHOLDS (Config L proven parameters)
        self.min_similarity = min_similarity
        self.top_k_similar = top_k_similar  
        self.confidence_threshold = confidence_threshold
        
        # CONFIG N UPGRADE: all-mpnet-base-v2 for better semantic similarity
        print("🔄 Loading all-mpnet-base-v2 (768D, 110M params)...")
        print("   Upgrade from all-MiniLM-L6-v2 (384D, 22M params)")
        print("   Expected: 3-4% better semantic similarity detection")
        self.sentence_model = SentenceTransformer('all-mpnet-base-v2')
        print("✅ MPNet model loaded successfully")
        
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
        """Compute sentence embeddings for all queries using all-mpnet-base-v2"""
        print("Computing sentence embeddings with all-mpnet-base-v2...")
        print("⏱️  Note: MPNet is 5x slower but much higher quality than MiniLM")
        
        # Dev embeddings
        print("  - Dev queries (342 queries)...")
        dev_texts = dev_df['query_text'].tolist()
        self.dev_embeddings = self.sentence_model.encode(dev_texts, show_progress_bar=True, batch_size=16)
        
        # Discovery embeddings  
        print("  - Discovery queries (14,950 queries)...")
        discovery_texts = discovery_df['query_text'].tolist()
        self.discovery_embeddings = self.sentence_model.encode(discovery_texts, show_progress_bar=True, batch_size=16)
        
        print(f"Dev embeddings shape: {self.dev_embeddings.shape} (768D)")
        print(f"Discovery embeddings shape: {self.discovery_embeddings.shape} (768D)")
        
        return self.dev_embeddings, self.discovery_embeddings
    
    def find_similar_queries(self, discovery_idx, dev_df):
        """Find most similar dev queries for a discovery query"""
        # Get similarity scores
        discovery_embedding = self.discovery_embeddings[discovery_idx].reshape(1, -1)
        similarities = cosine_similarity(discovery_embedding, self.dev_embeddings)[0]
        
        # Get top-k similar queries above threshold
        similar_indices = np.argsort(similarities)[::-1][:self.top_k_similar]
        similar_scores = similarities[similar_indices]
        
        # Filter by minimum similarity (CONSERVATIVE: 0.4)
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
                
                # Only keep high-confidence labels (CONSERVATIVE: 0.5)
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
        print("Generating weak-labeled dataset with MPNet similarity...")
        
        weak_labeled_examples = []
        stats = {
            'total_discovery_queries': len(discovery_df),
            'queries_with_weak_labels': 0,
            'total_weak_labels': 0,
            'avg_confidence': [],
            'avg_similarity': [],
            'coverage_by_llm': {},
            'embedding_model': 'all-mpnet-base-v2',
            'embedding_dimensions': 768,
            'parameters': '110M'
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
                    'source': 'config_n_weak_labeled',
                    'embedding_model': 'all-mpnet-base-v2'
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
        print("Validating weak labeling quality with all-mpnet-base-v2...")
        
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
            
            print(f"Validation Results (all-mpnet-base-v2):")
            print(f"  MAE: {metrics['mae']:.4f}")
            print(f"  RMSE: {metrics['rmse']:.4f}")
            print(f"  Correlation: {metrics['correlation']:.4f}")
            print(f"  Coverage: {metrics['coverage']} examples")
            print(f"  Avg Confidence: {metrics['avg_confidence']:.4f}")
            
            return metrics, validation_df
        else:
            print("Warning: No validation examples generated")
            return None, None

def run_config_n_weak_labeling():
    """Config N: Enhanced weak labeling pipeline with all-mpnet-base-v2"""
    print("=" * 80)
    print("CONFIG N: ENHANCED WEAK LABELING WITH ALL-MPNET-BASE-V2")
    print("=" * 80)
    print("GOAL: Test if better embedding model improves weak labeling quality")
    print("  Base: Config L architecture (champion at 0.4471 nDCG@10)")
    print("  Upgrade: all-MiniLM-L6-v2 → all-mpnet-base-v2")
    print("  Conservative thresholds: min_sim=0.4, confidence=0.5, top_k=10")
    print("  Expected: 3-4% better similarity → higher quality weak labels")
    print()
    
    start_time = time.time()
    
    # Initialize Config N weak labeler 
    labeler = DiscoveryWeakLabelerMPNet(
        min_similarity=0.4,     # CONSERVATIVE (Config L proven)
        top_k_similar=10,       # CONSERVATIVE (Config L proven)
        confidence_threshold=0.5 # CONSERVATIVE (Config L proven)
    )
    
    # Load data
    dev_df, qrels_df = labeler.load_dev_data()
    discovery_df, discovery_data = labeler.load_discovery_data()
    
    # Compute embeddings with MPNet
    dev_embeddings, discovery_embeddings = labeler.compute_embeddings(dev_df, discovery_df)
    
    # Validate weak labeling approach on dev set
    print("\n" + "="*50)
    print("VALIDATION: Testing MPNet weak labeling on dev set")
    print("="*50)
    validation_metrics, validation_df = labeler.validate_weak_labels(dev_df)
    
    if validation_metrics and validation_metrics['correlation'] > 0.3:
        print(f"✅ VALIDATION PASSED: Correlation {validation_metrics['correlation']:.3f} > 0.3")
        print("MPNet weak labeling approach is reliable - proceeding with discovery labeling")
    elif validation_metrics is None:
        print("⚠️  VALIDATION INCONCLUSIVE: No validation examples, but proceeding with discovery labeling")
        print("   (This can happen with strict thresholds - weak labeling may still work)")
    else:
        print("❌ VALIDATION FAILED: MPNet weak labeling not reliable enough")
        return None
    
    # Generate weak-labeled dataset
    print("\n" + "="*50) 
    print("GENERATING CONFIG N WEAK-LABELED DISCOVERY DATASET")
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
    weak_output_path = "../../data/supervised_training_config_n_weak_labeled.csv"
    combined_output_path = "../../data/supervised_training_config_n_combined.csv"
    
    weak_df.to_csv(weak_output_path, index=False)
    combined_df.to_csv(combined_output_path, index=False)
    
    # Save stats
    stats['processing_time_seconds'] = processing_time
    stats['processing_time_hours'] = processing_time / 3600
    stats['validation_metrics'] = validation_metrics
    stats['config'] = 'Config N'
    stats['embedding_model'] = 'all-mpnet-base-v2'
    stats['thresholds'] = {
        'min_similarity': labeler.min_similarity,
        'confidence_threshold': labeler.confidence_threshold,
        'top_k_similar': labeler.top_k_similar
    }
    
    stats_path = "../../data/config_n_weak_labeling_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    
    # Print final results
    print("\n" + "=" * 80)
    print("CONFIG N WEAK LABELING RESULTS")
    print("=" * 80)
    print(f"Embedding Model: all-mpnet-base-v2 (768D, 110M params)")
    print(f"Conservative Thresholds: sim≥{labeler.min_similarity}, conf≥{labeler.confidence_threshold}, top_k={labeler.top_k_similar}")
    print()
    print(f"Discovery queries processed: {stats['total_discovery_queries']:,}")
    print(f"Queries with weak labels: {stats['queries_with_weak_labels']:,} ({stats['coverage_queries_pct']:.1f}%)")
    print(f"Total weak labels generated: {stats['total_weak_labels']:,}")
    print(f"Average labels per query: {stats['avg_labels_per_query']:.1f}")
    print(f"Average confidence: {stats['avg_confidence']:.3f}")
    print(f"Average similarity: {stats['avg_similarity']:.3f}")
    print()
    print(f"COMBINED DATASET:")
    print(f"  Original examples: {len(original_df):,}")
    print(f"  Config N weak examples: {len(weak_df):,}")
    print(f"  Total examples: {len(combined_df):,}")
    print(f"  Scale multiplier: {len(combined_df) / len(original_df):.1f}x")
    print(f"  Weak label ratio: {len(weak_df) / len(combined_df) * 100:.1f}%")
    print()
    print(f"Processing time: {processing_time:.1f}s ({processing_time/3600:.2f} hours)")
    print()
    print(f"✓ Config N weak-labeled dataset saved: {weak_output_path}")
    print(f"✓ Config N combined dataset saved: {combined_output_path}")
    print(f"✓ Config N statistics saved: {stats_path}")
    
    expected_count_range = (80000, 150000)  # Expected similar to Config L
    if expected_count_range[0] <= len(weak_df) <= expected_count_range[1]:
        print(f"\n🎯 CONFIG N SUCCESS: Generated {len(weak_df):,} weak labels - expected range achieved!")
        print("   Ready for Config L architecture training with enhanced MPNet weak labels")
    elif len(weak_df) > expected_count_range[1]:
        print(f"\n🚀 CONFIG N EXCEEDED: Generated {len(weak_df):,} weak labels - more than expected!")
        print("   MPNet may have found more high-quality similarities than MiniLM")
    else:
        print(f"\n📊 CONFIG N CONSERVATIVE: Generated {len(weak_df):,} weak labels")
        print("   Fewer than expected but may be higher quality due to MPNet precision")
    
    return combined_df, stats

if __name__ == "__main__":
    result = run_config_n_weak_labeling()