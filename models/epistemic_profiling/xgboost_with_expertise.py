#!/usr/bin/env python3
"""
XGBoost with expertise profile features.
Integrates bilateral profiles and LLM-specific expertise into XGBoost ranking.
"""

import json
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import time
import sys
sys.path.append('../..')
from shared.utils.evaluation import calculate_metrics, save_standardized_results

class XGBoostWithExpertise:
    """XGBoost ranker augmented with expertise profile features"""
    
    def __init__(self):
        # Text processing
        self.tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
        self.llm_encoder = LabelEncoder()
        self.query_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load profiles
        self.bilateral_profiles = {}
        self.expertise_profiles = {}
        self.cluster_assignments = {}
        self.load_all_profiles()
        
    def load_all_profiles(self):
        """Load all profile data"""
        print("Loading profile data...")
        
        # Load bilateral profiles
        try:
            with open('complete_profiles.json', 'r') as f:
                profiles = json.load(f)
                # Convert to more usable format
                for llm_id, profile in profiles.items():
                    self.bilateral_profiles[llm_id] = {
                        'inconsistent': float(profile[0]),      # (t,t)
                        'confident_correct': float(profile[2]), # (t,f)
                        'overconfident_wrong': float(profile[6]), # (f,t)
                        'uncertain': float(profile[8]),         # (f,f)
                        'reliability': float(profile[2]) - float(profile[6])
                    }
            print(f"  Loaded {len(self.bilateral_profiles)} bilateral profiles")
        except:
            print("  Warning: Could not load bilateral profiles")
        
        # Load expertise profiles
        try:
            with open('all_llm_expertise_profiles.json', 'r') as f:
                self.expertise_profiles = json.load(f)
            print(f"  Loaded expertise for {len(self.expertise_profiles)} LLMs")
        except:
            print("  Warning: Could not load expertise profiles")
        
        # Load cluster assignments
        try:
            with open('llm_clusters.json', 'r') as f:
                self.cluster_assignments = json.load(f)
            print(f"  Loaded cluster assignments for {len(self.cluster_assignments)} LLMs")
        except:
            print("  Warning: Could not load cluster assignments")
    
    def find_best_expertise_match(self, query_text, llm_id):
        """Find best matching expertise cluster for a query-LLM pair"""
        if llm_id not in self.expertise_profiles:
            return None
        
        profile = self.expertise_profiles[llm_id]
        if 'expertise' not in profile:
            return None
        
        # Encode query
        query_embedding = self.query_encoder.encode([query_text], show_progress_bar=False)[0]
        
        best_match = None
        best_score = -1
        second_best_score = -1
        
        for cluster_id, cluster_info in profile['expertise'].items():
            # Calculate similarity
            centroid = np.array(cluster_info['centroid'])
            similarity = cosine_similarity([query_embedding], [centroid])[0][0]
            
            if similarity > best_score:
                second_best_score = best_score
                best_score = similarity
                best_match = {
                    'cluster_id': cluster_id,
                    'similarity': similarity,
                    'quality': cluster_info['quality'],
                    'features': cluster_info['features'],
                    'size': cluster_info['size']
                }
        
        if best_match:
            best_match['match_confidence'] = best_score - second_best_score
        
        return best_match
    
    def extract_features(self, query_text, llm_id, tfidf_features):
        """Extract all features for a query-LLM pair"""
        features = []
        
        # 1. Original TF-IDF features (already computed)
        features.extend(tfidf_features)
        
        # 2. LLM ID encoding (categorical)
        try:
            llm_id_encoded = self.llm_encoder.transform([llm_id])[0]
        except:
            llm_id_encoded = -1  # Unknown LLM
        features.append(llm_id_encoded)
        
        # 3. Bilateral profile features
        if llm_id in self.bilateral_profiles:
            bp = self.bilateral_profiles[llm_id]
            features.extend([
                bp['confident_correct'],
                bp['overconfident_wrong'],
                bp['uncertain'],
                bp['inconsistent'],
                bp['reliability']
            ])
        else:
            features.extend([0.5, 0.5, 0.0, 0.0, 0.0])  # Default neutral values
        
        # 4. Cluster assignment features
        if llm_id in self.cluster_assignments:
            cluster = self.cluster_assignments[llm_id]
            # One-hot encode cluster (0-4)
            cluster_features = [0] * 5
            if 0 <= cluster < 5:
                cluster_features[cluster] = 1
            features.extend(cluster_features)
            # Is elite (clusters 1 or 2)
            features.append(1 if cluster in [1, 2] else 0)
        else:
            features.extend([0, 0, 0, 0, 0, 0])  # No cluster, not elite
        
        # 5. Expertise match features
        match = self.find_best_expertise_match(query_text, llm_id)
        if match:
            features.extend([
                match['similarity'],
                match['quality'],
                match['similarity'] * match['quality'],  # Combined score
                match['match_confidence'],
                match['features']['avg_answer_length'] / 1000.0,  # Normalized
                match['features']['non_answer_rate'],
                match['features']['code_rate'],
                match['features']['list_rate'],
                match['size'] / 100.0  # Normalized cluster size
            ])
        else:
            features.extend([0.0] * 9)  # No expertise data
        
        return features
    
    def prepare_data(self):
        """Load and prepare training data with all features"""
        print("\nPreparing data...")
        
        # Load training data
        train_df = pd.read_csv('../../data/supervised_training_full.csv')
        print(f"Loaded {len(train_df)} training examples")
        
        # Get unique queries for TF-IDF fitting
        unique_queries = train_df['query_text'].unique()
        print(f"Found {len(unique_queries)} unique queries")
        
        # Fit TF-IDF on all unique queries
        self.tfidf.fit(unique_queries)
        
        # Fit LLM encoder
        all_llms = train_df['llm_id'].unique()
        self.llm_encoder.fit(all_llms)
        print(f"Found {len(all_llms)} unique LLMs")
        
        # Pre-compute TF-IDF for all queries (for efficiency)
        query_to_tfidf = {}
        for query in tqdm(unique_queries, desc="Computing TF-IDF"):
            query_to_tfidf[query] = self.tfidf.transform([query]).toarray()[0]
        
        # Extract features for all examples
        X = []
        y = []
        queries = []
        
        print("\nExtracting features...")
        for idx, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Processing"):
            query_text = row['query_text']
            llm_id = row['llm_id']
            qrel = row['qrel']
            
            # Get pre-computed TF-IDF
            tfidf_features = query_to_tfidf[query_text]
            
            # Extract all features
            features = self.extract_features(query_text, llm_id, tfidf_features)
            
            X.append(features)
            y.append(qrel)
            queries.append(row['query_id'])
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Features per example: {X.shape[1]}")
        
        # Feature breakdown
        print("\nFeature breakdown:")
        print(f"  TF-IDF features: 1000")
        print(f"  LLM ID: 1")
        print(f"  Bilateral profile: 5")
        print(f"  Cluster features: 6")
        print(f"  Expertise match: 9")
        print(f"  Total: {1000 + 1 + 5 + 6 + 9}")
        
        return X, y, queries
    
    def run_cv_evaluation(self, X, y, queries):
        """Run 10-fold cross-validation"""
        print("\n" + "="*80)
        print("RUNNING 10-FOLD CROSS-VALIDATION")
        print("="*80)
        
        # Group by query for proper CV
        unique_queries = sorted(set(queries))
        query_to_indices = defaultdict(list)
        for i, q in enumerate(queries):
            query_to_indices[q].append(i)
        
        # Setup cross-validation
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        
        fold_results = []
        start_time = time.time()
        
        for fold, (train_queries_idx, val_queries_idx) in enumerate(kf.split(unique_queries), 1):
            print(f"\nFold {fold}/10")
            
            # Get train/val queries
            train_queries = [unique_queries[i] for i in train_queries_idx]
            val_queries = [unique_queries[i] for i in val_queries_idx]
            
            # Get indices for these queries
            train_idx = []
            val_idx = []
            for q in train_queries:
                train_idx.extend(query_to_indices[q])
            for q in val_queries:
                val_idx.extend(query_to_indices[q])
            
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            print(f"  Training: {len(X_train)} examples from {len(train_queries)} queries")
            print(f"  Validation: {len(X_val)} examples from {len(val_queries)} queries")
            
            # Train model
            model = XGBRegressor(
                n_estimators=100,
                max_depth=10,
                learning_rate=0.1,
                objective='reg:squarederror',
                random_state=42,
                n_jobs=-1
            )
            
            fold_start = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - fold_start
            
            # Predict
            y_pred = model.predict(X_val)
            
            # Group predictions by query for evaluation
            val_queries_in_fold = [queries[i] for i in val_idx]
            y_true_by_query = defaultdict(list)
            y_pred_by_query = defaultdict(list)
            
            for i, q in enumerate(val_queries_in_fold):
                y_true_by_query[q].append(y_val[i])
                y_pred_by_query[q].append(y_pred[i])
            
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
                'pipeline': 'XGBoost with Expertise Profiles',
                'features': 'TF-IDF + Bilateral + Clusters + Expertise Match',
                'n_features': X.shape[1],
                'dataset': 'TREC 2025 Million LLMs Track',
                'total_examples': len(X),
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
        
        with open('xgboost_expertise_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Results saved to xgboost_expertise_results.json")
        
        return results
    
    def analyze_feature_importance(self, X, y):
        """Train on full data and analyze feature importance"""
        print("\n" + "="*80)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*80)
        
        # Train on all data
        model = XGBRegressor(
            n_estimators=100,
            max_depth=10,
            learning_rate=0.1,
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1
        )
        
        print("Training on full dataset...")
        model.fit(X, y)
        
        # Get feature importance
        importance = model.feature_importances_
        
        # Feature names
        feature_names = []
        feature_names.extend([f'tfidf_{i}' for i in range(1000)])
        feature_names.append('llm_id')
        feature_names.extend(['bilateral_confident_correct', 'bilateral_overconfident_wrong',
                             'bilateral_uncertain', 'bilateral_inconsistent', 'bilateral_reliability'])
        feature_names.extend(['cluster_0', 'cluster_1', 'cluster_2', 'cluster_3', 'cluster_4', 'is_elite'])
        feature_names.extend(['expertise_similarity', 'expertise_quality', 'expertise_combined',
                             'expertise_confidence', 'expertise_answer_length', 'expertise_non_answer_rate',
                             'expertise_code_rate', 'expertise_list_rate', 'expertise_cluster_size'])
        
        # Sort by importance
        importance_pairs = list(zip(feature_names, importance))
        importance_pairs.sort(key=lambda x: x[1], reverse=True)
        
        print("\nTop 20 most important features:")
        for i, (name, imp) in enumerate(importance_pairs[:20]):
            print(f"  {i+1:2d}. {name:40s}: {imp:.4f}")
        
        # Summarize by feature group
        tfidf_importance = sum(imp for name, imp in importance_pairs if name.startswith('tfidf_'))
        bilateral_importance = sum(imp for name, imp in importance_pairs if name.startswith('bilateral_'))
        cluster_importance = sum(imp for name, imp in importance_pairs if name.startswith('cluster_') or name == 'is_elite')
        expertise_importance = sum(imp for name, imp in importance_pairs if name.startswith('expertise_'))
        
        print("\nImportance by feature group:")
        print(f"  TF-IDF features:     {tfidf_importance:.4f} ({tfidf_importance*100:.1f}%)")
        print(f"  Bilateral profiles:  {bilateral_importance:.4f} ({bilateral_importance*100:.1f}%)")
        print(f"  Cluster features:    {cluster_importance:.4f} ({cluster_importance*100:.1f}%)")
        print(f"  Expertise matching:  {expertise_importance:.4f} ({expertise_importance*100:.1f}%)")
        print(f"  LLM ID:             {importance_pairs[feature_names.index('llm_id')][1]:.4f}")

def main():
    """Run XGBoost with expertise features"""
    print("="*80)
    print("XGBOOST WITH EXPERTISE PROFILES")
    print("="*80)
    
    # Initialize
    ranker = XGBoostWithExpertise()
    
    # Prepare data
    X, y, queries = ranker.prepare_data()
    
    # Run evaluation
    results = ranker.run_cv_evaluation(X, y, queries)
    
    # Analyze feature importance
    ranker.analyze_feature_importance(X, y)
    
    # Compare to baseline
    print("\n" + "="*80)
    print("COMPARISON TO BASELINES")
    print("="*80)
    
    print("\nOriginal XGBoost (TF-IDF only):     nDCG@10 = 0.3925")
    print(f"XGBoost with expertise:              nDCG@10 = {results['performance_metrics']['ndcg_10']['mean']:.4f}")
    
    improvement = (results['performance_metrics']['ndcg_10']['mean'] - 0.3925) / 0.3925 * 100
    print(f"\nImprovement: {improvement:+.1f}%")

if __name__ == "__main__":
    main()