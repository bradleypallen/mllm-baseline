#!/usr/bin/env python3
"""
Two-phase ranking approach:
1. Use existing model to get initial top-k candidates
2. Rerank using LLM-specific expertise profiles
"""

import json
import numpy as np
import pandas as pd
import sys
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import pickle

sys.path.append('../..')
from shared.utils.evaluation import calculate_metrics

class TwoPhaseReranker:
    """Rerank top-k candidates using expertise profiles"""
    
    def __init__(self, initial_k=20, final_k=10):
        self.initial_k = initial_k  # How many to get from base model
        self.final_k = final_k      # How many to return after reranking
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load expertise profiles - try complete profiles first
        self.expertise = {}
        if os.path.exists('all_llm_expertise_profiles.json'):
            with open('all_llm_expertise_profiles.json', 'r') as f:
                self.expertise = json.load(f)
            print(f"Loaded complete expertise profiles for {len(self.expertise)} LLMs")
        elif os.path.exists('llm_specific_expertise.json'):
            with open('llm_specific_expertise.json', 'r') as f:
                self.expertise = json.load(f)
            print(f"Loaded expertise profiles for {len(self.expertise)} LLMs")
        
        # Load cluster assignments for non-elite LLMs
        self.clusters = {}
        if os.path.exists('llm_clusters.json'):
            with open('llm_clusters.json', 'r') as f:
                self.clusters = json.load(f)
        
        # Identify elite LLMs
        self.elite_llms = set([llm for llm, cluster in self.clusters.items() 
                               if cluster in [1, 2]])
        
    def get_initial_ranking(self, query_id, method='random'):
        """Get initial ranking from existing model or random"""
        
        if method == 'neural':
            # Load neural model predictions if available
            predictions_file = '../neural_two_tower/predictions.json'
            if os.path.exists(predictions_file):
                with open(predictions_file, 'r') as f:
                    predictions = json.load(f)
                if query_id in predictions:
                    return predictions[query_id][:self.initial_k]
        
        # Fallback to all LLMs in random order
        all_llms = list(self.clusters.keys())
        np.random.shuffle(all_llms)
        return all_llms[:self.initial_k]
    
    def compute_expertise_score(self, query_text, llm_id):
        """Compute expertise-based score for query-LLM pair"""
        
        # If no expertise profile, use cluster-based scoring
        if llm_id not in self.expertise:
            cluster = self.clusters.get(llm_id, 0)
            # Cluster 0: very bad, 3-4: mediocre, 1-2: good
            cluster_scores = {0: 0.1, 1: 0.7, 2: 0.8, 3: 0.3, 4: 0.4}
            return cluster_scores.get(cluster, 0.2)
        
        # Encode query
        query_embedding = self.encoder.encode([query_text])[0]
        
        # Find best matching cluster
        profile = self.expertise[llm_id]
        best_score = 0
        best_match = None
        
        for cluster_id, cluster_info in profile['expertise'].items():
            # Get cluster centroid
            centroid = np.array(cluster_info['centroid'])
            
            # Compute similarity
            similarity = cosine_similarity([query_embedding], [centroid])[0][0]
            
            # Get cluster quality
            quality = cluster_info['quality']
            
            # Combined score (weighted combination)
            # Higher weight on quality since similarity is often noisy
            score = (0.3 * similarity + 0.7 * quality) * quality
            
            if score > best_score:
                best_score = score
                best_match = {
                    'cluster_id': cluster_id,
                    'similarity': similarity,
                    'quality': quality,
                    'keywords': cluster_info['features']['question_keywords'][:5]
                }
        
        return best_score
    
    def rerank_candidates(self, query_text, initial_ranking):
        """Rerank top-k candidates using expertise profiles"""
        
        reranked = []
        
        for llm_id in initial_ranking:
            score = self.compute_expertise_score(query_text, llm_id)
            reranked.append({
                'llm_id': llm_id,
                'score': score,
                'is_elite': llm_id in self.elite_llms
            })
        
        # Sort by score
        reranked.sort(key=lambda x: x['score'], reverse=True)
        
        # Return top-k
        return [item['llm_id'] for item in reranked[:self.final_k]]
    
    def evaluate_on_dev_set(self, use_existing_rankings=False):
        """Evaluate two-phase ranking on development set"""
        
        print("\n" + "="*80)
        print("TWO-PHASE RERANKING EVALUATION")
        print("="*80)
        
        # Load dev data
        dev_queries = pd.read_csv('../../data/llm_dev_data.tsv', sep='\t', header=None)
        dev_queries.columns = ['query_id', 'query_text']
        
        # Load qrels
        qrels = pd.read_csv('../../data/llm_dev_qrels.txt', sep=' ', header=None)
        qrels.columns = ['query_id', 'zero', 'llm_id', 'qrel']
        
        # For comparison, also evaluate baseline
        baseline_ndcg = []
        reranked_ndcg = []
        
        print(f"\nEvaluating on {len(dev_queries)} queries...")
        print(f"Initial candidates: {self.initial_k}, Final ranking: {self.final_k}")
        
        for idx, row in tqdm(dev_queries.iterrows(), total=len(dev_queries), desc="Processing"):
            query_id = row['query_id']
            query_text = row['query_text']
            
            # Get ground truth
            query_qrels = qrels[qrels['query_id'] == query_id]
            true_scores = {r['llm_id']: r['qrel'] for _, r in query_qrels.iterrows()}
            
            # Skip if no positive qrels
            if not any(score > 0 for score in true_scores.values()):
                continue
            
            # Get initial ranking (simulate existing model)
            if use_existing_rankings:
                initial_ranking = self.get_initial_ranking(query_id, method='neural')
            else:
                # Use a simple heuristic: elite LLMs first, then others
                all_llms = list(self.clusters.keys())
                elite = [llm for llm in all_llms if llm in self.elite_llms]
                non_elite = [llm for llm in all_llms if llm not in self.elite_llms]
                np.random.shuffle(elite)
                np.random.shuffle(non_elite)
                initial_ranking = (elite + non_elite)[:self.initial_k]
            
            # Baseline: use initial ranking as-is
            baseline_ranking = initial_ranking[:self.final_k]
            
            # Reranked: apply expertise-based reranking
            reranked_ranking = self.rerank_candidates(query_text, initial_ranking)
            
            # Calculate nDCG@10 for both
            # Baseline
            dcg = sum(true_scores.get(llm_id, 0) / np.log2(i + 2) 
                     for i, llm_id in enumerate(baseline_ranking))
            ideal_scores = sorted(true_scores.values(), reverse=True)[:self.final_k]
            idcg = sum(score / np.log2(i + 2) for i, score in enumerate(ideal_scores))
            baseline_ndcg.append(dcg / idcg if idcg > 0 else 0)
            
            # Reranked
            dcg = sum(true_scores.get(llm_id, 0) / np.log2(i + 2) 
                     for i, llm_id in enumerate(reranked_ranking))
            reranked_ndcg.append(dcg / idcg if idcg > 0 else 0)
        
        # Results
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        
        print(f"\nBaseline (initial ranking):")
        print(f"  nDCG@{self.final_k}: {np.mean(baseline_ndcg):.4f} ± {np.std(baseline_ndcg):.4f}")
        
        print(f"\nWith expertise reranking:")
        print(f"  nDCG@{self.final_k}: {np.mean(reranked_ndcg):.4f} ± {np.std(reranked_ndcg):.4f}")
        
        improvement = (np.mean(reranked_ndcg) - np.mean(baseline_ndcg)) / np.mean(baseline_ndcg) * 100
        print(f"\nImprovement: {improvement:+.1f}%")
        
        # Statistical significance test
        from scipy import stats
        t_stat, p_value = stats.ttest_rel(reranked_ndcg, baseline_ndcg)
        print(f"Paired t-test: t={t_stat:.3f}, p={p_value:.4f}")
        if p_value < 0.05:
            print("✅ Improvement is statistically significant!")
        else:
            print("❌ Improvement is not statistically significant")
        
        return {
            'baseline_ndcg': np.mean(baseline_ndcg),
            'reranked_ndcg': np.mean(reranked_ndcg),
            'improvement': improvement,
            'p_value': p_value
        }

def main():
    """Test two-phase reranking"""
    
    # First check if we have expertise profiles
    if not os.path.exists('llm_specific_expertise.json'):
        print("ERROR: No expertise profiles found. Run build_llm_expertise_clusters.py first")
        return
    
    # Initialize reranker
    reranker = TwoPhaseReranker(initial_k=30, final_k=10)
    
    # Run evaluation
    results = reranker.evaluate_on_dev_set(use_existing_rankings=False)
    
    # Save results
    with open('two_phase_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to two_phase_results.json")
    
    # Analysis
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    
    print("\nKey Insights:")
    print("1. Two-phase approach allows combining any base ranker with expertise")
    print("2. Reranking only top-k is computationally efficient")
    print("3. Expertise profiles help identify which LLMs to promote/demote")
    
    if results['improvement'] > 0:
        print(f"4. Expertise reranking improved nDCG by {results['improvement']:.1f}%")
    else:
        print("4. Current expertise profiles need refinement")

if __name__ == "__main__":
    main()