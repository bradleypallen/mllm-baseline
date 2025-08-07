#!/usr/bin/env python3
"""
Shared evaluation utilities for all models in the TREC 2025 Million LLMs Track baseline framework.

Provides standardized evaluation functions to ensure consistent metrics across all baseline implementations.
"""

import numpy as np
from sklearn.metrics import ndcg_score
import json
from datetime import datetime
from pathlib import Path


def mean_reciprocal_rank(y_true_by_query, y_pred_by_query):
    """Calculate Mean Reciprocal Rank across all queries"""
    reciprocal_ranks = []
    
    for query_id in y_true_by_query.keys():
        y_true = np.array(y_true_by_query[query_id])
        y_pred = np.array(y_pred_by_query[query_id])
        
        # Sort by predicted scores (descending)
        sorted_indices = np.argsort(y_pred)[::-1]
        sorted_true = y_true[sorted_indices]
        
        # Find first relevant item (relevance > 0)
        relevant_indices = np.where(sorted_true > 0)[0]
        if len(relevant_indices) > 0:
            first_relevant_rank = relevant_indices[0] + 1  # 1-indexed
            reciprocal_ranks.append(1.0 / first_relevant_rank)
        else:
            reciprocal_ranks.append(0.0)
    
    return np.mean(reciprocal_ranks)


def evaluate_ndcg_at_k(y_true_by_query, y_pred_by_query, k=10):
    """Calculate nDCG@k across all queries"""
    ndcg_scores = []
    skipped_queries = 0
    
    for query_id in y_true_by_query.keys():
        y_true = y_true_by_query[query_id]
        y_pred = y_pred_by_query[query_id]
        
        # Need at least 2 documents for meaningful nDCG
        if len(y_true) < 2:
            skipped_queries += 1
            continue
        
        # Check if there's any relevance signal (not all zeros)
        if np.sum(y_true) == 0:
            # No relevant documents, nDCG is 0
            ndcg_scores.append(0.0)
        else:
            try:
                ndcg = ndcg_score([y_true], [y_pred], k=k)
                ndcg_scores.append(ndcg)
            except ValueError:
                # Handle any sklearn nDCG issues
                skipped_queries += 1
                continue
    
    if len(ndcg_scores) == 0:
        print(f"WARNING: No valid queries for nDCG@{k} calculation (skipped {skipped_queries})")
        return 0.0
    
    return np.mean(ndcg_scores)


def calculate_metrics(y_true_by_query, y_pred_by_query):
    """Calculate all standard evaluation metrics"""
    return {
        'ndcg_10': evaluate_ndcg_at_k(y_true_by_query, y_pred_by_query, k=10),
        'ndcg_5': evaluate_ndcg_at_k(y_true_by_query, y_pred_by_query, k=5),
        'mrr': mean_reciprocal_rank(y_true_by_query, y_pred_by_query)
    }


def aggregate_cv_results(fold_results, runtime_seconds=None):
    """Aggregate cross-validation results with statistics"""
    ndcg_10_scores = [r['ndcg_10'] for r in fold_results]
    ndcg_5_scores = [r['ndcg_5'] for r in fold_results] 
    mrr_scores = [r['mrr'] for r in fold_results]
    
    def calc_stats(scores):
        return {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'min': np.min(scores),
            'max': np.max(scores),
            'confidence_interval_95': [
                np.mean(scores) - 1.96 * np.std(scores),
                np.mean(scores) + 1.96 * np.std(scores)
            ]
        }
    
    results = {
        'performance_metrics': {
            'ndcg_10': calc_stats(ndcg_10_scores),
            'ndcg_5': calc_stats(ndcg_5_scores),
            'mrr': calc_stats(mrr_scores)
        },
        'fold_by_fold_results': fold_results
    }
    
    if runtime_seconds:
        results['evaluation_info'] = {
            'timestamp': datetime.now().isoformat(),
            'total_runtime_seconds': runtime_seconds,
            'total_runtime_minutes': runtime_seconds / 60,
            'total_runtime_hours': runtime_seconds / 3600
        }
    
    return results


def save_standardized_results(results, model_name, output_dir='../../data/results'):
    """Save results in standardized format"""
    output_path = Path(output_dir) / f"{model_name}_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return str(output_path)


def load_results(model_name, results_dir='../../data/results'):
    """Load standardized results for a model"""
    results_path = Path(results_dir) / f"{model_name}_results.json"
    
    if not results_path.exists():
        return None
        
    with open(results_path, 'r') as f:
        return json.load(f)


def get_qrel_mapping():
    """Get standardized qrel encoding mapping"""
    return {0: 0.0, 1: 1.0, 2: 0.7}


if __name__ == "__main__":
    # Test evaluation functions
    print("Testing shared evaluation utilities...")
    
    # Sample data for testing
    y_true_by_query = {
        'q1': np.array([1.0, 0.7, 0.0, 0.0]),
        'q2': np.array([0.0, 1.0, 0.0, 0.7]),
        'q3': np.array([0.7, 0.0, 0.0, 1.0])
    }
    
    y_pred_by_query = {
        'q1': np.array([0.9, 0.8, 0.3, 0.1]),
        'q2': np.array([0.2, 0.9, 0.1, 0.8]),
        'q3': np.array([0.7, 0.2, 0.1, 0.9])
    }
    
    metrics = calculate_metrics(y_true_by_query, y_pred_by_query)
    print(f"Test metrics: {metrics}")
    print("Evaluation utilities working correctly!")