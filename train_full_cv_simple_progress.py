#!/usr/bin/env python3
"""
10-fold cross-validation on FULL dataset with simple, reliable progress tracking.
No tqdm dependency - uses simple print statements that work in all terminals.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import ndcg_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
import time
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_full_data():
    """Load the complete supervised training data"""
    print("Loading FULL training dataset...")
    df = pd.read_csv('data/supervised_training_full.csv')
    print(f"✓ Loaded {len(df):,} examples")
    print(f"  Queries: {df.query_id.nunique()}")
    print(f"  LLMs: {df.llm_id.nunique()}")
    return df

def create_features(df, tfidf_vectorizer=None, llm_encoder=None, fit=True):
    """Extract features from query text and LLM ID"""
    
    # Text features: TF-IDF of query text
    if fit:
        tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        query_features = tfidf_vectorizer.fit_transform(df['query_text'])
    else:
        query_features = tfidf_vectorizer.transform(df['query_text'])
    
    # LLM features: Label encoding
    if fit:
        llm_encoder = LabelEncoder()
        llm_encoded = llm_encoder.fit_transform(df['llm_id'])
    else:
        llm_encoded = llm_encoder.transform(df['llm_id'])
    
    # Combine features
    X_text = query_features.toarray()
    X_llm = llm_encoded.reshape(-1, 1)
    X = np.hstack([X_text, X_llm])
    
    # Target: Convert qrel scores to [0,1] interval
    y = df['qrel'].values / 2.0
    
    return X, y, tfidf_vectorizer, llm_encoder

def evaluate_ndcg_at_k(y_true_by_query, y_pred_by_query, k=10):
    """Calculate nDCG@k across all queries"""
    ndcg_scores = []
    
    for query_id in y_true_by_query.keys():
        y_true = y_true_by_query[query_id]
        y_pred = y_pred_by_query[query_id]
        
        ndcg = ndcg_score([y_true], [y_pred], k=k)
        ndcg_scores.append(ndcg)
    
    return np.mean(ndcg_scores), ndcg_scores

def evaluate_mrr(y_true_by_query, y_pred_by_query):
    """Calculate Mean Reciprocal Rank (MRR) across all queries"""
    rr_scores = []
    
    for query_id in y_true_by_query.keys():
        y_true = y_true_by_query[query_id]
        y_pred = y_pred_by_query[query_id]
        
        # Sort by predicted scores (descending)
        sorted_indices = np.argsort(y_pred)[::-1]
        sorted_true = y_true[sorted_indices]
        
        # Find first relevant item (relevance > 0)
        rr = 0.0
        for rank, relevance in enumerate(sorted_true, 1):
            if relevance > 0:
                rr = 1.0 / rank
                break
        
        rr_scores.append(rr)
    
    return np.mean(rr_scores), rr_scores

def evaluate_fold(y_true, y_pred, query_ids, llm_ids, fold_num):
    """Evaluate a single fold with TREC submission format"""
    
    # Basic regression metrics
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    # Group by query for ranking metrics
    y_true_by_query = {}
    y_pred_by_query = {}
    
    for i, (query_id, llm_id) in enumerate(zip(query_ids, llm_ids)):
        if query_id not in y_true_by_query:
            y_true_by_query[query_id] = []
            y_pred_by_query[query_id] = []
        
        y_true_by_query[query_id].append(y_true[i])
        y_pred_by_query[query_id].append(y_pred[i])
    
    # Convert grouped data to numpy arrays for nDCG calculation
    for query_id in y_true_by_query.keys():
        y_true_by_query[query_id] = np.array(y_true_by_query[query_id])
        y_pred_by_query[query_id] = np.array(y_pred_by_query[query_id])
    
    # Calculate nDCG and MRR metrics
    print("    Calculating ranking metrics...")
    ndcg_5_mean, _ = evaluate_ndcg_at_k(y_true_by_query, y_pred_by_query, k=5)
    ndcg_10_mean, ndcg_10_scores = evaluate_ndcg_at_k(y_true_by_query, y_pred_by_query, k=10)
    mrr_mean, mrr_scores = evaluate_mrr(y_true_by_query, y_pred_by_query)
    
    return {
        'mse': mse,
        'mae': mae,
        'ndcg_5': ndcg_5_mean,
        'ndcg_10': ndcg_10_mean,
        'mrr': mrr_mean,
        'n_queries': len(y_true_by_query)
    }

def cross_validate_model(df, n_folds=10, random_state=42):
    """Perform k-fold cross-validation on queries"""
    print(f"\n=== {n_folds}-FOLD CROSS-VALIDATION (FULL DATASET) ===")
    print(f"Dataset: {len(df):,} examples, {df.query_id.nunique()} queries, {df.llm_id.nunique()} LLMs")
    print(f"Expected runtime: ~{n_folds * 6:.0f} minutes")
    
    # Get unique queries for CV splitting
    unique_queries = df['query_id'].unique()
    np.random.seed(random_state)
    np.random.shuffle(unique_queries)
    
    kf = KFold(n_splits=n_folds, shuffle=False)  # Already shuffled above
    
    fold_results = []
    overall_start_time = time.time()
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(unique_queries)):
        fold_start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"FOLD {fold + 1}/{n_folds} - {((fold + 1) / n_folds * 100):.0f}% Complete")
        print(f"{'='*60}")
        
        # Split queries
        train_queries = unique_queries[train_idx]
        test_queries = unique_queries[test_idx]
        
        # Create train/test dataframes
        train_mask = df['query_id'].isin(train_queries)
        test_mask = df['query_id'].isin(test_queries)
        
        train_df = df[train_mask].reset_index(drop=True)
        test_df = df[test_mask].reset_index(drop=True)
        
        print(f"Train: {len(train_df):,} examples ({len(train_queries)} queries)")
        print(f"Test:  {len(test_df):,} examples ({len(test_queries)} queries)")
        
        # Create features
        print("  Creating features...")
        X_train, y_train, tfidf_vectorizer, llm_encoder = create_features(train_df, fit=True)
        X_test, y_test, _, _ = create_features(test_df, tfidf_vectorizer, llm_encoder, fit=False)
        print(f"  ✓ Features created: {X_train.shape}")
        
        # Train model
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1
        )
        
        print(f"  Training Random Forest (100 trees) on {len(X_train):,} examples...")
        train_start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - train_start
        print(f"  ✓ Training complete: {train_time:.1f} seconds")
        
        # Predict and evaluate
        print(f"  Generating predictions for {len(X_test):,} examples...")
        pred_start = time.time()
        y_pred = model.predict(X_test)
        y_pred = np.clip(y_pred, 0, 1)
        pred_time = time.time() - pred_start
        print(f"  ✓ Predictions complete: {pred_time:.1f} seconds")
        
        # Evaluate metrics
        fold_metrics = evaluate_fold(y_test, y_pred, test_df['query_id'].values, 
                                   test_df['llm_id'].values, fold + 1)
        fold_metrics['train_time'] = train_time
        fold_metrics['pred_time'] = pred_time
        fold_metrics['fold'] = fold + 1
        
        fold_results.append(fold_metrics)
        
        # Show fold results
        fold_total_time = time.time() - fold_start_time
        print(f"\n  RESULTS:")
        print(f"    nDCG@10: {fold_metrics['ndcg_10']:.4f}")
        print(f"    nDCG@5:  {fold_metrics['ndcg_5']:.4f}")
        print(f"    MRR:     {fold_metrics['mrr']:.4f}")
        print(f"    MSE:     {fold_metrics['mse']:.4f}")
        print(f"  TIMING:")
        print(f"    Train: {train_time:.1f}s, Predict: {pred_time:.1f}s, Total: {fold_total_time:.1f}s")
        
        # Progress and time estimates
        elapsed_total = time.time() - overall_start_time
        if fold > 0:  # After first fold, we can estimate
            avg_time_per_fold = elapsed_total / (fold + 1)
            remaining_folds = n_folds - fold - 1
            if remaining_folds > 0:
                estimated_remaining = avg_time_per_fold * remaining_folds
                print(f"  PROGRESS:")
                print(f"    Completed: {fold + 1}/{n_folds} folds")
                print(f"    Elapsed: {elapsed_total/60:.1f} minutes")
                print(f"    Estimated remaining: {estimated_remaining/60:.1f} minutes")
                print(f"    Estimated total: {(elapsed_total + estimated_remaining)/60:.1f} minutes")
    
    return fold_results

def analyze_cv_results(fold_results):
    """Analyze cross-validation results"""
    print(f"\n{'='*60}")
    print("CROSS-VALIDATION RESULTS SUMMARY")
    print(f"{'='*60}")
    
    # Convert to DataFrame for easier analysis
    results_df = pd.DataFrame(fold_results)
    
    # Calculate means and standard deviations
    metrics = ['ndcg_10', 'ndcg_5', 'mrr', 'mse', 'mae', 'train_time']
    
    print("\nPerformance across folds:")
    print("-" * 60)
    print(f"{'Metric':<12} {'Mean':<8} {'Std':<8} {'Min':<8} {'Max':<8}")
    print("-" * 60)
    
    summary_stats = {}
    
    for metric in metrics:
        values = results_df[metric].values
        mean_val = np.mean(values)
        std_val = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)
        
        summary_stats[metric] = {
            'mean': mean_val,
            'std': std_val,
            'min': min_val,
            'max': max_val
        }
        
        if metric == 'train_time':
            print(f"{metric.upper():<12} {mean_val:<8.1f} {std_val:<8.1f} {min_val:<8.1f} {max_val:<8.1f}")
        else:
            print(f"{metric.upper():<12} {mean_val:<8.4f} {std_val:<8.4f} {min_val:<8.4f} {max_val:<8.4f}")
    
    # Final summary with confidence intervals
    print(f"\n{'='*60}")
    print("FINAL PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    
    ndcg_10_mean = summary_stats['ndcg_10']['mean']
    ndcg_10_std = summary_stats['ndcg_10']['std']
    mrr_mean = summary_stats['mrr']['mean']
    mrr_std = summary_stats['mrr']['std']
    
    print(f"nDCG@10: {ndcg_10_mean:.4f} ± {ndcg_10_std:.4f}")
    print(f"MRR:     {mrr_mean:.4f} ± {mrr_std:.4f}")
    print(f"")
    print(f"95% Confidence Intervals:")
    print(f"  nDCG@10: [{ndcg_10_mean - 1.96*ndcg_10_std:.4f}, {ndcg_10_mean + 1.96*ndcg_10_std:.4f}]")
    print(f"  MRR:     [{mrr_mean - 1.96*mrr_std:.4f}, {mrr_mean + 1.96*mrr_std:.4f}]")
    
    return summary_stats

def generate_comprehensive_report(fold_results, summary_stats, total_time, df):
    """Generate comprehensive evaluation report"""
    print(f"\n{'='*60}")
    print("GENERATING COMPREHENSIVE REPORT")
    print(f"{'='*60}")
    
    # Create comprehensive report
    report = {
        "evaluation_info": {
            "timestamp": datetime.now().isoformat(),
            "pipeline": "10-fold Cross-Validation on Full Dataset",
            "dataset": "TREC 2025 Million LLMs Track - Complete Dataset",
            "total_examples": len(df),
            "unique_queries": int(df.query_id.nunique()),
            "unique_llms": int(df.llm_id.nunique()),
            "total_runtime_seconds": round(total_time, 1),
            "total_runtime_minutes": round(total_time / 60, 1),
            "total_runtime_hours": round(total_time / 3600, 2)
        },
        
        "performance_metrics": {
            "ndcg_10": {
                "mean": round(summary_stats['ndcg_10']['mean'], 4),
                "std": round(summary_stats['ndcg_10']['std'], 4),
                "min": round(summary_stats['ndcg_10']['min'], 4),
                "max": round(summary_stats['ndcg_10']['max'], 4),
                "confidence_interval_95": [
                    round(summary_stats['ndcg_10']['mean'] - 1.96 * summary_stats['ndcg_10']['std'], 4),
                    round(summary_stats['ndcg_10']['mean'] + 1.96 * summary_stats['ndcg_10']['std'], 4)
                ]
            },
            "ndcg_5": {
                "mean": round(summary_stats['ndcg_5']['mean'], 4),
                "std": round(summary_stats['ndcg_5']['std'], 4),
                "min": round(summary_stats['ndcg_5']['min'], 4),
                "max": round(summary_stats['ndcg_5']['max'], 4),
                "confidence_interval_95": [
                    round(summary_stats['ndcg_5']['mean'] - 1.96 * summary_stats['ndcg_5']['std'], 4),
                    round(summary_stats['ndcg_5']['mean'] + 1.96 * summary_stats['ndcg_5']['std'], 4)
                ]
            },
            "mrr": {
                "mean": round(summary_stats['mrr']['mean'], 4),
                "std": round(summary_stats['mrr']['std'], 4),
                "min": round(summary_stats['mrr']['min'], 4),
                "max": round(summary_stats['mrr']['max'], 4),
                "confidence_interval_95": [
                    round(summary_stats['mrr']['mean'] - 1.96 * summary_stats['mrr']['std'], 4),
                    round(summary_stats['mrr']['mean'] + 1.96 * summary_stats['mrr']['std'], 4)
                ]
            }
        },
        
        "fold_by_fold_results": []
    }
    
    # Add fold-by-fold results
    for fold_result in fold_results:
        fold_data = {
            "fold": fold_result['fold'],
            "ndcg_10": round(fold_result['ndcg_10'], 4),
            "ndcg_5": round(fold_result['ndcg_5'], 4), 
            "mrr": round(fold_result['mrr'], 4),
            "mse": round(fold_result['mse'], 4),
            "mae": round(fold_result['mae'], 4),
            "n_queries": fold_result['n_queries'],
            "train_time": round(fold_result['train_time'], 1)
        }
        report["fold_by_fold_results"].append(fold_data)
    
    # Save JSON report
    with open('data/full_evaluation_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("✓ data/full_evaluation_report.json - Complete results with confidence intervals")
    
    return report

def main():
    """Main pipeline: 10-fold CV on full dataset with comprehensive reporting"""
    print("="*80)
    print("TREC 2025 MILLION LLMS TRACK - FULL DATASET EVALUATION")
    print("="*80)
    print("10-fold Cross-Validation on Complete Dataset (All 1131 LLMs)")
    print("Expected runtime: ~60-70 minutes")
    print()
    
    overall_start = time.time()
    
    # Load complete dataset
    df = load_full_data()
    
    # Run cross-validation
    fold_results = cross_validate_model(df, n_folds=10)
    
    # Analyze results
    summary_stats = analyze_cv_results(fold_results)
    
    total_time = time.time() - overall_start
    
    # Generate comprehensive report
    report = generate_comprehensive_report(fold_results, summary_stats, total_time, df)
    
    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"🎯 Final Results:")
    print(f"   nDCG@10: {summary_stats['ndcg_10']['mean']:.4f} ± {summary_stats['ndcg_10']['std']:.4f}")
    print(f"   MRR: {summary_stats['mrr']['mean']:.4f} ± {summary_stats['mrr']['std']:.4f}")
    print(f"   Total runtime: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"   Dataset: {len(df):,} examples, {df.query_id.nunique()} queries, {df.llm_id.nunique()} LLMs")
    
    return fold_results, summary_stats, report

if __name__ == "__main__":
    fold_results, summary_stats, report = main()