#!/usr/bin/env python3
"""
Quick TREC submission pipeline - single fold, optimized for speed.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import ndcg_score
import json
import time

def quick_train_and_predict():
    """Fast training and prediction pipeline"""
    print("=== QUICK TREC SUBMISSION PIPELINE ===")
    
    # Load full dataset but use smaller model for speed
    print("Loading full training data...")
    df = pd.read_csv('data/supervised_training_full.csv')
    print(f"✓ Loaded {len(df):,} examples ({df.query_id.nunique()} queries, {df.llm_id.nunique()} LLMs)")
    
    # Create features with smaller vocabulary for speed
    print("Creating features...")
    tfidf_vectorizer = TfidfVectorizer(
        max_features=500,  # Reduced from 1000
        stop_words='english',
        ngram_range=(1, 1),  # Only unigrams for speed
        min_df=3  # Higher threshold
    )
    
    query_features = tfidf_vectorizer.fit_transform(df['query_text'])
    llm_encoder = LabelEncoder()
    llm_encoded = llm_encoder.fit_transform(df['llm_id'])
    
    X_text = query_features.toarray()
    X_llm = llm_encoded.reshape(-1, 1)
    X = np.hstack([X_text, X_llm])
    y = df['qrel'].values / 2.0
    
    print(f"✓ Features: {X.shape}")
    
    # Fast model training
    print("Training lightweight model...")
    model = RandomForestRegressor(
        n_estimators=50,  # Reduced trees
        max_depth=10,     # Shallower trees
        min_samples_split=10,
        random_state=42,
        n_jobs=-1
    )
    
    start_time = time.time()
    model.fit(X, y)
    train_time = time.time() - start_time
    print(f"✓ Trained in {train_time:.1f} seconds")
    
    return model, tfidf_vectorizer, llm_encoder, df

def generate_quick_submission(model, tfidf_vectorizer, llm_encoder, df):
    """Generate TREC submission with sample queries"""
    print("\n=== GENERATING SUBMISSION ===")
    
    # Use actual dev queries as test examples (simulate real TREC test queries)
    dev_sample = df[['query_id', 'query_text']].drop_duplicates().head(3)
    
    test_queries = {}
    for _, row in dev_sample.iterrows():
        test_queries[row['query_id']] = row['query_text']
    
    print(f"Using actual dev queries as test examples:")
    for qid, query in test_queries.items():
        print(f"  {qid}: {query[:60]}...")
    
    all_llm_ids = llm_encoder.classes_
    print(f"Test queries: {len(test_queries)}")
    print(f"LLMs to rank: {len(all_llm_ids)}")
    
    # Generate predictions efficiently
    trec_lines = []
    
    for query_id, query_text in test_queries.items():
        print(f"  Processing {query_id}...")
        
        # Create batch of examples for this query
        query_batch = [query_text] * len(all_llm_ids)
        llm_batch = list(all_llm_ids)
        
        # Create features
        temp_df = pd.DataFrame({
            'query_text': query_batch,
            'llm_id': llm_batch
        })
        
        query_features = tfidf_vectorizer.transform(temp_df['query_text'])
        llm_encoded = llm_encoder.transform(temp_df['llm_id'])
        
        X_text = query_features.toarray()
        X_llm = llm_encoded.reshape(-1, 1)
        X = np.hstack([X_text, X_llm])
        
        # Predict and rank
        scores = model.predict(X)
        scores = np.clip(scores, 0, 1)
        
        # Create ranked list
        llm_scores = list(zip(all_llm_ids, scores))
        llm_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Generate TREC format lines
        for rank, (llm_id, score) in enumerate(llm_scores, 1):
            line = f"{query_id} Q0 {llm_id} {rank} {score:.6f} quick_baseline"
            trec_lines.append(line)
    
    # Write submission file
    with open("quick_submission.txt", "w") as f:
        f.write('\n'.join(trec_lines))
    
    print(f"✓ Submission generated: {len(trec_lines):,} lines")
    
    # Show sample output
    print("\nSample submission lines:")
    for line in trec_lines[:10]:
        print(f"  {line}")
    
    return "quick_submission.txt"

def validate_quick_submission():
    """Quick validation of submission format"""
    print("\n=== VALIDATION ===")
    
    with open("quick_submission.txt", "r") as f:
        lines = f.readlines()
    
    print(f"Total lines: {len(lines):,}")
    
    # Check format
    sample_line = lines[0].strip().split()
    if len(sample_line) == 6:
        print(f"✓ Format correct: 6 columns")
        print(f"  Query ID: {sample_line[0]}")
        print(f"  Q0: {sample_line[1]}")
        print(f"  LLM ID: {sample_line[2]}")
        print(f"  Rank: {sample_line[3]}")
        print(f"  Score: {sample_line[4]}")
        print(f"  Run ID: {sample_line[5]}")
    
    # Check queries and rankings
    queries = set()
    for line in lines:
        queries.add(line.split()[0])
    
    print(f"✓ Queries in submission: {len(queries)}")
    
    return True

def main():
    """Quick submission pipeline"""
    overall_start = time.time()
    
    # Train model
    model, tfidf_vectorizer, llm_encoder, df = quick_train_and_predict()
    
    # Generate submission
    submission_file = generate_quick_submission(model, tfidf_vectorizer, llm_encoder, df)
    
    # Validate
    validate_quick_submission()
    
    total_time = time.time() - overall_start
    
    print(f"\n=== QUICK PIPELINE COMPLETE ===")
    print(f"Total time: {total_time:.1f} seconds")
    print(f"Submission file: {submission_file}")
    print(f"Ready for TREC Million LLMs Track!")
    
    return submission_file

if __name__ == "__main__":
    submission_file = main()