#!/usr/bin/env python3
"""
Create supervised training set from TREC MLLM dev data.

Combines query text (data/llm_dev_data.tsv) with relevance judgments (data/llm_dev_qrels.txt)
to create a clean training dataset with ground truth labels.
"""

import pandas as pd

def main():
    """Generate supervised training set"""
    print("=== CREATING SUPERVISED TRAINING SET ===")
    
    # Load dev queries
    print("Loading query text...")
    dev_queries_df = pd.read_csv('data/llm_dev_data.tsv', sep='\t', header=None,
                                names=['query_id', 'query_text'],
                                dtype={'query_id': str})
    
    # Load relevance judgments
    print("Loading relevance judgments...")
    qrels_df = pd.read_csv('data/llm_dev_qrels.txt', sep=' ', header=None,
                          names=['query_id', 'iteration', 'llm_id', 'qrel'],
                          dtype={'query_id': str, 'llm_id': str})
    
    # Combine query text with judgments
    print("Joining data...")
    training_df = qrels_df.merge(dev_queries_df, on='query_id', how='left')
    
    # Clean data
    training_df = training_df.dropna(subset=['query_text'])
    training_df = training_df.drop('iteration', axis=1)  # Always 0, not needed
    
    # Save dataset
    print("Saving supervised training set...")
    training_df.to_csv('data/supervised_training_full.csv', index=False)
    
    print(f"✓ Generated supervised training set: {len(training_df):,} examples")
    print(f"✓ Queries: {training_df.query_id.nunique()}")
    print(f"✓ LLMs: {training_df.llm_id.nunique()}")
    print("✓ Saved to: data/supervised_training_full.csv")

if __name__ == "__main__":
    main()