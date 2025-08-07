#!/usr/bin/env python3
"""
Create a pure supervised training set from TREC MLLM dev data.

Combines query text (data/llm_dev_data.tsv) with relevance judgments (data/llm_dev_qrels.txt)
to create a clean training dataset with ground truth labels.
"""

import pandas as pd
import json

def create_full_supervised_set():
    """Create complete supervised training set"""
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
    
    print(f"✓ Complete training set: {len(training_df):,} examples")
    print(f"✓ Queries: {training_df.query_id.nunique()}")
    print(f"✓ LLMs: {training_df.llm_id.nunique()}")
    
    return training_df

def create_subset(training_df, n_queries=1000, n_llms=100):
    """Create smaller subset for faster experimentation"""
    print(f"\n=== CREATING SUBSET ({n_queries} queries, {n_llms} LLMs) ===")
    
    # Select top LLMs by expertise rate
    llm_stats = training_df.groupby('llm_id').agg({
        'qrel': ['count', 'sum', 'mean']
    })
    llm_stats.columns = ['total_queries', 'expert_count', 'expert_rate']
    llm_stats = llm_stats.sort_values('expert_rate', ascending=False)
    
    top_llms = llm_stats.head(n_llms).index.tolist()
    print(f"✓ Selected top {n_llms} LLMs (expertise rate: {llm_stats.iloc[n_llms-1]['expert_rate']:.3f} - {llm_stats.iloc[0]['expert_rate']:.3f})")
    
    # Select queries - prioritize those with more expert LLMs
    query_stats = training_df[training_df.qrel > 0].groupby('query_id').size().sort_values(ascending=False)
    
    if n_queries < len(query_stats):
        # Take top queries by number of experts
        selected_queries = query_stats.head(n_queries).index.tolist()
        print(f"✓ Selected {n_queries} queries with most expert LLMs")
    else:
        # Take all available queries
        selected_queries = training_df.query_id.unique().tolist()
        print(f"✓ Using all {len(selected_queries)} available queries")
    
    # Filter training data
    subset_df = training_df[
        (training_df.query_id.isin(selected_queries)) & 
        (training_df.llm_id.isin(top_llms))
    ]
    
    print(f"✓ Subset: {len(subset_df):,} examples")
    
    return subset_df, top_llms, selected_queries

def analyze_dataset(df, name="dataset"):
    """Print dataset statistics"""
    print(f"\n=== {name.upper()} STATISTICS ===")
    print(f"Total examples: {len(df):,}")
    print(f"Queries: {df.query_id.nunique()}")
    print(f"LLMs: {df.llm_id.nunique()}")
    
    # Relevance distribution
    print("\nRelevance distribution:")
    rel_dist = df.qrel.value_counts().sort_index()
    total = len(df)
    for rel, count in rel_dist.items():
        pct = count/total*100
        if rel == 0:
            print(f"  Rank {rel} (Not relevant): {count:,} ({pct:.1f}%)")
        else:
            print(f"  Rank {rel} (#{rel} most relevant): {count:,} ({pct:.1f}%)")
    
    # Query statistics
    examples_per_query = df.groupby('query_id').size()
    print(f"\nPer query:")
    print(f"  Examples per query: {examples_per_query.mean():.0f} avg, {examples_per_query.std():.0f} std")
    print(f"  Range: {examples_per_query.min()} - {examples_per_query.max()}")
    
    # Expert statistics
    expert_df = df[df.qrel > 0]
    if len(expert_df) > 0:
        experts_per_query = expert_df.groupby('query_id').size()
        print(f"\nExpert LLMs:")
        print(f"  Experts per query: {experts_per_query.mean():.1f} avg")
        print(f"  Queries with experts: {len(experts_per_query)}/{df.query_id.nunique()}")

def main():
    """Generate training sets"""
    
    # Create full supervised set
    full_training_df = create_full_supervised_set()
    
    # Create subset for experimentation  
    subset_df, top_llms, selected_queries = create_subset(full_training_df, 
                                                         n_queries=1000, n_llms=100)
    
    # Save datasets
    print("\n=== SAVING FILES ===")
    full_training_df.to_csv('data/supervised_training_full.csv', index=False)
    subset_df.to_csv('data/supervised_training_subset.csv', index=False)
    
    # Save metadata
    metadata = {
        'top_100_llms': top_llms,
        'subset_queries': selected_queries,
        'full_stats': {
            'total_examples': len(full_training_df),
            'queries': int(full_training_df.query_id.nunique()),
            'llms': int(full_training_df.llm_id.nunique())
        },
        'subset_stats': {
            'total_examples': len(subset_df),  
            'queries': int(subset_df.query_id.nunique()),
            'llms': int(subset_df.llm_id.nunique())
        }
    }
    
    with open('data/training_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("✓ data/supervised_training_full.csv - Complete dataset")
    print("✓ data/supervised_training_subset.csv - 1000 query subset")  
    print("✓ data/training_metadata.json - Dataset metadata")
    
    # Analysis
    analyze_dataset(full_training_df, "Full Dataset")
    analyze_dataset(subset_df, "Subset")

if __name__ == "__main__":
    main()