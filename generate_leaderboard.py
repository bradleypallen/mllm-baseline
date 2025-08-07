#!/usr/bin/env python3
"""
Leaderboard Generation Script for TREC 2025 Million LLMs Track

Automatically generates a comparison leaderboard from all model results in data/results/
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import argparse


def load_model_results(results_dir='data/results'):
    """Load all model results from the results directory"""
    results_path = Path(results_dir)
    model_results = {}
    
    if not results_path.exists():
        print(f"Results directory {results_dir} not found!")
        return model_results
    
    for results_file in results_path.glob("*_results.json"):
        model_name = results_file.stem.replace('_results', '')
        
        try:
            with open(results_file, 'r') as f:
                data = json.load(f)
            model_results[model_name] = data
            print(f"Loaded results for: {model_name}")
        except Exception as e:
            print(f"Error loading {results_file}: {e}")
    
    return model_results


def extract_leaderboard_data(model_results):
    """Extract key metrics for leaderboard comparison"""
    leaderboard_data = []
    
    for model_name, data in model_results.items():
        # Get performance metrics
        metrics = data.get('performance_metrics', {})
        eval_info = data.get('evaluation_info', {})
        
        row = {
            'Model': model_name.replace('_', ' ').title(),
            'nDCG@10': metrics.get('ndcg_10', {}).get('mean', 0.0),
            'nDCG@10_std': metrics.get('ndcg_10', {}).get('std', 0.0),
            'nDCG@5': metrics.get('ndcg_5', {}).get('mean', 0.0), 
            'nDCG@5_std': metrics.get('ndcg_5', {}).get('std', 0.0),
            'MRR': metrics.get('mrr', {}).get('mean', 0.0),
            'MRR_std': metrics.get('mrr', {}).get('std', 0.0),
            'Runtime_hours': eval_info.get('total_runtime_hours', 0.0),
            'Timestamp': eval_info.get('timestamp', 'Unknown')
        }
        
        leaderboard_data.append(row)
    
    return leaderboard_data


def generate_markdown_leaderboard(leaderboard_data, output_file='leaderboard.md'):
    """Generate markdown leaderboard"""
    # Sort by nDCG@10 (descending)
    sorted_data = sorted(leaderboard_data, key=lambda x: x['nDCG@10'], reverse=True)
    
    markdown_content = f"""# TREC 2025 Million LLMs Track - Model Leaderboard

*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*

## Performance Comparison

Ranking models by nDCG@10 performance on 10-fold cross-validation.

| Rank | Model | nDCG@10 | nDCG@5 | MRR | Runtime | 
|------|--------|---------|--------|-----|---------|
"""
    
    for rank, row in enumerate(sorted_data, 1):
        model_name = row['Model']
        ndcg10 = f"{row['nDCG@10']:.4f} ± {row['nDCG@10_std']:.3f}"
        ndcg5 = f"{row['nDCG@5']:.4f} ± {row['nDCG@5_std']:.3f}" 
        mrr = f"{row['MRR']:.4f} ± {row['MRR_std']:.3f}"
        runtime = f"{row['Runtime_hours']:.2f}h"
        
        markdown_content += f"| {rank} | **{model_name}** | {ndcg10} | {ndcg5} | {mrr} | {runtime} |\n"
    
    markdown_content += f"""

## Evaluation Protocol

- **Dataset**: TREC 2025 Million LLMs Track (342 queries, 1,131 LLMs)  
- **Cross-Validation**: 10-fold, query-based splitting to prevent data leakage
- **Metrics**: nDCG@10, nDCG@5, Mean Reciprocal Rank (MRR)
- **Qrel Encoding**: 0→0.0 (not relevant), 1→1.0 (most relevant), 2→0.7 (second-most relevant)

## Model Details

"""
    
    for row in sorted_data:
        model_name = row['Model']
        markdown_content += f"### {model_name}\n"
        
        if 'Random Forest' in model_name:
            markdown_content += """
- **Architecture**: Random Forest Regressor (100 trees, max_depth=15)
- **Features**: TF-IDF (1000 features) + LLM ID encoding  
- **Training**: Scikit-learn with standard hyperparameters
"""
        elif 'Neural Two Tower' in model_name:
            markdown_content += """
- **Architecture**: Dual encoder with sentence transformers
- **Query Tower**: all-MiniLM-L6-v2 → Dense [384→256→128→64]
- **LLM Tower**: Learned embeddings → Dense [64→128→64] 
- **Training**: 20 epochs/fold, margin-based pairwise loss
"""
        
        markdown_content += f"- **Performance**: nDCG@10={row['nDCG@10']:.4f}, MRR={row['MRR']:.4f}\n"
        markdown_content += f"- **Runtime**: {row['Runtime_hours']:.2f} hours\n\n"
    
    markdown_content += """## Usage

To add a new model to the leaderboard:

1. Implement your model in `models/your_model_name/`
2. Save results to `data/results/your_model_name_results.json` using the standardized format
3. Run `python generate_leaderboard.py` to update this leaderboard

## Results Files

"""
    
    for row in sorted_data:
        model_file = row['Model'].lower().replace(' ', '_') + '_results.json'
        markdown_content += f"- `data/results/{model_file}` - {row['Model']} detailed results\n"
    
    # Save leaderboard
    with open(output_file, 'w') as f:
        f.write(markdown_content)
    
    return markdown_content


def generate_csv_leaderboard(leaderboard_data, output_file='leaderboard.csv'):
    """Generate CSV leaderboard for easy analysis"""
    df = pd.DataFrame(leaderboard_data)
    df = df.sort_values('nDCG@10', ascending=False)
    df.to_csv(output_file, index=False)
    return df


def main():
    parser = argparse.ArgumentParser(description='Generate model comparison leaderboard')
    parser.add_argument('--results-dir', default='data/results', 
                        help='Directory containing model results (default: data/results)')
    parser.add_argument('--output', default='leaderboard.md',
                        help='Output leaderboard file (default: leaderboard.md)')
    parser.add_argument('--csv', action='store_true',
                        help='Also generate CSV leaderboard')
    
    args = parser.parse_args()
    
    print("Generating TREC 2025 Million LLMs Track Leaderboard...")
    
    # Load all model results
    model_results = load_model_results(args.results_dir)
    
    if not model_results:
        print("No model results found! Make sure results are saved in the correct format.")
        return
    
    # Extract leaderboard data
    leaderboard_data = extract_leaderboard_data(model_results)
    
    # Generate markdown leaderboard
    markdown_content = generate_markdown_leaderboard(leaderboard_data, args.output)
    print(f"Markdown leaderboard saved to: {args.output}")
    
    # Generate CSV leaderboard if requested
    if args.csv:
        csv_file = args.output.replace('.md', '.csv')
        generate_csv_leaderboard(leaderboard_data, csv_file)
        print(f"CSV leaderboard saved to: {csv_file}")
    
    # Show summary
    print(f"\nLeaderboard Summary ({len(model_results)} models):")
    sorted_data = sorted(leaderboard_data, key=lambda x: x['nDCG@10'], reverse=True)
    for rank, row in enumerate(sorted_data, 1):
        print(f"{rank}. {row['Model']:.<20} nDCG@10={row['nDCG@10']:.4f}")


if __name__ == "__main__":
    main()