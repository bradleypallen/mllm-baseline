#!/usr/bin/env python3
"""
Neural Model Evaluation and Comparison

Utilities for evaluating the Two-Tower neural model and comparing
performance against the Random Forest baseline.
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os


def load_evaluation_results(neural_path='../data/neural_evaluation_report.json',
                          rf_path='../data/full_evaluation_report.json'):
    """Load evaluation results from both models"""
    
    results = {}
    
    # Load neural results
    try:
        with open(neural_path, 'r') as f:
            results['neural'] = json.load(f)
        print(f"✓ Loaded neural results from {neural_path}")
    except FileNotFoundError:
        print(f"✗ Neural results not found at {neural_path}")
        results['neural'] = None
    
    # Load Random Forest results
    try:
        with open(rf_path, 'r') as f:
            results['random_forest'] = json.load(f)
        print(f"✓ Loaded Random Forest results from {rf_path}")
    except FileNotFoundError:
        print(f"✗ Random Forest results not found at {rf_path}")
        results['random_forest'] = None
    
    return results


def compare_performance(results):
    """Compare performance metrics between models"""
    
    if results['neural'] is None or results['random_forest'] is None:
        print("Cannot compare - missing results for one or both models")
        return
    
    neural_metrics = results['neural']['performance_metrics']
    rf_metrics = results['random_forest']['performance_metrics']
    
    print("\n=== PERFORMANCE COMPARISON ===")
    print(f"{'Metric':<12} {'Neural':<20} {'Random Forest':<20} {'Improvement':<12}")
    print("-" * 70)
    
    metrics = ['ndcg_10', 'ndcg_5', 'mrr']
    improvements = {}
    
    for metric in metrics:
        neural_mean = neural_metrics[metric]['mean']
        neural_std = neural_metrics[metric]['std']
        rf_mean = rf_metrics[metric]['mean']
        rf_std = rf_metrics[metric]['std']
        
        improvement = ((neural_mean - rf_mean) / rf_mean) * 100
        improvements[metric] = improvement
        
        print(f"{metric.upper():<12} {neural_mean:.4f}±{neural_std:.4f:<8} "
              f"{rf_mean:.4f}±{rf_std:.4f:<8} {improvement:+.2f}%")
    
    # Runtime comparison
    neural_runtime = results['neural']['evaluation_info']['total_runtime_hours']
    rf_runtime = results['random_forest']['evaluation_info']['total_runtime_hours']
    runtime_ratio = neural_runtime / rf_runtime
    
    print(f"\nRUNTIME COMPARISON:")
    print(f"Neural:        {neural_runtime:.2f} hours")
    print(f"Random Forest: {rf_runtime:.2f} hours")
    print(f"Neural is {runtime_ratio:.1f}x {'slower' if runtime_ratio > 1 else 'faster'}")
    
    return improvements


def analyze_fold_stability(results):
    """Analyze fold-by-fold performance stability"""
    
    print("\n=== FOLD-BY-FOLD STABILITY ANALYSIS ===")
    
    for model_name, model_results in results.items():
        if model_results is None:
            continue
            
        fold_results = model_results['fold_by_fold_results']
        
        print(f"\n{model_name.upper()} Model:")
        
        # Extract fold scores
        ndcg_10_scores = [fold['ndcg_10'] for fold in fold_results]
        ndcg_5_scores = [fold['ndcg_5'] for fold in fold_results]
        mrr_scores = [fold['mrr'] for fold in fold_results]
        
        # Calculate stability metrics
        ndcg_10_cv = np.std(ndcg_10_scores) / np.mean(ndcg_10_scores)
        ndcg_5_cv = np.std(ndcg_5_scores) / np.mean(ndcg_5_scores)
        mrr_cv = np.std(mrr_scores) / np.mean(mrr_scores)
        
        print(f"  nDCG@10 CV: {ndcg_10_cv:.3f} (lower is more stable)")
        print(f"  nDCG@5 CV:  {ndcg_5_cv:.3f}")
        print(f"  MRR CV:     {mrr_cv:.3f}")
        
        # Best and worst folds
        best_fold = max(fold_results, key=lambda x: x['ndcg_10'])
        worst_fold = min(fold_results, key=lambda x: x['ndcg_10'])
        
        print(f"  Best fold:  {best_fold['fold']} (nDCG@10: {best_fold['ndcg_10']:.4f})")
        print(f"  Worst fold: {worst_fold['fold']} (nDCG@10: {worst_fold['ndcg_10']:.4f})")
        print(f"  Range:      {best_fold['ndcg_10'] - worst_fold['ndcg_10']:.4f}")


def create_comparison_plots(results, save_dir='../plots'):
    """Create comparison plots"""
    
    # Create plots directory
    os.makedirs(save_dir, exist_ok=True)
    
    if results['neural'] is None or results['random_forest'] is None:
        print("Cannot create plots - missing results")
        return
    
    # Set up plot style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Fold-by-fold comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Model Performance Comparison Across Folds', fontsize=16)
    
    metrics = ['ndcg_10', 'ndcg_5', 'mrr']
    metric_names = ['nDCG@10', 'nDCG@5', 'MRR']
    
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        # Extract scores
        neural_scores = [fold[metric] for fold in results['neural']['fold_by_fold_results']]
        rf_scores = [fold[metric] for fold in results['random_forest']['fold_by_fold_results']]
        folds = range(1, len(neural_scores) + 1)
        
        # Plot
        axes[i].plot(folds, neural_scores, 'o-', label='Two-Tower Neural', linewidth=2, markersize=6)
        axes[i].plot(folds, rf_scores, 's-', label='Random Forest', linewidth=2, markersize=6)
        axes[i].set_xlabel('Fold')
        axes[i].set_ylabel(name)
        axes[i].set_title(f'{name} by Fold')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        
        # Add mean lines
        axes[i].axhline(np.mean(neural_scores), color='C0', linestyle='--', alpha=0.7)
        axes[i].axhline(np.mean(rf_scores), color='C1', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/model_comparison_folds.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Performance distribution box plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Performance Distribution Comparison', fontsize=16)
    
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        neural_scores = [fold[metric] for fold in results['neural']['fold_by_fold_results']]
        rf_scores = [fold[metric] for fold in results['random_forest']['fold_by_fold_results']]
        
        data = [neural_scores, rf_scores]
        labels = ['Two-Tower Neural', 'Random Forest']
        
        bp = axes[i].boxplot(data, labels=labels, patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')
        
        axes[i].set_ylabel(name)
        axes[i].set_title(f'{name} Distribution')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/performance_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Plots saved to {save_dir}/")


def generate_comparison_report(results, output_path='../neural_baseline_report.md'):
    """Generate markdown comparison report"""
    
    if results['neural'] is None:
        print("Cannot generate report - neural results missing")
        return
    
    neural_metrics = results['neural']['performance_metrics']
    neural_info = results['neural']['evaluation_info']
    
    # Calculate improvements if RF results available
    improvements = {}
    if results['random_forest'] is not None:
        rf_metrics = results['random_forest']['performance_metrics']
        for metric in ['ndcg_10', 'ndcg_5', 'mrr']:
            neural_mean = neural_metrics[metric]['mean']
            rf_mean = rf_metrics[metric]['mean']
            improvements[metric] = ((neural_mean - rf_mean) / rf_mean) * 100
    
    # Generate report
    report = f"""# Two-Tower Neural Network Baseline Report

**Generated**: {datetime.now().strftime('%B %d, %Y at %H:%M UTC')}  
**Evaluation**: {neural_info['pipeline']}  
**Dataset**: {neural_info['dataset']}  
**Runtime**: {neural_info['total_runtime_hours']:.2f} hours  

## Model Architecture

- **Query Tower**: Sentence transformer (all-MiniLM-L6-v2) + Dense layers [384→256→128→64]
- **LLM Tower**: Learned embeddings + Dense layers [64→128→64] 
- **Similarity**: Cosine similarity between 64-dimensional embeddings
- **Loss**: Margin-based pairwise ranking loss
- **Training**: {neural_info['epochs_per_fold']} epochs per fold, batch size {neural_info['batch_size']}

## Performance Results

### Cross-Validation Performance (10-Fold)

| Metric | Mean | Std | 95% CI | Min | Max |
|---------|------|-----|--------|-----|-----|
| **nDCG@10** | {neural_metrics['ndcg_10']['mean']:.4f} | {neural_metrics['ndcg_10']['std']:.4f} | [{neural_metrics['ndcg_10']['confidence_interval_95'][0]:.4f}, {neural_metrics['ndcg_10']['confidence_interval_95'][1]:.4f}] | {neural_metrics['ndcg_10']['min']:.4f} | {neural_metrics['ndcg_10']['max']:.4f} |
| **nDCG@5** | {neural_metrics['ndcg_5']['mean']:.4f} | {neural_metrics['ndcg_5']['std']:.4f} | [{neural_metrics['ndcg_5']['confidence_interval_95'][0]:.4f}, {neural_metrics['ndcg_5']['confidence_interval_95'][1]:.4f}] | {neural_metrics['ndcg_5']['min']:.4f} | {neural_metrics['ndcg_5']['max']:.4f} |
| **MRR** | {neural_metrics['mrr']['mean']:.4f} | {neural_metrics['mrr']['std']:.4f} | [{neural_metrics['mrr']['confidence_interval_95'][0]:.4f}, {neural_metrics['mrr']['confidence_interval_95'][1]:.4f}] | {neural_metrics['mrr']['min']:.4f} | {neural_metrics['mrr']['max']:.4f} |

"""
    
    # Add comparison section if available
    if improvements:
        report += f"""
### Comparison with Random Forest Baseline

| Metric | Neural Network | Random Forest | Improvement |
|---------|----------------|---------------|-------------|
| **nDCG@10** | {neural_metrics['ndcg_10']['mean']:.4f} | {rf_metrics['ndcg_10']['mean']:.4f} | {improvements['ndcg_10']:+.2f}% |
| **nDCG@5** | {neural_metrics['ndcg_5']['mean']:.4f} | {rf_metrics['ndcg_5']['mean']:.4f} | {improvements['ndcg_5']:+.2f}% |
| **MRR** | {neural_metrics['mrr']['mean']:.4f} | {rf_metrics['mrr']['mean']:.4f} | {improvements['mrr']:+.2f}% |

"""
    
    # Add fold-by-fold results
    report += """
### Fold-by-Fold Results

| Fold | nDCG@10 | nDCG@5 | MRR | Queries | Train Time (min) | Best Epoch |
|------|---------|--------|-----|---------|------------------|------------|
"""
    
    for fold in results['neural']['fold_by_fold_results']:
        report += f"| {fold['fold']} | {fold['ndcg_10']:.4f} | {fold['ndcg_5']:.4f} | {fold['mrr']:.4f} | {fold['n_queries']} | {fold['train_time']/60:.1f} | {fold['best_epoch']} |\n"
    
    report += f"""

## Key Findings

### Neural Network Advantages
- **Deep Feature Learning**: Learns dense representations for queries and LLMs
- **Semantic Understanding**: Leverages pre-trained sentence transformers
- **End-to-End Training**: Optimizes directly for ranking objectives

### Implementation Details
- **Device**: {neural_info['device']}
- **Total Training Time**: {neural_info['total_runtime_hours']:.2f} hours
- **Parameters**: ~{neural_info['batch_size']} batch size, {neural_info['learning_rate']} learning rate
- **Architecture**: Two-tower design with 64-dimensional embeddings

This neural baseline demonstrates the feasibility of deep learning approaches for the TREC 2025 Million LLMs ranking task.
"""
    
    # Save report
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"✓ Report saved to {output_path}")


def main():
    """Main evaluation function"""
    print("=== NEURAL BASELINE EVALUATION ===")
    
    # Load results
    results = load_evaluation_results()
    
    # Compare performance
    if results['neural'] and results['random_forest']:
        improvements = compare_performance(results)
        analyze_fold_stability(results)
        
        # Create plots
        create_comparison_plots(results)
    
    # Generate report
    generate_comparison_report(results)
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()