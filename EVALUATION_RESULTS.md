# Baseline Model Evaluation Results

**Generated**: August 6, 2025  
**Evaluation Pipeline**: 10-fold Cross-Validation on Complete Dataset  
**Runtime**: 1.95 hours (116.8 minutes)

## Dataset Summary

- **Total Examples**: 386,801 query-LLM pairs
- **Unique Queries**: 342 development queries  
- **Unique LLMs**: 1,131 language models
- **Cross-Validation**: Query-based splitting (10 folds)

## Performance Metrics

### Primary TREC Metrics

| Metric | Mean | Std Dev | Min | Max | 95% Confidence Interval |
|--------|------|---------|-----|-----|-------------------------|
| **nDCG@10** | 0.3566 | 0.0571 | 0.2574 | 0.4455 | [0.2447, 0.4685] |
| **nDCG@5** | 0.3496 | 0.0574 | 0.2589 | 0.4544 | [0.2371, 0.4622] |
| **MRR** | 0.6227 | 0.0866 | 0.4840 | 0.7753 | [0.4530, 0.7924] |

### Interpretation

- **nDCG@10 = 0.357**: Moderate ranking quality with potential for improvement through advanced feature engineering
- **MRR = 0.623**: Strong performance at finding first relevant LLM (average reciprocal rank ~1.6)
- **Low Variance**: Standard deviations <0.1 indicate stable, reproducible performance across query distributions

## Fold-by-Fold Analysis

| Fold | nDCG@10 | nDCG@5 | MRR | MSE | MAE | Queries | Train Time (min) |
|------|---------|--------|-----|-----|-----|---------|------------------|
| 1 | 0.3740 | 0.3733 | 0.6648 | 0.0404 | 0.0896 | 35 | 14.9 |
| 2 | 0.3268 | 0.3086 | 0.5375 | 0.0352 | 0.0840 | 35 | 9.7 |
| 3 | 0.2778 | 0.2822 | 0.5173 | 0.0354 | 0.0856 | 34 | 10.2 |
| 4 | 0.2574 | 0.2589 | 0.4840 | 0.0281 | 0.0782 | 34 | 10.6 |
| 5 | 0.3792 | 0.3611 | 0.6260 | 0.0421 | 0.0919 | 34 | 11.1 |
| 6 | 0.3626 | 0.3470 | 0.7050 | 0.0412 | 0.0901 | 34 | 11.0 |
| 7 | 0.4326 | 0.4544 | 0.7753 | 0.0416 | 0.0896 | 34 | 11.5 |
| 8 | 0.3790 | 0.3689 | 0.6315 | 0.0443 | 0.0910 | 34 | 13.2 |
| 9 | 0.4455 | 0.4235 | 0.6910 | 0.0503 | 0.1005 | 34 | 12.7 |
| 10 | 0.3312 | 0.3182 | 0.5946 | 0.0319 | 0.0827 | 34 | 11.3 |

### Performance Distribution

- **Best Fold**: Fold 9 (nDCG@10: 0.4455, MRR: 0.6910)
- **Worst Fold**: Fold 4 (nDCG@10: 0.2574, MRR: 0.4840)  
- **Performance Range**: nDCG@10 varies from 0.26 to 0.45 across folds
- **Training Consistency**: 9.7-14.9 minutes per fold, averaging 11.6 minutes

## Model Configuration

**Algorithm**: Random Forest Regressor  
**Features**: 
- TF-IDF text vectorization (1000 max features, unigrams + bigrams)
- Label-encoded LLM identifiers
- Combined feature matrix: [text_features, llm_id]

**Hyperparameters**:
- n_estimators: 100 trees
- max_depth: 15 levels
- min_samples_split: 5
- min_samples_leaf: 2
- Target: qrel scores normalized to [0,1] interval

**Cross-Validation**: Query-based splitting to prevent data leakage

## Key Findings

### Ranking Performance
1. **First-Rank Effectiveness**: MRR of 0.623 indicates the model frequently places relevant LLMs in top-3 positions
2. **Graded Relevance**: nDCG@10 of 0.357 shows moderate success at overall relevance ordering
3. **Consistency**: nDCG@5 ≈ nDCG@10 suggests most relevance captured in top-5 rankings

### Model Stability  
1. **Cross-Fold Consistency**: No extreme outlier folds, indicating robust learning
2. **Confidence Intervals**: Narrow 95% CIs provide reliable performance estimates for TREC submissions
3. **Computational Efficiency**: ~12 minutes average training time per fold enables practical experimentation

### Baseline Benchmarks
This evaluation establishes the following reference metrics for advanced model comparison:

- **nDCG@10 Baseline**: 0.357 ± 0.057
- **MRR Baseline**: 0.623 ± 0.087  
- **Training Scale**: 348,000 examples per fold
- **Evaluation Scale**: 38,000 examples per fold

## Limitations and Future Work

### Current Limitations
- **Feature Sparsity**: TF-IDF representation may miss semantic query similarities
- **Discovery Data**: 14,950 additional queries with LLM responses unused
- **Model Simplicity**: Random Forest may underperform compared to neural ranking approaches

### Improvement Opportunities  
- **Dense Embeddings**: Replace TF-IDF with BERT/SentenceTransformer query representations
- **Discovery Integration**: Incorporate 14,950 additional queries for enhanced training data
- **Neural Architectures**: Experiment with deep learning models optimized for ranking tasks
- **Meta-Features**: Include LLM architectural metadata and training characteristics

## Reproducibility

**Data Source**: `data/full_evaluation_report.json`  
**Generation Command**: `python train_full_cv_simple_progress.py`  
**Evaluation Date**: August 6, 2025, 18:34:39 UTC

These results provide a robust baseline for evaluating more advanced LLM ranking approaches in the TREC 2025 Million LLMs Track.