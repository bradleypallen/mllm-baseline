# Epistemic Profiling Results Summary

## Overview
This document summarizes the results of integrating epistemic profiling (bilateral truth evaluation and expertise matching) into LLM ranking models for the TREC 2025 Million LLMs Track.

## Approach
We developed three key components:
1. **Bilateral Truth Profiles**: 4D epistemic profiles measuring LLM behavior patterns
2. **LLM Clustering**: Grouped 1,131 LLMs into 5 clusters based on epistemic profiles
3. **Expertise Profiles**: Domain-specific expertise for 117 elite LLMs

## Models Tested

### 1. XGBoost with Expertise Features
- **Baseline XGBoost**: nDCG@10 = 0.3925
- **XGBoost + Expertise**: nDCG@10 = 0.3994 ± 0.0434
- **Improvement**: +1.8%
- **Runtime**: ~15 minutes for 10-fold CV
- **Features**: 1021 total (1000 TF-IDF + 1 LLM ID + 5 bilateral + 6 cluster + 9 expertise)

### 2. Two-Tower with Expertise Features (Partial Results)
- **Baseline Two-Tower**: nDCG@10 = 0.4022
- **Two-Tower + Expertise**: nDCG@10 ≈ 0.41 (estimated from 9 folds)
- **Improvement**: ~2%
- **Runtime**: >2 hours on CPU (terminated early)
- **Architecture**: Augmented LLM tower with 20 additional profile features

### 3. Two-Phase Reranking
- **Baseline (random elite)**: nDCG@10 = 0.2801
- **With expertise reranking**: nDCG@10 = 0.3467
- **Improvement**: +23.8% (but over weak baseline)
- **Note**: This baseline was misleading - just random selection of elite LLMs

## Key Findings

### Bilateral Profiles Analysis
- **Cluster 0** (822 LLMs): "Confabulators" - high overconfidence, poor reliability
- **Clusters 1-2** (117 LLMs): Elite performers - high confidence with correctness
- **Clusters 3-4** (192 LLMs): Mediocre - mixed performance

### Feature Importance (from XGBoost)
The expertise features contributed but didn't dominate:
- TF-IDF features remained most important
- Bilateral reliability score showed moderate importance
- Expertise matching similarity had limited impact

### Performance Summary
| Model | nDCG@10 | Improvement | Status |
|-------|---------|-------------|--------|
| Neural Two-Tower (baseline) | 0.4022 | - | Complete |
| XGBoost + Expertise | 0.3994 | +1.8% over XGBoost | Complete |
| Two-Tower + Expertise | ~0.41 | ~+2% over Two-Tower | Partial |
| XGBoost (baseline) | 0.3925 | - | Complete |
| Random Forest | 0.3860 | - | Complete |

## Conclusions

### What Worked
1. **Bilateral profiles** successfully identified unreliable LLMs (Cluster 0)
2. **Elite LLM identification** was accurate (Clusters 1-2)
3. **Feature integration** was technically successful across architectures
4. **Modest improvements** were consistent across different models

### What Didn't Work
1. **Limited impact**: Expertise features provided only 1-2% improvement
2. **Computational cost**: Two-Tower training became prohibitively slow
3. **Expertise coverage**: Only 117/1131 LLMs had expertise profiles
4. **Quality estimation**: Heuristic-based quality scores were noisy

## Recommendations

### For Future Work
1. **Expand coverage**: Generate expertise profiles for all 1,131 LLMs
2. **Improve quality metrics**: Use actual bilateral truth evaluation instead of heuristics
3. **Feature engineering**: Explore interaction features between profiles and queries
4. **Ensemble approach**: Combine expertise-enhanced models with base models

### Technical Improvements
1. **GPU acceleration**: Essential for neural models with additional features
2. **Batch processing**: More efficient bilateral truth evaluation
3. **Caching**: Pre-compute expertise similarities for common queries
4. **Sparse features**: Use sparse representations for cluster/expertise features

## Data Assets Created
- `complete_profiles.json`: Bilateral profiles for all 1,131 LLMs
- `llm_clusters.json`: Cluster assignments based on epistemic behavior
- `all_llm_expertise_profiles.json`: Expertise profiles for 117 elite LLMs
- `xgboost_expertise_results.json`: Complete XGBoost evaluation results

## Code Artifacts
- `bilateral_truth_clustering.py`: Clustering based on epistemic profiles
- `xgboost_with_expertise.py`: XGBoost with full feature integration
- `two_tower_with_expertise.py`: Neural model with profile features
- `two_phase_reranker.py`: Expertise-based reranking approach

## Final Assessment
While epistemic profiling provided meaningful insights into LLM behavior and modest performance improvements, the gains were not substantial enough to justify the added complexity for production use. The approach successfully identified unreliable LLMs and elite performers, but translating these insights into ranking improvements proved challenging. The 1-2% improvements suggest the profiles capture some useful signal, but more sophisticated integration methods or better quality estimation may be needed for breakthrough performance.