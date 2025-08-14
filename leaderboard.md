# TREC 2025 Million LLMs Track - Model Leaderboard

*Generated on 2025-08-14 04:38:05*

## Performance Comparison

Ranking models by nDCG@10 performance on 10-fold cross-validation.

| Rank | Model | nDCG@10 | nDCG@5 | MRR | Runtime | 
|------|--------|---------|--------|-----|---------|
| 1 | **Tier2 Cpu Optimized** | 0.4306 ± 0.055 | 0.4347 ± 0.058 | 0.7263 ± 0.070 | 3.11h |
| 2 | **Tier3 Cross Encoder** | 0.4259 ± 0.049 | 0.4378 ± 0.051 | 0.7141 ± 0.076 | 21.92h |
| 3 | **Enhanced Neural Two Tower** | 0.4256 ± 0.050 | 0.4287 ± 0.056 | 0.7113 ± 0.074 | 2.95h |
| 4 | **Tier2 With Profiles Simplified** | 0.4128 ± 0.031 | 0.4232 ± 0.037 | 0.7083 ± 0.053 | 6.23h |
| 5 | **Xgboost Hybrid** | 0.4107 ± 0.044 | 0.4216 ± 0.048 | 0.6979 ± 0.060 | 0.00h |
| 6 | **Neural Two Tower** | 0.4022 ± 0.028 | 0.4135 ± 0.034 | 0.6761 ± 0.057 | 6.95h |
| 7 | **Xgboost Smart Imputation** | 0.4017 ± 0.037 | 0.4098 ± 0.039 | 0.6690 ± 0.049 | 0.00h |
| 8 | **Enhanced Two Tower Epistemic** | 0.4009 ± 0.061 | 0.4093 ± 0.074 | 0.6777 ± 0.090 | 1.30h |
| 9 | **Xgboost Ensemble** | 0.4003 ± 0.040 | 0.4109 ± 0.039 | 0.6844 ± 0.056 | 0.00h |
| 10 | **Xgboost Deeper** | 0.3997 ± 0.031 | 0.4017 ± 0.022 | 0.6630 ± 0.056 | 0.00h |
| 11 | **Xgboost Expertise** | 0.3994 ± 0.043 | 0.4077 ± 0.048 | 0.6842 ± 0.082 | 0.25h |
| 12 | **Xgboost Interactions** | 0.3977 ± 0.049 | 0.4058 ± 0.056 | 0.6578 ± 0.065 | 0.00h |
| 13 | **Xgboost Twostage** | 0.3957 ± 0.048 | 0.4056 ± 0.045 | 0.6827 ± 0.062 | 0.00h |
| 14 | **Simplified Epistemic 300** | 0.3952 ± 0.061 | 0.4040 ± 0.064 | 0.6721 ± 0.092 | 1.37h |
| 15 | **Xgboost Epistemic** | 0.3927 ± 0.049 | 0.4062 ± 0.048 | 0.6691 ± 0.045 | 0.01h |
| 16 | **Xgboost Full Profiles** | 0.3910 ± 0.037 | 0.4009 ± 0.042 | 0.6953 ± 0.068 | 0.00h |
| 17 | **Xgboost Minimal Reliability** | 0.3886 ± 0.048 | 0.3919 ± 0.054 | 0.6157 ± 0.070 | 0.00h |
| 18 | **Xgboost Complete Profiles** | 0.3874 ± 0.034 | 0.4001 ± 0.045 | 0.6885 ± 0.062 | 0.00h |
| 19 | **Random Forest** | 0.3860 ± 0.044 | 0.3871 ± 0.050 | 0.6701 ± 0.081 | 1.37h |
| 20 | **Xgboost** | 0.3824 ± 0.045 | 0.3808 ± 0.047 | 0.6206 ± 0.052 | 0.03h |


## Evaluation Protocol

- **Dataset**: TREC 2025 Million LLMs Track (342 queries, 1,131 LLMs)  
- **Cross-Validation**: 10-fold, query-based splitting to prevent data leakage
- **Metrics**: nDCG@10, nDCG@5, Mean Reciprocal Rank (MRR)
- **Qrel Encoding**: 0→0.0 (not relevant), 1→1.0 (most relevant), 2→0.7 (second-most relevant)

## Model Details

### Tier2 Cpu Optimized
- **Performance**: nDCG@10=0.4306, MRR=0.7263
- **Runtime**: 3.11 hours

### Tier3 Cross Encoder
- **Performance**: nDCG@10=0.4259, MRR=0.7141
- **Runtime**: 21.92 hours

### Enhanced Neural Two Tower

- **Architecture**: Dual encoder with sentence transformers
- **Query Tower**: all-MiniLM-L6-v2 → Dense [384→256→128→64]
- **LLM Tower**: Learned embeddings → Dense [64→128→64] 
- **Training**: 20 epochs/fold, margin-based pairwise loss
- **Performance**: nDCG@10=0.4256, MRR=0.7113
- **Runtime**: 2.95 hours

### Tier2 With Profiles Simplified
- **Performance**: nDCG@10=0.4128, MRR=0.7083
- **Runtime**: 6.23 hours

### Xgboost Hybrid
- **Performance**: nDCG@10=0.4107, MRR=0.6979
- **Runtime**: 0.00 hours

### Neural Two Tower

- **Architecture**: Dual encoder with sentence transformers
- **Query Tower**: all-MiniLM-L6-v2 → Dense [384→256→128→64]
- **LLM Tower**: Learned embeddings → Dense [64→128→64] 
- **Training**: 20 epochs/fold, margin-based pairwise loss
- **Performance**: nDCG@10=0.4022, MRR=0.6761
- **Runtime**: 6.95 hours

### Xgboost Smart Imputation
- **Performance**: nDCG@10=0.4017, MRR=0.6690
- **Runtime**: 0.00 hours

### Enhanced Two Tower Epistemic
- **Performance**: nDCG@10=0.4009, MRR=0.6777
- **Runtime**: 1.30 hours

### Xgboost Ensemble
- **Performance**: nDCG@10=0.4003, MRR=0.6844
- **Runtime**: 0.00 hours

### Xgboost Deeper
- **Performance**: nDCG@10=0.3997, MRR=0.6630
- **Runtime**: 0.00 hours

### Xgboost Expertise
- **Performance**: nDCG@10=0.3994, MRR=0.6842
- **Runtime**: 0.25 hours

### Xgboost Interactions
- **Performance**: nDCG@10=0.3977, MRR=0.6578
- **Runtime**: 0.00 hours

### Xgboost Twostage
- **Performance**: nDCG@10=0.3957, MRR=0.6827
- **Runtime**: 0.00 hours

### Simplified Epistemic 300
- **Performance**: nDCG@10=0.3952, MRR=0.6721
- **Runtime**: 1.37 hours

### Xgboost Epistemic
- **Performance**: nDCG@10=0.3927, MRR=0.6691
- **Runtime**: 0.01 hours

### Xgboost Full Profiles
- **Performance**: nDCG@10=0.3910, MRR=0.6953
- **Runtime**: 0.00 hours

### Xgboost Minimal Reliability
- **Performance**: nDCG@10=0.3886, MRR=0.6157
- **Runtime**: 0.00 hours

### Xgboost Complete Profiles
- **Performance**: nDCG@10=0.3874, MRR=0.6885
- **Runtime**: 0.00 hours

### Random Forest

- **Architecture**: Random Forest Regressor (100 trees, max_depth=15)
- **Features**: TF-IDF (1000 features) + LLM ID encoding  
- **Training**: Scikit-learn with standard hyperparameters
- **Performance**: nDCG@10=0.3860, MRR=0.6701
- **Runtime**: 1.37 hours

### Xgboost
- **Performance**: nDCG@10=0.3824, MRR=0.6206
- **Runtime**: 0.03 hours

## Usage

To add a new model to the leaderboard:

1. Implement your model in `models/your_model_name/`
2. Save results to `data/results/your_model_name_results.json` using the standardized format
3. Run `python generate_leaderboard.py` to update this leaderboard

## Results Files

- `data/results/tier2_cpu_optimized_results.json` - Tier2 Cpu Optimized detailed results
- `data/results/tier3_cross_encoder_results.json` - Tier3 Cross Encoder detailed results
- `data/results/enhanced_neural_two_tower_results.json` - Enhanced Neural Two Tower detailed results
- `data/results/tier2_with_profiles_simplified_results.json` - Tier2 With Profiles Simplified detailed results
- `data/results/xgboost_hybrid_results.json` - Xgboost Hybrid detailed results
- `data/results/neural_two_tower_results.json` - Neural Two Tower detailed results
- `data/results/xgboost_smart_imputation_results.json` - Xgboost Smart Imputation detailed results
- `data/results/enhanced_two_tower_epistemic_results.json` - Enhanced Two Tower Epistemic detailed results
- `data/results/xgboost_ensemble_results.json` - Xgboost Ensemble detailed results
- `data/results/xgboost_deeper_results.json` - Xgboost Deeper detailed results
- `data/results/xgboost_expertise_results.json` - Xgboost Expertise detailed results
- `data/results/xgboost_interactions_results.json` - Xgboost Interactions detailed results
- `data/results/xgboost_twostage_results.json` - Xgboost Twostage detailed results
- `data/results/simplified_epistemic_300_results.json` - Simplified Epistemic 300 detailed results
- `data/results/xgboost_epistemic_results.json` - Xgboost Epistemic detailed results
- `data/results/xgboost_full_profiles_results.json` - Xgboost Full Profiles detailed results
- `data/results/xgboost_minimal_reliability_results.json` - Xgboost Minimal Reliability detailed results
- `data/results/xgboost_complete_profiles_results.json` - Xgboost Complete Profiles detailed results
- `data/results/random_forest_results.json` - Random Forest detailed results
- `data/results/xgboost_results.json` - Xgboost detailed results
