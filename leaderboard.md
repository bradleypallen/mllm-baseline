# TREC 2025 Million LLMs Track - Model Leaderboard

*Generated on 2025-08-22*

## 🏆 Key Achievement

**Config O** achieves **0.4482 nDCG@10** through weak labeling from discovery data, representing **16.1% improvement** over Random Forest baseline.

### Performance Breakdown
- **Weak labeling contribution**: +11.1% (Config L: 0.4289 vs baseline 0.3860)
- **Architecture optimization**: +4.3% (Config O: 0.4482 vs Config L 0.4289)
- **Total improvement**: +16.1% over Random Forest baseline

### Key Insights
- **What Works**: Simple heuristic weak labeling from discovery data provides massive gains
- **What Doesn't**: Complex approaches (pseudo-labeling, epistemic profiling) add noise
- **Sweet Spot**: 256D embeddings with 4-layer depth optimal for LLM tower

## Performance Comparison

Ranking models by nDCG@10 performance on 10-fold cross-validation.

| Rank | Model | nDCG@10 | nDCG@5 | MRR | Runtime | 
|------|--------|---------|--------|-----|---------|
| 1 | **🏆 Config O (Champion)** | 0.4482 ± 0.038 | 0.4622 ± 0.042 | 0.7242 ± 0.073 | 4.74h |
| 2 | **Config P (512D)** | 0.4418 ± 0.038 | 0.4535 ± 0.044 | 0.7186 ± 0.062 | 4.5h |
| 3 | **Tier2 Config J** | 0.4417 ± 0.039 | 0.4598 ± 0.060 | 0.7281 ± 0.063 | 4.63h |
| 4 | **Tier2 Config M** | 0.4410 ± 0.042 | 0.4553 ± 0.041 | 0.7209 ± 0.064 | 5.42h |
| 5 | **Config Q (Enhanced Query)** | 0.4375 ± 0.044 | 0.4413 ± 0.053 | 0.7178 ± 0.069 | 5h |
| 6 | **Tier2 Config I** | 0.4327 ± 0.043 | 0.4383 ± 0.042 | 0.6933 ± 0.062 | 2.95h |
| 7 | **Tier2 Cpu Optimized** | 0.4306 ± 0.055 | 0.4347 ± 0.058 | 0.7263 ± 0.070 | 3.11h |
| 8 | **Tier2 Cpu Optimized Config G** | 0.4303 ± 0.039 | 0.4335 ± 0.042 | 0.7030 ± 0.053 | 3.06h |
| 9 | **Tier2 Config H Cv Epistemic** | 0.4297 ± 0.050 | 0.4346 ± 0.052 | 0.7098 ± 0.078 | 1.32h |
| 10 | **Tier2 Cpu Optimized Config F** | 0.4292 ± 0.051 | 0.4393 ± 0.054 | 0.6884 ± 0.079 | 3.22h |
| 11 | **Config L (Weak Labels)** | 0.4289 ± 0.040 | 0.4312 ± 0.048 | 0.7162 ± 0.072 | 5.25h |
| 12 | **Tier2 Cpu Optimized Config A** | 0.4278 ± 0.052 | 0.4399 ± 0.054 | 0.7162 ± 0.066 | 4.54h |
| 13 | **Tier2 Cpu Optimized Config B** | 0.4261 ± 0.052 | 0.4356 ± 0.058 | 0.6783 ± 0.071 | 4.73h |
| 14 | **Tier3 Cross Encoder** | 0.4259 ± 0.049 | 0.4378 ± 0.051 | 0.7141 ± 0.076 | 21.92h |
| 15 | **Enhanced Neural Two Tower** | 0.4256 ± 0.050 | 0.4287 ± 0.056 | 0.7113 ± 0.074 | 2.95h |
| 16 | **Config R (Failed Pseudo)** | 0.4241 ± 0.056 | 0.4142 ± 0.073 | 0.7133 ± 0.070 | 3h |
| 17 | **Tier2 Cpu Optimized Config D** | 0.4219 ± 0.046 | 0.4346 ± 0.053 | 0.7153 ± 0.073 | 2.38h |
| 18 | **Tier2 Cpu Optimized Tuned** | 0.4194 ± 0.048 | 0.4328 ± 0.047 | 0.7004 ± 0.061 | 2.71h |
| 19 | **Tier2 With Profiles Simplified** | 0.4128 ± 0.031 | 0.4232 ± 0.037 | 0.7083 ± 0.053 | 6.23h |
| 20 | **Xgboost Hybrid** | 0.4107 ± 0.044 | 0.4216 ± 0.048 | 0.6979 ± 0.060 | 0.00h |
| 21 | **Tier2 Config K** | 0.4067 ± 0.048 | 0.4125 ± 0.052 | 0.6782 ± 0.065 | 6.38h |
| 22 | **Neural Two Tower (Original)** | 0.4022 ± 0.028 | 0.4135 ± 0.034 | 0.6761 ± 0.057 | 6.95h |
| 23 | **Xgboost Smart Imputation** | 0.4017 ± 0.037 | 0.4098 ± 0.039 | 0.6690 ± 0.049 | 0.00h |
| 24 | **Enhanced Two Tower Epistemic** | 0.4009 ± 0.061 | 0.4093 ± 0.074 | 0.6777 ± 0.090 | 1.30h |
| 25 | **Xgboost Ensemble** | 0.4003 ± 0.040 | 0.4109 ± 0.039 | 0.6844 ± 0.056 | 0.00h |
| 26 | **Xgboost Deeper** | 0.3997 ± 0.031 | 0.4017 ± 0.022 | 0.6630 ± 0.056 | 0.00h |
| 27 | **Xgboost Weighted Ensemble** | 0.3994 ± 0.029 | 0.4145 ± 0.035 | 0.6905 ± 0.044 | 0.33h |
| 28 | **Xgboost Expertise** | 0.3994 ± 0.043 | 0.4077 ± 0.048 | 0.6842 ± 0.082 | 0.25h |
| 29 | **Xgboost Interactions** | 0.3977 ± 0.049 | 0.4058 ± 0.056 | 0.6578 ± 0.065 | 0.00h |
| 30 | **Xgboost Fast Advanced** | 0.3962 ± 0.037 | 0.4042 ± 0.040 | 0.6785 ± 0.075 | 0.03h |
| 31 | **Xgboost Twostage** | 0.3957 ± 0.048 | 0.4056 ± 0.045 | 0.6827 ± 0.062 | 0.00h |
| 32 | **Simplified Epistemic 300** | 0.3952 ± 0.061 | 0.4040 ± 0.064 | 0.6721 ± 0.092 | 1.37h |
| 33 | **Xgboost Discovery Optimized** | 0.3938 ± 0.028 | 0.4127 ± 0.036 | 0.7042 ± 0.054 | 0.01h |
| 34 | **Xgboost Epistemic** | 0.3927 ± 0.049 | 0.4062 ± 0.048 | 0.6691 ± 0.045 | 0.01h |
| 35 | **Xgboost Full Profiles** | 0.3910 ± 0.037 | 0.4009 ± 0.042 | 0.6953 ± 0.068 | 0.00h |
| 36 | **Xgboost Discovery** | 0.3891 ± 0.029 | 0.3965 ± 0.037 | 0.6809 ± 0.046 | 0.04h |
| 37 | **Xgboost Minimal Reliability** | 0.3886 ± 0.048 | 0.3919 ± 0.054 | 0.6157 ± 0.070 | 0.00h |
| 38 | **Xgboost Complete Profiles** | 0.3874 ± 0.034 | 0.4001 ± 0.045 | 0.6885 ± 0.062 | 0.00h |
| 39 | **Random Forest (Baseline)** | 0.3860 ± 0.044 | 0.3871 ± 0.050 | 0.6701 ± 0.081 | 1.37h |
| 40 | **Xgboost Massive Synthetic** | 0.3828 ± 0.048 | 0.3815 ± 0.048 | 0.6305 ± 0.058 | 0.54h |
| 41 | **Xgboost** | 0.3824 ± 0.045 | 0.3808 ± 0.047 | 0.6206 ± 0.052 | 0.03h |
| 42 | **Xgboost Fair Comparison** | 0.3652 ± 0.040 | 0.3618 ± 0.041 | 0.6087 ± 0.065 | 0.06h |


## Evaluation Protocol

- **Dataset**: TREC 2025 Million LLMs Track (342 queries, 1,131 LLMs)  
- **Cross-Validation**: 10-fold, query-based splitting to prevent data leakage
- **Metrics**: nDCG@10, nDCG@5, Mean Reciprocal Rank (MRR)
- **Qrel Encoding**: 0→0.0 (not relevant), 1→1.0 (most relevant), 2→0.7 (second-most relevant)

⚠️ **Note**: Only models with proper 10-fold cross-validation are included in rankings. Single-fold results (e.g., Config H with 0.4491) are not comparable and excluded from the leaderboard.

## Model Details

### Config O (Champion) 🏆
- **Performance**: nDCG@10=0.4482, MRR=0.7242
- **Architecture**: 4-layer LLM tower (256→512→384→256→128) with 256D embeddings
- **Key Features**: 501K examples (387K original + 114K weak labels), collaborative pre-training, 4 attention heads
- **Innovation**: Combines weak labeling breakthrough with optimized architecture depth
- **Runtime**: 4.74 hours

### Config P (512D Embeddings)
- **Performance**: nDCG@10=0.4418, MRR=0.7186  
- **Architecture**: 4-layer LLM tower with 512D embeddings (512→768→512→256→128)
- **Key Finding**: Wider embeddings led to overfitting compared to Config O's 256D
- **Runtime**: 4.5 hours

### Config Q (Enhanced Query Tower)
- **Performance**: nDCG@10=0.4375, MRR=0.7178
- **Architecture**: 4-layer query tower + 6-layer LLM tower
- **Key Finding**: Diminishing returns from deeper architectures
- **Runtime**: 5 hours

### Config L (Weak Labels Breakthrough)
- **Performance**: nDCG@10=0.4289, MRR=0.7162
- **Architecture**: 3-layer LLM tower (256→512→256→128) with weak labeled data
- **Key Innovation**: First to use weak labeling from discovery data (490K weak labels)
- **Runtime**: 5.25 hours

### Config R (Failed Pseudo-Labeling)
- **Performance**: nDCG@10=0.4241, MRR=0.7133
- **Architecture**: Config O architecture with additional 4K pseudo-labels
- **Key Finding**: Pseudo-labeling degraded performance due to circular validation
- **Runtime**: 3 hours

### Tier2 Config J
- **Performance**: nDCG@10=0.4417, MRR=0.7281
- **Runtime**: 4.63 hours

### Tier2 Config M
- **Performance**: nDCG@10=0.4410, MRR=0.7209
- **Runtime**: 5.42 hours

### Tier2 Config I
- **Performance**: nDCG@10=0.4327, MRR=0.6933
- **Runtime**: 2.95 hours

### Tier2 Cpu Optimized
- **Performance**: nDCG@10=0.4306, MRR=0.7263
- **Runtime**: 3.11 hours

### Tier2 Cpu Optimized Config G
- **Performance**: nDCG@10=0.4303, MRR=0.7030
- **Runtime**: 3.06 hours

### Tier2 Config H Cv Epistemic
- **Performance**: nDCG@10=0.4297, MRR=0.7098
- **Runtime**: 1.32 hours

### Tier2 Cpu Optimized Config F
- **Performance**: nDCG@10=0.4292, MRR=0.6884
- **Runtime**: 3.22 hours

### Tier2 Cpu Optimized Config A
- **Performance**: nDCG@10=0.4278, MRR=0.7162
- **Runtime**: 4.54 hours

### Tier2 Cpu Optimized Config B
- **Performance**: nDCG@10=0.4261, MRR=0.6783
- **Runtime**: 4.73 hours

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

### Tier2 Cpu Optimized Config D
- **Performance**: nDCG@10=0.4219, MRR=0.7153
- **Runtime**: 2.38 hours

### Tier2 Cpu Optimized Tuned
- **Performance**: nDCG@10=0.4194, MRR=0.7004
- **Runtime**: 2.71 hours

### Tier2 With Profiles Simplified
- **Performance**: nDCG@10=0.4128, MRR=0.7083
- **Runtime**: 6.23 hours

### Xgboost Hybrid
- **Performance**: nDCG@10=0.4107, MRR=0.6979
- **Runtime**: 0.00 hours

### Tier2 Config K
- **Performance**: nDCG@10=0.4067, MRR=0.6782
- **Runtime**: 6.38 hours

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

### Xgboost Weighted Ensemble
- **Performance**: nDCG@10=0.3994, MRR=0.6905
- **Runtime**: 0.33 hours

### Xgboost Expertise
- **Performance**: nDCG@10=0.3994, MRR=0.6842
- **Runtime**: 0.25 hours

### Xgboost Interactions
- **Performance**: nDCG@10=0.3977, MRR=0.6578
- **Runtime**: 0.00 hours

### Xgboost Fast Advanced
- **Performance**: nDCG@10=0.3962, MRR=0.6785
- **Runtime**: 0.03 hours

### Xgboost Twostage
- **Performance**: nDCG@10=0.3957, MRR=0.6827
- **Runtime**: 0.00 hours

### Simplified Epistemic 300
- **Performance**: nDCG@10=0.3952, MRR=0.6721
- **Runtime**: 1.37 hours

### Xgboost Discovery Optimized
- **Performance**: nDCG@10=0.3938, MRR=0.7042
- **Runtime**: 0.01 hours

### Xgboost Epistemic
- **Performance**: nDCG@10=0.3927, MRR=0.6691
- **Runtime**: 0.01 hours

### Xgboost Full Profiles
- **Performance**: nDCG@10=0.3910, MRR=0.6953
- **Runtime**: 0.00 hours

### Xgboost Discovery
- **Performance**: nDCG@10=0.3891, MRR=0.6809
- **Runtime**: 0.04 hours

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

### Xgboost Massive Synthetic
- **Performance**: nDCG@10=0.3828, MRR=0.6305
- **Runtime**: 0.54 hours

### Xgboost
- **Performance**: nDCG@10=0.3824, MRR=0.6206
- **Runtime**: 0.03 hours

### Xgboost Fair Comparison
- **Performance**: nDCG@10=0.3652, MRR=0.6087
- **Runtime**: 0.06 hours

## Key Findings

### What Works ✅
1. **Weak Labeling**: Simple heuristic-based weak labeling from discovery data provides +11.1% improvement
2. **Collaborative Pre-training**: Using discovery data for query encoder pre-training improves initialization
3. **Optimal Architecture**: 256D embeddings with 4-layer depth for LLM tower hits the sweet spot
4. **Multi-head Attention**: 4 attention heads provide best balance of performance and efficiency

### What Doesn't Work ❌
1. **Pseudo-Labeling**: Teacher-student approaches fail due to distribution mismatch (-5.4% vs Config O)
2. **Epistemic Profiling**: Complex profiling methods can't extract reliable signals from unsupervised data
3. **Synthetic Data**: Artificial patterns don't transfer to real relevance judgments
4. **Wider Embeddings**: 512D embeddings lead to overfitting without performance gains
5. **Excessive Depth**: Beyond 4 layers shows diminishing returns

## Future Directions

### Scaling Weak Labeling
- Current: 490K weak labels → +11.1% improvement
- Next Steps: Scale to 2M, 5M, 16M weak labels with GPU acceleration
- Expected: Additional 3-7% improvements based on scaling patterns

### AWS GPU Training Plan
- Documented in `models/neural_two_tower/aws_gpu_training_plan.md`
- Cost-effective approach using p3.2xlarge instances
- Estimated 10-15x speedup for large-scale weak labeling

## Usage

To add a new model to the leaderboard:

1. Implement your model in `models/your_model_name/`
2. Save results to `data/results/your_model_name_results.json` using the standardized format
3. Run `python generate_leaderboard.py` to update this leaderboard

## Results Files

- `data/results/tier2_config_h_epistemic_results.json` - Tier2 Config H Epistemic detailed results
- `data/results/tier2_config_o_results.json` - Config O (Champion) detailed results
- `data/results/tier2_config_p_results.json` - Config P (512D) detailed results
- `data/results/tier2_config_q_results.json` - Config Q (Enhanced Query) detailed results
- `data/results/tier2_config_l_results.json` - Config L (Weak Labels) detailed results
- `data/results/tier2_config_r_results.json` - Config R (Failed Pseudo) detailed results
- `data/results/tier2_config_j_results.json` - Tier2 Config J detailed results
- `data/results/tier2_config_m_results.json` - Tier2 Config M detailed results
- `data/results/tier2_config_i_results.json` - Tier2 Config I detailed results
- `data/results/tier2_cpu_optimized_results.json` - Tier2 Cpu Optimized detailed results
- `data/results/tier2_cpu_optimized_config_g_results.json` - Tier2 Cpu Optimized Config G detailed results
- `data/results/tier2_config_h_cv_epistemic_results.json` - Tier2 Config H Cv Epistemic detailed results
- `data/results/tier2_cpu_optimized_config_f_results.json` - Tier2 Cpu Optimized Config F detailed results
- `data/results/tier2_cpu_optimized_config_a_results.json` - Tier2 Cpu Optimized Config A detailed results
- `data/results/tier2_cpu_optimized_config_b_results.json` - Tier2 Cpu Optimized Config B detailed results
- `data/results/tier3_cross_encoder_results.json` - Tier3 Cross Encoder detailed results
- `data/results/enhanced_neural_two_tower_results.json` - Enhanced Neural Two Tower detailed results
- `data/results/tier2_cpu_optimized_config_d_results.json` - Tier2 Cpu Optimized Config D detailed results
- `data/results/tier2_cpu_optimized_tuned_results.json` - Tier2 Cpu Optimized Tuned detailed results
- `data/results/tier2_with_profiles_simplified_results.json` - Tier2 With Profiles Simplified detailed results
- `data/results/xgboost_hybrid_results.json` - Xgboost Hybrid detailed results
- `data/results/tier2_config_k_results.json` - Tier2 Config K detailed results
- `data/results/neural_two_tower_results.json` - Neural Two Tower detailed results
- `data/results/xgboost_smart_imputation_results.json` - Xgboost Smart Imputation detailed results
- `data/results/enhanced_two_tower_epistemic_results.json` - Enhanced Two Tower Epistemic detailed results
- `data/results/xgboost_ensemble_results.json` - Xgboost Ensemble detailed results
- `data/results/xgboost_deeper_results.json` - Xgboost Deeper detailed results
- `data/results/xgboost_weighted_ensemble_results.json` - Xgboost Weighted Ensemble detailed results
- `data/results/xgboost_expertise_results.json` - Xgboost Expertise detailed results
- `data/results/xgboost_interactions_results.json` - Xgboost Interactions detailed results
- `data/results/xgboost_fast_advanced_results.json` - Xgboost Fast Advanced detailed results
- `data/results/xgboost_twostage_results.json` - Xgboost Twostage detailed results
- `data/results/simplified_epistemic_300_results.json` - Simplified Epistemic 300 detailed results
- `data/results/xgboost_discovery_optimized_results.json` - Xgboost Discovery Optimized detailed results
- `data/results/xgboost_epistemic_results.json` - Xgboost Epistemic detailed results
- `data/results/xgboost_full_profiles_results.json` - Xgboost Full Profiles detailed results
- `data/results/xgboost_discovery_results.json` - Xgboost Discovery detailed results
- `data/results/xgboost_minimal_reliability_results.json` - Xgboost Minimal Reliability detailed results
- `data/results/xgboost_complete_profiles_results.json` - Xgboost Complete Profiles detailed results
- `data/results/random_forest_results.json` - Random Forest detailed results
- `data/results/xgboost_massive_synthetic_results.json` - Xgboost Massive Synthetic detailed results
- `data/results/xgboost_results.json` - Xgboost detailed results
- `data/results/xgboost_fair_comparison_results.json` - Xgboost Fair Comparison detailed results
