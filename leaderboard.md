# TREC 2025 Million LLMs Track - Model Leaderboard

*Generated on 2025-08-09 13:31:47*

## Performance Comparison

Ranking models by nDCG@10 performance on 10-fold cross-validation.

| Rank | Model | nDCG@10 | nDCG@5 | MRR | Runtime | 
|------|--------|---------|--------|-----|---------|
| 1 | **Tier2 Cpu Optimized** | 0.4306 ± 0.055 | 0.4347 ± 0.058 | 0.7263 ± 0.070 | 3.11h |
| 2 | **Tier3 Cross Encoder** | 0.4259 ± 0.049 | 0.4378 ± 0.051 | 0.7141 ± 0.076 | 21.92h |
| 3 | **Enhanced Neural Two Tower** | 0.4256 ± 0.050 | 0.4287 ± 0.056 | 0.7113 ± 0.074 | 2.95h |
| 4 | **Neural Two Tower** | 0.4022 ± 0.028 | 0.4135 ± 0.034 | 0.6761 ± 0.057 | 6.95h |
| 5 | **Random Forest** | 0.3860 ± 0.044 | 0.3871 ± 0.050 | 0.6701 ± 0.081 | 1.37h |
| 6 | **Xgboost** | 0.3824 ± 0.045 | 0.3808 ± 0.047 | 0.6206 ± 0.052 | 0.03h |


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

- **Architecture**: Cross-encoder with joint query-LLM encoding
- **Transformer**: DistilBERT-base-uncased with query-LLM attention
- **Query Processing**: Tokenized text → DistilBERT → [CLS] token representation 
- **LLM Integration**: Learned LLM embeddings concatenated with query representation
- **Training**: 15 epochs/fold, BCE loss for direct relevance prediction, batch_size=48
- **Advanced Features**: Joint attention modeling, end-to-end relevance optimization
- **Performance**: nDCG@10=0.4259, MRR=0.7141  
- **Runtime**: 21.92 hours (7x slower than Tier 2 for comparable performance)
- **Efficiency Trade-off**: High computational cost with marginal performance gains

### Enhanced Neural Two Tower

- **Architecture**: Dual encoder with sentence transformers
- **Query Tower**: all-MiniLM-L6-v2 → Dense [384→256→128→64]
- **LLM Tower**: Learned embeddings → Dense [64→128→64] 
- **Training**: 20 epochs/fold, margin-based pairwise loss
- **Performance**: nDCG@10=0.4256, MRR=0.7113
- **Runtime**: 2.95 hours

### Neural Two Tower

- **Architecture**: Dual encoder with sentence transformers
- **Query Tower**: all-MiniLM-L6-v2 → Dense [384→256→128→64]
- **LLM Tower**: Learned embeddings → Dense [64→128→64] 
- **Training**: 20 epochs/fold, margin-based pairwise loss
- **Performance**: nDCG@10=0.4022, MRR=0.6761
- **Runtime**: 6.95 hours

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
- `data/results/neural_two_tower_results.json` - Neural Two Tower detailed results
- `data/results/random_forest_results.json` - Random Forest detailed results
- `data/results/xgboost_results.json` - Xgboost detailed results
