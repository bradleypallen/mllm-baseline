# TREC 2025 Million LLMs Track - Model Leaderboard

*Generated on 2025-08-07 15:40:26*

## Performance Comparison

Ranking models by nDCG@10 performance on 10-fold cross-validation.

| Rank | Model | nDCG@10 | nDCG@5 | MRR | Runtime | 
|------|--------|---------|--------|-----|---------|
| 1 | **Neural Two Tower** | 0.4022 ± 0.028 | 0.4135 ± 0.034 | 0.6761 ± 0.057 | 6.95h |
| 2 | **Random Forest** | 0.3860 ± 0.044 | 0.3871 ± 0.050 | 0.6701 ± 0.081 | 1.37h |


## Evaluation Protocol

- **Dataset**: TREC 2025 Million LLMs Track (342 queries, 1,131 LLMs)  
- **Cross-Validation**: 10-fold, query-based splitting to prevent data leakage
- **Metrics**: nDCG@10, nDCG@5, Mean Reciprocal Rank (MRR)
- **Qrel Encoding**: 0→0.0 (not relevant), 1→1.0 (most relevant), 2→0.7 (second-most relevant)

## Model Details

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

## Usage

To add a new model to the leaderboard:

1. Implement your model in `models/your_model_name/`
2. Save results to `data/results/your_model_name_results.json` using the standardized format
3. Run `python generate_leaderboard.py` to update this leaderboard

## Results Files

- `data/results/neural_two_tower_results.json` - Neural Two Tower detailed results
- `data/results/random_forest_results.json` - Random Forest detailed results
