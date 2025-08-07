# Random Forest Baseline

Random Forest regression baseline for LLM ranking using TF-IDF features and LLM identifier encoding.

## Architecture

- **Features**: TF-IDF vectorization (max_features=1000, ngram_range=(1,2)) + LLM ID encoding
- **Model**: Random Forest Regressor (100 trees, max_depth=15)
- **Target**: Corrected qrel encoding (0→0.0, 2→0.7, 1→1.0)

## Files

- `evaluate_10fold_cv.py` - 10-fold cross-validation experimental evaluation script
- `README.md` - This file

## Usage

```bash
# Generate training data
cd data
python create_supervised_training_set.py

# Run 10-fold cross-validation experimental evaluation
cd ../models/random_forest
python evaluate_10fold_cv.py
```

## Performance

**10-Fold Cross-Validation Results**:
- **nDCG@10**: 0.386 ± 0.044
- **nDCG@5**: 0.387 ± 0.050  
- **MRR**: 0.670 ± 0.081
- **Runtime**: 1.37 hours

Results saved to `../../data/results/random_forest_results.json`