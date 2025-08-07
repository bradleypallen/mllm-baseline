# XGBoost Baseline Model

**Model Type**: Gradient Boosting Regressor  
**Framework**: XGBoost  
**Purpose**: High-performance baseline with gradient boosting for LLM ranking

## Model Architecture

### XGBoost Configuration
- **Algorithm**: XGBoost Regressor with gradient boosting
- **Estimators**: 100 boosting rounds
- **Max Depth**: 6 levels per tree
- **Learning Rate**: 0.1 for stable convergence
- **Subsample**: 0.8 for regularization
- **Colsample_bytree**: 0.8 to prevent overfitting

### Feature Engineering

**Text Features (TF-IDF)**:
- **Dimensionality**: 1,000 features maximum
- **N-gram Range**: Unigrams and bigrams (1,2)
- **Preprocessing**: English stop word removal, minimum document frequency = 2
- **Purpose**: Capture semantic query patterns

**LLM Identity Features**:
- **Encoding**: Label encoding of LLM identifiers
- **Purpose**: Learn relative performance patterns across different LLMs
- **Integration**: Single additional feature column

**Target Mapping**:
- **Qrel Encoding**: 0→0.0 (not relevant), 1→1.0 (most relevant), 2→0.7 (second-most relevant)
- **Objective**: Squared error minimization with bounded predictions [0,1]

## Evaluation Protocol

### Cross-Validation Strategy
- **Method**: 10-fold cross-validation
- **Splitting**: Query-based to prevent data leakage
- **Training Set**: ~90% of queries (~308 queries, ~348K examples per fold)
- **Test Set**: ~10% of queries (~34 queries, ~38K examples per fold)

### Performance Metrics
- **Primary**: nDCG@10 (normalized Discounted Cumulative Gain at rank 10)
- **Secondary**: nDCG@5, Mean Reciprocal Rank (MRR)
- **Regression**: Mean Squared Error (MSE) for comparison with other baselines

## Model Strengths

### Performance Advantages
- **Gradient Boosting**: Iterative learning can capture complex patterns
- **Regularization**: Built-in techniques prevent overfitting
- **Feature Importance**: Provides interpretable feature rankings
- **Efficiency**: Fast training and prediction with optimized C++ implementation

### Use Cases
- **Baseline Comparison**: Strong traditional ML baseline for neural models
- **Feature Analysis**: Understanding most predictive query and LLM patterns
- **Rapid Prototyping**: Fast experimentation with different hyperparameters
- **Ensemble Components**: Can be combined with other models

## Implementation Details

### Dependencies
```bash
pip install xgboost  # Core XGBoost library
# Uses same core dependencies as other baselines: pandas, numpy, scikit-learn
```

### Running the Model
```bash
cd models/xgboost
python evaluate_10fold_cv.py
```

### Expected Performance
- **Training Time**: Expected ~3-5 minutes per fold
- **Total Runtime**: ~0.5-1 hour for full 10-fold CV
- **Memory Usage**: Similar to Random Forest baseline
- **Expected nDCG@10**: Competitive with or better than Random Forest

## Hyperparameter Choices

### Core Parameters
- **n_estimators=100**: Balanced between performance and training time
- **max_depth=6**: Moderate depth to prevent overfitting
- **learning_rate=0.1**: Conservative rate for stable convergence
- **subsample=0.8**: Sample 80% of data per tree for regularization
- **colsample_bytree=0.8**: Use 80% of features per tree

### Regularization
- **Built-in L2 regularization**: Default XGBoost regularization
- **Early stopping**: Could be added for optimal tree count
- **Feature subsampling**: Reduces overfitting on high-dimensional TF-IDF features

## Comparison with Other Baselines

### vs Random Forest
- **Advantages**: Sequential learning, better handling of feature interactions
- **Trade-offs**: Slightly more complex hyperparameter tuning
- **Expected**: Similar or better performance with comparable training time

### vs Neural Two-Tower
- **Advantages**: Much faster training, interpretable feature importance
- **Trade-offs**: No learned query embeddings, traditional feature engineering
- **Expected**: Competitive performance with significantly faster runtime

## Extension Opportunities

### Hyperparameter Optimization
```python
# Example: Grid search for optimal parameters
param_grid = {
    'max_depth': [4, 6, 8],
    'learning_rate': [0.05, 0.1, 0.2],
    'n_estimators': [100, 200, 300]
}
```

### Advanced Features
```python
# Example: Additional feature engineering
# - Query length, complexity metrics
# - LLM metadata integration
# - Interaction features between query and LLM characteristics
```

### Ensemble Integration
```python
# Example: Combine with other baselines
# - Weighted averaging with Random Forest and Neural predictions
# - Stacking with XGBoost as meta-learner
```

## Results Format

Results are saved to `../../data/results/xgboost_results.json` using the standardized format:

```json
{
  "evaluation_info": {
    "pipeline": "XGBoost Regressor (100 trees, max_depth=6)",
    "total_runtime_hours": "~0.5-1h"
  },
  "performance_metrics": {
    "ndcg_10": {"mean": 0.0, "std": 0.0, "confidence_interval_95": [0.0, 0.0]},
    "mrr": {"mean": 0.0, "std": 0.0, "confidence_interval_95": [0.0, 0.0]}
  },
  "fold_by_fold_results": [...]
}
```

This XGBoost implementation provides a strong gradient boosting baseline that leverages the same feature engineering pipeline as Random Forest while potentially achieving better performance through sequential learning and advanced regularization techniques.