# Shared Utilities

Common utilities and evaluation functions used across all model implementations in the TREC 2025 Million LLMs Track baseline framework.

## Files

- **`utils/evaluation.py`**: Standardized evaluation functions (nDCG, MRR, aggregation)
- **`utils/__init__.py`**: Package initialization with imports

## Usage

```python
from shared.utils import calculate_metrics, save_standardized_results

# Calculate standard metrics
metrics = calculate_metrics(y_true_by_query, y_pred_by_query)

# Save results in standardized format  
save_standardized_results(results, 'model_name', 'data/results/')
```

## Standardized Functions

### Evaluation Metrics
- `evaluate_ndcg_at_k()` - Calculate nDCG@k with proper error handling
- `mean_reciprocal_rank()` - Calculate MRR across all queries  
- `calculate_metrics()` - Get all metrics (nDCG@10, nDCG@5, MRR)

### Results Management
- `aggregate_cv_results()` - Aggregate fold results with statistics
- `save_standardized_results()` - Save in consistent JSON format
- `load_results()` - Load results for comparison

### Data Utilities
- `get_qrel_mapping()` - Standard qrel encoding (0→0.0, 1→1.0, 2→0.7)

This ensures all models use identical evaluation protocols for fair comparison.