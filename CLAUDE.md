# CLAUDE.md

This file provides guidance to Claude Code when working with this TREC 2025 Million LLMs Track multi-model baseline framework.

## Project Overview

This repository contains a comprehensive multi-model machine learning framework for ranking Large Language Models (LLMs) by predicted expertise on user queries. The task is to rank 1,131 LLMs based on their expected ability to answer queries without actually querying them, using development data with ground truth relevance judgments.

**Framework Features**:
- Multiple baseline implementations for comparison
- Standardized evaluation protocols across all models  
- Automatic leaderboard generation
- Shared utilities to prevent code duplication
- Easy extension for new model approaches

## Current Repository Structure

```
├── README.md                          # Main framework documentation
├── CLAUDE.md                          # This development guide
├── leaderboard.md                     # Model comparison leaderboard
├── generate_leaderboard.py            # Leaderboard generation script
├── models/                            # Model implementations
│   ├── random_forest/                 # Random Forest baseline
│   │   ├── evaluate_10fold_cv.py      # 10-fold CV experimental evaluation
│   │   └── README.md                  # Model documentation
│   └── neural_two_tower/              # Neural Two-Tower baseline
│       ├── evaluate_10fold_cv.py      # 10-fold CV experimental evaluation
│       ├── model.py                   # Two-tower architecture
│       ├── data_loader.py             # Neural data loading
│       ├── evaluate_neural.py         # Evaluation utilities
│       ├── requirements_neural.txt    # Additional dependencies
│       ├── performance.md             # Detailed performance analysis
│       └── README.md                  # Model documentation
├── shared/                            # Shared utilities
│   └── utils/                         # Common evaluation functions
│       ├── evaluation.py              # Standardized metrics
│       └── __init__.py                # Package initialization
└── data/                              # All data files
    ├── llm_dev_data.tsv               # 342 development queries
    ├── llm_dev_qrels.txt              # 386,802 relevance judgments
    ├── supervised_training_full.csv   # Processed training dataset
    └── results/                       # Standardized model results
        ├── random_forest_results.json        # Random Forest CV results
        └── neural_two_tower_results.json     # Neural baseline CV results
```

## Data Files

### Development Data (Used in All Baselines)
- **data/llm_dev_data.tsv**: 342 queries with format `<query_id>\t<query_text>`
- **data/llm_dev_qrels.txt**: 386,802 relevance judgments with format `<query_id> 0 <llm_id> <qrel_score>`
  - qrel_score: 0 (not relevant), 1 (most relevant), 2 (second most relevant)
  - Distribution: 92.4% score=0, 3.7% score=1, 3.8% score=2

### Discovery Data (Available for Advanced Models)
- **Discovery dataset**: 14,950 queries with LLM responses (gitignored due to large file sizes)

### Generated Files
- **data/supervised_training_full.csv**: Combined training data (query_text, llm_id, qrel)
- **data/results/*.json**: Standardized results for each model
- **leaderboard.md**: Automatically generated model comparison

## Current Model Performance

**Multi-Model Leaderboard (10-Fold Cross-Validation)**:

| Rank | Model | nDCG@10 | nDCG@5 | MRR | Runtime |
|------|--------|---------|--------|-----|---------|
| 1 | **Neural Two Tower** | 0.4022 ± 0.028 | 0.4135 ± 0.034 | 0.6761 ± 0.057 | 6.95h |
| 2 | **Random Forest** | 0.3860 ± 0.044 | 0.3871 ± 0.050 | 0.6701 ± 0.081 | 1.37h |

### Performance Analysis
- **Best Performance**: Neural Two-Tower with 4.2% nDCG@10 improvement
- **Best Efficiency**: Random Forest 5x faster training time
- **Consistency**: Neural approach shows lower variance across folds
- **Strong MRR**: Both models achieve ~0.67 MRR (average reciprocal rank ~1.5)

## Common Development Tasks

### 1. Generate Training Data
```bash
cd data
python create_supervised_training_set.py
```
- Reads `llm_dev_data.tsv` and `llm_dev_qrels.txt` from current directory
- Creates `supervised_training_full.csv` with 386,802 examples
- Uses corrected qrel encoding (0→0.0, 2→0.7, 1→1.0)

### 2. Run Baseline Models

**Random Forest Baseline**:
```bash
# First generate shared training data
cd data
python create_supervised_training_set.py

# Then run Random Forest 10-fold CV evaluation
cd ../models/random_forest
python evaluate_10fold_cv.py
```
- 10-fold cross-validation with query-based splitting
- TF-IDF features (1000 max features) + LLM ID encoding
- Random Forest (100 trees, max_depth=15)
- Outputs `../../data/results/random_forest_results.json`

**Neural Two-Tower Baseline**:
```bash
# Generate shared training data first (if not already done)
cd data
python create_supervised_training_set.py

# Run neural model 10-fold CV evaluation
cd ../models/neural_two_tower
pip install -r requirements_neural.txt
python evaluate_10fold_cv.py
```
- Two-tower architecture with sentence transformers
- Query tower: all-MiniLM-L6-v2 → Dense [384→256→128→64]
- LLM tower: Learned embeddings → Dense [64→128→64]
- Margin-based pairwise ranking loss
- Outputs `../../data/results/neural_two_tower_results.json`

### 3. Generate Updated Leaderboard
```bash
python generate_leaderboard.py
```
- Automatically scans `data/results/` for all model results
- Generates `leaderboard.md` with performance comparison
- Supports CSV output with `--csv` flag
- Updates rankings based on nDCG@10 performance

### 4. Add New Model Implementation

**Step 1: Create Model Directory**
```bash
mkdir -p models/your_model_name
cd models/your_model_name
```

**Step 2: Implement Evaluation Script**
- Create `evaluate_10fold_cv.py` for 10-fold cross-validation experimental evaluation
- Use `shared.utils.evaluation` for consistent metrics
- Follow standardized results format (see below)
- Save results to `../../data/results/your_model_name_results.json`

**Step 3: Update Leaderboard**
```bash
python ../../generate_leaderboard.py
```

## Standardized Results Format

All models must save results in this JSON format:

```json
{
  "evaluation_info": {
    "timestamp": "2025-08-07T12:00:00",
    "pipeline": "Model Description",
    "dataset": "TREC 2025 Million LLMs Track - Complete Dataset",
    "total_examples": 386801,
    "unique_queries": 342,
    "unique_llms": 1131,
    "total_runtime_seconds": 0.0,
    "total_runtime_hours": 0.0
  },
  "performance_metrics": {
    "ndcg_10": {
      "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0,
      "confidence_interval_95": [0.0, 0.0]
    },
    "ndcg_5": {
      "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0,
      "confidence_interval_95": [0.0, 0.0]
    },
    "mrr": {
      "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0,
      "confidence_interval_95": [0.0, 0.0]
    }
  },
  "fold_by_fold_results": [
    {"fold": 1, "ndcg_10": 0.0, "ndcg_5": 0.0, "mrr": 0.0, "train_time": 0.0, "n_queries": 0}
  ]
}
```

## Shared Utilities

### Evaluation Functions (`shared/utils/evaluation.py`)
```python
from shared.utils import calculate_metrics, save_standardized_results

# Calculate standard metrics
metrics = calculate_metrics(y_true_by_query, y_pred_by_query)

# Save in standardized format
save_standardized_results(results, 'model_name', '../../data/results/')
```

**Available Functions**:
- `evaluate_ndcg_at_k()` - nDCG calculation with error handling
- `mean_reciprocal_rank()` - MRR across all queries
- `calculate_metrics()` - All metrics (nDCG@10, nDCG@5, MRR)
- `aggregate_cv_results()` - Statistics from fold results
- `save_standardized_results()` - Consistent JSON output
- `get_qrel_mapping()` - Standard encoding (0→0.0, 1→1.0, 2→0.7)

## Technical Implementation Standards

### Cross-Validation Protocol
- **Splitting**: Query-based 10-fold to prevent data leakage
- **Training**: ~90% of queries (~308 queries, ~348K examples)
- **Validation**: ~10% of queries (~34 queries, ~38K examples)
- **Reproducibility**: Fixed random seed (42) across all models

### Qrel Encoding (CRITICAL)
- **Correct Mapping**: 0→0.0 (not relevant), 1→1.0 (most relevant), 2→0.7 (second most relevant)
- **Rationale**: Reflects proper ranking order where 1 > 2 > 0
- **Historical Note**: Previous incorrect mapping (1→0.5, 2→1.0) was corrected

### Evaluation Metrics
- **Primary**: nDCG@10 for model ranking
- **Secondary**: nDCG@5, MRR for comprehensive evaluation
- **Aggregation**: Mean ± std across 10 folds with 95% confidence intervals
- **Query Grouping**: Essential for proper ranking evaluation

## Development Commands

```bash
# Check data integrity
wc -l data/llm_dev_data.tsv        # Should be 342
wc -l data/llm_dev_qrels.txt       # Should be 386,802

# View current results
python -m json.tool data/results/random_forest_results.json
python -m json.tool data/results/neural_two_tower_results.json

# Generate leaderboard variations
python generate_leaderboard.py                    # Markdown leaderboard
python generate_leaderboard.py --csv             # Also generate CSV
python generate_leaderboard.py --output custom.md # Custom output file

# Quick data exploration
head -5 data/supervised_training_full.csv
cut -d' ' -f4 data/llm_dev_qrels.txt | sort | uniq -c

# Test shared utilities
cd shared/utils && python evaluation.py
```

## Extension Points for Advanced Models

### 1. Gradient Boosting Models
```python
# models/xgboost/train.py
from xgboost import XGBRegressor
from shared.utils import calculate_metrics, save_standardized_results

model = XGBRegressor(objective='reg:squarederror')
# Use same data pipeline as Random Forest
```

### 2. Advanced Neural Architectures
```python
# models/transformer_ranker/model.py
from transformers import AutoModel
import torch.nn as nn

class TransformerRanker(nn.Module):
    def __init__(self):
        super().__init__()
        self.query_encoder = AutoModel.from_pretrained('bert-base-uncased')
        # Add cross-attention layers for query-LLM interaction
```

### 3. Discovery Data Integration
```python
# Example: Load discovery data for advanced models
import json

# Discovery data contains 14,950 additional queries with LLM responses
# Can be used for pre-training, auxiliary tasks, or data augmentation
discovery_queries = json.load(open('data/llm_discovery_data_1.json'))
```

### 4. Meta-Learning Approaches
```python
# Incorporate LLM metadata, architectural features
llm_metadata = {
    'llm_0000': {'architecture': 'transformer', 'parameters': '7B'},
    'llm_0001': {'architecture': 'rnn', 'parameters': '1B'}
}
```

### 5. Ensemble Methods
```python
# models/ensemble/train.py
from shared.utils import load_results

rf_results = load_results('random_forest')
neural_results = load_results('neural_two_tower')
# Combine predictions with learned weights
```

## Best Practices for Contributors

### Code Organization
- **Model Directory**: Each model in separate directory under `models/`
- **Shared Code**: Use `shared/utils/` for common functions
- **Documentation**: Include model-specific README.md with architecture details
- **Dependencies**: Separate requirements files for model-specific dependencies

### Evaluation Standards
- **Consistent Metrics**: Use shared evaluation functions
- **Reproducible Results**: Fixed random seeds, documented hyperparameters
- **Fair Comparison**: Same data splits, preprocessing, evaluation protocol
- **Statistical Rigor**: Report confidence intervals, fold-by-fold results

### Performance Reporting
- **Runtime Tracking**: Include training time in results
- **Resource Usage**: Document memory/compute requirements
- **Hyperparameter Sensitivity**: Report key hyperparameter choices
- **Error Analysis**: Include analysis of failure cases when relevant

This multi-model framework serves as a foundation for systematic comparison of different approaches to automated LLM ranking and selection for query answering tasks.
- Make evaluation scripts be completely consistent in their output during 10-fold CV runs, including per-fold validation performance metrics.
- Use the bilateral-truth package for LLM judges