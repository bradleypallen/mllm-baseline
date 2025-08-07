# CLAUDE.md

This file provides guidance to Claude Code when working with this TREC 2025 Million LLMs Track baseline framework.

## Project Overview

This repository contains a baseline machine learning framework for ranking Large Language Models (LLMs) by predicted expertise on user queries. The task is to rank 1,131 LLMs based on their expected ability to answer queries without actually querying them, using only development data with ground truth relevance judgments.

## Current Repository Structure

```
├── README.md                          # Main documentation
├── CLAUDE.md                          # This development guide
├── create_supervised_training_set.py  # Data preprocessing pipeline
├── train_full_cv_simple_progress.py   # 10-fold CV evaluation
├── trec_submission_output.py          # TREC submission generation
└── data/                              # All data files
    ├── llm_dev_data.tsv               # 342 development queries
    ├── llm_dev_qrels.txt              # 386,802 relevance judgments
    ├── supervised_training_full.csv   # Processed training dataset
    ├── full_evaluation_report.json    # Actual CV results
    ├── llm_discovery_data_1.json      # Discovery dataset (unused in baseline)
    ├── llm_discovery_data_2.json      # Discovery dataset (unused in baseline)
    └── llm_discovery_metadata_1.json  # Discovery metadata (unused in baseline)
```

## Data Files

### Development Data (Used in Baseline)
- **data/llm_dev_data.tsv**: 342 queries with format `<query_id>\t<query_text>`
- **data/llm_dev_qrels.txt**: 386,802 relevance judgments with format `<query_id> 0 <llm_id> <qrel_score>`
  - qrel_score: 0 (not relevant), 1 (most relevant), 2 (second most relevant)
  - Distribution: 92.4% score=0, 3.7% score=1, 3.8% score=2

### Discovery Data (Not Used in Baseline)
- **data/llm_discovery_data_*.json**: 14,950 queries with LLM responses
- **data/llm_discovery_metadata_1.json**: LLM metadata and characteristics

### Generated Files
- **data/supervised_training_full.csv**: Combined training data (query_text, llm_id, qrel)
- **data/full_evaluation_report.json**: Actual 10-fold CV results with performance metrics
- **quick_submission.txt**: Sample TREC submission (generated when running trec_submission_output.py)

## Baseline Model Performance

Based on the actual results in `data/full_evaluation_report.json`:

**10-Fold Cross-Validation Results (All 1,131 LLMs)**:
- **nDCG@10**: 0.3566 ± 0.0571 (95% CI: [0.2447, 0.4685])
- **nDCG@5**: 0.3496 ± 0.0574 (95% CI: [0.2371, 0.4622]) 
- **MRR**: 0.6227 ± 0.0866 (95% CI: [0.4530, 0.7924])
- **Runtime**: 1.95 hours total, ~11 minutes per fold

## Common Development Tasks

### 1. Generate Training Data
```bash
python create_supervised_training_set.py
```
- Reads `data/llm_dev_data.tsv` and `data/llm_dev_qrels.txt`
- Creates `data/supervised_training_full.csv` with 386,802 examples

### 2. Run 10-Fold Cross-Validation
```bash
python train_full_cv_simple_progress.py
```
- Performs query-based CV to prevent data leakage
- Trains Random Forest Regressor (100 trees, max_depth=15)
- Uses TF-IDF features (1000 max features) + LLM ID encoding
- Outputs `data/full_evaluation_report.json` with detailed results

### 3. Generate TREC Submission
```bash
python trec_submission_output.py
```
- Trains on complete dataset
- Generates `quick_submission.txt` in TREC format
- Format: `<query_id> Q0 <llm_id> <rank> <score> <run_id>`

## Technical Implementation Details

### Feature Engineering
- **Text Features**: TF-IDF vectorization (max_features=1000, ngram_range=(1,2))
- **LLM Features**: Label encoding of LLM identifiers  
- **Target**: qrel scores normalized to [0,1] (0→0.0, 1→0.5, 2→1.0)

### Model Configuration
- **Algorithm**: Random Forest Regressor
- **Parameters**: n_estimators=100, max_depth=15, min_samples_split=5, min_samples_leaf=2
- **Cross-Validation**: 10-fold, query-based splitting
- **Evaluation**: nDCG@10, nDCG@5, MRR, MSE

### Data Pipeline
1. Load development queries and relevance judgments
2. Join on query_id to create supervised dataset
3. Split queries (not examples) for CV to prevent leakage
4. Train Random Forest on ~348,000 examples per fold
5. Evaluate on ~38,000 examples per fold

## Performance Analysis

### Current Baseline Strengths
- **MRR > nDCG**: Good at finding first relevant LLM (MRR=0.623 vs nDCG@10=0.357)
- **Low Variance**: Consistent performance across folds (std < 0.1)
- **Complete Coverage**: Uses all 1,131 LLMs and 342 queries

### Areas for Improvement
- **Feature Engineering**: Current TF-IDF could be enhanced with dense embeddings
- **Discovery Data**: 14,950 additional queries unused in baseline
- **Model Architecture**: Random Forest could be replaced with neural ranking models

## Development Commands

```bash
# Check data integrity
wc -l data/llm_dev_data.tsv        # Should be 342
wc -l data/llm_dev_qrels.txt       # Should be 386,802

# View current results
python -m json.tool data/full_evaluation_report.json

# Quick data exploration
head -5 data/supervised_training_full.csv
tail -20 quick_submission.txt

# Count relevance distribution  
cut -d' ' -f4 data/llm_dev_qrels.txt | sort | uniq -c
```

## Extension Points for Advanced Models

### Discovery Data Integration
```python
# Example: Load discovery data
import json
with open('data/llm_discovery_data_1.json', 'r') as f:
    discovery_queries = json.load(f)
# Contains 14,950 additional queries with LLM responses
```

### Feature Engineering
```python
# Example: Dense embeddings
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(queries)
```

### Alternative Models
```python
# Example: XGBoost
from xgboost import XGBRegressor
model = XGBRegressor(objective='reg:squarederror')
```

This baseline serves as a foundation for more sophisticated approaches that may incorporate the discovery dataset, advanced feature engineering, or neural architectures.