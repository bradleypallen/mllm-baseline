# TREC 2025 Million LLMs Track - Multi-Model Baseline Framework

A comprehensive machine learning framework for ranking Large Language Models (LLMs) by predicted expertise on user queries, developed for the [TREC 2025 Million LLMs Track](https://trec-mllm.github.io/) with multiple baseline implementations for comparison.

## Framework Overview

**Problem Statement**: Given a user query, predict and rank 1,131 LLMs by their ability to provide high-quality answers without actually querying the models.

**Multi-Model Approach**: This framework implements and compares multiple baseline approaches, providing standardized evaluation protocols and performance benchmarks for understanding different approaches to the LLM ranking task.

**Current Baselines**:
- **Random Forest Regressor**: Traditional ML with TF-IDF features and LLM identity encoding
- **Two-Tower Neural Network**: Deep learning approach with sentence transformers and learned embeddings

## Current Leaderboard

| Rank | Model | nDCG@10 | nDCG@5 | MRR | Runtime |
|------|--------|---------|--------|-----|---------|
| 1 | **Neural Two Tower** | 0.4022 ± 0.028 | 0.4135 ± 0.034 | 0.6761 ± 0.057 | 6.95h |
| 2 | **Random Forest** | 0.3860 ± 0.044 | 0.3871 ± 0.050 | 0.6701 ± 0.081 | 1.37h |

*See [leaderboard.md](leaderboard.md) for detailed comparison*

## Repository Structure

```
├── README.md                          # This documentation
├── CLAUDE.md                          # Development guidance  
├── NAMING_CONVENTIONS.md              # Model implementation standards
├── TESTING.md                         # Testing and verification guide
├── requirements.txt                   # Core dependencies (pandas, numpy, scikit-learn)
├── test_models.py                     # Quick testing suite (< 3 seconds)
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
    ├── create_supervised_training_set.py     # Data preprocessing script
    ├── llm_dev_data.tsv               # 342 development queries
    ├── llm_dev_qrels.txt              # 386,802 relevance judgments
    ├── supervised_training_full.csv   # Processed training dataset
    └── results/                       # Standardized model results
        ├── random_forest_results.json        # Random Forest CV results
        └── neural_two_tower_results.json     # Neural baseline CV results
```

## Dataset and Evaluation Protocol

### Source Data
- **Development Queries**: 342 queries from `data/llm_dev_data.tsv` (used in baselines)
- **Relevance Judgments**: 386,802 query-LLM pairs from `data/llm_dev_qrels.txt` (used in baselines)
- **Discovery Queries**: 14,950 queries with LLM responses (available for advanced models)
- **Ground Truth Labels**: Three-level relevance scale (0=not relevant, 1=most relevant, 2=second most relevant)

### Standardized Evaluation
- **Cross-Validation**: 10-fold, query-based splitting to prevent data leakage
- **Metrics**: nDCG@10, nDCG@5, Mean Reciprocal Rank (MRR)
- **Qrel Encoding**: Corrected mapping (0→0.0, 2→0.7, 1→1.0) reflecting proper ranking order
- **Results Format**: Standardized JSON with confidence intervals and fold-by-fold details

## Model Implementations

### 1. Random Forest Baseline (`models/random_forest/`)

**Architecture**:
- **Features**: TF-IDF (1000 features) + LLM ID encoding
- **Model**: Random Forest Regressor (100 trees, max_depth=15)
- **Performance**: nDCG@10=0.386, Runtime=1.37h

**Strengths**: Fast training, interpretable, strong baseline performance
**Use Case**: Comparison benchmark, rapid prototyping

### 2. Neural Two-Tower Baseline (`models/neural_two_tower/`)

**Architecture**:
- **Query Tower**: all-MiniLM-L6-v2 → Dense layers [384→256→128→64]
- **LLM Tower**: Learned embeddings → Dense layers [64→128→64]
- **Similarity**: Cosine similarity with margin-based pairwise loss
- **Performance**: nDCG@10=0.402, Runtime=6.95h

**Strengths**: Semantic understanding, better ranking quality, learned representations
**Use Case**: Deep learning foundation, semantic query understanding

## Getting Started

### 1. Setup Environment
```bash
git clone <repository-url>
cd mllm-baseline
pip install -r requirements.txt  # Core dependencies (pandas, numpy, scikit-learn)
```

### 2. Quick Testing (Optional but Recommended)
```bash
python test_models.py  # Verify setup in < 3 seconds
```

### 3. Generate Training Data
```bash
cd data
python create_supervised_training_set.py
```

### 4. Run Baseline Models

**Random Forest**:
```bash
cd models/random_forest
python evaluate_10fold_cv.py
```

**Neural Two-Tower** (requires additional dependencies):
```bash
cd models/neural_two_tower
pip install -r requirements_neural.txt
python evaluate_10fold_cv.py
```

### 5. Generate Updated Leaderboard
```bash
python generate_leaderboard.py
```

## Testing and Verification

### Quick Testing Suite
```bash
python test_models.py
```
**What it verifies** (in < 3 seconds):
- ✅ Data preprocessing pipeline
- ✅ Model imports and dependencies  
- ✅ Data loading paths
- ✅ Shared utilities functionality
- ✅ Leaderboard generation

### Detailed Testing
For comprehensive testing strategies, see [TESTING.md](TESTING.md):
- Subset testing with minimal data (< 30 seconds)
- Smoke tests for individual components (< 1 minute)  
- Error diagnosis and troubleshooting
- Pre-flight checks before full evaluation

## Adding New Models

### 1. Create Model Directory
```bash
mkdir -p models/your_model_name
```

### 2. Implement Your Model
- Create `evaluate_10fold_cv.py` for 10-fold cross-validation experimental evaluation
- Follow [NAMING_CONVENTIONS.md](NAMING_CONVENTIONS.md) for consistent structure
- Use shared evaluation utilities from `shared/utils/`
- Save results in standardized format to `data/results/your_model_name_results.json`

### 3. Update Leaderboard
```bash
python generate_leaderboard.py
```

### 4. Standardized Results Format
```json
{
  "evaluation_info": {
    "timestamp": "ISO timestamp",
    "pipeline": "Model description",
    "total_runtime_hours": 0.0
  },
  "performance_metrics": {
    "ndcg_10": {"mean": 0.0, "std": 0.0, "confidence_interval_95": [0.0, 0.0]},
    "ndcg_5": {"mean": 0.0, "std": 0.0, "confidence_interval_95": [0.0, 0.0]},
    "mrr": {"mean": 0.0, "std": 0.0, "confidence_interval_95": [0.0, 0.0]}
  },
  "fold_by_fold_results": [{"fold": 1, "ndcg_10": 0.0, "ndcg_5": 0.0, "mrr": 0.0}]
}
```

## Performance Analysis

### Current Results Summary
- **Best nDCG@10**: Neural Two-Tower (0.402) with 4.2% improvement over Random Forest
- **Best Runtime**: Random Forest (1.37h) is 5x faster than Neural approach
- **Most Consistent**: Neural Two-Tower shows lower variance (±0.028 vs ±0.044)
- **MRR Performance**: Both models achieve strong first-relevant ranking (MRR ~0.67)

### Model Trade-offs
- **Random Forest**: Fast, interpretable, good baseline, limited semantic understanding
- **Neural Two-Tower**: Better performance, semantic features, longer training time, GPU beneficial

## Requirements

### Core Dependencies (Required for All Models)
```bash
pip install -r requirements.txt
# Installs: pandas, numpy, scikit-learn
```

### Model-Specific Dependencies
**Random Forest**: No additional dependencies (uses core requirements only)

**Neural Two-Tower**: Additional dependencies for deep learning
```bash
cd models/neural_two_tower
pip install -r requirements_neural.txt
# Installs: torch, sentence-transformers, transformers, etc.
```

### System Requirements
- **RAM**: 8GB+ for full dataset processing
- **CPU**: Multi-core recommended for parallel training
- **GPU**: Optional but beneficial for neural models

## Future Directions

This multi-model framework enables systematic comparison of different approaches for LLM ranking. Future extensions may include:

- **Discovery Dataset Integration**: Leverage 14,950 additional queries with LLM responses
- **Advanced Neural Architectures**: Transformer-based ranking, cross-attention models
- **Meta-Learning Approaches**: Incorporate LLM metadata, architectural features
- **Ensemble Methods**: Combine multiple baseline approaches
- **Transfer Learning**: Fine-tune large language models for ranking tasks

## Contributing

1. **Implement Model**: Create new model in `models/your_model_name/`
2. **Follow Standards**: Use shared evaluation utilities, standardized results format
3. **Update Documentation**: Add model description, performance analysis
4. **Generate Leaderboard**: Run `python generate_leaderboard.py` to update comparisons

This framework provides a solid foundation for advancing the state of automated LLM ranking and selection for query answering tasks.

## Documentation

### Core Documentation
- **[README.md](README.md)** - Main framework overview and getting started guide
- **[CLAUDE.md](CLAUDE.md)** - Detailed development guidance and technical implementation details
- **[NAMING_CONVENTIONS.md](NAMING_CONVENTIONS.md)** - Standards for implementing new models
- **[TESTING.md](TESTING.md)** - Testing strategies and verification procedures

### Model Documentation  
- **[leaderboard.md](leaderboard.md)** - Automatically generated model comparison results
- **[models/random_forest/README.md](models/random_forest/README.md)** - Random Forest baseline documentation
- **[models/neural_two_tower/README.md](models/neural_two_tower/README.md)** - Neural Two-Tower baseline documentation

### Quick Reference
- **Quick Setup**: `pip install -r requirements.txt`
- **Quick Test**: `python test_models.py` (< 3 seconds)
- **Data Generation**: `cd data && python create_supervised_training_set.py`
- **Run Models**: `cd models/[model_name] && python evaluate_10fold_cv.py`
- **Update Results**: `python generate_leaderboard.py`