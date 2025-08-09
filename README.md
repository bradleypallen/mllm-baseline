# TREC 2025 Million LLMs Track - Multi-Model Baseline Framework

A comprehensive machine learning framework for ranking Large Language Models (LLMs) by predicted expertise on user queries, developed for the [TREC 2025 Million LLMs Track](https://trec-mllm.github.io/) with multiple baseline implementations for comparison.

## Framework Overview

**Problem Statement**: Given a user query, predict and rank 1,131 LLMs by their ability to provide high-quality answers without actually querying the models.

**Multi-Model Approach**: This framework implements and compares multiple baseline approaches, providing standardized evaluation protocols and performance benchmarks for understanding different approaches to the LLM ranking task.

**Current Baselines**:
- **Tier 2 CPU Optimized**: State-of-the-art neural architecture with multi-head attention, hard negative mining, and advanced training
- **Tier 3 Cross-Encoder**: Joint query-LLM encoding with transformer attention for direct relevance prediction
- **Enhanced Neural Two-Tower**: Advanced deep learning with ContrastiveLoss, 128D embeddings, and hard negative mining
- **Neural Two-Tower**: Deep learning approach with sentence transformers and learned embeddings
- **Random Forest Regressor**: Traditional ML with TF-IDF features and LLM identity encoding
- **XGBoost Regressor**: Gradient boosting with advanced regularization and fast training

## Current Leaderboard

| Rank | Model | nDCG@10 | nDCG@5 | MRR | Runtime |
|------|--------|---------|--------|-----|---------|
| 1 | **Tier 2 CPU Optimized** | 0.4306 ± 0.055 | 0.4347 ± 0.058 | 0.7263 ± 0.070 | 3.11h |
| 2 | **Tier 3 Cross-Encoder** | 0.4259 ± 0.049 | 0.4378 ± 0.051 | 0.7141 ± 0.076 | 21.92h |
| 3 | **Enhanced Neural Two Tower** | 0.4256 ± 0.050 | 0.4287 ± 0.056 | 0.7113 ± 0.074 | 2.95h |
| 4 | **Neural Two Tower** | 0.4022 ± 0.028 | 0.4135 ± 0.034 | 0.6761 ± 0.057 | 6.95h |
| 5 | **Random Forest** | 0.3860 ± 0.044 | 0.3871 ± 0.050 | 0.6701 ± 0.081 | 1.37h |
| 6 | **XGBoost** | 0.3824 ± 0.045 | 0.3808 ± 0.047 | 0.6206 ± 0.052 | 0.03h |

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
│   ├── neural_two_tower/              # Neural Two-Tower baseline
│   │   ├── evaluate_10fold_cv.py      # 10-fold CV experimental evaluation
│   │   ├── evaluate_enhanced_10fold_cv.py  # Enhanced model evaluation
│   │   ├── evaluate_tier2_10fold_cv.py     # Tier 2 model evaluation (GPU)
│   │   ├── evaluate_tier2_cpu.py      # Tier 2 model evaluation (CPU optimized)
│   │   ├── evaluate_tier3_cross_encoder.py  # Tier 3 cross-encoder evaluation
│   │   ├── model.py                   # Multi-tier architectures with enhancements
│   │   ├── data_loader.py             # Neural data loading with hard negative mining
│   │   ├── evaluate_neural.py         # Evaluation utilities
│   │   ├── requirements_neural.txt    # Additional dependencies
│   │   ├── performance.md             # Detailed performance analysis
│   │   └── README.md                  # Model documentation
│   └── xgboost/                       # XGBoost baseline
│       ├── evaluate_10fold_cv.py      # 10-fold CV experimental evaluation
│       ├── requirements_xgboost.txt   # XGBoost dependencies
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
        ├── tier2_cpu_optimized_results.json        # Tier 2 CPU optimized CV results
        ├── tier3_cross_encoder_results.json        # Tier 3 cross-encoder CV results
        ├── enhanced_neural_two_tower_results.json  # Enhanced Neural baseline CV results
        ├── neural_two_tower_results.json       # Neural baseline CV results  
        ├── random_forest_results.json          # Random Forest CV results
        └── xgboost_results.json                # XGBoost CV results
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

### 1. Tier 2 CPU Optimized (`models/neural_two_tower/`)

**Architecture**:
- **Query Tower**: all-MiniLM-L6-v2 → Multi-head attention (3 heads) → Dense layers [384→256→192→128]
- **LLM Tower**: Learned embeddings → Enhanced Dense layers [128→192→128]
- **Loss Function**: ContrastiveLoss (InfoNCE) with head diversity regularization
- **Training**: Active hard negative mining (60% hard, 40% easy), curriculum learning
- **Performance**: nDCG@10=0.4306, Runtime=3.11h

**Tier 2 Enhancements**:
- **Multi-head Query Attention**: 3 specialized attention heads with emergent specialization
- **Active Hard Negative Mining**: Intelligent selection of challenging training examples
- **Head Diversity Regularization**: Encourages different heads to learn complementary aspects
- **Curriculum Learning**: Progressive training difficulty with hard negatives activated from epoch 3

**Strengths**: State-of-the-art performance, advanced neural architecture, efficient CPU optimization
**Use Case**: Best-in-class ranking performance, research baseline for advanced multi-head approaches

### 2. Tier 3 Cross-Encoder (`models/neural_two_tower/`)

**Architecture**:
- **Transformer Backbone**: DistilBERT-base-uncased for joint query-LLM encoding
- **Query Processing**: Tokenized text → DistilBERT → [CLS] token representation (768D)
- **LLM Integration**: Learned LLM embeddings (192D) concatenated with query representation
- **Classification Head**: Dense layers [960→768→384→1] with ReLU activation and dropout
- **Training**: BCE loss for direct relevance prediction, batch_size=48, 15 epochs
- **Performance**: nDCG@10=0.4259, Runtime=21.92h

**Tier 3 Cross-Encoder Features**:
- **Joint Attention**: Direct transformer attention between query tokens and LLM representation
- **End-to-End Optimization**: Direct relevance prediction vs. embedding similarity matching
- **Memory Optimization**: batch_size=48 optimized for 24GB unified memory systems
- **Advanced Loss**: BCE with logits for stable binary classification training

**Strengths**: Sophisticated transformer architecture, direct query-LLM interaction modeling
**Limitations**: 7x slower training than Tier 2 for comparable performance, computationally expensive
**Use Case**: Research exploration of cross-encoder limits, comparison baseline for attention mechanisms

### 3. Enhanced Neural Two-Tower (`models/neural_two_tower/`)

**Architecture**:
- **Query Tower**: all-MiniLM-L6-v2 → Dense layers [384→256→192→128]
- **LLM Tower**: Learned embeddings → Dense layers [128→192→128]
- **Loss Function**: ContrastiveLoss (InfoNCE) with temperature=0.1
- **Training**: Hard negative mining capability, 128D embeddings
- **Performance**: nDCG@10=0.4256, Runtime=2.95h

**Key Enhancements**:
- **ContrastiveLoss**: InfoNCE-based loss for better representation learning
- **128D Embeddings**: Enhanced representational capacity vs 64D baseline
- **Hard Negative Mining**: Intelligent selection of challenging training examples
- **Efficient Training**: 2x faster convergence than baseline neural model

**Strengths**: State-of-the-art performance, advanced deep learning techniques, efficient training
**Use Case**: Best-in-class ranking performance, research baseline for advanced neural approaches

### 2. Neural Two-Tower Baseline (`models/neural_two_tower/`)

**Architecture**:
- **Query Tower**: all-MiniLM-L6-v2 → Dense layers [384→256→128→64]
- **LLM Tower**: Learned embeddings → Dense layers [64→128→64]
- **Similarity**: Cosine similarity with margin-based pairwise loss
- **Performance**: nDCG@10=0.4022, Runtime=6.95h

**Strengths**: Semantic understanding, strong baseline performance, learned representations
**Use Case**: Deep learning foundation, semantic query understanding, comparison baseline

### 3. Random Forest Baseline (`models/random_forest/`)

**Architecture**:
- **Features**: TF-IDF (1000 features) + LLM ID encoding
- **Model**: Random Forest Regressor (100 trees, max_depth=15)
- **Performance**: nDCG@10=0.3860, Runtime=1.37h

**Strengths**: Fast training, interpretable, solid baseline performance
**Use Case**: Traditional ML comparison benchmark, rapid prototyping

### 4. XGBoost Baseline (`models/xgboost/`)

**Architecture**:
- **Features**: TF-IDF (1000 features) + LLM ID encoding (same as Random Forest)
- **Model**: XGBoost Regressor (100 trees, max_depth=6, learning_rate=0.1)
- **Performance**: nDCG@10=0.382, Runtime=0.03h

**Strengths**: Extremely fast training, gradient boosting, built-in regularization
**Use Case**: Rapid experimentation, efficient baseline, ensemble component

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

**Tier 2 CPU Optimized** (best performance):
```bash
cd models/neural_two_tower
pip install -r requirements_neural.txt
python evaluate_tier2_cpu.py
```

**Tier 3 Cross-Encoder** (transformer-based):
```bash
cd models/neural_two_tower
pip install -r requirements_neural.txt
python evaluate_tier3_cross_encoder.py
```

**Enhanced Neural Two-Tower** (Tier 1 baseline):
```bash
cd models/neural_two_tower
pip install -r requirements_neural.txt
python evaluate_enhanced_10fold_cv.py
```

**Neural Two-Tower** (baseline neural model):
```bash
cd models/neural_two_tower
pip install -r requirements_neural.txt
python evaluate_10fold_cv.py
```

**Random Forest**:
```bash
cd models/random_forest
python evaluate_10fold_cv.py
```

**XGBoost** (requires additional dependencies):
```bash
cd models/xgboost
pip install -r requirements_xgboost.txt
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
- **Best nDCG@10**: Tier 2 CPU Optimized (0.4306) leads by 1.1% over Tier 3, 1.2% over Enhanced Neural
- **Best MRR**: Tier 2 CPU Optimized (0.7263) achieves strongest first-relevant-item performance  
- **Best Efficiency**: Tier 2 achieves top performance in 3.11h vs Tier 3's 21.92h (7x faster)
- **Most Advanced Architecture**: Tie between Tier 2 (multi-head + hard negatives) and Tier 3 (cross-encoder)
- **Performance Plateau**: Top 3 models within 0.5% nDCG@10, suggesting architectural limits with current features

### Model Trade-offs
- **Tier 2 CPU Optimized**: State-of-the-art performance, advanced multi-head architecture, efficient CPU implementation
- **Tier 3 Cross-Encoder**: Competitive performance, sophisticated transformer architecture, 7x training time
- **Enhanced Neural Two-Tower**: Strong Tier 1 performance, ContrastiveLoss and 128D embeddings, good baseline
- **Neural Two-Tower**: Solid baseline performance, semantic features, longer training time 
- **Random Forest**: Good balance of performance and interpretability, moderate training time
- **XGBoost**: Ultra-fast training, competitive performance, excellent for rapid experimentation

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

**XGBoost**: Additional dependency for gradient boosting
```bash
cd models/xgboost
pip install -r requirements_xgboost.txt
# Installs: xgboost (and libomp on macOS)
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