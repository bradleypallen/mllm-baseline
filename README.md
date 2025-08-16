# TREC 2025 Million LLMs Track - Multi-Model Baseline Framework

A comprehensive machine learning framework for ranking Large Language Models (LLMs) by predicted expertise on user queries, developed for the [TREC 2025 Million LLMs Track](https://trec-mllm.github.io/) with multiple baseline implementations for comparison.

## Framework Overview

**Problem Statement**: Given a user query, predict and rank 1,131 LLMs by their ability to provide high-quality answers without actually querying the models.

**Multi-Model Approach**: This framework implements and compares multiple baseline approaches, providing standardized evaluation protocols and performance benchmarks for understanding different approaches to the LLM ranking task.

**Current Baselines**:
- **Tier2 CPU Optimized**: Neural two-tower architecture with multi-head attention (3 heads) achieving nDCG@10=0.4306
- **Tier2 Config A & B**: Hyperparameter-optimized variants with 6-8 attention heads achieving nDCG@10=0.4278/0.4261
- **Tier3 Cross-Encoder**: Joint query-LLM encoding with transformer attention for direct relevance prediction
- **Enhanced Neural Two-Tower**: Deep learning with ContrastiveLoss, 128D embeddings, and hard negative mining
- **Neural Two-Tower**: Deep learning approach with sentence transformers and learned embeddings
- **Random Forest Regressor**: Traditional ML with TF-IDF features and LLM identity encoding
- **XGBoost Regressor**: Gradient boosting with regularization and fast training

## Current Leaderboard

| Rank | Model | nDCG@10 | nDCG@5 | MRR | Runtime |
|------|--------|---------|--------|-----|---------|
| 1 | **Tier2 Cpu Optimized** | 0.4306 ± 0.055 | 0.4347 ± 0.058 | 0.7263 ± 0.070 | 3.11h |
| 2 | **Tier2 Cpu Optimized Config A** | 0.4278 ± 0.052 | 0.4399 ± 0.054 | 0.7162 ± 0.066 | 4.54h |
| 3 | **Tier2 Cpu Optimized Config B** | 0.4261 ± 0.052 | 0.4356 ± 0.058 | 0.6783 ± 0.071 | 4.73h |
| 4 | **Tier3 Cross Encoder** | 0.4259 ± 0.049 | 0.4378 ± 0.051 | 0.7141 ± 0.076 | 21.92h |
| 5 | **Enhanced Neural Two Tower** | 0.4256 ± 0.050 | 0.4287 ± 0.056 | 0.7113 ± 0.074 | 2.95h |
| 6 | **Neural Two Tower** | 0.4022 ± 0.028 | 0.4135 ± 0.034 | 0.6761 ± 0.057 | 6.95h |

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
├── docs/                              # Additional documentation
│   └── neural-architecture-evolution.md  # Neural architecture development notes
├── models/                            # Model implementations
│   ├── random_forest/                 # Random Forest baseline
│   │   ├── evaluate_10fold_cv.py      # 10-fold CV experimental evaluation
│   │   └── README.md                  # Model documentation
│   ├── neural_two_tower/              # Neural Two-Tower baseline
│   │   ├── evaluate_10fold_cv.py      # Original neural two-tower evaluation
│   │   ├── evaluate_enhanced_10fold_cv.py  # Enhanced model evaluation
│   │   ├── evaluate_tier2_10fold_cv.py     # Tier 2 model evaluation (GPU)
│   │   ├── evaluate_tier2_cpu.py      # Tier 2 model evaluation (CPU optimized)
│   │   ├── evaluate_tier2_config_a.py # Hyperparameter Config A (6 heads)
│   │   ├── evaluate_tier2_config_b.py # Hyperparameter Config B (8 heads)
│   │   ├── evaluate_tier2_tuned.py    # Tuned hyperparameter variant
│   │   ├── tune_tier2_hyperparams.py  # Hyperparameter search script
│   │   ├── evaluate_tier2_with_profiles.py  # Epistemic profile integration
│   │   ├── evaluate_tier3_cross_encoder.py  # Tier 3 cross-encoder evaluation
│   │   ├── model.py                   # Multi-tier architectures with enhancements
│   │   ├── data_loader.py             # Neural data loading with hard negative mining
│   │   ├── evaluate_neural.py         # Evaluation utilities
│   │   ├── requirements_neural.txt    # Additional dependencies
│   │   ├── performance.md             # Detailed performance analysis
│   │   └── README.md                  # Model documentation
│   ├── xgboost/                       # XGBoost baseline with variants
│   │   ├── evaluate_10fold_cv.py      # Original XGBoost evaluation
│   │   ├── evaluate_xgboost_hybrid.py # Hybrid feature engineering
│   │   ├── evaluate_xgboost_ensemble.py   # Ensemble methods
│   │   ├── evaluate_xgboost_discovery.py  # Discovery data integration
│   │   ├── evaluate_xgboost_epistemic.py  # Epistemic feature integration
│   │   ├── evaluate_xgboost_expertise.py  # Expertise-based features
│   │   ├── evaluate_xgboost_interactions.py  # Feature interactions
│   │   ├── evaluate_xgboost_smart_imputation.py  # Advanced imputation
│   │   ├── evaluate_xgboost_deeper.py     # Deeper tree configurations
│   │   ├── evaluate_xgboost_twostage.py   # Two-stage training
│   │   ├── evaluate_xgboost_weighted_ensemble.py  # Weighted ensemble
│   │   ├── requirements_xgboost.txt   # XGBoost dependencies
│   │   └── README.md                  # Model documentation
│   └── epistemic_profiling/           # Epistemic profiling experiments
│       ├── EPISTEMIC_PROFILING_OVERVIEW.md  # Comprehensive overview
│       ├── RESULTS_SUMMARY.md         # Results analysis and insights
│       ├── PROFILE_FILES_README.md    # Guide to profile file formats
│       ├── bilateral_profile_extractor.py  # Bilateral truth profiling
│       ├── epistemic_profile_extractor.py  # Epistemic behavior analysis
│       ├── cluster_llm_profiles.py    # LLM clustering algorithms
│       ├── build_expertise_profiles.py     # Domain expertise extraction
│       ├── xgboost_with_expertise.py  # XGBoost with expertise features
│       ├── two_tower_with_expertise.py     # Neural model with profiles
│       ├── enhanced_two_tower.py      # Enhanced model with epistemic features
│       ├── tier2_with_discovery.py    # Tier2 with discovery integration
│       ├── simplified_epistemic_model.py   # Lightweight epistemic model
│       ├── analyze_complete_profiles.py    # Profile analysis tools
│       ├── compare_evaluation_methods.py   # Method comparison studies
│       ├── strategic_300_llms.json    # Strategic LLM subset profiles
│       ├── complete_1131_llms_profiles.json # Complete LLM profiles
│       ├── bilateral_epistemic_profiles.json # Bilateral truth profiles
│       ├── qrel_expertise_profiles.json    # Q&A-based expertise profiles
│       └── [100+ additional experimental files] # Checkpoints, analyses, results
├── shared/                            # Shared utilities
│   └── utils/                         # Common evaluation functions
│       ├── evaluation.py              # Standardized metrics
│       └── __init__.py                # Package initialization
└── data/                              # All data files
    ├── create_supervised_training_set.py     # Data preprocessing script
    ├── llm_dev_data.tsv               # 342 development queries
    ├── llm_dev_qrels.txt              # 386,802 relevance judgments
    ├── llm_discovery_data_1.json      # Discovery dataset (14,950 queries)
    ├── llm_discovery_data_2.json      # Discovery dataset continuation
    ├── llm_discovery_metadata_1.json  # Discovery metadata
    ├── supervised_training_full.csv   # Processed training dataset
    └── results/                       # Standardized model results (25+ files)
        ├── tier2_cpu_optimized_results.json        # Tier2 CPU optimized
        ├── tier2_cpu_optimized_config_a_results.json # Config A (6 heads)
        ├── tier2_cpu_optimized_config_b_results.json # Config B (8 heads)
        ├── tier2_cpu_optimized_tuned_results.json    # Tuned hyperparameters
        ├── tier3_cross_encoder_results.json        # Tier3 cross-encoder
        ├── enhanced_neural_two_tower_results.json  # Enhanced neural baseline
        ├── neural_two_tower_results.json       # Original neural baseline
        ├── random_forest_results.json          # Random Forest baseline
        ├── xgboost_results.json                # Original XGBoost baseline
        ├── xgboost_hybrid_results.json         # XGBoost hybrid features
        ├── xgboost_ensemble_results.json       # XGBoost ensemble methods
        ├── xgboost_epistemic_results.json      # XGBoost epistemic features
        ├── xgboost_expertise_results.json      # XGBoost expertise features
        ├── simplified_epistemic_300_results.json   # Simplified epistemic model
        └── [additional XGBoost variant results] # 15+ XGBoost experimental results
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

### 1. Tier2 CPU Optimized (`models/neural_two_tower/`)

**Architecture**:
- **Query Tower**: all-MiniLM-L6-v2 → Multi-head attention (3 heads) → Dense layers [384→256→192→128]
- **LLM Tower**: Learned embeddings → Dense layers [128→192→128]
- **Loss Function**: ContrastiveLoss (InfoNCE) with head diversity regularization
- **Training**: Hard negative mining (60% hard, 40% easy), curriculum learning
- **Performance**: nDCG@10=0.4306, Runtime=3.11h

**Implementation Details**:
- **Multi-head Query Attention**: 3 attention heads with specialization regularization
- **Hard Negative Mining**: Dynamic selection of challenging training examples
- **Head Diversity Regularization**: Loss term encouraging complementary head representations
- **Curriculum Learning**: Hard negatives activated from epoch 3 onwards

**Characteristics**: Highest measured performance (nDCG@10=0.4306), multi-head neural architecture, CPU-optimized implementation
**Use Case**: Performance baseline for multi-head attention approaches in LLM ranking

### 2. Tier3 Cross-Encoder (`models/neural_two_tower/`)

**Architecture**:
- **Transformer Backbone**: DistilBERT-base-uncased for joint query-LLM encoding
- **Query Processing**: Tokenized text → DistilBERT → [CLS] token representation (768D)
- **LLM Integration**: Learned LLM embeddings (192D) concatenated with query representation
- **Classification Head**: Dense layers [960→768→384→1] with ReLU activation and dropout
- **Training**: BCE loss for direct relevance prediction, batch_size=48, 15 epochs
- **Performance**: nDCG@10=0.4259, Runtime=21.92h

**Implementation Details**:
- **Joint Attention**: Transformer attention between query tokens and LLM representation
- **Direct Optimization**: Relevance prediction without embedding similarity constraints
- **Memory Configuration**: Batch size 48 for memory efficiency
- **Binary Classification**: BCE loss with logits for training stability

**Characteristics**: Transformer-based architecture, direct query-LLM interaction modeling
**Trade-offs**: 7x longer training time (21.92h vs 3.11h) for comparable performance (nDCG@10=0.4259 vs 0.4306)
**Use Case**: Comparison baseline for cross-encoder approaches in LLM ranking

### 3. Enhanced Neural Two-Tower (`models/neural_two_tower/`)

**Architecture**:
- **Query Tower**: all-MiniLM-L6-v2 → Dense layers [384→256→192→128]
- **LLM Tower**: Learned embeddings → Dense layers [128→192→128]
- **Loss Function**: ContrastiveLoss (InfoNCE) with temperature=0.1
- **Training**: Hard negative mining, 128D embeddings
- **Performance**: nDCG@10=0.4256, Runtime=2.95h

**Implementation Details**:
- **ContrastiveLoss**: InfoNCE-based loss for representation learning
- **128D Embeddings**: Increased dimensionality compared to 64D baseline
- **Hard Negative Mining**: Dynamic selection of challenging training examples
- **Training Efficiency**: 2.4x faster convergence than baseline neural model

**Characteristics**: ContrastiveLoss implementation with 128D embeddings, 2.4x faster convergence than baseline
**Use Case**: Baseline for contrastive learning approaches in neural LLM ranking

### 4. Neural Two-Tower Baseline (`models/neural_two_tower/`)

**Architecture**:
- **Query Tower**: all-MiniLM-L6-v2 → Dense layers [384→256→128→64]
- **LLM Tower**: Learned embeddings → Dense layers [64→128→64]
- **Similarity**: Cosine similarity with margin-based pairwise loss
- **Performance**: nDCG@10=0.4022, Runtime=6.95h

**Characteristics**: Semantic encoding with learned representations, margin-based pairwise loss
**Use Case**: Baseline for two-tower architectures in neural LLM ranking

### 5. Random Forest Baseline (`models/random_forest/`)

**Architecture**:
- **Features**: TF-IDF (1000 features) + LLM ID encoding
- **Model**: Random Forest Regressor (100 trees, max_depth=15)
- **Performance**: nDCG@10=0.3860, Runtime=1.37h

**Characteristics**: Traditional ML approach with TF-IDF features, 1.37h training time
**Use Case**: Baseline for traditional machine learning approaches in LLM ranking

### 4. XGBoost Baseline (`models/xgboost/`)

**Architecture**:
- **Features**: TF-IDF (1000 features) + LLM ID encoding (same as Random Forest)
- **Model**: XGBoost Regressor (100 trees, max_depth=6, learning_rate=0.1)
- **Performance**: nDCG@10=0.382, Runtime=0.03h

**Characteristics**: Gradient boosting with regularization, 0.03h training time
**Use Case**: Fast experimentation, efficient baseline, ensemble component

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

**Tier2 CPU Optimized** (nDCG@10=0.4306):
```bash
cd models/neural_two_tower
pip install -r requirements_neural.txt
python evaluate_tier2_cpu.py
```

**Tier2 Config A** (6 heads, nDCG@10=0.4278):
```bash
cd models/neural_two_tower
pip install -r requirements_neural.txt
python evaluate_tier2_config_a.py
```

**Tier2 Config B** (8 heads, nDCG@10=0.4261):
```bash
cd models/neural_two_tower
pip install -r requirements_neural.txt
python evaluate_tier2_config_b.py
```

**Tier3 Cross-Encoder** (transformer-based):
```bash
cd models/neural_two_tower
pip install -r requirements_neural.txt
python evaluate_tier3_cross_encoder.py
```

**Enhanced Neural Two-Tower** (nDCG@10=0.4256):
```bash
cd models/neural_two_tower
pip install -r requirements_neural.txt
python evaluate_enhanced_10fold_cv.py
```

**Neural Two-Tower** (nDCG@10=0.4022):
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

**XGBoost** (nDCG@10=0.3824):
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
- **Highest nDCG@10**: Tier2 CPU Optimized (0.4306 ± 0.055) with 3-head attention architecture
- **Performance Distribution**: Top 5 models cluster within 1.2% nDCG@10 range (0.4306 to 0.4256)
- **Attention Head Scaling**: Config A (6 heads, 0.4278) and Config B (8 heads, 0.4261) show consistent performance gains
- **MRR Performance**: Tier2 CPU Optimized (0.7263) and Config A (0.7162) achieve highest mean reciprocal rank
- **Training Efficiency**: Original Tier2 CPU completes evaluation in 3.11h vs. hyperparameter variants' 4.5-4.7h
- **Multi-head Attention**: Systematic scaling from 3→6→8 heads demonstrates measurable performance improvements

### Model Trade-offs
- **Tier2 CPU Optimized**: Highest performance (nDCG@10=0.4306), 3-head attention, 3.11h training time
- **Tier2 Config A (6 heads)**: Peak performance up to 0.5221 nDCG@10, 6-head attention, 4.54h training time
- **Tier2 Config B (8 heads)**: Consistent performance (nDCG@10=0.4261), 8-head attention, 4.73h training time
- **Tier3 Cross-Encoder**: Comparable performance (nDCG@10=0.4259), transformer architecture, 21.92h training time
- **Enhanced Neural Two-Tower**: ContrastiveLoss implementation with 128D embeddings, 2.95h training time
- **Neural Two-Tower**: Baseline two-tower architecture with semantic features, 6.95h training time
- **Random Forest**: Traditional ML approach with interpretable features, 1.37h training time
- **XGBoost**: Gradient boosting implementation with fast training, 0.03h training time

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

## Experimental Extensions

### Epistemic Profiling (`models/epistemic_profiling/`)
Experiments integrating epistemic profiles (bilateral truth evaluation) and expertise matching into LLM ranking:
- **Bilateral Truth Profiles**: 4D epistemic profiles measuring LLM behavior patterns (confident_correct, overconfident_wrong, uncertain, inconsistent)
- **LLM Clustering**: Grouped 1,131 LLMs into 5 clusters based on epistemic reliability
- **Expertise Profiles**: Domain-specific expertise for 117 LLMs using Q&A clustering
- **Results**: XGBoost with expertise features achieved nDCG@10=0.3994 (+1.8% over baseline XGBoost)
- See [models/epistemic_profiling/RESULTS_SUMMARY.md](models/epistemic_profiling/RESULTS_SUMMARY.md) for detailed analysis

## Future Directions

This multi-model framework enables systematic comparison of different approaches for LLM ranking. Potential extensions include:

- **Discovery Dataset Integration**: Leverage 14,950 additional queries with LLM responses
- **Alternative Neural Architectures**: Transformer-based ranking, cross-attention models
- **Meta-Learning Approaches**: Incorporate LLM metadata, architectural features
- **Ensemble Methods**: Combine multiple baseline approaches
- **Transfer Learning**: Fine-tune pre-trained language models for ranking tasks

## Contributing

1. **Implement Model**: Create new model in `models/your_model_name/`
2. **Follow Standards**: Use shared evaluation utilities, standardized results format
3. **Update Documentation**: Add model description, performance analysis
4. **Generate Leaderboard**: Run `python generate_leaderboard.py` to update comparisons

This framework provides a foundation for research in automated LLM ranking and selection for query answering tasks.

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