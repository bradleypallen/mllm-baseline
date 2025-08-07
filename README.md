# TREC 2025 Million LLMs Track - Random Forest Regressor Baseline

A machine learning framework for ranking Large Language Models (LLMs) by predicted expertise on user queries, developed for the [TREC 2025 Million LLMs Track](https://trec-mllm.github.io/) as a baseline for understanding how to approach the task.

## Experimental Overview

**Problem Statement**: Given a user query, predict and rank 1,131 LLMs by their ability to provide high-quality answers without actually querying the models.

**Approach**: This baseline implementation uses supervised learning with Random Forest Regression, combining TF-IDF query features and LLM identity encoding to predict relevance scores, validated through 10-fold cross-validation.

**Baseline Purpose**: This framework establishes performance benchmarks for comparison with more advanced models under development. The approach prioritizes reproducibility and interpretability over optimization.

**Data Scope**: This experiment uses only the development dataset (342 queries with ground truth relevance judgments). The discovery dataset (14,950 queries) is not utilized in this baseline implementation.

## Dataset and Data Ingestion

### Source Data
- **Development Queries**: 342 queries from `data/llm_dev_data.tsv` (used in this baseline)
- **Relevance Judgments**: 386,802 query-LLM pairs from `data/llm_dev_qrels.txt` (used in this baseline)
- **Discovery Queries**: 14,950 queries with LLM responses (not used in this baseline implementation)
- **Ground Truth Labels**: Three-level relevance scale (0=not relevant, 1=most relevant, 2=second most relevant)

### Data Distribution
- **Class Imbalance**: 92.4% non-relevant (0), 3.7% most relevant (1), 3.8% second most relevant (2)
- **Coverage**: Complete relevance judgments for all 342 queries × 1,131 LLMs = 386,802 examples
- **Quality**: No missing values, consistent formatting across all entries

### Data Preprocessing Pipeline
```bash
python create_supervised_training_set.py
```

**Preprocessing Steps**:
1. **Query Text Integration**: Join development queries with relevance judgments by query_id
2. **Target Normalization**: Convert qrel scores to relevance values (0→0.0, 2→0.7, 1→1.0)
3. **Feature Matrix Creation**: Combine TF-IDF text features with label-encoded LLM identifiers
4. **Training Set Generation**: Output 386,802 examples to `data/supervised_training_full.csv` with query_text, llm_id, and normalized qrel columns

## Model Architecture

### Model Type: Random Forest Regressor

**Architecture Configuration**:
- **Estimators**: 100 decision trees
- **Max Depth**: 15 levels to balance expressiveness and overfitting prevention
- **Min Samples Split**: 5 examples required for internal node splits
- **Min Samples Leaf**: 2 examples required at terminal nodes
- **Parallelization**: Multi-threaded training with n_jobs=-1

### Feature Engineering

**Text Features (TF-IDF Vectorization)**:
- **Dimensionality**: 1,000 features maximum
- **N-gram Range**: Unigrams and bigrams (1,2) for semantic richness
- **Preprocessing**: English stop word removal, minimum document frequency = 2
- **Output**: Sparse matrix of normalized term frequencies capturing query semantics

**LLM Identity Features**:
- **Encoding**: Label encoding of LLM identifiers (llm_0000 → 0, llm_0001 → 1, ...)
- **Purpose**: Enable model to learn relative performance patterns across LLMs
- **Integration**: Single additional feature column appended to TF-IDF matrix

**Regression Objective**:
- **Target**: Continuous relevance scores (0.0=not relevant, 0.7=second-most relevant, 1.0=most relevant)
- **Loss Function**: Mean squared error minimization
- **Prediction Clipping**: Outputs bounded to [0,1] range
- **Ensemble Averaging**: Final predictions from mean of 100 tree outputs

### Training Strategy

**Supervised Learning Framework**:
- **Data Split**: Query-based partitioning to prevent information leakage
- **Cross-Validation**: 10-fold stratified by unique queries (not examples)
- **Training Objective**: Learn query-LLM relevance mapping from labeled examples
- **Hyperparameters**: Fixed configuration optimized for ranking performance

## Experimental Design: 10-Fold Cross-Validation

### Validation Methodology
```bash
python train_full_cv_simple_progress.py
```

**Cross-Validation Protocol**:
- **Splitting Strategy**: Partition 342 unique queries into 10 folds (~34 queries per fold)
- **Data Leakage Prevention**: Ensure no query appears in both train and test within same fold
- **Training Set**: ~90% of queries (~308 queries, ~348,000 examples per fold)
- **Test Set**: ~10% of queries (~34 queries, ~38,000 examples per fold)

**Model Training Process**:
1. **Feature Creation**: Fit TF-IDF vectorizer and LLM encoder on training queries
2. **Model Fitting**: Train Random Forest on ~348,000 training examples (~11 minutes per fold)
3. **Prediction**: Generate relevance scores for ~38,000 test examples
4. **Evaluation**: Calculate nDCG@10 and MRR metrics on test queries

### Evaluation Metrics

**Primary Metrics (TREC Standard)**:
- **nDCG@10**: Normalized Discounted Cumulative Gain at rank 10
- **MRR**: Mean Reciprocal Rank of first relevant LLM

**Metric Calculation**:
1. **Query-Level Grouping**: Group predictions by query_id for ranking evaluation
2. **LLM Ranking**: Sort LLMs by predicted relevance scores (descending)
3. **Relevance Mapping**: Use ground truth qrel scores for gain calculation
4. **Aggregation**: Compute mean performance across all test queries

## Performance Results

### 10-Fold Cross-Validation Performance
| Metric | Mean | Std | 95% Confidence Interval | Min | Max |
|---------|------|-----|-------------------------|-----|-----|
| **nDCG@10** | 0.3860 | 0.0436 | [0.3006, 0.4715] | 0.3006 | 0.4570 |
| **nDCG@5** | 0.3871 | 0.0500 | [0.2891, 0.4852] | 0.2935 | 0.4786 |
| **MRR** | 0.6701 | 0.0809 | [0.5117, 0.8286] | 0.5140 | 0.8244 |

### Fold-by-Fold Detailed Results
| Fold | nDCG@10 | nDCG@5 | MRR | MSE | Queries | Train Time (min) |
|------|---------|--------|-----|-----|---------|------------------|
| 1 | 0.4063 | 0.3901 | 0.6204 | 0.0455 | 35 | 7.2 |
| 2 | 0.3579 | 0.3526 | 0.6363 | 0.0415 | 35 | 7.4 |
| 3 | 0.3405 | 0.3659 | 0.6845 | 0.0428 | 34 | 7.4 |
| 4 | 0.3006 | 0.2935 | 0.5140 | 0.0392 | 34 | 7.8 |
| 5 | 0.4570 | 0.4786 | 0.8244 | 0.0460 | 34 | 8.7 |
| 6 | 0.3710 | 0.3486 | 0.6271 | 0.0417 | 34 | 8.3 |
| 7 | 0.4289 | 0.4471 | 0.7385 | 0.0443 | 34 | 8.5 |
| 8 | 0.3932 | 0.3820 | 0.6331 | 0.0451 | 34 | 8.6 |
| 9 | 0.4215 | 0.4202 | 0.7435 | 0.0500 | 34 | 8.8 |
| 10 | 0.3834 | 0.3928 | 0.6795 | 0.0422 | 34 | 8.9 |

### Key Performance Insights

**Ranking Performance**:
- **nDCG@10 (0.386)**: Improved ranking quality with corrected qrel encoding
- **MRR (0.670)**: Average reciprocal rank of ~1.5 indicates strong top-ranking performance
- **Consistent nDCG@5 vs nDCG@10**: Performance plateau suggests top-5 ranking captures most relevance

**Model Stability**:
- **Low Variance**: Standard deviations <0.1 across all metrics demonstrate robust performance
- **Confidence Intervals**: 95% CIs provide reliable performance estimates for TREC submissions
- **Fold Consistency**: No extreme outlier folds, indicating stable learning across query distributions

**Computational Requirements**:
- **Training Time**: ~8 minutes per fold (348,000 examples) for full dataset training
- **Total Runtime**: 1.37 hours for complete 10-fold cross-validation
- **Inference Time**: Sub-second prediction for TREC submission generation

## Feature Importance Analysis

**Predictive Power Distribution**:
- **Query Text Features (TF-IDF)**: ~68% of total feature importance
- **LLM Identity**: ~32% of total feature importance

**Interpretation**:
- **Query-Driven Ranking**: Text features dominate, indicating strong query-specific expertise patterns
- **LLM-Specific Performance**: Identity features capture consistent relative performance across LLMs
- **Balanced Contribution**: Both feature types contribute significantly to ranking quality

## File Structure and Pipeline

```
├── README.md                          # This documentation
├── CLAUDE.md                          # Development guidance  
├── create_supervised_training_set.py  # Data preprocessing pipeline
├── train_full_cv_simple_progress.py   # 10-fold CV evaluation and reporting
└── data/                              # Data files
    ├── llm_dev_data.tsv               # 342 development queries
    ├── llm_dev_qrels.txt              # 386,802 relevance judgments
    ├── supervised_training_full.csv   # Processed training dataset
    └── full_evaluation_report.json    # CV results and performance metrics
```

## Requirements and Usage

### Dependencies
```bash
pip install scikit-learn pandas numpy
```

**System Requirements**:
- Python 3.8+
- 8GB+ RAM (for full dataset processing)
- Multi-core CPU recommended (for parallel Random Forest training)

### Reproduction Steps
```bash
# 1. Generate training dataset
python create_supervised_training_set.py

# 2. Run 10-fold cross-validation
python train_full_cv_simple_progress.py
```

## Limitations and Future Work

**Current Limitations**:
- **Feature Sparsity**: TF-IDF may miss semantic similarities between queries
- **Cold Start**: Limited ability to rank completely new LLMs not seen in training
- **Linear Ranking**: Assumes consistent relative LLM performance across query types

**Future Extensions**:
- **Dense Embeddings**: Replace TF-IDF with BERT/MPNet query representations
- **Meta-Learning**: Incorporate LLM architectural metadata and training details
- **Neural Ranking**: Experiment with deep learning architectures optimized for ranking
- **Multi-Task Learning**: Joint training on query classification and LLM ranking

## Baseline Framework for Future Work

This implementation serves as a baseline for developing more advanced ranking models for the TREC 2025 Million LLMs Track. The framework provides standardized evaluation protocols and performance benchmarks for comparison with improved approaches that may incorporate:

- **Discovery Dataset Integration**: Utilization of the 14,950 discovery queries with LLM responses
- **Advanced Feature Engineering**: Dense embeddings, metadata integration, and multi-modal features  
- **Sophisticated Architectures**: Neural ranking models, transformer-based approaches, and ensemble methods
- **Transfer Learning**: Pre-trained language models fine-tuned for LLM ranking tasks

