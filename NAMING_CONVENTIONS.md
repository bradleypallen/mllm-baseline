# Model Implementation Naming Conventions

This document establishes consistent naming conventions for implementing new models in the TREC 2025 Million LLMs Track baseline framework.

## Required Files for Each Model

### 1. **`evaluate_10fold_cv.py`** (REQUIRED)
- **Purpose**: Main experimental evaluation script that performs 10-fold cross-validation
- **Function**: Complete end-to-end evaluation including training, validation, and results generation
- **Output**: Saves results to `../../data/results/[model_name]_results.json` in standardized format
- **Usage**: `python evaluate_10fold_cv.py` (run from model directory)

### 2. **`README.md`** (REQUIRED)  
- **Purpose**: Model-specific documentation
- **Content**: Architecture description, usage instructions, performance results
- **Format**: Follow existing model README templates

### 3. **Model-Specific Architecture Files** (OPTIONAL)
- **Examples**: `model.py`, `architecture.py`, `network.py`
- **Purpose**: Define model architecture, layers, components
- **Usage**: Imported by `evaluate_10fold_cv.py`

### 4. **Data Loading Files** (OPTIONAL)
- **Examples**: `data_loader.py`, `preprocessing.py`, `dataset.py`
- **Purpose**: Custom data loading, preprocessing, batch generation
- **Usage**: Used by `evaluate_10fold_cv.py` for model-specific data handling

### 5. **Requirements File** (CONDITIONAL)
- **Name**: `requirements_[model_name].txt`
- **Purpose**: Model-specific dependencies beyond base framework requirements
- **Example**: `requirements_neural.txt` for neural models requiring PyTorch

## Directory Structure Convention

```
models/[model_name]/
├── evaluate_10fold_cv.py          # REQUIRED: Main evaluation script
├── README.md                      # REQUIRED: Model documentation
├── [model_specific_files].py      # OPTIONAL: Architecture/data files
├── requirements_[model_name].txt  # CONDITIONAL: Additional dependencies
└── [other_supporting_files]       # OPTIONAL: Utilities, configs, etc.
```

## Standardized Workflow

### 1. **Data Generation** (Run Once)
```bash
cd data
python create_supervised_training_set.py
```

### 2. **Model Evaluation** (Per Model)
```bash
cd models/[model_name]
python evaluate_10fold_cv.py
```

### 3. **Leaderboard Update** (After New Results)
```bash
cd ../..  # Back to repository root
python generate_leaderboard.py
```

## Implementation Guidelines

### `evaluate_10fold_cv.py` Requirements
- **Import shared utilities**: Use `from shared.utils import calculate_metrics, save_standardized_results`
- **Load training data**: Use `pd.read_csv('../../data/supervised_training_full.csv')`
- **Query-based CV**: Split by unique queries, not individual examples
- **Standardized metrics**: Calculate nDCG@10, nDCG@5, MRR using shared functions
- **Results output**: Save to `../../data/results/[model_name]_results.json`
- **Qrel encoding**: Use standard mapping (0→0.0, 1→1.0, 2→0.7)

### Results Format
All models must output results in the standardized JSON format (see CLAUDE.md for full specification).

### Performance Reporting
- **Runtime tracking**: Include total training time in results
- **Statistical rigor**: Report mean ± std with 95% confidence intervals
- **Fold-by-fold details**: Include individual fold results for transparency

## Examples

### Existing Models
- `models/random_forest/evaluate_10fold_cv.py` - Traditional ML baseline
- `models/neural_two_tower/evaluate_10fold_cv.py` - Deep learning baseline

### Future Model Examples
- `models/xgboost/evaluate_10fold_cv.py` - Gradient boosting approach
- `models/transformer_ranker/evaluate_10fold_cv.py` - Transformer-based ranking
- `models/ensemble/evaluate_10fold_cv.py` - Multi-model ensemble

## Benefits of Consistent Naming

1. **Predictable Structure**: Developers know exactly what to expect in each model directory
2. **Easy Automation**: Scripts can automatically find and run evaluations across models
3. **Clear Purpose**: File names clearly indicate their experimental evaluation function
4. **Maintainability**: Consistent patterns make the codebase easier to maintain
5. **Documentation**: Self-documenting structure with clear naming conventions

## Getting Started with New Models

1. **Create directory**: `mkdir -p models/your_model_name`
2. **Copy template**: Use existing `evaluate_10fold_cv.py` as starting point
3. **Implement model**: Create architecture and data loading files as needed
4. **Follow standards**: Use shared utilities and standardized output format
5. **Document**: Create comprehensive README.md following existing templates
6. **Test**: Ensure evaluation runs successfully and produces valid results
7. **Update leaderboard**: Run `python generate_leaderboard.py` to include new results

This naming convention ensures all models follow consistent patterns while allowing flexibility for model-specific implementation details.