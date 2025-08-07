# Testing Guide for Model Reorganization

This document provides strategies for testing model implementations without running full 10-fold cross-validation experiments.

## Quick Testing Strategy (< 5 seconds)

### **Automated Test Suite**
```bash
python test_models.py
```
**What it tests**:
- ✅ Data preprocessing pipeline
- ✅ Model imports and dependencies  
- ✅ Data loading from new directory paths
- ✅ Shared utilities functionality
- ✅ Minimal model instantiation
- ✅ Leaderboard generation

### **Manual Verification Steps**

#### **1. Core Dependencies**
```bash
pip install -r requirements.txt
# Should install: pandas, numpy, scikit-learn
```

#### **2. Data Pipeline** 
```bash
cd data
python create_supervised_training_set.py
# Should complete in ~5 seconds, create supervised_training_full.csv
```

#### **3. Random Forest Minimal Test**
```bash
cd models/random_forest
python -c "
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
df = pd.read_csv('../../data/supervised_training_full.csv')
print(f'✅ Loaded {len(df)} examples from correct path')
model = RandomForestRegressor(n_estimators=2)
print('✅ Model creation works')
"
```

#### **4. Neural Two-Tower Import Test**
```bash
cd models/neural_two_tower
python -c "
from model import create_model
from data_loader import load_data
df = load_data('../../data/supervised_training_full.csv')
print(f'✅ Neural imports and data loading work: {len(df)} examples')
"
```

#### **5. Leaderboard Generation**
```bash
python generate_leaderboard.py
# Should regenerate leaderboard.md with existing results
```

## Testing Before Full Evaluation

### **Subset Testing (< 30 seconds per model)**

For more thorough testing without full CV, modify evaluation scripts temporarily:

#### **Random Forest Quick Test**
```python
# In models/random_forest/evaluate_10fold_cv.py, modify main():
def main():
    # Load just a small subset
    df = load_full_data()
    df_small = df.head(5000)  # Just 5K examples
    
    # Run 2-fold instead of 10-fold
    fold_results = cross_validate_model(df_small, n_folds=2)
    
    # Rest of analysis...
```

#### **Neural Model Quick Test**  
```python
# In models/neural_two_tower/evaluate_10fold_cv.py, modify run_cross_validation():
def run_cross_validation(df, n_folds=2, epochs=1, batch_size=32):  # Minimal config
    # Use small subset
    df_small = df.head(1000)
    # Run with 2 folds, 1 epoch for speed
```

### **Smoke Tests (< 1 minute each)**

Test each model can complete one training iteration:

#### **Random Forest Smoke Test**
```bash
cd models/random_forest
python -c "
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Load small sample
df = pd.read_csv('../../data/supervised_training_full.csv').head(1000)

# Feature extraction
tfidf = TfidfVectorizer(max_features=50)
X_text = tfidf.fit_transform(df['query_text'])

encoder = LabelEncoder()
X_llm = encoder.fit_transform(df['llm_id']).reshape(-1, 1)

# Combine features
import scipy.sparse
X = scipy.sparse.hstack([X_text, X_llm])

# Target mapping
qrel_mapping = {0: 0.0, 1: 1.0, 2: 0.7}
y = df['qrel'].map(qrel_mapping).values

# Train tiny model
model = RandomForestRegressor(n_estimators=2, max_depth=3)
model.fit(X, y)

print('✅ Random Forest smoke test passed')
"
```

## Error Diagnosis

### **Common Issues and Solutions**

#### **Import Errors**
- **Problem**: Module not found
- **Solution**: Check `requirements.txt` installation, verify file paths
- **Test**: Run import tests individually

#### **Path Errors**
- **Problem**: File not found when loading data
- **Solution**: Verify relative paths (`../../data/`) are correct
- **Test**: Run data loading tests from each model directory

#### **Missing Dependencies**
- **Problem**: Neural model dependencies missing
- **Solution**: Install model-specific requirements
```bash
cd models/neural_two_tower
pip install -r requirements_neural.txt
```

#### **Results Path Errors**
- **Problem**: Can't save results to `../../data/results/`
- **Solution**: Ensure results directory exists
```bash
mkdir -p data/results
```

## Validation Checklist

Before running full evaluations, ensure:

- [ ] ✅ `python test_models.py` passes all tests
- [ ] ✅ Data preprocessing creates correct output
- [ ] ✅ Both models can import all dependencies  
- [ ] ✅ Both models can load training data from new paths
- [ ] ✅ Results directories exist and are writable
- [ ] ✅ Leaderboard generation works with existing results
- [ ] ✅ Shared utilities import and function correctly

## Full Evaluation Testing

When ready for full evaluation, run one model first:

```bash
# Test Random Forest first (faster - ~1.5 hours)
cd models/random_forest  
python evaluate_10fold_cv.py

# If successful, test Neural Two-Tower (~7 hours)
cd ../neural_two_tower
python evaluate_10fold_cv.py

# Update leaderboard
cd ../..
python generate_leaderboard.py
```

## Continuous Testing

For ongoing development, the `test_models.py` script should be run:
- After any directory reorganization
- Before implementing new models
- After updating shared utilities
- Before running expensive full evaluations

This ensures the framework integrity without computational cost.