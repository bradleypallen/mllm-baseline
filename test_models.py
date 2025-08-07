#!/usr/bin/env python3
"""
Quick test script to verify model evaluation scripts work correctly after reorganization.
Tests imports, data loading, and basic functionality without running full 10-fold CV.
"""

import os
import sys
import subprocess
import pandas as pd
import numpy as np
import time
from pathlib import Path


def test_data_preprocessing():
    """Test data preprocessing pipeline"""
    print("🧪 Testing data preprocessing...")
    
    # Check if supervised_training_full.csv exists
    data_file = Path('data/supervised_training_full.csv')
    if not data_file.exists():
        print("❌ supervised_training_full.csv not found - run data preprocessing first")
        return False
    
    # Load and validate data
    try:
        df = pd.read_csv(data_file)
        print(f"✅ Loaded training data: {len(df)} examples, {df.query_id.nunique()} queries")
        return True
    except Exception as e:
        print(f"❌ Error loading training data: {e}")
        return False


def test_random_forest_imports():
    """Test Random Forest model imports and basic functionality"""
    print("🧪 Testing Random Forest imports...")
    
    try:
        # Change to model directory
        os.chdir('models/random_forest')
        
        # Test imports by running Python with import statements
        import_test = """
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import ndcg_score
from sklearn.model_selection import KFold
print("✅ All Random Forest imports successful")
"""
        
        result = subprocess.run([sys.executable, '-c', import_test], 
                               capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Random Forest imports work correctly")
            return True
        else:
            print(f"❌ Random Forest import error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Random Forest test error: {e}")
        return False
    finally:
        os.chdir('../..')


def test_data_loading():
    """Test that models can load data from new paths"""
    print("🧪 Testing data loading from model directories...")
    
    try:
        # Test from random forest directory
        os.chdir('models/random_forest')
        df = pd.read_csv('../../data/supervised_training_full.csv')
        print(f"✅ Random Forest can load data: {len(df)} examples")
        os.chdir('../..')
        
        # Test from neural directory  
        os.chdir('models/neural_two_tower')
        df = pd.read_csv('../../data/supervised_training_full.csv')
        print(f"✅ Neural Two-Tower can load data: {len(df)} examples")
        os.chdir('../..')
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading error: {e}")
        return False


def test_shared_utilities():
    """Test shared evaluation utilities"""
    print("🧪 Testing shared utilities...")
    
    try:
        # Test import
        from shared.utils import calculate_metrics, get_qrel_mapping
        
        # Test with sample data
        y_true_by_query = {
            'q1': np.array([1.0, 0.7, 0.0]),
            'q2': np.array([0.0, 1.0, 0.7])
        }
        y_pred_by_query = {
            'q1': np.array([0.9, 0.8, 0.1]),
            'q2': np.array([0.2, 0.9, 0.8])
        }
        
        metrics = calculate_metrics(y_true_by_query, y_pred_by_query)
        print(f"✅ Shared utilities work: nDCG@10={metrics['ndcg_10']:.3f}")
        return True
        
    except Exception as e:
        print(f"❌ Shared utilities error: {e}")
        return False


def test_minimal_random_forest():
    """Test Random Forest with tiny subset of data"""
    print("🧪 Testing Random Forest with minimal data...")
    
    try:
        os.chdir('models/random_forest')
        
        # Create minimal test script
        test_script = """
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# Load tiny subset of data
df = pd.read_csv('../../data/supervised_training_full.csv')
df_small = df.head(1000)  # Just 1000 examples

# Test feature creation
tfidf = TfidfVectorizer(max_features=10)
query_features = tfidf.fit_transform(df_small['query_text'])

encoder = LabelEncoder()
llm_features = encoder.fit_transform(df_small['llm_id'])

# Test qrel mapping
qrel_mapping = {0: 0.0, 1: 1.0, 2: 0.7}
y = df_small['qrel'].map(qrel_mapping).values

# Test model creation (don't train)
model = RandomForestRegressor(n_estimators=2, max_depth=2)
print("✅ Random Forest minimal test passed")
"""
        
        result = subprocess.run([sys.executable, '-c', test_script], 
                               capture_output=True, text=True, cwd=os.getcwd())
        
        if result.returncode == 0:
            print("✅ Random Forest minimal functionality works")
            return True
        else:
            print(f"❌ Random Forest minimal test failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Random Forest minimal test error: {e}")
        return False
    finally:
        os.chdir('../..')


def test_leaderboard_generation():
    """Test leaderboard generation with existing results"""
    print("🧪 Testing leaderboard generation...")
    
    try:
        result = subprocess.run([sys.executable, 'generate_leaderboard.py'], 
                               capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Leaderboard generation works")
            print(f"Generated leaderboard with existing results")
            return True
        else:
            print(f"❌ Leaderboard generation failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Leaderboard test error: {e}")
        return False


def main():
    """Run all quick tests"""
    print("🚀 TESTING MODEL REORGANIZATION")
    print("=" * 50)
    
    start_time = time.time()
    tests_passed = 0
    total_tests = 6
    
    # Run tests
    tests = [
        test_data_preprocessing,
        test_shared_utilities, 
        test_data_loading,
        test_random_forest_imports,
        test_minimal_random_forest,
        test_leaderboard_generation
    ]
    
    for test_func in tests:
        if test_func():
            tests_passed += 1
        print("-" * 30)
    
    # Summary
    elapsed = time.time() - start_time
    print(f"\n📊 TEST SUMMARY")
    print(f"Tests passed: {tests_passed}/{total_tests}")
    print(f"Time elapsed: {elapsed:.1f} seconds")
    
    if tests_passed == total_tests:
        print("🎉 ALL TESTS PASSED - Reorganization successful!")
        return True
    else:
        print(f"⚠️  {total_tests - tests_passed} tests failed - check issues above")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)