# Two-Tower Neural Network Baseline

This directory contains a neural network baseline implementation using a Two-Tower architecture for ranking LLMs by predicted expertise on user queries.

## Architecture Overview

**Two-Tower Design**: Separate neural networks for query and LLM representations with similarity-based ranking.

### Query Tower
- **Input**: Query text processed through pre-trained sentence transformer
- **Model**: all-MiniLM-L6-v2 (384-dim embeddings)
- **Architecture**: Dense layers [384→256→128→64] with ReLU activation
- **Output**: 64-dimensional query embedding

### LLM Tower
- **Input**: LLM identifier (learned embedding)
- **Architecture**: Embedding layer + Dense layers [64→128→64]
- **Output**: 64-dimensional LLM embedding

### Similarity & Ranking
- **Similarity**: Cosine similarity between query and LLM embeddings
- **Loss**: Margin-based pairwise ranking loss
- **Training**: Batch sampling with 4 negative samples per positive

## Files

- **`evaluate_10fold_cv.py`**: 10-fold cross-validation experimental evaluation script
- **`model.py`**: Two-Tower architecture implementation
- **`data_loader.py`**: Neural data loading with positive/negative sampling
- **`evaluate_neural.py`**: Evaluation and comparison utilities
- **`requirements_neural.txt`**: Additional dependencies for neural baseline

## Usage

### 1. Install Dependencies
```bash
cd models/neural_two_tower
pip install -r requirements_neural.txt
```

### 2. Run 10-fold Cross-Validation Experimental Evaluation
```bash
python evaluate_10fold_cv.py
```

### 3. Evaluate and Compare
```bash
python evaluate_neural.py
```

## Training Configuration

- **Epochs**: 20 per fold
- **Batch Size**: 64
- **Learning Rate**: 0.001 with ReduceLROnPlateau scheduling
- **Negative Sampling**: 4 negatives per positive example
- **Margin Loss**: margin=1.0
- **Optimizer**: Adam with weight_decay=1e-5

## Expected Performance

The neural baseline is designed to:
- Match or exceed Random Forest performance (nDCG@10 ≥ 0.386)
- Leverage semantic understanding through sentence transformers
- Provide learned representations for queries and LLMs
- Demonstrate deep learning feasibility for the ranking task

## Comparison Framework

Results are directly comparable to the Random Forest baseline using:
- Same 10-fold cross-validation protocol
- Same evaluation metrics (nDCG@10, nDCG@5, MRR)
- Same corrected qrel encoding (0→0.0, 2→0.7, 1→1.0)

## Output Files

- **`../../data/results/neural_two_tower_results.json`**: Detailed CV results
- **`../../leaderboard.md`**: Model comparison leaderboard
- **`../../plots/`**: Performance comparison visualizations

This neural implementation provides a foundation for more sophisticated deep learning approaches to LLM ranking.