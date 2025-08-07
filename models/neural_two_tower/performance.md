# Neural Two-Tower Performance Results

## 10-Fold Cross-Validation Results

**Two-Tower Neural Network**:
- **nDCG@10**: 0.402 ± 0.028 
- **nDCG@5**: 0.413 ± 0.034
- **MRR**: 0.676 ± 0.057
- **Runtime**: 6.95 hours (25,023 seconds)

## Training Configuration
- **Architecture**: Dual encoder with sentence transformer
- **Query Model**: all-MiniLM-L6-v2 (384→256→128→64)
- **LLM Model**: Learned embeddings (64→128→64)
- **Epochs**: 20 per fold
- **Batch Size**: 64
- **Learning Rate**: 0.001 with ReduceLROnPlateau
- **Loss**: Margin-based pairwise ranking (margin=1.0)
- **Negative Sampling**: 4 negatives per positive

## Comparison with Random Forest
- **nDCG@10 Improvement**: +4.2% (0.402 vs 0.386)
- **Lower Variance**: More consistent across folds
- **Training Time**: ~5x longer than Random Forest
- **Better Semantic Understanding**: Leverages sentence transformers