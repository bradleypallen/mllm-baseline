# Neural Architecture Evolution for LLM Ranking

*A comprehensive analysis of the evolution from two-tower baseline to cross-encoder architectures for automated LLM selection*

## Abstract

This document presents the systematic evolution of neural architectures for ranking Large Language Models (LLMs) by predicted expertise on user queries. Through four architectural variants—from baseline two-tower to advanced cross-encoder—we demonstrate the progression from simple similarity learning to sophisticated multi-head attention with curriculum learning. Our findings reveal a performance plateau around nDCG@10≈0.43, with Tier 2's multi-head attention achieving optimal efficiency-performance balance, while Tier 3's cross-encoder encounters diminishing returns due to feature sparsity constraints.

## Evolution of Neural Architecture for LLM Ranking: From Two-Tower Baseline to Cross-Encoder

### Core Two-Tower Architecture

The foundation of our neural approach builds upon the established two-tower architecture, a dual-encoder paradigm that has proven effective for large-scale retrieval tasks. The architecture consists of two independent neural networks: a Query Tower that processes user queries through a pre-trained sentence transformer (all-MiniLM-L6-v2) followed by dense layers [384→256→128→64], and an LLM Tower that learns embeddings for the 1,131 LLMs through an embedding layer followed by dense transformations [64→128→64]. The models are trained using margin-based pairwise ranking loss, optimizing for cosine similarity between query and LLM embeddings. This baseline approach achieved nDCG@10=0.4022 with a runtime of 6.95 hours, establishing a strong foundation that leverages semantic understanding from pre-trained transformers while learning task-specific LLM representations.

### Tier 1 Enhancements: Contrastive Learning and Architectural Refinements

Our first major enhancement, designated Enhanced Tier 1, introduced three critical improvements motivated by representation learning theory and training efficiency concerns. First, we replaced the margin-based ranking loss with InfoNCE (contrastive loss), providing better gradient signals and more stable training dynamics. Second, we expanded the embedding dimensionality from 64D to 128D across both towers, increasing representational capacity while maintaining computational feasibility. Third, we implemented hard negative mining capability, enabling the model to focus on challenging examples during training. These refinements, combined with architectural improvements to the dense layers [384→256→192→128] in the query tower, resulted in nDCG@10=0.4256 with significantly faster convergence (2.95 hours), demonstrating that principled loss function design and strategic capacity increases yield substantial performance gains.

### Tier 2: Multi-Head Attention and Advanced Training Strategies

The Tier 2 enhancement represented our most significant architectural innovation, introducing multi-head query attention to capture diverse semantic aspects of user queries. Motivated by the success of attention mechanisms in transformer architectures, we implemented 3 specialized attention heads with emergent specialization—rather than hand-engineering different head functions, we allowed training to naturally differentiate head representations. Each head processes the sentence transformer output through dedicated pathways [384→96→96] before fusion [288→384]. This multi-head processing is combined with active hard negative mining (60% hard, 40% easy negatives) and head diversity regularization that encourages complementary learning across heads. The training curriculum activates hard negatives after epoch 2, providing stable early training followed by challenging optimization. Despite being CPU-optimized to resolve MPS GPU compatibility issues, Tier 2 achieved our best performance: nDCG@10=0.4306 in just 3.11 hours, establishing the optimal balance of architectural sophistication and computational efficiency.

### Tier 3: Cross-Encoder with Joint Query-LLM Encoding

Our final architectural exploration, Tier 3, abandoned the two-tower paradigm entirely in favor of a cross-encoder approach using DistilBERT as the backbone. This design processes queries and LLM IDs jointly through transformer attention, allowing direct interaction modeling rather than independent encoding followed by similarity computation. The architecture tokenizes query text, processes it through DistilBERT to obtain [CLS] token representations (768D), concatenates these with learned LLM embeddings (192D), and applies a classification head [960→768→384→1] for direct relevance prediction using BCE loss. While theoretically superior due to cross-attention capabilities, the practical results were sobering: nDCG@10=0.4259 with a training time of 21.92 hours. This represents competitive but not superior performance at 7x the computational cost, suggesting that the theoretical advantages of cross-attention are limited by our feature-sparse environment where only LLM IDs are available.

### Comparative Analysis and Architectural Insights

The evolution reveals several key insights about neural architecture design for ranking tasks. First, we observe a performance plateau around nDCG@10≈0.43, with the top three models clustered within 0.5%, suggesting that architectural sophistication encounters diminishing returns given our feature constraints. Second, Tier 2's multi-head attention with hard negative mining achieves the optimal efficiency-performance balance, outperforming both simpler two-tower variants and the computationally expensive cross-encoder. Third, the failure of Tier 3 to substantially outperform Tier 2 highlights a critical limitation: cross-attention's advantages are most pronounced when rich feature interactions are possible, but with only sparse LLM ID information, the added complexity doesn't translate to meaningful performance gains. The results strongly suggest that for this task, the bottleneck lies in feature availability rather than architectural capacity, pointing toward data enrichment strategies as the most promising direction for future performance improvements.

## Architectural Diagrams

### Baseline Two-Tower (nDCG@10=0.4022)

```
Query: "What is ML?"                    LLM_ID: 42
        |                                  |
    [Sentence Transformer]            [Embedding Layer]
    all-MiniLM-L6-v2                      |
        |                            [64] LLM Embed
    [384] Query Embed                      |
        |                            [Linear 64→128]
    [Linear 384→256]                       |
        |                               [ReLU]
    [ReLU + Dropout]                       |
        |                            [Linear 128→64]
    [Linear 256→128]                       |
        |                            [64] LLM Output
    [ReLU + Dropout]                       |
        |                                  |
    [Linear 128→64]              ┌─────────┘
        |                        │
    [64] Query Output            │
        |                        │
        └────────┬───────────────┘
                 │
            [Cosine Similarity]
                 │
            Margin Ranking Loss
```

### Enhanced Tier 1 (nDCG@10=0.4256)

```
Query: "What is ML?"                    LLM_ID: 42
        |                                  |
    [Sentence Transformer]            [Embedding Layer]
    all-MiniLM-L6-v2                      |
        |                            [128] LLM Embed  ←── Expanded!
    [384] Query Embed                      |
        |                            [Linear 128→192] ←── Enhanced!
    [Linear 384→256]                       |
        |                               [ReLU]
    [ReLU + Dropout]                       |
        |                            [Linear 192→128]
    [Linear 256→192] ←── Added Layer       |
        |                            [128] LLM Output ←── Expanded!
    [ReLU + Dropout]                       |
        |                                  |
    [Linear 192→128]                       |
        |                        ┌─────────┘
    [128] Query Output ←── Expanded!       │
        |                        │
        └────────┬───────────────┘
                 │
            [Cosine Similarity]
                 │
            InfoNCE Loss ←── Contrastive!
                 │
        + Hard Negative Mining ←── New!
```

### Tier 2 CPU Optimized (nDCG@10=0.4306) - CHAMPION

```
Query: "What is ML?"                         LLM_ID: 42
        |                                       |
    [Sentence Transformer]                 [Embedding Layer]
    all-MiniLM-L6-v2                           |
        |                                 [128] LLM Embed
    [384] Query Embed                          |
        |                                 [Linear 128→192]
    ┌───┴────┬────────┐ ←── Multi-Head!        |
    │        │        │                    [ReLU + Dropout]
[Head 1] [Head 2] [Head 3]                     |
384→96   384→96   384→96                 [Linear 192→128]
  ReLU     ReLU     ReLU                       |
  Drop     Drop     Drop                  [128] LLM Output
384→96   384→96   384→96                       |
    │        │        │                        |
    └───┬────┴────┬───┘               ┌────────┘
        │         │                   │
    [Concatenate 288]                 │
        |                             │
    [Linear 288→384] ←── Fusion       │
        |                             │
    [ReLU + Dropout]                  │
        |                             │
    [Linear 384→256]                  │
        |                             │
    [Linear 256→192]                  │
        |                             │
    [Linear 192→128]                  │
        |                             │
    [128] Query Output                │
        |                             │
        └──────┬──────────────────────┘
               │
          [Cosine Similarity]
               │
    InfoNCE + Diversity Loss ←── Advanced!
               │
    + Active Hard Neg Mining (60/40) ←── Curriculum!
    + Head Diversity Regularization
```

### Tier 3 Cross-Encoder (nDCG@10=0.4259)

```
Query: "What is ML?"    +    LLM_ID: 42
        |                       |
    [Tokenizer]            [LLM Embedding]
        |                       |
[input_ids, attention_mask] [192] LLM Embed
        |                       |
    [DistilBERT Transformer] ←── Joint Processing!
        |                       |
[batch, seq_len, 768]          |
        |                       |
    [[CLS] Token]               |
        |                       |
    [768] Query Repr            |
        |                       |
        └───────┬───────────────┘
                │
        [Concatenate 960] ←── Joint Encoding!
                │
        [Linear 960→768]
                │
        [ReLU + Dropout]
                │
        [Linear 768→384]
                │
        [ReLU + Dropout]
                │
        [Linear 384→1] ←── Direct Prediction!
                │
        [Relevance Score]
                │
        BCE Loss ←── Binary Classification!

Note: 21.92 hours (7x slower than Tier 2)
```

### Architecture Evolution Summary

```
Evolution Path:
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Baseline   │ -> │ Enhanced T1 │ -> │   Tier 2    │ -> │   Tier 3    │
│  Two-Tower  │    │   (128D +   │    │ Multi-Head  │    │Cross-Encoder│
│             │    │ Contrastive)│    │ Attention   │    │   (Joint)   │
│ nDCG: 0.402 │    │ nDCG: 0.426 │    │ nDCG: 0.431 │    │ nDCG: 0.426 │
│  6.95h      │    │   2.95h     │    │   3.11h     │    │  21.92h     │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
      |                    |                   |                   |
   Similarity         Contrastive        Multi-Head           Joint
   Learning           + Capacity         + Mining           Attention
```

**Key Insight**: Tier 2's multi-head architecture achieves optimal performance/efficiency balance, while Tier 3's sophistication hits diminishing returns due to sparse LLM features.

## Glossary

### Core Components

**[Sentence Transformer]** - Pre-trained encoder model (all-MiniLM-L6-v2) that converts text queries into dense 384-dimensional semantic embeddings. Uses BERT-like attention to capture contextual meaning.

**[Embedding Layer]** - Learnable lookup table that maps discrete LLM IDs (0-1130) to dense vector representations. Allows the model to learn task-specific LLM characteristics.

**[Linear X→Y]** - Fully connected (dense) layer transforming input dimension X to output dimension Y through learned weight matrix and bias term.

**[ReLU]** - Rectified Linear Unit activation function: f(x) = max(0, x). Introduces non-linearity while being computationally efficient.

**[Dropout]** - Regularization technique that randomly sets a fraction of input units to 0 during training to prevent overfitting.

### Architecture Types

**Two-Tower** - Dual-encoder architecture where queries and LLMs are processed independently through separate "towers" (neural networks), then combined via similarity computation.

**Cross-Encoder** - Joint architecture where query and LLM information are processed together through shared transformer layers, allowing direct interaction modeling.

**Multi-Head Attention** - Parallel attention mechanisms that learn different aspects of the input. Each "head" focuses on different semantic relationships.

### Training & Loss Functions

**Margin Ranking Loss** - Pairwise loss function: max(0, margin - (positive_score - negative_score)). Ensures positive examples score higher than negatives by at least the margin.

**InfoNCE Loss** - Information Noise Contrastive Estimation. Contrastive learning objective that maximizes agreement between positive pairs while minimizing agreement with negatives using temperature-scaled softmax.

**BCE Loss** - Binary Cross-Entropy loss for binary classification. Used in cross-encoder to directly predict relevance probability.

**Contrastive Loss** - Learning paradigm that brings similar examples closer in embedding space while pushing dissimilar examples apart.

### Advanced Training Techniques

**Hard Negative Mining** - Training strategy that selects challenging negative examples (those the model currently ranks incorrectly high) to improve learning efficiency and model robustness.

**Curriculum Learning** - Training approach that gradually increases task difficulty. In our case, starting with random negatives then introducing hard negatives after initial epochs.

**Head Diversity Regularization** - Loss term that encourages different attention heads to learn complementary representations rather than redundant ones.

**Active Hard Negative Mining** - Dynamic selection of hard negatives during training based on current model predictions, updated periodically.

### Model Architecture Elements

**[CLS] Token** - Special classification token from transformer models (DistilBERT) that aggregates sequence-level information for classification tasks.

**Query Tower** - Neural network branch that processes user queries into embeddings for similarity matching.

**LLM Tower** - Neural network branch that processes LLM identifiers into embeddings for similarity matching.

**Fusion Layer** - Component that combines multiple representations (e.g., multi-head outputs) into a single unified representation.

**Classification Head** - Final layers that transform learned representations into task-specific outputs (rankings, probabilities).

### Similarity & Evaluation Metrics

**Cosine Similarity** - Measure of similarity between two vectors: (A·B)/(||A||×||B||). Range [-1,1], with 1 indicating identical direction.

**nDCG@10** - Normalized Discounted Cumulative Gain at rank 10. Evaluation metric that rewards relevant items ranked higher, with logarithmic position discounting.

**MRR** - Mean Reciprocal Rank. Average of 1/rank for the first relevant item across all queries. Measures how quickly users find relevant results.

### Technical Specifications

**384D, 128D, 96D** - Dimensional sizes of vector representations. Higher dimensions provide more representational capacity but increase computational cost.

**Batch Size** - Number of training examples processed simultaneously. Larger batches provide better gradient estimates but require more memory.

**Temperature Scaling** - Parameter in contrastive learning that controls the "sharpness" of the probability distribution. Lower values create more confident predictions.

**60/40 Hard/Easy** - Ratio indicating 60% of negative examples are hard negatives (challenging) and 40% are easy negatives (random sampling).

### Performance Indicators

**Runtime** - Total training time for 10-fold cross-validation. Indicates computational efficiency and practical usability.

**7x slower** - Comparative performance indicating one approach takes 7 times longer than the baseline for training.

**Performance Plateau** - Point where architectural improvements yield diminishing returns, suggesting feature limitations rather than model capacity constraints.

## Performance Summary

| Architecture | nDCG@10 | nDCG@5 | MRR | Runtime | Key Innovation |
|-------------|---------|--------|-----|---------|----------------|
| **Baseline Two-Tower** | 0.4022 ± 0.028 | 0.4135 ± 0.034 | 0.6761 ± 0.057 | 6.95h | Semantic similarity learning |
| **Enhanced Tier 1** | 0.4256 ± 0.050 | 0.4287 ± 0.056 | 0.7113 ± 0.074 | 2.95h | Contrastive learning + 128D |
| **Tier 2 (Champion)** | 0.4306 ± 0.055 | 0.4347 ± 0.058 | 0.7263 ± 0.070 | 3.11h | Multi-head attention + mining |
| **Tier 3 Cross-Encoder** | 0.4259 ± 0.049 | 0.4378 ± 0.051 | 0.7141 ± 0.076 | 21.92h | Joint transformer encoding |

### Key Findings

1. **Performance Plateau**: Top models cluster within 0.5% nDCG@10, suggesting feature limitations
2. **Efficiency Champion**: Tier 2 achieves best performance/time ratio at 3.11 hours
3. **Diminishing Returns**: Tier 3's 7x computational cost yields minimal gains
4. **Architectural Insights**: Multi-head attention with curriculum learning outperforms complex cross-attention
5. **Future Directions**: Feature enrichment more promising than architectural complexity

## Implementation Details

All models implemented with:
- **Dataset**: TREC 2025 Million LLMs Track (342 queries, 1,131 LLMs, 386,801 examples)
- **Evaluation**: 10-fold cross-validation with query-based splitting
- **Hardware**: Apple M3 with 24GB unified memory
- **Framework**: PyTorch with sentence-transformers and transformers libraries

Code available in the [mllm-baseline repository](https://github.com/bradleypallen/mllm-baseline) under `models/neural_two_tower/`.

---

*This analysis demonstrates the systematic exploration of neural architectures for LLM ranking, establishing clear performance benchmarks and revealing the optimal balance between architectural sophistication and computational efficiency.*