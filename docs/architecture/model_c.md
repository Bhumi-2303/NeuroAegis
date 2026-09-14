# Model C Architecture

**Parameters**: 91,858
**Composition**: CNN + Spatial GNN + Causal GRU

## Specifications
- **Input**: 23 channels × 1280 samples (5s @ 256Hz)
- **CNN Backbone**: 1 → 16 → 32 → 64 → 64 (per channel)
- **GNN Backbone**: 23 nodes, 64 features, 40 edges, Pearson graph, threshold = 0.30, 2 layers
- **Pooling**: Global mean (64) + Global max (64) = 128-dimensional spatial embedding
- **GRU**: Sequence length = 8, input = 128, hidden = 64, layers = 1, unidirectional/causal
- **Classifier**: 64 → 32 → 1

*Note: This architecture does not contain self-attention mechanisms.*
