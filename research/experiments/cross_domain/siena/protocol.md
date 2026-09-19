# NeuroAegis Experiment 6 — Zero-Shot Cross-Domain Protocol

## Invariant Scientific Principles
1. **Zero Retraining**: Model C weights (91,858 parameters) are strictly frozen.
2. **Zero Siena Label Exposure**: No Siena annotations were used for channel selection, threshold tuning, preprocessing design, or postprocessing tuning.
3. **Harmonized Domain Conversion**:
   - Anti-aliased Chebyshev Decimation: $512\text{ Hz} \to 256\text{ Hz}$
   - Bandpass Filter: $0.5 - 40.0\text{ Hz}$ zero-phase Butterworth SOS
   - European Notch Filter: $50.0\text{ Hz}$ zero-phase IIR ($Q=30.0$)
   - Recording-local Z-Score Normalization: $\frac{x - \mu}{\sigma}$ per channel
4. **Inference Streaming**: $5.0\text{s}$ windows (stride $2.5\text{s}$) $\to$ 128-D spatial embeddings $\to$ causal $L=8$ GRU sequence $\to$ linear logit $\to$ sigmoid $\to$ $\tau=0.50$.