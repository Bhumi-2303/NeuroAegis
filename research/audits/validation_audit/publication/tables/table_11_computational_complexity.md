### Table 11: Computational Runtime, Latency, and Memory Footprint Profile

| Architecture | Parameters | Forward Latency (MPS) | Forward Latency (CPU) | FLOPs / Window | Memory Footprint | Real-Time Factor |
| --- | --- | --- | --- | --- | --- | --- |
| Model A (1D-CNN) | 173601 | 0.45 ms | 1.12 ms | 34.2 MFLOPs | 18.4 MB | 5,555x |
| Model B (CNN+GNN) | 52497 | 0.82 ms | 2.05 ms | 42.8 MFLOPs | 22.6 MB | 3,048x |
| Model C (CNN+GNN+GRU) | 91858 | 1.42 ms | 3.68 ms | 51.4 MFLOPs | 26.8 MB | 1,760x |
