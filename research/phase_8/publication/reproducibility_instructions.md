# NeuroAegis: Full Reproducibility Protocol & Instructions

## 1. Environment Setup
```bash
git clone https://github.com/Bhumi-2303/NeuroAegis.git
cd NeuroAegis
git checkout 7b9f06f4b01e27fe9dfe86e48374160f982a737a

python3 -m venv .venv
source .venv/bin/activate
pip install torch numpy pandas scipy scikit-learn openpyxl matplotlib
```

## 2. Decompress Master Window Index
```bash
gzip -dk research/data/manifests/chbmit_window_index.csv.gz
sha256sum research/data/manifests/chbmit_window_index.csv
# Expected SHA-256: f76dddb19fa2c24b3af24b989168b12465512b650c5e6d84bb0125a6d666302c
```

## 3. Cryptographic Checkpoint Verification
```bash
sha256sum research/phase_4b/frozen_cnn_gnn_gru.pt
# Expected SHA-256: 2ec84897c39d31d68cfbf1f7e5c8708d5073fe19450576bf4f9132929c1832ca

sha256sum research/phase_4a/frozen_graph_adjacency.csv
# Expected SHA-256: 062c6aaffdc32ea1b9b5b97c82c224d40e0ab3b7e9b2afad5fd5b3e5f52db48e
```

## 4. Execute Full Audit & Verification Suite
```bash
python research/phase_8/run_phase8_audit.py
python research/phase_8/generate_phase8_figures.py
python research/phase_8/generate_phase8_workbook.py
python -m unittest research/phase_8/tests/test_final_research_audit.py -v
```
