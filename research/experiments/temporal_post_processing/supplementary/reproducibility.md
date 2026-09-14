# NeuroAegis: Reproducibility Guide & Forensic Manifest

This document specifies the exact environment, frozen software artifacts, cryptographic checksums, and execution instructions required to fully reproduce all experimental results, figures, tables, and audits presented in the manuscript.

---

## 1. Cryptographic Artifact Checksums

| Artifact Category | Local File Path | Cryptographic Hash (SHA-256) | Authoritative Parameter / Value |
| :--- | :--- | :--- | :--- |
| **Model Weights** | `artifacts/checkpoints/frozen_cnn_gnn_gru.pt` | `2ec84897c39d31d68cfbf1f7e5c8708d5073fe19450576bf4f9132929c1832ca` | Total Parameters: 91,858 |
| **Spatial Graph** | `research/phase_4a/frozen_graph_adjacency.csv` | `062c6aaffdc32ea1b9b5b97c82c224d40e0ab3b7e9b2afad5fd5b3e5f52db48e` | $\theta = 0.30$, 23 Nodes, 40 Edges |
| **Test Predictions** | `research/phase_4b/results/final_test_predictions.csv` | `611598f8b50f7572ceea3193e25d262d10738eb2c1f93f619b02353fe66245e3` | 219,909 rows, TP=534, FP=399 |
| **Authoritative JSON**| `research/phase_8/final_results/authoritative_final_metrics.json` | `5cbf603a115456f9175c5e8964789508bc0a1e05d0337d117cb83ae2ae42ebba` | Primary Freeze Record |

---

## 2. Software & Hardware Environment

- **Operating System:** macOS Darwin 24.3.0 (Apple Silicon ARM64)
- **Python Runtime:** Python 3.11.11 (`.venv`)
- **Key Dependencies:**
  - `torch == 2.6.0` (MPS acceleration enabled)
  - `numpy == 2.2.3`
  - `scipy == 1.15.2`
  - `pandas == 2.2.3`
  - `scikit-learn == 1.6.1`
  - `matplotlib == 3.10.1`
  - `seaborn == 0.13.2`
  - `openpyxl == 3.1.5`

---

## 3. End-to-End Reproduction Steps

### Step 1: Environment Activation
```bash
source .venv/bin/activate
export MPLCONFIGDIR=/tmp/matplotlib_cache
```

### Step 2: Verify Cryptographic Integrity
```bash
python3 -c "
import hashlib
def verify(path, expected):
    h = hashlib.sha256(open(path, 'rb').read()).hexdigest()
    assert h == expected, f'Hash mismatch for {path}: got {h}'
    print(f'PASS: {path}')

verify('artifacts/checkpoints/frozen_cnn_gnn_gru.pt', '2ec84897c39d31d68cfbf1f7e5c8708d5073fe19450576bf4f9132929c1832ca')
verify('research/phase_4a/frozen_graph_adjacency.csv', '062c6aaffdc32ea1b9b5b97c82c224d40e0ab3b7e9b2afad5fd5b3e5f52db48e')
"
```

### Step 3: Execute Authoritative Cross-Phase Audit
```bash
python3 research/phase_8/tests/test_final_research_audit.py
```
*Expected output: Ran 17 tests in ~0.5s -> OK (100% passing).*

### Step 4: Regenerate Publication Figures (300 DPI)
```bash
python3 research/phase_8/generate_figures.py
```
*Generates 18 high-resolution figures in `research/phase_8/figures/`.*

### Step 5: Regenerate Excel Evidence Master & Tables
```bash
python3 research/phase_8/generate_excel_and_tables.py
```
*Creates `research/phase_8/NeuroAegis_Final_Research_Results.xlsx` containing all 23 sheets.*

### Step 6: Validate Manuscript Claims Against Ground Truth
```bash
python3 -m unittest research/phase_9/tests/test_paper_integrity.py
```
*Verifies 0.0 drift between the manuscript narrative and lowest-level machine artifacts.*
