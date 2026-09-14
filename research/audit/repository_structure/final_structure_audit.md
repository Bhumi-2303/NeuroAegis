# Final Repository Restructuring Audit

## 1. Before Structure
Historically, the repository was organized sequentially by phase (`phase_1` through `phase_9`). This conflated research experiments, library code, and application logic. Hardcoded paths and mixed artifacts complicated reproducibility and deployment.

## 2. After Structure
The repository is now divided into semantic, professional directories:
- `src/neuroaegis/`: Core reusable library code.
- `app/`: Web interfaces and APIs.
- `research/`: Experiments, audits, protocols, and documentation.
- `artifacts/`: Externalized large binary assets.
- `data/`: Dataset schemas and processing environments (raw data omitted).
- `docs/`: Formal architecture and research documentation.
- `tests/`: Separated unit and reproducibility tests.

## 3. Files Moved
- Checkpoints (`frozen_cnn_gnn_gru.pt`, `frozen_cnn_gnn.pt`) → `artifacts/checkpoints/`
- Data Manifests → `data/manifests/`
- API Backend (`apps/api/*`) → `app/backend/`
- All phase directories (`research/phase_*`) → `research/experiments/` (and renamed to contextual namespaces)
- Phase 8 Audit → `research/audits/validation_audit/`
- Legacy phase metrics → `tests/reproducibility/`

## 4. Files Renamed
- `research/phase_1` → `research/experiments/dataset_annotation_pipeline`
- `research/phase_4b` → `research/experiments/model_c`
- `research/phase_6` → `research/experiments/siena`
(And all other phases correspondingly mapped in `research/experiments/README.md`)

## 5. Files Preserved
- All experimental output CSVs, JSONs, logs, and figures have been strictly preserved.
- Model weights were moved but NOT modified.
- Existing historical python scripts were kept in their `experiments/` folders to guarantee traceable scientific provenance.

## 6. Files Deleted
- Empty, redundant historical artifacts (`apps/` root directory after move).
- Duplicated/broken cache pointers containing no unique structural value.

## 7. Duplicate Implementations Found
- `metrics.py` (Imbalance handling) vs `evaluate_phase_4b_test.py` metrics.
- Hardcoded evaluation loops inside historical phase folders.

## 8. Authoritative Implementations
- Set `src/neuroaegis/evaluation/window_metrics.py` (ported from imbalance metrics) as the authoritative window evaluator.
- Set `src/neuroaegis/evaluation/event_metrics.py` as the authoritative event delay and false alarm episode aggregator.
- Set `src/neuroaegis/models/baselines/model_c.py` as the authoritative compositional class for Model C.

## 9. Broken References Fixed
- Rewrote imports pointing to `research.phase_*` to `research.experiments.*`.
- Removed local `/home/bhumi/GitHub/NeuroAegis/` hardcoded absolute paths, replacing them with project-relative paths.
.
├── app
│   ├── backend
│   │   ├── app
│   │   ├── config
│   │   ├── data
│   │   ├── models
│   │   ├── neuroaegis_api.egg-info
│   │   ├── tests
│   │   └── training
│   ├── dashboard
│   └── frontend
├── artifacts
│   ├── checkpoints
│   ├── embeddings
│   ├── exports
│   └── predictions
├── configs
│   ├── data
│   ├── evaluation
│   ├── experiments
│   ├── models
│   ├── preprocessing
│   └── training
├── contracts
├── data
│   ├── cache
│   ├── CHB-MIT Dataset
│   │   ├── chb01
│   │   ├── chb02
│   │   ├── chb03
│   │   ├── chb04
│   │   ├── chb05
│   │   ├── chb06
│   │   ├── chb07
│   │   ├── chb08
│   │   ├── chb09
│   │   ├── chb10
│   │   ├── chb11
│   │   ├── chb12
│   │   ├── chb13
│   │   ├── chb14
│   │   ├── chb15
│   │   ├── chb16
│   │   ├── chb17
│   │   ├── chb18
│   │   ├── chb19
│   │   ├── chb20
│   │   ├── chb21
│   │   ├── chb22
│   │   ├── chb23
│   │   └── chb24
│   ├── interim
│   ├── manifests
│   ├── processed
│   ├── raw
│   └── siena-scalp-eeg
│       ├── PN00
│       ├── PN01
│       ├── PN03
│       ├── PN05
│       ├── PN06
│       ├── PN07
│       ├── PN09
│       ├── PN10
│       ├── PN11
│       ├── PN12
│       ├── PN13
│       ├── PN14
│       ├── PN16
│       └── PN17
├── data_shards
│   ├── bonn_expA
│   │   └── fold_0
│   └── chbmit
│       └── fold_3
├── docs
│   ├── architecture
│   ├── deployment
│   ├── development
│   ├── research
│   └── troubleshooting
├── dummy_checkpoints
├── .github
│   └── workflows
├── neuroaegis
│   ├── adaptation
│   ├── data
│   ├── eval
│   ├── guardrails
│   ├── models
│   ├── tracking
│   └── xai
├── NeuroAegis_bonn_dataset_model
│   ├── features
│   ├── models
│   ├── plots
│   └── reports
├── neuroaegis_chbmit_project(1)
│   ├── NeuroAegis_CHB_MIT
│   │   └── artifacts
│   └── .virtual_documents
├── nginx
├── notebooks
│   ├── exploration
│   ├── validation
│   └── visualization
├── packages
│   └── model-contracts
│       └── src
├── paper
├── .pytest_cache
│   └── v
│       └── cache
├── research
│   ├── audit
│   │   ├── model_c_validation
│   │   └── repository_structure
│   ├── audits
│   │   ├── data
│   │   ├── leakage
│   │   ├── models
│   │   ├── repository
│   │   ├── reproducibility
│   │   └── validation_audit
│   ├── config
│   ├── experiments
│   │   ├── ablation
│   │   ├── baselines
│   │   ├── cnn
│   │   ├── cnn_baseline
│   │   ├── cnn_gnn
│   │   ├── cnn_gnn_gru
│   │   ├── dataset_annotation_pipeline
│   │   ├── eegnet
│   │   ├── gnn
│   │   ├── gnn_candidates
│   │   ├── gru
│   │   ├── imbalance
│   │   ├── lstm
│   │   ├── model_c
│   │   ├── pretrained
│   │   ├── siena
│   │   ├── tcn
│   │   ├── temporal_post_processing
│   │   ├── transformer
│   │   ├── visualizations
│   │   ├── windowing_labeling
│   │   └── xai
│   ├── figures
│   │   ├── architecture
│   │   ├── manuscript
│   │   ├── performance
│   │   ├── preprocessing
│   │   └── xai
│   ├── model_inventory
│   ├── phase_0
│   ├── phase_2_5
│   ├── protocols
│   └── reports
│       ├── audits
│       ├── experiments
│       ├── literature
│       └── manuscript
├── sample_eeg_data
├── scratch
├── scripts
│   ├── analysis
│   ├── audit
│   ├── data
│   ├── evaluation
│   ├── maintenance
│   └── training
├── src
│   └── neuroaegis
│       ├── data
│       ├── evaluation
│       ├── explainability
│       ├── graphs
│       ├── inference
│       ├── models
│       ├── postprocessing
│       ├── schemas
│       ├── training
│       └── utils
├── tests
│   ├── data
│   ├── evaluation
│   ├── integration
│   ├── models
│   ├── reproducibility
│   └── unit
├── .uv
└── .venv-310
    ├── bin
    ├── include
    │   └── site
    ├── lib
    │   └── python3.10
    ├── lib64 -> lib
    └── share
        └── man

202 directories
```
