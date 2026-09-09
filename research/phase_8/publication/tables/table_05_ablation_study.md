### Table 5: Progressive Component Ablation and False Alarm Suppression

| ablation_stage | component_added | active_modules | parameters | param_delta | event_sensitivity | event_sens_delta | fa_24h | fa_delta_pct | auroc | auroc_delta | auprc | auprc_delta | f1_score | f1_delta | clinical_finding |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline: Temporal Conv | 1D CNN | Conv1D Temporal | 173601 | 0 | 0.5455 | 0.0 | 1946.6 | 0.0 | 0.36389 | 0.0 | 0.04148 | 0.0 | 0.01553 | 0.0 | High sensitivity but catastrophic false alarm rate (1,946.56 FA/24h) due to missing spatial context. |
| Step 1: + Spatial Topology | Spatial GNN (theta=0.30) | Conv1D + GNN | 52497 | -121104 | 0.2727 | -0.2728 | 200.39 | -89.71 | 0.19431 | -0.16958 | 0.00492 | -0.03656 | 0.02784 | 0.01231 | Spatial filtering suppresses false alarms by 89.7% but causes sensitivity collapse (27.27%) without temporal memory. |
| Step 2: + Temporal Sequence | Causal GRU (L=8, 22.5s context) | Conv1D + GNN + Causal GRU | 91858 | 39361 | 0.9545 | 0.6818 | 62.66 | -68.73 | 0.9897 | 0.79539 | 0.80681 | 0.80189 | 0.68025 | 0.65241 | Temporal context restores event sensitivity to 95.45% (21/22), cuts FA to 62.66/24h (-96.8% vs CNN), and skyrockets AUPRC to 0.8068. |
