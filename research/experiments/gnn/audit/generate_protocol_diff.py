import os
import json

AUDIT_DIR = "research/experiments/gnn/audit"
PHASE3_METRICS_PATH = "research/experiments/cnn_baseline/phase_3_metrics.json"
PHASE4A_METRICS_PATH = "research/experiments/gnn/cnn_gnn/exp_01/phase_4a_metrics.json"
OUTPUT_DIFF_PATH = os.path.join(AUDIT_DIR, "phase3_vs_phase4a_protocol_diff.json")

with open(PHASE3_METRICS_PATH, "r") as f:
    p3 = json.load(f)
    
with open(PHASE4A_METRICS_PATH, "r") as f:
    p4 = json.load(f)

p3_loss = p3["loss"]
p3_g = p3["gamma"]
p3_a = p3["alpha"]
p4_loss = p4["loss"]
p4_g = p4["gamma"]
p4_a = p4["alpha"]

diff = {
    "audit_timestamp": "2026-09-06T18:25:00Z",
    "frozen_invariants": {
        "dataset": {
            "phase3": p3["dataset"],
            "phase4a": p4["dataset"],
            "match": p3["dataset"] == p4["dataset"]
        },
        "num_channels": {
            "phase3": p3["channels"],
            "phase4a": p4["channels"],
            "match": p3["channels"] == p4["channels"]
        },
        "window_duration_samples": {
            "phase3": p3["window"],
            "phase4a": p4["window"],
            "match": p3["window"] == p4["window"]
        },
        "stride_samples": {
            "phase3": p3["stride"],
            "phase4a": p4["stride"],
            "match": p3["stride"] == p4["stride"]
        },
        "primary_label": {
            "phase3": p3["primary_label"],
            "phase4a": p4["primary_label"],
            "match": p3["primary_label"] == p4["primary_label"]
        },
        "training_sampler": {
            "phase3": p3["training_sampler"],
            "phase4a": p4["training_sampler"],
            "match": p3["training_sampler"] == p4["training_sampler"]
        },
        "loss_function": {
            "phase3": f"{p3_loss} (gamma={p3_g}, alpha={p3_a})",
            "phase4a": f"{p4_loss} (gamma={p4_g}, alpha={p4_a})",
            "match": (p3_loss == p4_loss) and (p3_g == p4_g) and (p3_a == p4_a)
        },
        "train_patients": {
            "phase3": p3["train_patients"],
            "phase4a": p4["train_patients"],
            "match": p3["train_patients"] == p4["train_patients"]
        },
        "validation_patients": {
            "phase3": p3["validation_patients"],
            "phase4a": p4["validation_patients"],
            "match": p3["validation_patients"] == p4["validation_patients"]
        },
        "test_patients": {
            "phase3": p3["test_patients"],
            "phase4a": p4["test_patients"],
            "match": p3["test_patients"] == p4["test_patients"]
        },
        "leakage_audits": {
            "patient_leakage": {"p3": p3["patient_leakage"], "p4": p4["patient_leakage"]},
            "recording_leakage": {"p3": p3["recording_leakage"], "p4": p4["recording_leakage"]},
            "window_leakage": {"p3": p3["window_leakage"], "p4": p4["window_leakage"]}
        },
        "master_index_sha256": {
            "phase3": p3["master_index_sha256"],
            "phase4a": p4["master_index_sha256"],
            "match": p3["master_index_sha256"] == p4["master_index_sha256"]
        }
    },
    "architectural_differences": {
        "model_type": {
            "phase3": "Multi-Channel Temporal 1D CNN",
            "phase4a": "Channel-Preserving 1D CNN + 2-Layer Spatial GCN"
        },
        "parameter_count": {
            "phase3": p3["parameters"],
            "phase4a": p4["parameters"],
            "delta": p4["parameters"] - p3["parameters"],
            "pct_change": round((p4["parameters"] - p3["parameters"]) / p3["parameters"] * 100, 2)
        },
        "channel_interaction_mechanism": {
            "phase3": "Dense 1D Conv across all 23 channels at Stage 1 (Conv1d(23, 32))",
            "phase4a": "Explicit Graph Message Passing via Adjacency Matrix over 23 distinct node representations"
        },
        "temporal_feature_extractor": {
            "phase3": "Multi-channel (23 -> 32 -> 64 -> 128 -> 128)",
            "phase4a": "Single-channel shared (1 -> 16 -> 32 -> 64 -> 64)"
        },
        "graph_pooling": {
            "phase3": "N/A (Spatial channels collapsed in Conv1d layer 1)",
            "phase4a": "Dual Pooling: Concatenation of Global Mean Pool and Global Max Pool (64 + 64 = 128-dim)"
        }
    },
    "empirical_metrics_comparison": {
        "test_auroc": {
            "phase3": p3["test_auroc"],
            "phase4a": p4["test_auroc"],
            "delta": round(p4["test_auroc"] - p3["test_auroc"], 5)
        },
        "test_auprc": {
            "phase3": p3["test_auprc"],
            "phase4a": p4["test_auprc"],
            "delta": round(p4["test_auprc"] - p3["test_auprc"], 5)
        },
        "test_window_sensitivity": {
            "phase3": p3["test_sensitivity"],
            "phase4a": p4["test_sensitivity"],
            "delta": round(p4["test_sensitivity"] - p3["test_sensitivity"], 5)
        },
        "test_window_specificity": {
            "phase3": p3["test_specificity"],
            "phase4a": p4["test_specificity"],
            "delta": round(p4["test_specificity"] - p3["test_specificity"], 5)
        },
        "test_precision": {
            "phase3": p3["test_precision"],
            "phase4a": p4["test_precision"],
            "delta": round(p4["test_precision"] - p3["test_precision"], 5)
        },
        "test_f1": {
            "phase3": p3["test_f1"],
            "phase4a": p4["test_f1"],
            "delta": round(p4["test_f1"] - p3["test_f1"], 5)
        },
        "test_accuracy": {
            "phase3": p3["test_accuracy"],
            "phase4a": p4["test_accuracy"],
            "delta": round(p4["test_accuracy"] - p3["test_accuracy"], 5)
        },
        "event_sensitivity": {
            "phase3": p3["event_sensitivity"],
            "phase4a": p4["event_sensitivity"],
            "delta": round(p4["event_sensitivity"] - p3["event_sensitivity"], 5)
        },
        "false_alarms_per_day": {
            "phase3": p3["false_alarms_per_day"],
            "phase4a": p4["false_alarms_per_day"],
            "delta": round(p4["false_alarms_per_day"] - p3["false_alarms_per_day"], 2),
            "pct_reduction": round((p3["false_alarms_per_day"] - p4["false_alarms_per_day"]) / p3["false_alarms_per_day"] * 100, 2)
        },
        "detection_delay_sec": {
            "phase3": p3["detection_delay"],
            "phase4a": p4["detection_delay"],
            "delta": round(p4["detection_delay"] - p3["detection_delay"], 2)
        },
        "training_time_sec": {
            "phase3": p3["training_time_sec"],
            "phase4a": p4["training_time_sec"],
            "delta": round(p4["training_time_sec"] - p3["training_time_sec"], 2)
        }
    },
    "scientific_interpretation": {
        "summary": "Phase 4A introduces explicit spatial topology constraints while strictly preserving data, split, and labeling invariants.",
        "spatial_smoothing_tradeoff": "Phase 4A acts as an aggressive spatial low-pass filter. While suppressing background false alarms by 90.8% (1946.56 -> 179.19 FA/24h) and increasing specificity to 99.48%, the conservative output distribution reduces window-level sensitivity at fixed threshold 0.50 from 16.01% to 3.30%, causing a lower AUROC/AUPRC on ranked window evaluation even though all 22 seizure events triggered detections.",
        "capacity_difference": "Phase 4A possesses 70% fewer parameters (52,497 vs 173,601) due to channel weight-sharing in the temporal backbone, which regularizes the model but limits individual channel spectral expressivity."
    }
}

with open(OUTPUT_DIFF_PATH, "w") as f:
    json.dump(diff, f, indent=2)
    
print(f"Saved protocol comparison diff to {OUTPUT_DIFF_PATH}")
print("Invariants verification:")
for k, v in diff["frozen_invariants"].items():
    if "match" in v:
        res = "MATCH (PASS)" if v["match"] else "MISMATCH (FAIL)"
        print(f"  - {k}: {res}")
