import json
import os

mapping_data = {
    "source_dataset": "CHB-MIT",
    "target_dataset": "Siena Scalp EEG Database",
    "source_montage_type": "Bipolar Double-Banana (23 channels)",
    "target_montage_type": "Unipolar Referential (29 EEG channels)",
    "reconstruction_method": "Differential Referential Subtraction (V_a - V_b)",
    "sampling_rate_source_hz": 256.0,
    "sampling_rate_target_hz": 512.0,
    "decimation_factor": 2,
    "mappings": [
        {
            "index": 0,
            "source_channel": "FP1-F7",
            "target_derivation": "EEG Fp1 - EEG F7",
            "source_anode": "FP1",
            "source_cathode": "F7",
            "target_anode": "FP1",
            "target_cathode": "F7",
            "mapping_method": "Exact 1:1 anatomical match",
            "availability": "100% (41/41 recordings)",
            "justification": "Direct subtraction of referential FP1 and F7 electrodes."
        },
        {
            "index": 1,
            "source_channel": "F7-T7",
            "target_derivation": "EEG F7 - EEG T3",
            "source_anode": "F7",
            "source_cathode": "T7",
            "target_anode": "F7",
            "target_cathode": "T3",
            "mapping_method": "10-20 Standard Equivalent (T3 == T7)",
            "availability": "100% (41/41 recordings)",
            "justification": "International 10-20 standard designates T3 as the mid-temporal electrode, renamed T7 in modern 10-10 nomenclature."
        },
        {
            "index": 2,
            "source_channel": "T7-P7",
            "target_derivation": "EEG T3 - EEG T5",
            "source_anode": "T7",
            "source_cathode": "P7",
            "target_anode": "T3",
            "target_cathode": "T5",
            "mapping_method": "10-20 Standard Equivalent (T3 == T7, T5 == P7)",
            "availability": "100% (41/41 recordings)",
            "justification": "T3 corresponds to T7; T5 corresponds to posterior-temporal P7."
        },
        {
            "index": 3,
            "source_channel": "P7-O1",
            "target_derivation": "EEG T5 - EEG O1",
            "source_anode": "P7",
            "source_cathode": "O1",
            "target_anode": "T5",
            "target_cathode": "O1",
            "mapping_method": "10-20 Standard Equivalent (T5 == P7, O1 == O1)",
            "availability": "100% (41/41 recordings)",
            "justification": "T5 corresponds to P7; O1 is direct 1:1 match."
        },
        {
            "index": 4,
            "source_channel": "FP1-F3",
            "target_derivation": "EEG Fp1 - EEG F3",
            "source_anode": "FP1",
            "source_cathode": "F3",
            "target_anode": "FP1",
            "target_cathode": "F3",
            "mapping_method": "Exact 1:1 anatomical match",
            "availability": "100% (41/41 recordings)",
            "justification": "Direct subtraction of referential FP1 and F3 electrodes."
        },
        {
            "index": 5,
            "source_channel": "F3-C3",
            "target_derivation": "EEG F3 - EEG C3",
            "source_anode": "F3",
            "source_cathode": "C3",
            "target_anode": "F3",
            "target_cathode": "C3",
            "mapping_method": "Exact 1:1 anatomical match",
            "availability": "100% (41/41 recordings)",
            "justification": "Direct subtraction of referential F3 and C3 electrodes."
        },
        {
            "index": 6,
            "source_channel": "C3-P3",
            "target_derivation": "EEG C3 - EEG P3",
            "source_anode": "C3",
            "source_cathode": "P3",
            "target_anode": "C3",
            "target_cathode": "P3",
            "mapping_method": "Exact 1:1 anatomical match",
            "availability": "100% (41/41 recordings)",
            "justification": "Direct subtraction of referential C3 and P3 electrodes."
        },
        {
            "index": 7,
            "source_channel": "P3-O1",
            "target_derivation": "EEG P3 - EEG O1",
            "source_anode": "P3",
            "source_cathode": "O1",
            "target_anode": "P3",
            "target_cathode": "O1",
            "mapping_method": "Exact 1:1 anatomical match",
            "availability": "100% (41/41 recordings)",
            "justification": "Direct subtraction of referential P3 and O1 electrodes."
        },
        {
            "index": 8,
            "source_channel": "FP2-F4",
            "target_derivation": "EEG Fp2 - EEG F4",
            "source_anode": "FP2",
            "source_cathode": "F4",
            "target_anode": "FP2",
            "target_cathode": "F4",
            "mapping_method": "Exact 1:1 anatomical match",
            "availability": "100% (41/41 recordings)",
            "justification": "Direct subtraction of referential FP2 and F4 electrodes."
        },
        {
            "index": 9,
            "source_channel": "F4-C4",
            "target_derivation": "EEG F4 - EEG C4",
            "source_anode": "F4",
            "source_cathode": "C4",
            "target_anode": "F4",
            "target_cathode": "C4",
            "mapping_method": "Exact 1:1 anatomical match",
            "availability": "100% (41/41 recordings)",
            "justification": "Direct subtraction of referential F4 and C4 electrodes."
        },
        {
            "index": 10,
            "source_channel": "C4-P4",
            "target_derivation": "EEG C4 - EEG P4",
            "source_anode": "C4",
            "source_cathode": "P4",
            "target_anode": "C4",
            "target_cathode": "P4",
            "mapping_method": "Exact 1:1 anatomical match",
            "availability": "100% (41/41 recordings)",
            "justification": "Direct subtraction of referential C4 and P4 electrodes."
        },
        {
            "index": 11,
            "source_channel": "P4-O2",
            "target_derivation": "EEG P4 - EEG O2",
            "source_anode": "P4",
            "source_cathode": "O2",
            "target_anode": "P4",
            "target_cathode": "O2",
            "mapping_method": "Exact 1:1 anatomical match",
            "availability": "100% (41/41 recordings)",
            "justification": "Direct subtraction of referential P4 and O2 electrodes."
        },
        {
            "index": 12,
            "source_channel": "FP2-F8",
            "target_derivation": "EEG Fp2 - EEG F8",
            "source_anode": "FP2",
            "source_cathode": "F8",
            "target_anode": "FP2",
            "target_cathode": "F8",
            "mapping_method": "Exact 1:1 anatomical match",
            "availability": "100% (41/41 recordings)",
            "justification": "Direct subtraction of referential FP2 and F8 electrodes."
        },
        {
            "index": 13,
            "source_channel": "F8-T8",
            "target_derivation": "EEG F8 - EEG T4",
            "source_anode": "F8",
            "source_cathode": "T8",
            "target_anode": "F8",
            "target_cathode": "T4",
            "mapping_method": "10-20 Standard Equivalent (T4 == T8)",
            "availability": "100% (41/41 recordings)",
            "justification": "T4 corresponds to right mid-temporal electrode T8."
        },
        {
            "index": 14,
            "source_channel": "T8-P8",
            "target_derivation": "EEG T4 - EEG T6",
            "source_anode": "T8",
            "source_cathode": "P8",
            "target_anode": "T4",
            "target_cathode": "T6",
            "mapping_method": "10-20 Standard Equivalent (T4 == T8, T6 == P8)",
            "availability": "100% (41/41 recordings)",
            "justification": "T4 corresponds to T8; T6 corresponds to right posterior-temporal P8."
        },
        {
            "index": 15,
            "source_channel": "P8-O2",
            "target_derivation": "EEG T6 - EEG O2",
            "source_anode": "P8",
            "source_cathode": "O2",
            "target_anode": "T6",
            "target_cathode": "O2",
            "mapping_method": "10-20 Standard Equivalent (T6 == P8, O2 == O2)",
            "availability": "100% (41/41 recordings)",
            "justification": "T6 corresponds to P8; O2 is direct 1:1 match."
        },
        {
            "index": 16,
            "source_channel": "FZ-CZ",
            "target_derivation": "EEG Fz - EEG Cz",
            "source_anode": "FZ",
            "source_cathode": "CZ",
            "target_anode": "FZ",
            "target_cathode": "CZ",
            "mapping_method": "Exact 1:1 anatomical match",
            "availability": "100% (41/41 recordings)",
            "justification": "Midline frontal-central bipolar derivation."
        },
        {
            "index": 17,
            "source_channel": "CZ-PZ",
            "target_derivation": "EEG Cz - EEG Pz",
            "source_anode": "CZ",
            "source_cathode": "PZ",
            "target_anode": "CZ",
            "target_cathode": "PZ",
            "mapping_method": "Exact 1:1 anatomical match",
            "availability": "100% (41/41 recordings)",
            "justification": "Midline central-parietal bipolar derivation."
        },
        {
            "index": 18,
            "source_channel": "P7-T7",
            "target_derivation": "EEG T5 - EEG T3",
            "source_anode": "P7",
            "source_cathode": "T7",
            "target_anode": "T5",
            "target_cathode": "T3",
            "mapping_method": "10-20 Standard Equivalent (T5 == P7, T3 == T7)",
            "availability": "100% (41/41 recordings)",
            "justification": "Inversion derivation of temporal-parietal link (T5 - T3 = -(T3 - T5))."
        },
        {
            "index": 19,
            "source_channel": "T7-FT9",
            "target_derivation": "EEG T3 - EEG F9",
            "source_anode": "T7",
            "source_cathode": "FT9",
            "target_anode": "T3",
            "target_cathode": "F9",
            "mapping_method": "10-20 Standard Equivalent (T3 == T7, F9 == FT9)",
            "availability": "100% (41/41 recordings)",
            "justification": "F9 corresponds to the anterior inferior temporal electrode FT9."
        },
        {
            "index": 20,
            "source_channel": "FT9-FT10",
            "target_derivation": "EEG F9 - EEG F10",
            "source_anode": "FT9",
            "source_cathode": "FT10",
            "target_anode": "F9",
            "target_cathode": "F10",
            "mapping_method": "10-20 Standard Equivalent (F9 == FT9, F10 == FT10)",
            "availability": "100% (41/41 recordings)",
            "justification": "Cross-hemispheric anterior temporal bridge."
        },
        {
            "index": 21,
            "source_channel": "FT10-T8",
            "target_derivation": "EEG F10 - EEG T4",
            "source_anode": "FT10",
            "source_cathode": "T8",
            "target_anode": "F10",
            "target_cathode": "T4",
            "mapping_method": "10-20 Standard Equivalent (F10 == FT10, T4 == T8)",
            "availability": "100% (41/41 recordings)",
            "justification": "Right anterior temporal to mid-temporal link."
        },
        {
            "index": 22,
            "source_channel": "T8-P8",
            "target_derivation": "EEG T4 - EEG T6",
            "source_anode": "T8",
            "source_cathode": "P8",
            "target_anode": "T4",
            "target_cathode": "T6",
            "mapping_method": "10-20 Standard Equivalent (T4 == T8, T6 == P8)",
            "availability": "100% (41/41 recordings)",
            "justification": "Canonical CHB-MIT channel 23 is identical to channel 14 (T8-P8)."
        }
    ]
}

out_path = "research/experiments/siena/config/siena_channel_mapping.json"
with open(out_path, "w") as f:
    json.dump(mapping_data, f, indent=2)

print(f"Saved {out_path} with {len(mapping_data['mappings'])} mapped channels.")
