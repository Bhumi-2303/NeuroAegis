# Scientific Contributions of NeuroAegis

This work establishes five principal methodological and empirical contributions to patient-independent automated epileptic seizure detection:

1. **Lightweight Patient-Independent Spatio-Temporal Architecture:**  
   We design a compact, channel-preserving neural network combining a 4-layer 1D convolutional feature extractor, a 2-layer spatial Graph Convolutional Network (GCN) constructed over validation-frozen functional connectivity ($\theta=0.30$, 23 nodes, 40 undirected edges), and a 1-layer causal unidirectional Gated Recurrent Unit (GRU, $L=8$, 22.5 s context). Comprising only 91,858 parameters, the architecture operates with an inference latency of 1.42 ms per window on Apple Silicon MPS (1,760x real-time margin), enabling efficient execution without inter-patient retraining.

2. **Systematic Characterization of the Sensitivity–False-Alarm Trade-off:**  
   Through rigorous 3-stage ablation, we uncover an essential structural interaction in EEG seizure modeling: static spatial graph convolutions dramatically suppress spurious non-epileptic artifacts (-89.7% false alarm reduction), but cause severe sensitivity collapse (event sensitivity dropping to 27.27%) by over-smoothing localized seizure onsets. Incorporating causal recurrent temporal memory successfully recovers temporal continuity, elevating event sensitivity to 95.45% while further reducing false alarms to 62.66 FA/24h (-96.8% relative to the 1D-CNN baseline).

3. **Clinically Grounded, False-Alarm-Aware Event Evaluation:**  
   Moving beyond misleading window-level accuracy under extreme class imbalance (344.2:1), we evaluate the system across 152.82 continuous unsegmented monitoring hours using clinical event metrics: event sensitivity (95.45%, detecting 21 of 22 seizures), operational false alarm rate (62.66 FA/24h), and empirical detection latency (median: 9.0 seconds, with 85.7% of detections occurring within 15 seconds of electrographic onset).

4. **Cross-Domain Transfer and Lightweight Target Calibration:**  
   We provide external generalization evidence on an evaluated Siena Scalp EEG benchmark subset (2 patients, 4 recordings, 4 seizures, 2.46 hours). NeuroAegis achieves 100% zero-shot event sensitivity with 0.0 false alarms per 24 hours (AUROC 0.9120, AUPRC 0.7135). Furthermore, a lightweight, non-invasive temperature and threshold calibration protocol derived on patient PN00 recovers test F1 on held-out patient PN12 from 0.3043 to 0.3750 (+23.2% relative gain) without retraining model parameters.

5. **Quantitative XAI Attribution and Perturbation Faithfulness:**  
   We implement axiomatic feature attribution via Integrated Gradients (50 Riemann steps) to rank channel and temporal contributions across seizure windows. Attribution saliency concentrated heavily in temporal-parietal leads (`T7-P7`, `P3-O1`, `FP1-F7`). We quantitatively validate the faithfulness of these explanations through systematic deletion perturbation experiments, demonstrating a monotonic model confidence degradation from 0.8093 to 0.2011 upon progressive removal of top-attributed features.
