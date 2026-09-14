# EEG Preprocessing Visualization

This folder contains a visualization of the actual preprocessing pipeline used in NeuroAegis on a real EDF file from the CHB-MIT dataset.

## Provenance
- **Dataset**: CHB-MIT
- **Patient**: chb01
- **EDF**: `chb01_03.edf`
- **Channel**: FP1-F7
- **Extraction Timestamps**: 2993 to 3003 seconds (10 seconds total). The seizure in this recording is located at 2996 to 3036 seconds.
- **Preprocessing Code/Config Used**: `research/phase_2/chbmit_preprocessor.py` (CHBMITSignalFilter, CHBMITNormalizer)

## Pipeline Parameters
- **Bandpass Filter**: 0.5-40 Hz (zero-phase SOS filter, order 4)
- **Notch Filter**: 60 Hz (zero-phase IIR, Q=30)
- **Normalization**: Per-channel z-score (recording-local, zero mean, unit variance)
- **Window Parameters**: 5 seconds length (1280 samples at 256 Hz), 23 channels (canonical montage order)

## Validation Notes
- The 23 channels were extracted successfully matching the actual configuration. The source EDF actually has duplicate `T8-P8` channels which `CHBMITChannelManager` handles correctly.
- The 60 Hz Notch filter's effect was evaluated. The bandpass filter (which cuts off at 40 Hz) already heavily attenuates the 60 Hz component by > 99.9%. The subsequent notch filter technically reduces the microscopic remaining 60 Hz energy further, but practically is redundant given the 0.5-40 Hz bandpass bounds.

## Files
- `metadata.json`: Contains numerical metadata parameters
- `*_signal.csv`: The actual computed values in time-domain.
- `model_input.npy`: The 23x1280 final tensor.
- `figure_*.png`: Resulting publication figures showing time-domain, frequency-domain, notch analysis, and final tensor inputs.

*Generated: 2026-09-14*
