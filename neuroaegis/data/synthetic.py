import numpy as np
from typing import List, Tuple
from neuroaegis.data.schemas import WindowSample, PreprocessingConfig

def generate_synthetic_data(
    dataset: str, 
    patient_id: str, 
    recording_id: str, 
    config: PreprocessingConfig, 
    num_windows: int = 10
) -> List[WindowSample]:
    """
    Generates fake EEG window samples so other tracks can develop against this pipeline
    before real EDF files are wired in.
    """
    samples = []
    n_channels = len(config.common_channels)
    n_samples_per_window = int(config.windowing.size_sec * config.sampling_rate_hz)
    
    current_time = 0.0
    step = config.windowing.size_sec - config.windowing.overlap_sec
    
    for i in range(num_windows):
        # Fake EEG data (random noise)
        data = np.random.randn(n_channels, n_samples_per_window)
        
        # 10% chance of seizure label in synthetic data
        label = 1 if np.random.rand() > 0.9 else 0
        
        # If seizure, add some sine waves to make it visually distinct
        if label == 1:
            t = np.linspace(0, config.windowing.size_sec, n_samples_per_window)
            seizure_wave = 5 * np.sin(2 * np.pi * 6 * t)  # 6 Hz activity
            data += seizure_wave
            
        sample = WindowSample(
            patient_id=patient_id,
            recording_id=recording_id,
            window_start_sec=current_time,
            window_end_sec=current_time + config.windowing.size_sec,
            data=data,
            label=label,
            dataset_source=dataset
        )
        samples.append(sample)
        current_time += step
        
    return samples
