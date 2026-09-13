import os
import json
import numpy as np

def generate_synthetic_dataset(output_dir: str = "data"):
    """
    Generates predefined held-out events (e.g., 20 seizure, 20 non-seizure).
    Each event contains synthetic EEG signal data for 18 channels, over 5 seconds.
    """
    os.makedirs(output_dir, exist_ok=True)
    channels = ["FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", "C3-P3", "P3-O1", 
                "FP2-F4", "F4-C4", "C4-P4", "P4-O2", "FP2-F8", "F8-T8", "T8-P8", "P8-O2", 
                "FZ-CZ", "CZ-PZ"]
    
    events = []
    
    # 20 seizure, 20 non-seizure
    for i in range(40):
        is_seizure = i < 20
        event_id = f"{'seizure' if is_seizure else 'non_seizure'}_{i:03d}"
        
        # 5 seconds at 256 Hz = 1280 samples
        time_sec = 5.0
        n_samples = 1280
        
        # We will save the data as a simple numpy array
        data = np.random.randn(len(channels), n_samples)
        
        if is_seizure:
            # Add some 6 Hz sine waves to some channels randomly
            t = np.linspace(0, time_sec, n_samples)
            for c in range(len(channels)):
                if np.random.rand() > 0.5:
                    data[c] += 3.0 * np.sin(2 * np.pi * 6 * t)
                    
        filepath = os.path.join(output_dir, f"{event_id}.npy")
        np.save(filepath, data)
        
        events.append({
            "event_id": event_id,
            "filepath": filepath,
            "label": "Seizure" if is_seizure else "Non-Seizure",
            "duration_sec": time_sec,
            "channels": channels
        })
        
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(events, f, indent=4)
        
    print(f"Generated synthetic dataset at {output_dir}")

if __name__ == "__main__":
    generate_synthetic_dataset()
