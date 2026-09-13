import streamlit as st
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from schema import ClinicianAnnotation

st.set_page_config(page_title="NeuroAegis Clinician Annotation Tool", layout="wide")

st.title("NeuroAegis Clinician Annotation Tool")
st.markdown("""
Welcome to the clinical agreement study. Please review the EEG window and rank the importance 
of specific channels, time segments, and frequency bands that guided your diagnosis.
""")

# Load manifest
data_dir = "data"
manifest_path = os.path.join(data_dir, "manifest.json")
annotations_dir = "annotations"
os.makedirs(annotations_dir, exist_ok=True)

if not os.path.exists(manifest_path):
    st.warning("No data found. Please run `python generate_synthetic.py` first.")
    st.stop()

with open(manifest_path, "r") as f:
    events = json.load(f)

# State management
if "current_idx" not in st.session_state:
    st.session_state.current_idx = 0

clinician_id = st.sidebar.text_input("Clinician ID", value="Dr. Anon")

st.sidebar.markdown("---")
st.sidebar.write(f"**Progress:** {st.session_state.current_idx + 1} / {len(events)}")

event = events[st.session_state.current_idx]
event_id = event["event_id"]
channels = event["channels"]

st.header(f"Event: {event_id} ({event['label']})")

# Load and Plot Data
data = np.load(event["filepath"])
time_axis = np.linspace(0, event["duration_sec"], data.shape[1])

fig, axes = plt.subplots(len(channels), 1, figsize=(10, len(channels)*0.5), sharex=True)
if len(channels) == 1:
    axes = [axes]
    
for idx, ax in enumerate(axes):
    ax.plot(time_axis, data[idx], color='black', linewidth=0.5)
    ax.set_ylabel(channels[idx], rotation=0, labelpad=20, va="center")
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

axes[-1].set_xlabel("Time (seconds)")
st.pyplot(fig)

st.markdown("---")
st.subheader("Annotation Form")

# Helper function for options
def scale_to_int(val):
    return {"None": 0, "Low": 1, "Medium": 2, "High": 3}[val]

with st.form(key=f"annotation_form_{event_id}"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**1. Channel Importance**")
        channel_vals = {}
        for ch in channels:
            channel_vals[ch] = st.select_slider(
                f"{ch}", options=["None", "Low", "Medium", "High"], value="None"
            )
            
    with col2:
        st.write("**2. Time-Region Importance (1s intervals)**")
        time_vals = {}
        for i in range(int(event["duration_sec"])):
            t_label = f"{i}.0-{i+1}.0s"
            time_vals[t_label] = st.select_slider(
                t_label, options=["None", "Low", "Medium", "High"], value="None"
            )
            
    with col3:
        st.write("**3. Frequency-Band Importance**")
        bands = ["Delta (<4Hz)", "Theta (4-8Hz)", "Alpha (8-13Hz)", "Beta (13-30Hz)", "Gamma (>30Hz)"]
        freq_keys = ["delta", "theta", "alpha", "beta", "gamma"]
        freq_vals = {}
        for b, k in zip(bands, freq_keys):
            freq_vals[k] = st.select_slider(
                b, options=["None", "Low", "Medium", "High"], value="None"
            )

    submit_button = st.form_submit_button(label="Save & Next")

if submit_button:
    # Build schema
    annotation = ClinicianAnnotation(
        event_id=event_id,
        clinician_id=clinician_id,
        channel_importance={k: scale_to_int(v) for k, v in channel_vals.items()},
        time_importance={k: scale_to_int(v) for k, v in time_vals.items()},
        frequency_importance={k: scale_to_int(v) for k, v in freq_vals.items()}
    )
    
    # Save to disk
    out_file = os.path.join(annotations_dir, f"{event_id}_{clinician_id}.json")
    with open(out_file, "w") as f:
        f.write(annotation.model_dump_json(indent=4))
        
    st.success(f"Saved annotation for {event_id}!")
    
    # Move next
    if st.session_state.current_idx < len(events) - 1:
        st.session_state.current_idx += 1
        st.rerun()
    else:
        st.balloons()
        st.write("You have completed all events. Thank you!")
