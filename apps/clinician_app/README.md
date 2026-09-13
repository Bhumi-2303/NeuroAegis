# NeuroAegis Clinician Annotation Tool

A lightweight local web app built with Streamlit designed for non-technical clinician collaborators.

## How to Run

1. Generate the placeholder dataset (synthetic EEG windows):
   ```bash
   python generate_synthetic.py
   ```

2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Output Schema (For Track D)

The app outputs a JSON file for every annotated event in the `annotations/` directory. The structure perfectly aligns with the `ClinicianAnnotation` Pydantic schema, mapped to integers representing importance (`0: None`, `1: Low`, `2: Medium`, `3: High`):

```json
{
    "event_id": "seizure_000",
    "clinician_id": "Dr. Anon",
    "channel_importance": {
        "FP1-F7": 0,
        "F7-T7": 3,
        ...
    },
    "time_importance": {
        "0.0-1.0s": 0,
        "1.0-2.0s": 2,
        ...
    },
    "frequency_importance": {
        "delta": 0,
        "theta": 1,
        "alpha": 3,
        "beta": 0,
        "gamma": 0
    }
}
```

Track D's agreement metrics module can easily load these dictionaries, convert the integer values to a `torch.Tensor` or `numpy.array`, and run them directly against the `IntegratedGradients` or `Attention` vectors using the pre-built `spearman_correlation` and `jaccard_similarity` functions.
