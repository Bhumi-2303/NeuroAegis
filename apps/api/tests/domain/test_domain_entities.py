import pytest
from uuid import uuid4
from pydantic import ValidationError
from apps.api.app.domain.value_objects.confidence import ConfidenceScore, ConfidenceBand
from apps.api.app.domain.entities.prediction import Prediction
from apps.api.app.domain.entities.pipeline import Pipeline
from apps.api.app.domain.entities.job import Job, JobStatus

def test_confidence_score_valid():
    score = ConfidenceScore(value=0.85, band=ConfidenceBand.HIGH)
    assert score.value == 0.85
    assert score.band == ConfidenceBand.HIGH

def test_confidence_score_invalid_value():
    with pytest.raises(ValidationError):
        ConfidenceScore(value=1.5, band=ConfidenceBand.HIGH)
    
    with pytest.raises(ValidationError):
        ConfidenceScore(value=-0.1, band=ConfidenceBand.LOW)

def test_prediction_valid():
    pred = Prediction(
        id=uuid4(),
        recording_id=uuid4(),
        patient_id=uuid4(),
        model_version="v1.0.0",
        label="seizure",
        probabilities={"seizure": 0.8, "normal": 0.2}
    )
    assert pred.label == "seizure"

def test_prediction_invalid_probabilities():
    with pytest.raises(ValidationError):
        Prediction(
            id=uuid4(),
            recording_id=uuid4(),
            patient_id=uuid4(),
            model_version="v1.0.0",
            label="seizure",
            probabilities={"seizure": 0.8, "normal": 0.3} # Sums to 1.1
        )

def test_pipeline_valid():
    pipe = Pipeline(
        id=uuid4(),
        task_id="seizure_detection",
        dataset_id="bonn",
        configuration={
            "model": "lightgbm",
            "feature_extractors": ["time_domain"]
        }
    )
    assert pipe.task_id == "seizure_detection"

def test_pipeline_invalid_configuration():
    with pytest.raises(ValidationError):
        # Missing model
        Pipeline(
            id=uuid4(),
            task_id="seizure_detection",
            dataset_id="bonn",
            configuration={"feature_extractors": ["time_domain"]}
        )
    
    with pytest.raises(ValidationError):
        # Missing feature_extractors
        Pipeline(
            id=uuid4(),
            task_id="seizure_detection",
            dataset_id="bonn",
            configuration={"model": "lightgbm"}
        )

def test_job_defaults():
    job = Job(id=uuid4(), patient_id=uuid4())
    assert job.status == JobStatus.QUEUED
    assert job.progress == 0
