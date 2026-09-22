from __future__ import annotations
import json
import logging
import re
import tempfile
import uuid
from pathlib import Path

import mne
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import Patient, PredictionJob
from app.services.edf_validation import (
    EdfValidationResult,
    UploadValidationError,
    cleanup_temp_upload,
    edf_validation_service,
    sanitize_upload_filename,
    save_upload_to_temp,
    usable_eeg_channel_indices,
)
from app.services.job_service import run_prediction_pipeline
from app.services.model_service import ml_model_service
from app.services.prediction.prediction_router import prediction_router

router = APIRouter()
logger = logging.getLogger("neuroaegis")


def _validation_http_exception(result: EdfValidationResult, status_code: int) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={
            "message": "EDF validation failed",
            "validation": result.response(),
        },
    )

@router.post("/predict")
async def create_prediction_job(
    background_tasks: BackgroundTasks,
    name: str = Form(...),
    age: int = Form(...),
    gender: str = Form(...),
    weight: float = Form(...),
    height: float = Form(...),
    medical_history: str = Form(...), # JSON string
    vital_signs: str = Form(...), # JSON string
    file: UploadFile = File(...),
    sampling_rate: float | None = Form(None),
    channels: str | None = Form(None),
    db: Session = Depends(get_db)
):
    try:
        safe_filename = sanitize_upload_filename(file.filename)
    except UploadValidationError as exc:
        result = EdfValidationResult(
            validationStatus="invalid",
            fileName=file.filename or "",
            fileSizeBytes=0,
            errors=[str(exc)],
        )
        raise _validation_http_exception(result, 400) from exc

    if not ml_model_service.is_loaded:
        raise HTTPException(status_code=503, detail="Model is not loaded on the backend")

    temp_path: Path | None = None
    try:
        with tempfile.TemporaryDirectory(prefix="neuroaegis_edf_v2_") as temp_dir:
            temp_path, safe_filename, uploaded_size = await save_upload_to_temp(
                file,
                temp_root=temp_dir,
                max_upload_size=edf_validation_service.max_upload_size,
            )
            validation = edf_validation_service.validate_file(
                temp_path,
                file_name=safe_filename,
                file_size_bytes=uploaded_size,
            )
            if validation.validation_status == "invalid":
                status_code = 413 if any("size limit" in error for error in validation.errors) else 400
                raise _validation_http_exception(validation, status_code)
            if validation.dataset == "unknown":
                validation.validation_status = "invalid"
                validation.errors.append("Unable to identify the EDF dataset with sufficient confidence")
                raise _validation_http_exception(validation, 422)

            detected_dataset = validation.dataset
            try:
                prediction_router.get_predictor(detected_dataset)
            except ValueError as exc:
                validation.validation_status = "invalid"
                validation.errors.append(f"No prediction model is configured for dataset '{detected_dataset}'")
                raise _validation_http_exception(validation, 422) from exc

            final_sampling_rate = float(validation.sampling_rate or 0.0)
            if sampling_rate is not None and abs(sampling_rate - final_sampling_rate) > 0.5:
                validation.validation_status = "invalid"
                validation.errors.append("Provided sampling rate does not match the EDF header")
                raise _validation_http_exception(validation, 422)

            predictor_metadata = prediction_router.get_available_models().get(
                detected_dataset, {}
            ).get("dataset_info", {})
            window_length = predictor_metadata.get("window_length", 15360)
            if not isinstance(window_length, (int, float)) or window_length <= 0:
                window_length = 15360
            window_length = int(window_length)

            raw = mne.io.read_raw_edf(temp_path, preload=False, verbose="ERROR")
            try:
                eeg_picks = usable_eeg_channel_indices(raw)
                start_sample = 0
                if raw.n_times > window_length and detected_dataset == "chbmit":
                    file_match = re.search(r"(chb\d+)", safe_filename, flags=re.IGNORECASE)
                    if file_match:
                        patient_str = file_match.group(1).lower()
                        current_dir = Path(__file__).resolve().parent
                        summary_path = current_dir / "data" / "chbmit_subset" / patient_str / f"{patient_str}-summary.txt"
                        if summary_path.exists():
                            content = summary_path.read_text(encoding="utf-8")
                            file_idx = content.find(Path(safe_filename).stem)
                            if file_idx != -1:
                                next_file_idx = content.find("File Name:", file_idx + 1)
                                section = content[file_idx:next_file_idx if next_file_idx != -1 else len(content)]
                                start_match = re.search(r"Seizure\s+(?:\d+\s+)?Start Time:\s*(\d+)", section)
                                if start_match:
                                    start_sample = max(
                                        0,
                                        int(int(start_match.group(1)) * final_sampling_rate) - window_length // 2,
                                    )
                                    start_sample = min(start_sample, raw.n_times - window_length)
                eeg_data = raw.get_data(
                    picks=eeg_picks,
                    start=start_sample,
                    stop=min(raw.n_times, start_sample + window_length),
                )
                channel_names = [raw.ch_names[index] for index in eeg_picks]
            finally:
                raw.close()

            if channels and channels.strip():
                channel_names_input = [item.strip() for item in channels.split(",") if item.strip()]
                if channel_names_input != channel_names:
                    validation.validation_status = "invalid"
                    validation.errors.append("Provided channels do not match the EDF header")
                    raise _validation_http_exception(validation, 422)

            cleanup_temp_upload(temp_path)
            temp_path = None

        # Parse JSON fields
        try:
            parsed_medical_history = json.loads(medical_history)
            parsed_vital_signs = json.loads(vital_signs)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON in medical_history or vital_signs")
            
        # Create Patient
        patient_id = str(uuid.uuid4())
        patient = Patient(
            id=patient_id,
            name=name,
            age=age,
            gender=gender,
            weight=weight,
            height=height,
            medical_history=parsed_medical_history,
            vital_signs=parsed_vital_signs
        )
        db.add(patient)
        
        # Create Job
        job_id = str(uuid.uuid4())
        job = PredictionJob(
            id=job_id,
            patient_id=patient_id,
            status="Validating",
            progress=0
        )
        db.add(job)
        db.commit()
        
        # Start background task
        background_tasks.add_task(run_prediction_pipeline, job_id, eeg_data, channel_names, final_sampling_rate, detected_dataset)
        
        return {"job_id": job_id, "patient_id": patient_id}
        
    except HTTPException:
        raise
    except UploadValidationError as exc:
        result = EdfValidationResult(
            validationStatus="invalid",
            fileName=safe_filename,
            fileSizeBytes=exc.file_size_bytes,
            errors=[str(exc)],
        )
        raise _validation_http_exception(result, 413 if exc.file_size_bytes else 400) from exc
    except Exception:
        logger.exception("Failed to start EDF prediction job")
        raise HTTPException(status_code=500, detail="EDF prediction job could not be started")
    finally:
        if temp_path is not None:
            cleanup_temp_upload(temp_path)

@router.get("/predict/status/{job_id}")
async def get_job_status(job_id: str, db: Session = Depends(get_db)):
    job = db.query(PredictionJob).filter(PredictionJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
        
    response = {
        "job_id": job.id,
        "status": job.status,
        "progress": job.progress
    }
    
    if job.status == "Completed":
        response["result"] = {
            "prediction_label": job.prediction_label,
            "probability_seizure": job.probability_seizure,
            "confidence_band": job.confidence_band,
            "shap_explanation": job.shap_explanation
        }
        
    return response

@router.get("/history")
async def get_history(db: Session = Depends(get_db)):
    jobs = db.query(PredictionJob).order_by(PredictionJob.created_at.desc()).all()
    results = []
    for job in jobs:
        results.append({
            "job_id": job.id,
            "patient_name": job.patient.name,
            "created_at": job.created_at,
            "status": job.status,
            "prediction_label": job.prediction_label
        })
    return results

@router.get("/report/{job_id}")
async def get_report(job_id: str, db: Session = Depends(get_db)):
    job = db.query(PredictionJob).filter(PredictionJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return {
        "job": {
            "id": job.id,
            "status": job.status,
            "prediction_label": job.prediction_label,
            "probability_seizure": job.probability_seizure,
            "confidence_band": job.confidence_band,
            "shap_explanation": job.shap_explanation,
            "created_at": job.created_at,
            "completed_at": job.completed_at
        },
        "patient": {
            "name": job.patient.name,
            "age": job.patient.age,
            "gender": job.patient.gender,
            "weight": job.patient.weight,
            "height": job.patient.height,
            "medical_history": job.patient.medical_history,
            "vital_signs": job.patient.vital_signs
        }
    }
