from __future__ import annotations
import uuid
import re
import tempfile
from pathlib import Path
from typing import Any

import mne
import numpy as np
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

from app.core.config import settings
from app.db.database import SessionLocal, get_db
from app.db.models import Patient, PredictionJob, User
from app.services.prediction.prediction_router import prediction_router
from app.core.auth import require_role
from app.services.edf_validation import (
    EdfValidationResult,
    UploadValidationError,
    cleanup_temp_upload,
    edf_validation_service,
    sanitize_upload_filename,
    save_upload_to_temp,
    usable_eeg_channel_indices,
)
from app.services.eeg_visualization import (
    build_eeg_visualization_from_raw,
    build_window_visualization,
)

router = APIRouter()

import logging

logger = logging.getLogger("neuroaegis")


def build_eeg_visualization(eeg_data, channel_names, fs, max_points=1500):
    """Backward-compatible wrapper for callers with only a model window."""
    return build_window_visualization(eeg_data, channel_names, fs, max_points=max_points)


def process_and_save_prediction(
    job_id: str,
    eeg_data,
    channel_names,
    fs,
    dataset: str,
    model_name: str,
    eeg_visualization: dict[str, Any] | None = None,
):
    db = SessionLocal()
    try:
        # Update status to processing
        job = db.query(PredictionJob).filter(PredictionJob.id == job_id).first()
        if job:
            job.status = "Processing"
            job.progress = 50
            db.commit()

        logger.info(f"[{job_id}] Job started. Dataset: {dataset}, Model: {model_name}")

        # Run model
        predictor = prediction_router.get_predictor(dataset)
        logger.info(f"[{job_id}] Predictor retrieved successfully. Starting inference...")
        result = predictor.get_prediction(
            eeg_data=eeg_data,
            channel_names=channel_names,
            fs=fs,
            model_name=model_name
        )
        
        # Save results
        job = db.query(PredictionJob).filter(PredictionJob.id == job_id).first()
        if job:
            job.prediction_label = result["prediction"]["label"]
            job.probability_seizure = result["prediction"]["probabilities"]["seizure"]
            job.confidence_band = result["confidence"]["band"]
            job.shap_explanation = result["explanation"]
            job.eeg_visualization = eeg_visualization or build_window_visualization(
                eeg_data=eeg_data,
                channel_names=channel_names,
                fs=fs,
            )
            job.status = "Completed"
            job.progress = 100
            import datetime
            job.completed_at = datetime.datetime.utcnow()
            db.commit()
            logger.info(f"[{job_id}] Prediction finished successfully. Result: {job.prediction_label} ({job.probability_seizure:.4f})")
    except Exception as e:
        logger.error(f"[{job_id}] Prediction background task failed: {e}", exc_info=True)
        job = db.query(PredictionJob).filter(PredictionJob.id == job_id).first()
        if job:
            job.status = "Failed"
            job.progress = 0
            job.error = str(e)
            db.commit()
    finally:
        db.close()


def _validation_http_exception(result: EdfValidationResult, status_code: int) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={
            "message": "EDF validation failed",
            "validation": result.response(),
        },
    )


@router.post("/", response_model=dict[str, Any])
async def predict_eeg(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    sampling_rate: float | None = Form(None),
    channels: str | None = Form(None),
    patient_id: str | None = Form(None),
    dataset: str | None = Form(None),
    model: str | None = Form(None),
    db: Session = Depends(get_db)
):
    """Validate an EDF upload, then queue the existing prediction pipeline."""
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

    if not prediction_router.is_loaded:
        raise HTTPException(status_code=503, detail="Model is not loaded on the backend")

    logger.info("Received EDF upload. Size limit=%s bytes", settings.MAX_EEG_UPLOAD_BYTES)
    validation: EdfValidationResult
    detected_dataset: str
    confidence: float
    matched_rules: list[str]
    selected_model: str
    final_sampling_rate: float
    eeg_data: np.ndarray
    channel_names: list[str]
    eeg_visualization: dict[str, Any]

    with tempfile.TemporaryDirectory(prefix="neuroaegis_edf_") as temp_dir:
        temp_path: Path | None = None
        try:
            temp_path, safe_filename, uploaded_size = await save_upload_to_temp(
                file,
                temp_root=temp_dir,
                max_upload_size=settings.MAX_EEG_UPLOAD_BYTES,
            )
            validation = edf_validation_service.validate_file(
                temp_path,
                file_name=safe_filename,
                file_size_bytes=uploaded_size,
            )
            if validation.validation_status == "invalid":
                status_code = 413 if any("size limit" in error for error in validation.errors) else 400
                raise _validation_http_exception(validation, status_code)

            detected_dataset = validation.dataset
            confidence = validation.detection_confidence
            matched_rules = validation.matched_rules
            if detected_dataset == "unknown":
                validation.validation_status = "invalid"
                validation.errors.append("Unable to identify the EDF dataset with sufficient confidence")
                raise _validation_http_exception(validation, 422)
            if dataset and dataset.lower() != detected_dataset:
                validation.validation_status = "invalid"
                validation.errors.append("Provided dataset does not match the EDF metadata")
                raise _validation_http_exception(validation, 422)

            try:
                predictor = prediction_router.get_predictor(detected_dataset)
            except ValueError as exc:
                validation.validation_status = "invalid"
                validation.errors.append(f"No prediction model is configured for dataset '{detected_dataset}'")
                raise _validation_http_exception(validation, 422) from exc

            selected_model = model or predictor.default_model
            predictor_metadata = prediction_router.get_available_models().get(
                detected_dataset, {}
            ).get("dataset_info", {})
            window_length = predictor_metadata.get("window_length", 15360)
            if not isinstance(window_length, (int, float)) or window_length <= 0:
                window_length = 15360
            window_length = int(window_length)
            final_sampling_rate = float(validation.sampling_rate or 0.0)

            if sampling_rate is not None and abs(sampling_rate - final_sampling_rate) > 0.5:
                validation.validation_status = "invalid"
                validation.errors.append("Provided sampling rate does not match the EDF header")
                raise _validation_http_exception(validation, 422)

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
                            base_name = Path(safe_filename).stem
                            file_idx = content.find(base_name)
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
                if channels and channels.strip():
                    provided_channels = [item.strip() for item in channels.split(",") if item.strip()]
                    if provided_channels != channel_names:
                        validation.validation_status = "invalid"
                        validation.errors.append("Provided channels do not match the EDF header")
                        raise _validation_http_exception(validation, 422)
                eeg_visualization = build_eeg_visualization_from_raw(
                    raw,
                    file_name=safe_filename,
                    file_size_bytes=uploaded_size,
                    dataset=detected_dataset,
                )
            finally:
                raw.close()
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
            logger.exception("EDF upload processing failed")
            result = EdfValidationResult(
                validationStatus="invalid",
                fileName=safe_filename,
                fileSizeBytes=0,
                errors=["EDF upload could not be processed safely"],
            )
            raise _validation_http_exception(result, 400)
        finally:
            if temp_path is not None:
                cleanup_temp_upload(temp_path)
        
    try:
        # Verify patient exists if provided
        if patient_id:
            patient = db.query(Patient).filter(Patient.id == patient_id).first()
            if not patient:
                raise HTTPException(status_code=404, detail="Patient not found")
                
        # Create Job
        job_id = str(uuid.uuid4())
        job = PredictionJob(
            id=job_id,
            patient_id=patient_id,
            status="Validating",
            progress=0,
            # We assume the schema is updated to support these if possible, 
            # or we store in a JSON column if supported.
            # If the columns don't exist yet, we will add them.
            detected_dataset=detected_dataset,
            detection_confidence=confidence,
            selected_model=selected_model
        )
        db.add(job)
        db.commit()
        
        
        logger.info(f"[{job_id}] Validator passed. Dataset: {detected_dataset}, Confidence: {confidence:.2f}, Predictor: {selected_model}")
        
        background_tasks.add_task(
            process_and_save_prediction,
            job_id,
            eeg_data,
            channel_names,
            final_sampling_rate,
            detected_dataset,
            selected_model,
            eeg_visualization,
        )
        logger.info(f"[{job_id}] Response sent for prediction task.")
        
        return {
            "job_id": job_id,
            "detected_dataset": detected_dataset,
            "confidence": validation.detection_confidence,
            "matched_rules": validation.matched_rules,
            "selected_model": selected_model,
            "validation": validation.response(),
        }

        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred during prediction processing.")
