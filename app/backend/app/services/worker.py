from __future__ import annotations
import asyncio
import datetime
import json
import logging
import os
import socket
import uuid
from typing import Any

import numpy as np
from arq.connections import RedisSettings
from sqlalchemy import text

from app.core.config import settings
from app.db.database import SessionLocal, engine
from app.db.models import PredictionJob
from app.services.job_service import run_prediction_pipeline
from app.services.prediction.prediction_router import prediction_router

logger = logging.getLogger("neuroaegis.worker")

_WORKER_ID: str | None = None


def get_worker_id() -> str:
    """Return a unique runtime identity for this worker process."""
    global _WORKER_ID
    if _WORKER_ID is None:
        hostname = socket.gethostname()
        pid = os.getpid()
        token = uuid.uuid4().hex[:8]
        _WORKER_ID = f"{hostname}:{pid}:{token}"
    return _WORKER_ID


def claim_job(job_id: str, worker_id: str, lease_duration_seconds: int) -> bool:
    """
    Atomically claim ownership of a distributed job for this worker.
    Ensures an actively leased job owned by another worker is not stolen.
    """
    db = SessionLocal()
    try:
        job = db.query(PredictionJob).filter(PredictionJob.id == job_id).first()
        if not job:
            logger.error(f"Cannot claim job {job_id}: not found in database")
            return False

        if job.status in ("Completed", "Failed"):
            logger.warning(f"Cannot claim job {job_id}: already in terminal state '{job.status}'")
            return False

        now = datetime.datetime.utcnow()

        # If job is actively leased by another live worker, do not steal
        if job.worker_id and job.worker_id != worker_id:
            if job.lease_expires_at and job.lease_expires_at > now:
                logger.warning(
                    f"Cannot claim job {job_id}: actively leased by worker '{job.worker_id}' "
                    f"until {job.lease_expires_at.isoformat()}"
                )
                return False

        # Claim ownership and set initial lease
        job.worker_id = worker_id
        job.status = "Running"
        job.heartbeat_at = now
        job.lease_expires_at = now + datetime.timedelta(seconds=lease_duration_seconds)
        db.commit()
        logger.info(
            f"Worker claimed job: job_id={job_id}, worker_id={worker_id}, "
            f"lease_expires_at={job.lease_expires_at.isoformat()}"
        )
        return True
    except Exception as exc:
        db.rollback()
        logger.error(f"Failed to claim job {job_id}: {exc}", exc_info=True)
        return False
    finally:
        db.close()


def update_heartbeat(job_id: str, worker_id: str, lease_duration_seconds: int) -> bool:
    """
    Refresh the lease and heartbeat timestamp for an actively claimed job.
    Must only be called by the owning worker.
    """
    db = SessionLocal()
    try:
        job = db.query(PredictionJob).filter(PredictionJob.id == job_id).first()
        if not job:
            return False

        if job.worker_id != worker_id:
            logger.warning(
                f"Heartbeat rejected for job {job_id}: current owner is '{job.worker_id}', "
                f"caller is '{worker_id}'"
            )
            return False

        if job.status in ("Completed", "Failed"):
            return False

        now = datetime.datetime.utcnow()
        job.heartbeat_at = now
        job.lease_expires_at = now + datetime.timedelta(seconds=lease_duration_seconds)
        db.commit()
        logger.debug(f"Heartbeat updated: job_id={job_id}, worker_id={worker_id}")
        return True
    except Exception as exc:
        db.rollback()
        logger.error(f"Failed to record heartbeat for job {job_id}: {exc}")
        return False
    finally:
        db.close()


def fail_job(job_id: str, error_message: str, expected_worker_id: str | None = None) -> None:
    """Safe helper to record failure status and error message on PredictionJob."""
    try:
        db = SessionLocal()
        try:
            job = db.query(PredictionJob).filter(PredictionJob.id == job_id).first()
            if job:
                if expected_worker_id is not None and job.worker_id != expected_worker_id:
                    logger.warning(
                        f"Skipping fail_job for {job_id}: worker ownership mismatch "
                        f"(expected '{expected_worker_id}', current is '{job.worker_id}')"
                    )
                    return
                job.status = "Failed"
                job.progress = 0
                job.error = error_message
                job.completed_at = datetime.datetime.utcnow()
                db.commit()
                logger.info(f"Recorded job failure: job_id={job_id}, error='{error_message}'")
        finally:
            db.close()
    except Exception as exc:
        logger.error(f"Failed to update job {job_id} error status in database: {exc}")


def load_staged_payload(staged_path: str, dataset_name: str) -> tuple[np.ndarray, list[str], float, dict | None]:
    """
    Load staged EEG data package without passing arrays through Redis.
    Supports staged .npz packages and raw .edf recordings.
    """
    if not os.path.exists(staged_path):
        raise FileNotFoundError(f"Staged payload file does not exist: {staged_path}")

    _, ext = os.path.splitext(staged_path)
    ext = ext.lower()

    if ext == ".npz":
        from app.services.storage import storage_backend
        eeg_data, channel_names, fs, ds_name, eeg_visualization, _ = storage_backend.load_window(staged_path)
        return eeg_data, channel_names, fs, eeg_visualization

    elif ext == ".edf":
        import mne
        from app.services.edf_validation import usable_eeg_channel_indices
        from app.services.eeg_visualization import build_eeg_visualization_from_raw

        raw = mne.io.read_raw_edf(staged_path, preload=False, verbose="ERROR")
        try:
            picks = usable_eeg_channel_indices(raw)
            channel_names = [raw.ch_names[idx] for idx in picks]
            fs = float(raw.info["sfreq"])
            window_len = 15360 if dataset_name == "chbmit" else 4096
            eeg_data = raw.get_data(picks=picks, start=0, stop=min(raw.n_times, window_len))
            file_size = os.path.getsize(staged_path)
            eeg_visualization = build_eeg_visualization_from_raw(
                raw,
                file_name=os.path.basename(staged_path),
                file_size_bytes=file_size,
                dataset=dataset_name,
            )
            return eeg_data, channel_names, fs, eeg_visualization
        finally:
            raw.close()
    else:
        raise ValueError(f"Unsupported staged payload format '{ext}'. Must be .npz or .edf")


async def _heartbeat_loop(
    job_id: str,
    worker_id: str,
    stop_event: asyncio.Event,
    interval_seconds: int,
    lease_duration_seconds: int,
) -> None:
    """Async background task that refreshes the job lease while inference is running."""
    logger.info(f"Heartbeat loop started: job_id={job_id}, worker_id={worker_id}, interval={interval_seconds}s")
    try:
        while not stop_event.is_set():
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=interval_seconds)
                break
            except asyncio.TimeoutError:
                pass

            if stop_event.is_set():
                break

            refreshed = update_heartbeat(job_id, worker_id, lease_duration_seconds)
            if not refreshed:
                logger.warning(f"Heartbeat refresh could not be recorded for job {job_id}")
    finally:
        logger.info(f"Heartbeat loop ended: job_id={job_id}, worker_id={worker_id}")


async def run_prediction_task(
    ctx: dict[str, Any],
    job_id: str,
    staged_path: str,
    dataset_name: str = "bonn",
    medical_history: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    ARQ worker task entry point.
    Reuses existing prediction pipeline without duplicating ML/feature logic.
    Claims ownership, maintains lease heartbeat, and cleans up staged storage payload upon conclusion.
    """
    from app.services.storage import storage_backend

    worker_id = get_worker_id()
    logger.info(
        f"Worker received prediction task: worker_id={worker_id}, job_id={job_id}, "
        f"staged_path={staged_path}, dataset={dataset_name}"
    )

    if not job_id or not isinstance(job_id, str):
        err = "Invalid task arguments: job_id must be a non-empty string"
        logger.error(err)
        return {"status": "failed", "job_id": job_id, "error": err}

    if not staged_path or not isinstance(staged_path, str):
        err = "Invalid task arguments: staged_path must be a string path"
        logger.error(err)
        fail_job(job_id, err)
        return {"status": "failed", "job_id": job_id, "error": err}

    # Verify job existence in database
    db = SessionLocal()
    try:
        job = db.query(PredictionJob).filter(PredictionJob.id == job_id).first()
        if not job:
            err = f"Job {job_id} not found in database"
            logger.error(err)
            return {"status": "failed", "job_id": job_id, "error": err}
    finally:
        db.close()

    # Step 1: Claim job ownership with initial lease
    claimed = claim_job(job_id, worker_id, settings.JOB_LEASE_TIMEOUT_SECONDS)
    if not claimed:
        err = f"Worker {worker_id} could not claim job {job_id} (already owned or terminal)"
        logger.warning(err)
        return {"status": "failed", "job_id": job_id, "error": err}

    # Step 2: Launch heartbeat task concurrently with prediction execution
    stop_heartbeat = asyncio.Event()
    heartbeat_task = asyncio.create_task(
        _heartbeat_loop(
            job_id=job_id,
            worker_id=worker_id,
            stop_event=stop_heartbeat,
            interval_seconds=settings.WORKER_HEARTBEAT_INTERVAL_SECONDS,
            lease_duration_seconds=settings.JOB_LEASE_TIMEOUT_SECONDS,
        )
    )

    try:
        try:
            eeg_data, channel_names, fs, eeg_vis = load_staged_payload(staged_path, dataset_name)
        except Exception as exc:
            err = f"Failed to load staged payload for job {job_id}: {exc}"
            logger.error(err, exc_info=True)
            fail_job(job_id, err, expected_worker_id=worker_id)
            return {"status": "failed", "job_id": job_id, "error": err}

        try:
            # Delegate directly to the existing tested prediction pipeline
            await run_prediction_pipeline(
                job_id=job_id,
                eeg_data=eeg_data,
                channel_names=channel_names,
                fs=fs,
                dataset_name=dataset_name,
                eeg_visualization=eeg_vis,
                expected_worker_id=worker_id,
            )
            logger.info(f"Worker {worker_id} successfully processed prediction pipeline for job {job_id}")
            return {"status": "completed", "job_id": job_id, "dataset_name": dataset_name}
        except Exception as exc:
            err = f"Execution error in prediction pipeline for job {job_id}: {exc}"
            logger.error(err, exc_info=True)
            fail_job(job_id, err, expected_worker_id=worker_id)
            return {"status": "failed", "job_id": job_id, "error": err}

    finally:
        # Step 3: Stop heartbeat cleanly
        stop_heartbeat.set()
        try:
            await asyncio.wait_for(heartbeat_task, timeout=1.0)
        except Exception:
            pass

        # Step 4: Clean up staged storage package after completion or failure
        try:
            cleaned = storage_backend.delete_window(staged_path)
            if cleaned:
                logger.info(f"Cleaned up staged window package: {staged_path}")
        except Exception as clean_exc:
            logger.warning(f"Could not clean up staged path {staged_path}: {clean_exc}")


async def _worker_reaper_loop(stop_event: asyncio.Event) -> None:
    """Background loop that periodically reaps stale jobs and orphaned files."""
    from app.services.job_recovery import cleanup_orphaned_staged_files, reap_stale_jobs

    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=settings.REAPER_INTERVAL_SECONDS)
            break
        except asyncio.TimeoutError:
            pass

        if stop_event.is_set():
            break

        try:
            db = SessionLocal()
            try:
                reap_stale_jobs(db)
                cleanup_orphaned_staged_files(db=db)
            finally:
                db.close()
        except Exception as exc:
            logger.error(f"Error in worker background reaper loop: {exc}")


async def startup(ctx: dict[str, Any]) -> None:
    """Initialize worker dependencies, verify connectivity, and pre-load ML model artifacts."""
    worker_id = get_worker_id()
    logger.info(f"Initializing ARQ Worker with identity: {worker_id}")

    # Prompt 7.4 Architectural Rule: Worker must NOT mutate DB schema on startup
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("Worker database connection verified.")
    except Exception as exc:
        logger.warning(f"Worker database connectivity notice: {exc}")

    try:
        loaded = prediction_router.load_all_models()
        if loaded:
            logger.info("Worker ML model artifacts loaded successfully.")
        else:
            logger.warning("Worker ML models loaded with warnings or missing optional checkpoints.")
    except Exception as exc:
        logger.error(f"Worker failed to load models during startup: {exc}", exc_info=True)

    ctx["prediction_router"] = prediction_router

    # Launch background reaper loop for lifecycle maintenance
    reaper_stop = asyncio.Event()
    ctx["reaper_stop"] = reaper_stop
    ctx["reaper_task"] = asyncio.create_task(_worker_reaper_loop(reaper_stop))

    logger.info(
        f"ARQ Worker ready. WorkerID='{worker_id}', Queue='{settings.QUEUE_NAME}', "
        f"MaxJobs={settings.WORKER_MAX_JOBS}, JobTimeout={settings.JOB_TIMEOUT_SECONDS}s, "
        f"HeartbeatInterval={settings.WORKER_HEARTBEAT_INTERVAL_SECONDS}s, "
        f"LeaseTimeout={settings.JOB_LEASE_TIMEOUT_SECONDS}s."
    )


async def shutdown(ctx: dict[str, Any]) -> None:
    """Clean worker shutdown."""
    worker_id = get_worker_id()
    logger.info(f"ARQ Worker {worker_id} shutting down cleanly...")

    reaper_stop = ctx.get("reaper_stop")
    if reaper_stop:
        reaper_stop.set()
    reaper_task = ctx.get("reaper_task")
    if reaper_task:
        try:
            await asyncio.wait_for(reaper_task, timeout=2.0)
        except Exception:
            pass

    logger.info(f"ARQ Worker {worker_id} shutdown complete.")


class WorkerSettings:
    """ARQ Worker configuration specification."""
    functions = [run_prediction_task]
    redis_settings = RedisSettings.from_dsn(settings.REDIS_URL)
    queue_name = settings.QUEUE_NAME
    max_jobs = settings.WORKER_MAX_JOBS
    job_timeout = settings.JOB_TIMEOUT_SECONDS
    max_tries = 1
    retry_jobs = False
    on_startup = startup
    on_shutdown = shutdown
