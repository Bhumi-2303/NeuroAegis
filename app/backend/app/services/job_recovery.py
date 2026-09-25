from __future__ import annotations
import datetime
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.database import SessionLocal
from app.db.models import PredictionJob
from app.services.storage import SAFE_JOB_ID_REGEX, StorageSecurityError, storage_backend

logger = logging.getLogger("neuroaegis.lifecycle")

STALE_JOB_ERROR_MESSAGE = "Worker lease expired: distributed worker became unavailable"


def utc_timestamp(dt: datetime.datetime | None = None) -> float:
    """Safely convert a datetime (naive UTC or aware) to a UNIX epoch timestamp."""
    if dt is None:
        return time.time()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return dt.timestamp()



def is_job_stale(job: PredictionJob, now: datetime.datetime | None = None) -> bool:
    """
    Check if a PredictionJob is in an active non-terminal state with an expired worker lease.
    """
    if not job:
        return False
    if job.status in ("Completed", "Failed"):
        return False
    if job.lease_expires_at is None:
        return False

    current_time = now or datetime.datetime.utcnow()
    return job.lease_expires_at <= current_time


def recover_job_as_failed(job: PredictionJob, error_message: str = STALE_JOB_ERROR_MESSAGE, now: datetime.datetime | None = None) -> None:
    """
    Transition a stale job to the terminal 'Failed' state with structured error context.
    Does NOT rerun inference or move the job back to the queue.
    """
    transition_time = now or datetime.datetime.utcnow()
    job.status = "Failed"
    job.progress = 0
    job.error = error_message
    job.completed_at = transition_time
    logger.warning(
        f"Reaped stale job: job_id={job.id}, worker_id={job.worker_id}, "
        f"lease_expired_at={job.lease_expires_at}, reaped_at={transition_time.isoformat()}"
    )


def reap_job_if_stale(job_id: str, db: Session, now: datetime.datetime | None = None) -> bool:
    """
    Inspect a specific job. If it is stale, transition it to 'Failed' and commit.
    Returns True if reaped, False otherwise.
    """
    job = db.query(PredictionJob).filter(PredictionJob.id == job_id).first()
    if job and is_job_stale(job, now=now):
        recover_job_as_failed(job, now=now)
        db.commit()
        return True
    return False


def reap_stale_jobs(db: Session, now: datetime.datetime | None = None) -> list[str]:
    """
    Identify and recover all active jobs with expired worker leases.
    Returns the list of reaped job IDs.
    """
    current_time = now or datetime.datetime.utcnow()
    reaped_ids: list[str] = []

    try:
        # Find active jobs with expired lease
        stale_jobs = (
            db.query(PredictionJob)
            .filter(
                PredictionJob.status.notin_(["Completed", "Failed"]),
                PredictionJob.lease_expires_at.isnot(None),
                PredictionJob.lease_expires_at <= current_time,
            )
            .all()
        )

        for job in stale_jobs:
            recover_job_as_failed(job, now=current_time)
            reaped_ids.append(job.id)

        if reaped_ids:
            db.commit()
            logger.info(f"Reaper completed: recovered {len(reaped_ids)} stale job(s): {reaped_ids}")

    except Exception as exc:
        db.rollback()
        logger.error(f"Error during stale job reaper run: {exc}", exc_info=True)

    return reaped_ids


def cleanup_orphaned_staged_files(
    db: Session | None = None,
    backend=None,
    grace_seconds: int | None = None,
    now_ts: float | None = None,
) -> list[str]:
    """
    Conservative cleanup for orphaned staged payloads surviving worker termination.

    Safety rules enforced:
    1. Strict storage directory containment check (never delete outside STORAGE_DIR).
    2. Active jobs with valid unexpired leases are NEVER deleted.
    3. Files newer than the grace period are NEVER deleted.
    4. Terminal (Completed/Failed) jobs with age >= grace_seconds are eligible for deletion.
    5. Temporary partial files (.tmp_*) with age >= grace_seconds are eligible for deletion.
    6. Unknown files with age >= grace_seconds are eligible for deletion.
    """
    current_backend = backend or storage_backend
    current_grace = grace_seconds if grace_seconds is not None else settings.STORAGE_ORPHAN_GRACE_SECONDS
    current_now = now_ts if now_ts is not None else time.time()
    cleaned_files: list[str] = []

    storage_dir = Path(current_backend.storage_dir).resolve()
    if not storage_dir.exists() or not storage_dir.is_dir():
        return cleaned_files

    should_close_db = False
    active_db = db
    if active_db is None:
        active_db = SessionLocal()
        should_close_db = True

    try:
        for entry in storage_dir.iterdir():
            if not entry.is_file():
                continue

            # Rule 1: Storage directory containment check
            try:
                resolved_path = entry.resolve()
                resolved_path.relative_to(storage_dir)
            except (ValueError, StorageSecurityError) as sec_exc:
                logger.error(f"Security containment violation detected for {entry}: {sec_exc}")
                continue

            # Rule 2: File age check against grace threshold
            try:
                mtime = entry.stat().st_mtime
            except OSError:
                continue

            age_seconds = current_now - mtime
            if age_seconds < current_grace:
                # File is recent — preserve unconditionally
                continue

            file_name = entry.name

            # Check for leftover temporary atomic upload files
            if file_name.startswith(".tmp_") and file_name.endswith(".npz"):
                try:
                    entry.unlink(missing_ok=True)
                    cleaned_files.append(str(entry))
                    logger.info(f"Cleaned up aged temporary staged file: {file_name} (age: {age_seconds:.1f}s)")
                except Exception as exc:
                    logger.warning(f"Could not clean up temp file {entry}: {exc}")
                continue

            # Check for staged window packages: {job_id}.npz
            if file_name.endswith(".npz"):
                job_id = file_name[:-4]
                if not SAFE_JOB_ID_REGEX.match(job_id):
                    logger.warning(f"Skipping non-conforming filename in storage directory: {file_name}")
                    continue

                job = active_db.query(PredictionJob).filter(PredictionJob.id == job_id).first()
                if job:
                    # Job exists in database
                    if job.status in ("Completed", "Failed"):
                        # Job is already finished or failed; staged payload is orphaned
                        try:
                            entry.unlink(missing_ok=True)
                            cleaned_files.append(str(entry))
                            logger.info(f"Cleaned up orphaned staged payload for terminal job {job_id} ({job.status})")
                        except Exception as exc:
                            logger.warning(f"Could not delete staged file {entry}: {exc}")
                    elif is_job_stale(job, now=datetime.datetime.utcfromtimestamp(current_now)):
                        # Job was active but has expired lease; reap it and delete file
                        recover_job_as_failed(job, now=datetime.datetime.utcfromtimestamp(current_now))
                        active_db.commit()
                        try:
                            entry.unlink(missing_ok=True)
                            cleaned_files.append(str(entry))
                            logger.info(f"Reaped stale job {job_id} and cleaned staged payload")
                        except Exception as exc:
                            logger.warning(f"Could not delete staged file {entry}: {exc}")
                    else:
                        # Job is active with valid unexpired lease — PRESERVE
                        continue
                else:
                    # Job does not exist in DB and file exceeds grace threshold — eligible as unknown orphan
                    try:
                        entry.unlink(missing_ok=True)
                        cleaned_files.append(str(entry))
                        logger.warning(f"Cleaned up unknown orphaned staged file {file_name} (age: {age_seconds:.1f}s)")
                    except Exception as exc:
                        logger.warning(f"Could not delete unknown staged file {entry}: {exc}")

    finally:
        if should_close_db and active_db is not None:
            active_db.close()

    return cleaned_files
