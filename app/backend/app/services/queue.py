from __future__ import annotations
import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from arq import create_pool
from arq.connections import ArqRedis, RedisSettings

from app.core.config import settings

logger = logging.getLogger("neuroaegis.queue")


class QueueError(Exception):
    """Base exception for queue operations."""
    pass


class QueueConnectionError(QueueError):
    """Raised when queue connection to Redis fails."""
    pass


class QueuePayloadError(QueueError):
    """Raised when an invalid or oversized payload is submitted for queuing."""
    pass


class PredictionQueue(ABC):
    """Abstract interface for prediction job queue dispatch."""

    @abstractmethod
    async def enqueue_prediction_job(
        self,
        job_id: str,
        staged_path: str,
        dataset_name: str = "bonn",
        medical_history: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Enqueue a prediction job identifier and staged payload reference.
        Must NOT accept raw EEG arrays or binary buffers.
        """
        pass


def _validate_payload_metadata(
    job_id: str,
    staged_path: str,
    dataset_name: str,
    medical_history: dict[str, Any] | None,
    extra_kwargs: dict[str, Any],
) -> None:
    """Enforce architectural rule: No raw EEG signals, numpy arrays, or large binaries in Redis."""
    if not job_id or not isinstance(job_id, str):
        raise QueuePayloadError("job_id must be a non-empty string identifier")

    if not staged_path or not isinstance(staged_path, str):
        raise QueuePayloadError("staged_path must be a valid path/key reference string")

    # Guard against passing in-memory EEG data directly into Redis
    for key, val in [("staged_path", staged_path), ("medical_history", medical_history), *extra_kwargs.items()]:
        if isinstance(val, np.ndarray):
            raise QueuePayloadError(
                f"Prohibited payload argument '{key}': NumPy arrays must not be sent via Redis. "
                "Stage the payload to shared storage and pass a staged_path reference."
            )
        if isinstance(val, (bytes, bytearray)) and len(val) > 1024:
            raise QueuePayloadError(
                f"Prohibited payload argument '{key}': Large binary buffers (>1KB) must not be sent via Redis. "
                "Stage binary data to shared storage instead."
            )


class RedisPredictionQueue(PredictionQueue):
    """Production ARQ/Redis queue implementation."""

    def __init__(
        self,
        redis_url: str | None = None,
        queue_name: str | None = None,
    ) -> None:
        self.redis_url = redis_url or settings.REDIS_URL
        self.queue_name = queue_name or settings.QUEUE_NAME
        self._pool: ArqRedis | None = None

    async def get_pool(self) -> ArqRedis:
        """Lazy connection to ARQ Redis pool."""
        if self._pool is None:
            try:
                redis_settings = RedisSettings.from_dsn(self.redis_url)
                self._pool = await create_pool(redis_settings)
            except Exception as exc:
                logger.error(f"Failed to connect to Redis at {self.redis_url}: {exc}")
                raise QueueConnectionError(
                    f"Could not connect to Redis broker at '{self.redis_url}': {exc}"
                ) from exc
        return self._pool

    async def close(self) -> None:
        """Close Redis connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def enqueue_prediction_job(
        self,
        job_id: str,
        staged_path: str,
        dataset_name: str = "bonn",
        medical_history: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        """Enqueue task by reference to Redis via ARQ."""
        _validate_payload_metadata(job_id, staged_path, dataset_name, medical_history, kwargs)

        pool = await self.get_pool()
        try:
            arq_job = await pool.enqueue_job(
                "run_prediction_task",
                job_id=job_id,
                staged_path=staged_path,
                dataset_name=dataset_name,
                medical_history=medical_history,
                _queue_name=self.queue_name,
                **kwargs,
            )
            if arq_job is None:
                raise QueueError(f"Job {job_id} already exists or was dropped by Redis queue.")
            logger.info(f"Enqueued job {job_id} into queue '{self.queue_name}' (ARQ ID: {arq_job.job_id})")
            return arq_job.job_id
        except QueueError:
            raise
        except Exception as exc:
            logger.error(f"Failed to enqueue job {job_id}: {exc}")
            raise QueueError(f"Failed to enqueue prediction job {job_id}: {exc}") from exc


class InMemoryPredictionQueue(PredictionQueue):
    """Deterministic in-memory queue implementation for tests and isolation."""

    def __init__(self, queue_name: str = "in-memory:jobs") -> None:
        self.queue_name = queue_name
        self.enqueued_jobs: list[dict[str, Any]] = []

    async def enqueue_prediction_job(
        self,
        job_id: str,
        staged_path: str,
        dataset_name: str = "bonn",
        medical_history: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        """Record enqueued job in memory for test verification."""
        _validate_payload_metadata(job_id, staged_path, dataset_name, medical_history, kwargs)
        entry = {
            "job_id": job_id,
            "staged_path": staged_path,
            "dataset_name": dataset_name,
            "medical_history": medical_history,
            "extra_kwargs": kwargs,
        }
        self.enqueued_jobs.append(entry)
        return f"mock-{job_id}"

    def get_enqueued_jobs(self) -> list[dict[str, Any]]:
        return list(self.enqueued_jobs)

    def get_job_by_id(self, job_id: str) -> dict[str, Any] | None:
        for job in self.enqueued_jobs:
            if job["job_id"] == job_id:
                return job
        return None

    def clear(self) -> None:
        self.enqueued_jobs.clear()


# Default singleton instance (isolated from API v2 in this phase)
prediction_queue = RedisPredictionQueue()
