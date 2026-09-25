# Worker Lifecycle Management & Recovery Specification

## Overview

In distributed execution mode (Mode B: `ENABLE_DISTRIBUTED_QUEUE=True`), prediction jobs are dispatched via Redis/ARQ and executed by dedicated worker processes, reading staged `.npz` inference packages from shared persistent storage.

This document describes the worker lifecycle, lease/heartbeat management, failure recovery semantics, and storage cleanup policy established in Prompt 7.4.

---

## 1. Worker Identity & Ownership

- **Worker Identity**: Each worker process generates a unique runtime identifier on initialization:
  `worker_id = <hostname>:<process_id>:<runtime_token>`
- **Job Ownership Claim**: When a worker dequeues a prediction task, it atomically claims ownership in PostgreSQL:
  - Sets `worker_id = <worker_id>`
  - Sets `status = "Running"`
  - Sets `heartbeat_at = <utc_now>`
  - Sets `lease_expires_at = <utc_now> + JOB_LEASE_TIMEOUT_SECONDS`
- **Ownership Invariant**: A worker cannot claim an active job leased to another live worker. Multiple workers cannot concurrently finalize the same job.

---

## 2. Heartbeat & Lease Mechanism

- **Heartbeat Interval**: While inference executes in a thread pool executor, an asynchronous background task on the worker event loop refreshes the job's lease in PostgreSQL every `WORKER_HEARTBEAT_INTERVAL_SECONDS` (default: 5 seconds).
- **Lease Expiration**: `JOB_LEASE_TIMEOUT_SECONDS` (default: 30 seconds). If a worker process halts, crashes, or is killed via `SIGKILL`, heartbeats cease and the lease expires.
- **Heartbeat Isolation**: Heartbeat network or database hiccups do not trigger ML retries or interrupt deterministic model inference.

---

## 3. Stale-Job Detection & Reaper Semantics

- **Detection Rule**: Any job where `status not in ("Completed", "Failed")`, `lease_expires_at is not None`, and `lease_expires_at <= utc_now` is classified as **stale**.
- **Terminal State**: Stale jobs transition to `status = "Failed"` with structured error context:
  `error = "Worker lease expired: distributed worker became unavailable"`
- **Strict No-Retry Policy**: Stale jobs are never automatically retried, re-enqueued, or re-run. Prediction remains deterministic.
- **On-Demand & Periodic Execution**:
  - The API periodically runs the reaper loop in its lifespan background task.
  - Workers run the reaper loop in background maintenance tasks.
  - Any client polling `GET /api/v2/jobs/{job_id}` or `GET /api/v2/predict/status/{job_id}` triggers immediate on-demand evaluation, returning `Failed` instantly upon lease expiry without waiting for the next periodic cycle.

---

## 4. Staged Payload Cleanup Policy

Staged payloads (`{job_id}.npz`) are cleaned up according to a conservative safety policy:

1. **Storage Containment**: Deletion operations strictly verify path containment within `STORAGE_DIR`. Directory traversal attempts (e.g., `../`) raise `StorageSecurityError`.
2. **Active Job Protection**: Staged files belonging to active jobs with unexpired leases are never deleted.
3. **Grace Period**: Files modified more recently than `STORAGE_ORPHAN_GRACE_SECONDS` (default: 300 seconds) are preserved.
4. **Terminal & Orphan Cleanup**:
   - Files for jobs marked `Completed` or `Failed` are eligible for immediate cleanup upon job completion or reaper recovery.
   - Partial temporary files (`.tmp_*`) exceeding grace period retention are safely unlinked.
   - Files with no corresponding database record exceeding grace retention are unlinked as orphaned artifacts.

---

## 5. Graceful Startup & Shutdown

- **Startup**:
  - Worker validates database connectivity with a read query.
  - Worker does NOT mutate database schema or execute DDL on startup (schema migrations are exclusively handled by API/migration runners).
  - Worker pre-loads ML models and registers lifecycle hooks.
- **Shutdown**:
  - On `SIGTERM`/`SIGINT`, the worker ceases accepting new jobs from Redis.
  - Ongoing tasks are given a bounded shutdown window.
  - If a process is abruptly terminated mid-inference, it does NOT fabricate a successful result; the uncompleted job is safely recovered as `Failed` once its lease expires.

---

## 6. Operational Recovery Procedure

If a worker node or container fails in production:
1. Allow the lease timeout period to elapse (`JOB_LEASE_TIMEOUT_SECONDS`).
2. The stale job reaper (running in API and peer workers) automatically transitions running jobs of the deceased worker to `Failed`.
3. Staged payloads are cleaned up according to the orphan cleanup policy.
4. Restart the worker process or container (`docker compose up -d worker` or `systemctl restart neuroaegis-worker`).
5. The restarted worker immediately resumes processing newly enqueued jobs without corrupting database state.

---

## 7. Scope Boundaries & Current Limitations

- **Single Broker**: Designed for Redis-backed ARQ queues. Celery, Kafka, RabbitMQ, and multi-region brokers are out of scope.
- **Local Shared Storage**: Uses shared volume filesystem storage. S3, GCS, and MinIO backends are out of scope.
- **Manual Retry Only**: Automatic ML retries are explicitly deferred to maintain pipeline determinism and prevent infinite failure loops on corrupt data.
- **Single Host / Compose**: Designed for single-host Docker Compose deployment. Horizontal multi-host consensus is not established in this phase.
