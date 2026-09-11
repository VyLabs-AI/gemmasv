"""Small bounded background-job queue for certificate work."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
import secrets
import threading
import time
from typing import Any, Callable


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class JobRecord:
    job_id: str
    session_id: str
    status: JobStatus
    created_at: float
    updated_at: float
    result: dict[str, Any] | None = None
    error_code: str | None = None
    hidden: bool = False
    progress: float = 0.0


class JobNotFound(KeyError):
    pass


class JobQueue:
    """A process-local queue with bounded pending work and redacted failures."""

    def __init__(
        self,
        *,
        max_workers: int = 1,
        max_pending: int = 16,
        ttl_seconds: float = 60 * 60,
        clock: Callable[[], float] = time.monotonic,
    ):
        if max_workers <= 0 or max_pending <= 0:
            raise ValueError("job queue limits must be positive")
        self.max_pending = int(max_pending)
        self.ttl_seconds = float(ttl_seconds)
        self._clock = clock
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="hero-certificate",
        )
        self._lock = threading.RLock()
        self._jobs: dict[str, JobRecord] = {}
        self._futures: dict[str, Future] = {}

    def submit(
        self,
        session_id: str,
        fn: Callable[..., dict[str, Any]],
        /,
        *args,
        **kwargs,
    ) -> JobRecord:
        with self._lock:
            self.purge_expired()
            pending = sum(
                job.status in (JobStatus.QUEUED, JobStatus.RUNNING)
                for job in self._jobs.values()
            )
            if pending >= self.max_pending:
                raise RuntimeError("certificate queue is full")
            now = self._clock()
            job_id = secrets.token_urlsafe(18)
            record = JobRecord(job_id, session_id, JobStatus.QUEUED, now, now)
            self._jobs[job_id] = record
            self._futures[job_id] = self._executor.submit(
                self._run, job_id, fn, args, kwargs, False
            )
            return record

    def submit_with_progress(
        self,
        session_id: str,
        fn: Callable[..., dict[str, Any]],
        /,
        *args,
        **kwargs,
    ) -> JobRecord:
        """Submit ``fn(progress_callback, *args, **kwargs)``."""

        with self._lock:
            self.purge_expired()
            pending = sum(
                job.status in (JobStatus.QUEUED, JobStatus.RUNNING)
                for job in self._jobs.values()
            )
            if pending >= self.max_pending:
                raise RuntimeError("certificate queue is full")
            now = self._clock()
            job_id = secrets.token_urlsafe(18)
            record = JobRecord(job_id, session_id, JobStatus.QUEUED, now, now)
            self._jobs[job_id] = record
            self._futures[job_id] = self._executor.submit(
                self._run, job_id, fn, args, kwargs, True
            )
            return record

    def get(self, job_id: str, *, session_id: str) -> JobRecord:
        with self._lock:
            record = self._jobs.get(job_id)
            if (
                record is None
                or record.hidden
                or not secrets.compare_digest(record.session_id, session_id)
            ):
                raise JobNotFound(job_id)
            return record

    def delete_for_session(self, session_id: str) -> int:
        """Hide jobs immediately and clear any completed sensitive result."""

        with self._lock:
            count = 0
            for job_id, record in self._jobs.items():
                if not secrets.compare_digest(record.session_id, session_id):
                    continue
                count += 1
                record.hidden = True
                record.result = None
                record.error_code = None
                if record.status is JobStatus.QUEUED:
                    future = self._futures.get(job_id)
                    if future is not None and future.cancel():
                        record.status = JobStatus.CANCELLED
                record.updated_at = self._clock()
            return count

    def purge_expired(self) -> int:
        with self._lock:
            now = self._clock()
            expired = [
                job_id
                for job_id, record in self._jobs.items()
                if record.status
                not in (JobStatus.QUEUED, JobStatus.RUNNING)
                and now - record.updated_at > self.ttl_seconds
            ]
            for job_id in expired:
                self._jobs.pop(job_id, None)
                self._futures.pop(job_id, None)
            return len(expired)

    def shutdown(self, *, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=True)

    def _run(self, job_id: str, fn, args, kwargs, with_progress: bool) -> None:
        with self._lock:
            record = self._jobs[job_id]
            if record.status is JobStatus.CANCELLED:
                return
            record.status = JobStatus.RUNNING
            record.progress = max(record.progress, 0.01)
            record.updated_at = self._clock()
        try:
            if with_progress:
                result = fn(
                    lambda value: self._set_progress(job_id, value), *args, **kwargs
                )
            else:
                result = fn(*args, **kwargs)
        except Exception as exc:  # API receives a redacted type, never exception text.
            with self._lock:
                record = self._jobs[job_id]
                record.status = JobStatus.FAILED
                record.error_code = type(exc).__name__
                record.progress = 1.0
                record.updated_at = self._clock()
            return
        with self._lock:
            record = self._jobs[job_id]
            if record.hidden:
                record.result = None
                record.status = JobStatus.CANCELLED
            else:
                record.result = result
                record.status = JobStatus.SUCCEEDED
            record.progress = 1.0
            record.updated_at = self._clock()

    def _set_progress(self, job_id: str, value: float) -> None:
        value = min(0.99, max(0.01, float(value)))
        with self._lock:
            record = self._jobs[job_id]
            if not record.hidden and record.status is JobStatus.RUNNING:
                record.progress = max(record.progress, value)
                record.updated_at = self._clock()
