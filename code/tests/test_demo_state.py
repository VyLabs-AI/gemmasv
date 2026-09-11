import threading

import pytest

from gemma_sv.demo_server.jobs import JobNotFound, JobQueue, JobStatus
from gemma_sv.demo_server.state import SessionNotFound, SessionStore


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_session_store_expires_and_clears_sensitive_state():
    clock = Clock()
    store = SessionStore(ttl_seconds=10, clock=clock)
    record = store.create(domain="cybersecurity")
    record.memory_text = "synthetic codeword"
    record.engine_state["tokens"] = [1, 2, 3]

    clock.now = 11
    with pytest.raises(SessionNotFound):
        store.get(record.session_id)
    assert record.memory_text is None
    assert record.engine_state == {}


def test_session_store_enforces_capacity_after_purge():
    clock = Clock()
    store = SessionStore(ttl_seconds=5, max_sessions=1, clock=clock)
    store.create()
    with pytest.raises(RuntimeError, match="capacity"):
        store.create()
    clock.now = 6
    assert store.create().session_id


def test_job_queue_scopes_results_to_session_and_redacts_on_delete():
    gate = threading.Event()
    queue = JobQueue(max_workers=1)

    def work():
        gate.wait(timeout=2)
        return {"kl_nats": 1e-12}

    job = queue.submit("session-a", work)
    with pytest.raises(JobNotFound):
        queue.get(job.job_id, session_id="session-b")

    queue.delete_for_session("session-a")
    gate.set()
    queue.shutdown()
    with pytest.raises(JobNotFound):
        queue.get(job.job_id, session_id="session-a")


def test_job_queue_returns_completed_result():
    queue = JobQueue(max_workers=1)
    job = queue.submit("session-a", lambda: {"ok": True})
    # shutdown(wait=True) is a deterministic wait for this unit-size job.
    queue.shutdown()
    result = queue.get(job.job_id, session_id="session-a")
    assert result.status is JobStatus.SUCCEEDED
    assert result.result == {"ok": True}
