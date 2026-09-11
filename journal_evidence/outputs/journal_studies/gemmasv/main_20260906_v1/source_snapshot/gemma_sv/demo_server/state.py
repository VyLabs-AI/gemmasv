"""Ephemeral, thread-safe session state for the hosted demo."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import secrets
import threading
import time
from typing import Any, Callable


class DemoPhase(str, Enum):
    CREATED = "created"
    INGESTED = "ingested"
    RECALLED = "recalled"
    FORGOTTEN = "forgotten"
    ATTACKED = "attacked"
    TWIN_TESTED = "twin_tested"


@dataclass
class SessionRecord:
    session_id: str
    created_at: float
    touched_at: float
    phase: DemoPhase = DemoPhase.CREATED
    domain: str = "custom"
    memory_text: str | None = None
    secret_start: int | None = None
    secret_end: int | None = None
    deletion_ranges: tuple[tuple[int, int], ...] = ()
    deletion_scope: str = "field"
    record_id: str | None = None
    audit_probe: str | None = None
    engine_state: dict[str, Any] = field(default_factory=dict)
    results: dict[str, Any] = field(default_factory=dict)
    lock: threading.RLock = field(
        default_factory=threading.RLock,
        repr=False,
        compare=False,
    )

    def clear_sensitive(self) -> None:
        """Erase visitor-supplied text and model state before dropping a session."""

        with self.lock:
            self.memory_text = None
            self.audit_probe = None
            self.secret_start = None
            self.secret_end = None
            self.deletion_ranges = ()
            self.deletion_scope = "field"
            self.record_id = None
            self.engine_state.clear()
            self.results.clear()


class SessionNotFound(KeyError):
    pass


class SessionStore:
    """In-memory sessions with sliding expiry and bounded capacity."""

    def __init__(
        self,
        *,
        ttl_seconds: float = 30 * 60,
        max_sessions: int = 128,
        clock: Callable[[], float] = time.monotonic,
        on_remove: Callable[[SessionRecord], None] | None = None,
    ):
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if max_sessions <= 0:
            raise ValueError("max_sessions must be positive")
        self.ttl_seconds = float(ttl_seconds)
        self.max_sessions = int(max_sessions)
        self._clock = clock
        self._on_remove = on_remove
        self._lock = threading.RLock()
        self._sessions: dict[str, SessionRecord] = {}

    def create(self, *, domain: str = "custom") -> SessionRecord:
        with self._lock:
            self.purge_expired()
            if len(self._sessions) >= self.max_sessions:
                raise RuntimeError("session capacity reached")
            now = self._clock()
            session_id = secrets.token_urlsafe(24)
            record = SessionRecord(session_id, now, now, domain=domain)
            self._sessions[session_id] = record
            return record

    def get(self, session_id: str, *, touch: bool = True) -> SessionRecord:
        with self._lock:
            record = self._sessions.get(session_id)
            if record is None:
                raise SessionNotFound(session_id)
            now = self._clock()
            if now - record.touched_at > self.ttl_seconds:
                self._remove_locked(session_id)
                raise SessionNotFound(session_id)
            if touch:
                record.touched_at = now
            return record

    def delete(self, session_id: str) -> bool:
        with self._lock:
            if session_id not in self._sessions:
                return False
            self._remove_locked(session_id)
            return True

    def purge_expired(self) -> int:
        with self._lock:
            now = self._clock()
            expired = [
                session_id
                for session_id, record in self._sessions.items()
                if now - record.touched_at > self.ttl_seconds
            ]
            for session_id in expired:
                self._remove_locked(session_id)
            return len(expired)

    def snapshot_counts(self) -> dict[str, int]:
        with self._lock:
            counts = {phase.value: 0 for phase in DemoPhase}
            for record in self._sessions.values():
                counts[record.phase.value] += 1
            counts["total"] = len(self._sessions)
            return counts

    def _remove_locked(self, session_id: str) -> None:
        record = self._sessions.pop(session_id)
        record.clear_sensitive()
        if self._on_remove is not None:
            self._on_remove(record)
