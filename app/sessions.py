from __future__ import annotations

from dataclasses import dataclass, field
from secrets import token_urlsafe
from threading import RLock
from time import monotonic

from app.memory import MemoryStore
from app.tracking import PersonTracker


@dataclass(slots=True)
class Session:
    api_key: str | None
    demo: bool = False
    memory: MemoryStore = field(default_factory=MemoryStore)
    tracker: PersonTracker = field(default_factory=PersonTracker)
    created_at: float = field(default_factory=monotonic)
    last_used: float = field(default_factory=monotonic)
    _turn_busy: bool = False
    _turn_lock: RLock = field(default_factory=RLock, repr=False)

    def try_begin_turn(self) -> bool:
        with self._turn_lock:
            if self._turn_busy:
                return False
            self._turn_busy = True
            return True

    def end_turn(self) -> None:
        with self._turn_lock:
            self._turn_busy = False


class SessionRegistry:
    """Keeps BYOK secrets in process memory only; nothing is written to disk."""

    def __init__(self, ttl_seconds: int = 4 * 60 * 60) -> None:
        self._sessions: dict[str, Session] = {}
        self._ttl_seconds = ttl_seconds
        self._lock = RLock()

    def create(self, api_key: str | None, *, demo: bool = False) -> str:
        with self._lock:
            self._remove_expired()
            session_id = token_urlsafe(32)
            self._sessions[session_id] = Session(api_key=api_key, demo=demo)
            return session_id

    def get(self, session_id: str) -> Session | None:
        with self._lock:
            self._remove_expired()
            session = self._sessions.get(session_id)
            if session:
                session.last_used = monotonic()
            return session

    def delete(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def _remove_expired(self) -> None:
        cutoff = monotonic() - self._ttl_seconds
        expired = [key for key, value in self._sessions.items() if value.last_used < cutoff]
        for key in expired:
            del self._sessions[key]
