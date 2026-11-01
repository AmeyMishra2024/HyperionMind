from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from time import monotonic


@dataclass(slots=True)
class Turn:
    role: str
    text: str


@dataclass(slots=True)
class PersonMemory:
    person_id: str
    turns: list[Turn] = field(default_factory=list)
    last_seen: float = field(default_factory=monotonic)

    def add(self, role: str, text: str) -> None:
        clean = " ".join(text.split()).strip()
        if clean:
            self.turns.append(Turn(role=role, text=clean))
            # Bound local memory while retaining enough context for a natural chat.
            self.turns = self.turns[-24:]
        self.last_seen = monotonic()

    def as_openai_input(self) -> list[dict[str, str]]:
        return [{"role": turn.role, "content": turn.text} for turn in self.turns]


class MemoryStore:
    """Thread-safe, session-local conversation memory separated by person ID."""

    def __init__(self) -> None:
        self._people: dict[str, PersonMemory] = {}
        self._lock = RLock()

    def for_person(self, person_id: str) -> PersonMemory:
        with self._lock:
            if person_id not in self._people:
                self._people[person_id] = PersonMemory(person_id=person_id)
            memory = self._people[person_id]
            memory.last_seen = monotonic()
            return memory

    def add(self, person_id: str, role: str, text: str) -> None:
        with self._lock:
            self.for_person(person_id).add(role, text)

    def context(self, person_id: str) -> list[dict[str, str]]:
        with self._lock:
            return list(self.for_person(person_id).as_openai_input())

    def forget(self, person_id: str) -> bool:
        with self._lock:
            return self._people.pop(person_id, None) is not None

    def clear(self) -> None:
        with self._lock:
            self._people.clear()

    def snapshot(self) -> dict[str, list[dict[str, str]]]:
        with self._lock:
            return {
                person_id: memory.as_openai_input()
                for person_id, memory in self._people.items()
            }

