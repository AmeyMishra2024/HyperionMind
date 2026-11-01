from __future__ import annotations

from dataclasses import dataclass
from time import monotonic

Box = tuple[int, int, int, int]
Signature = tuple[float, ...]


def box_area(box: Box) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def intersection_over_union(a: Box, b: Box) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if not intersection:
        return 0.0
    return intersection / max(1, box_area(a) + box_area(b) - intersection)


def center_distance(a: Box, b: Box) -> float:
    ac = ((a[0] + a[2]) / 2, (a[1] + a[3]) / 2)
    bc = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    diagonal = max(1.0, ((b[2] - b[0]) ** 2 + (b[3] - b[1]) ** 2) ** 0.5)
    return (((ac[0] - bc[0]) ** 2 + (ac[1] - bc[1]) ** 2) ** 0.5) / diagonal


def size_similarity(a: Box, b: Box) -> float:
    smaller, larger = sorted((box_area(a), box_area(b)))
    return smaller / max(1, larger)


def appearance_similarity(a: Signature | None, b: Signature | None) -> float:
    if not a or not b or len(a) != len(b):
        return 0.5
    return max(0.0, min(1.0, sum(min(left, right) for left, right in zip(a, b))))


def blend_signature(old: Signature | None, new: Signature | None) -> Signature | None:
    if not new:
        return old
    if not old or len(old) != len(new):
        return new
    # Keep the original visual profile dominant. A brief occlusion or a crossed
    # detection should not quickly turn one person's profile into another's.
    values = [0.90 * left + 0.10 * right for left, right in zip(old, new)]
    total = sum(values)
    return tuple(value / total for value in values) if total else new


@dataclass(slots=True)
class Track:
    person_id: str
    box: Box
    last_seen: float
    signature: Signature | None = None
    raised_votes: int = 0
    lowered_votes: int = 0
    hand_raised: bool = False
    velocity: tuple[float, float] = (0.0, 0.0)


class PersonTracker:
    """Anonymous re-identification tracker for stable per-person conversation IDs."""

    def __init__(self, keep_seconds: float = 120.0) -> None:
        self.tracks: dict[str, Track] = {}
        self.keep_seconds = keep_seconds
        self._next_id = 1

    def update(self, detections: list[dict], now: float | None = None) -> list[dict]:
        timestamp = monotonic() if now is None else now
        self._expire(timestamp)
        prepared = [self._prepare(item) for item in detections]

        # Global matching avoids the old largest-box-first ID swaps when people cross.
        candidates: list[tuple[float, int, str]] = []
        for detection_index, detection in enumerate(prepared):
            for person_id, track in self.tracks.items():
                score = self._match_score(track, detection, timestamp)
                if score is not None:
                    candidates.append((score, detection_index, person_id))
        candidates.sort(reverse=True)

        detection_matches: dict[int, str] = {}
        matched_tracks: set[str] = set()
        for _, detection_index, person_id in candidates:
            if detection_index in detection_matches or person_id in matched_tracks:
                continue
            detection_matches[detection_index] = person_id
            matched_tracks.add(person_id)

        output: list[dict] = []
        for index, detection in enumerate(prepared):
            person_id = detection_matches.get(index)
            if person_id is None:
                person_id = f"Person {self._next_id}"
                self._next_id += 1
                self.tracks[person_id] = Track(
                    person_id=person_id,
                    box=detection["box"],
                    last_seen=timestamp,
                    signature=detection["signature"],
                )
            track = self.tracks[person_id]
            self._update_track(track, detection, timestamp)
            enriched = {key: value for key, value in detection.items() if key != "signature"}
            enriched.update(person_id=person_id, box=list(track.box), hand_raised=track.hand_raised)
            output.append(enriched)
        return output

    @staticmethod
    def _prepare(detection: dict) -> dict:
        prepared = dict(detection)
        prepared["box"] = tuple(int(value) for value in detection["box"])
        signature = detection.get("signature")
        prepared["signature"] = tuple(float(value) for value in signature) if signature else None
        return prepared

    @staticmethod
    def _match_score(track: Track, detection: dict, now: float) -> float | None:
        age = now - track.last_seen
        box = detection["box"]
        predicted = PersonTracker._predicted_box(track, age)
        overlap = intersection_over_union(predicted, box)
        distance = center_distance(predicted, box)
        appearance = appearance_similarity(track.signature, detection["signature"])
        size = size_similarity(predicted, box)
        if age <= 1.5:
            # Prediction allows a walking person to retain their ID between
            # frames. Appearance remains the strongest cue when paths cross.
            if overlap < 0.01 and distance > 1.35:
                return None
            return 2.25 * overlap + 1.15 * max(0.0, 1.0 - distance / 1.55) + 3.35 * appearance + 0.35 * size

        # After an occlusion, require appearance and size similarity. This prevents a
        # newcomer in the same spot from inheriting another person's private memory.
        if track.signature is None or detection["signature"] is None:
            return None
        if appearance < 0.68 or size < 0.38:
            return None
        return 2.25 * appearance + 0.55 * size - min(0.35, age * 0.003)

    @staticmethod
    def _predicted_box(track: Track, age: float) -> Box:
        """Extrapolate only a short horizon so stale tracks cannot roam."""
        seconds = max(0.0, min(age, 1.5))
        dx = int(round(track.velocity[0] * seconds))
        dy = int(round(track.velocity[1] * seconds))
        x1, y1, x2, y2 = track.box
        return x1 + dx, y1 + dy, x2 + dx, y2 + dy

    @staticmethod
    def _update_track(track: Track, detection: dict, timestamp: float) -> None:
        previous_box, previous_seen = track.box, track.last_seen
        elapsed = timestamp - previous_seen
        if 0.03 <= elapsed <= 2.0:
            old_center = ((previous_box[0] + previous_box[2]) / 2, (previous_box[1] + previous_box[3]) / 2)
            new_box = detection["box"]
            new_center = ((new_box[0] + new_box[2]) / 2, (new_box[1] + new_box[3]) / 2)
            measured = ((new_center[0] - old_center[0]) / elapsed, (new_center[1] - old_center[1]) / elapsed)
            track.velocity = (
                0.65 * track.velocity[0] + 0.35 * measured[0],
                0.65 * track.velocity[1] + 0.35 * measured[1],
            )
        elif elapsed > 2.0:
            track.velocity = (0.0, 0.0)
        track.box = detection["box"]
        track.last_seen = timestamp
        track.signature = blend_signature(track.signature, detection["signature"])
        if detection.get("hand_raised", False):
            track.raised_votes = min(5, track.raised_votes + 1)
            track.lowered_votes = 0
            if track.raised_votes >= 3:
                track.hand_raised = True
        else:
            track.lowered_votes = min(5, track.lowered_votes + 1)
            track.raised_votes = 0
            if track.lowered_votes >= 3:
                track.hand_raised = False

    def _expire(self, now: float) -> None:
        expired = [key for key, track in self.tracks.items() if now - track.last_seen > self.keep_seconds]
        for person_id in expired:
            del self.tracks[person_id]
