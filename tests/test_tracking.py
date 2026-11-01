from app.tracking import PersonTracker, intersection_over_union


def detection(box, raised=False, signature=None):
    return {"box": box, "confidence": 0.9, "hand_raised": raised, "signature": signature}


def test_iou() -> None:
    assert intersection_over_union((0, 0, 10, 10), (0, 0, 10, 10)) == 1
    assert intersection_over_union((0, 0, 10, 10), (20, 20, 30, 30)) == 0


def test_tracker_keeps_identity_when_person_moves() -> None:
    tracker = PersonTracker()
    first = tracker.update([detection([10, 10, 100, 200])], now=1.0)[0]
    second = tracker.update([detection([16, 12, 106, 202])], now=2.0)[0]
    assert first["person_id"] == second["person_id"]


def test_hand_raise_requires_three_frames_and_three_frames_to_lower() -> None:
    tracker = PersonTracker()
    results = [tracker.update([detection([10, 10, 100, 200], True)], now=float(i))[0] for i in range(3)]
    assert results[-1]["hand_raised"] is True
    lowered = [tracker.update([detection([10, 10, 100, 200], False)], now=float(i + 3))[0] for i in range(3)]
    assert lowered[-1]["hand_raised"] is False


def test_appearance_keeps_identities_when_people_cross() -> None:
    tracker = PersonTracker()
    red, blue = [1.0, 0.0], [0.0, 1.0]
    first = tracker.update(
        [detection([0, 0, 100, 200], signature=red), detection([110, 0, 210, 200], signature=blue)],
        now=0.0,
    )
    red_id, blue_id = first[0]["person_id"], first[1]["person_id"]

    crossed = tracker.update(
        [detection([110, 0, 210, 200], signature=red), detection([0, 0, 100, 200], signature=blue)],
        now=0.5,
    )
    assert crossed[0]["person_id"] == red_id
    assert crossed[1]["person_id"] == blue_id


def test_returning_person_is_reidentified_by_appearance() -> None:
    tracker = PersonTracker()
    signature = [0.7, 0.2, 0.1]
    original = tracker.update([detection([10, 10, 100, 200], signature=signature)], now=0.0)[0]
    tracker.update([], now=2.0)
    returned = tracker.update([detection([300, 20, 395, 215], signature=signature)], now=8.0)[0]
    assert returned["person_id"] == original["person_id"]


def test_new_appearance_does_not_inherit_old_memory_id() -> None:
    tracker = PersonTracker()
    original = tracker.update([detection([10, 10, 100, 200], signature=[1.0, 0.0])], now=0.0)[0]
    newcomer = tracker.update([detection([10, 10, 100, 200], signature=[0.0, 1.0])], now=8.0)[0]
    assert newcomer["person_id"] != original["person_id"]


def test_motion_prediction_keeps_identity_during_fast_movement() -> None:
    tracker = PersonTracker()
    signature = [0.2, 0.5, 0.3]
    first = tracker.update([detection([0, 10, 100, 210], signature=signature)], now=0.0)[0]
    tracker.update([detection([50, 10, 150, 210], signature=signature)], now=0.5)
    moved = tracker.update([detection([150, 10, 250, 210], signature=signature)], now=1.5)[0]
    assert moved["person_id"] == first["person_id"]
