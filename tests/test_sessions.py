from app.sessions import Session


def test_session_rejects_overlapping_turns() -> None:
    session = Session(api_key=None, demo=True)
    assert session.try_begin_turn() is True
    assert session.try_begin_turn() is False
    session.end_turn()
    assert session.try_begin_turn() is True
