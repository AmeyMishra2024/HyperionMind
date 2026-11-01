from fastapi.testclient import TestClient

from app.demo import demo_respond, remembered_name
from app.main import app


def test_demo_engine_remembers_a_name() -> None:
    history = [{"role": "user", "content": "My name is Maya"}]
    assert remembered_name(history) == "Maya"
    assert "Maya" in demo_respond(history, "What is my name?")


def test_demo_session_needs_no_api_key_and_keeps_people_separate() -> None:
    client = TestClient(app)
    session_id = client.post("/api/demo-session").json()["session_id"]

    first = client.post(
        "/api/text-turn",
        json={"session_id": session_id, "person_id": "Person 1", "text": "My name is Maya"},
    )
    second = client.post(
        "/api/text-turn",
        json={"session_id": session_id, "person_id": "Person 2", "text": "What is my name?"},
    )
    recall = client.post(
        "/api/text-turn",
        json={"session_id": session_id, "person_id": "Person 1", "text": "What is my name?"},
    )

    assert first.status_code == 200
    assert first.json()["audio"] is None
    assert "not told" in second.json()["reply"]
    assert "Maya" in recall.json()["reply"]

