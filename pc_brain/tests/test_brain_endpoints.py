from fastapi.testclient import TestClient

from app import main
from app.coordinator import BrainCoordinator
from app.journal import EventJournal


def test_manual_action_is_queryable_by_correlation_id(monkeypatch, tmp_path):
    coordinator = BrainCoordinator(EventJournal(tmp_path / "brain.db"))
    monkeypatch.setattr(main, "brain_coordinator", coordinator)
    monkeypatch.setattr(main, "brain_journal", coordinator.journal)

    async def fake_robot_post(path, body=None):
        return {"ok": True, "movement": body["direction"], "speed": body.get("speed")}

    monkeypatch.setattr(main, "robot_post", fake_robot_post)
    client = TestClient(main.app)
    response = client.post(
        "/robot/drive",
        headers={"x-correlation-id": "trace-me"},
        json={"move": "left", "speed": 90},
    )

    assert response.status_code == 200
    assert response.json()["correlation_id"] == "trace-me"
    events = client.get("/brain/events", params={"correlation_id": "trace-me"}).json()["events"]
    assert {event["event_type"] for event in events} >= {
        "action.proposed",
        "action.approved",
        "action.completed",
    }


def test_brain_event_limit_is_bounded():
    response = TestClient(main.app).get("/brain/events", params={"limit": 501})
    assert response.status_code == 422

