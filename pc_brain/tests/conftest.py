import pytest

from app import main
from app.coordinator import BrainCoordinator
from app.journal import EventJournal


@pytest.fixture(autouse=True)
def isolated_main_brain(monkeypatch, tmp_path):
    coordinator = BrainCoordinator(EventJournal(tmp_path / "main-brain.db"))
    monkeypatch.setattr(main, "brain_journal", coordinator.journal)
    monkeypatch.setattr(main, "brain_coordinator", coordinator)
    monkeypatch.setattr(main, "realtime_gateway", None)
    monkeypatch.setattr(main, "eye_controller", None)
    monkeypatch.setattr(main, "frame_broker", None)
    monkeypatch.setattr(main, "vision_service", None)
    monkeypatch.setattr(main, "tracking_service", None)
    monkeypatch.setattr(main, "robot_status_cache", None)
    monkeypatch.setattr(main, "robot_status_cache_at", 0.0)
    yield
