from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from io import BytesIO
from types import SimpleNamespace

from fastapi.testclient import TestClient
from PIL import Image

from app import main
from app.main import BackendStatus, PersonDetection, RFDetrBackend


@dataclass
class FakeBackend:
    status: BackendStatus

    def load(self) -> None:
        return None

    def detect(self, jpeg: bytes, threshold: float):
        assert jpeg == b"jpeg"
        assert threshold == 0.55
        return [PersonDetection(confidence=0.9, bounding_box=(0.1, 0.2, 0.8, 0.9))], 12.5


def test_health_and_detect(monkeypatch):
    fake = FakeBackend(BackendStatus(True, "cuda/bfloat16", None))
    monkeypatch.setattr(main, "backend", fake)
    with TestClient(main.app) as client:
        health = client.get("/health")
        assert health.json()["available"] is True

        response = client.post(
            "/detect",
            content=b"jpeg",
            headers={
                "Content-Type": "image/jpeg",
                "X-Robit-Frame-Id": "frame-1",
                "X-Robit-Captured-At": "2026-07-23T00:00:00Z",
            },
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["frame_id"] == "frame-1"
    assert payload["people"][0]["label"] == "person"
    assert payload["backend"] == "cuda/bfloat16"


def test_unavailable_is_clean_503(monkeypatch):
    fake = FakeBackend(BackendStatus(False, "unavailable", "model missing"))
    monkeypatch.setattr(main, "backend", fake)
    with TestClient(main.app) as client:
        response = client.post(
            "/detect",
            content=b"jpeg",
            headers={
                "X-Robit-Frame-Id": "frame-1",
                "X-Robit-Captured-At": "2026-07-23T00:00:00Z",
            },
        )
    assert response.status_code == 503
    assert response.json()["detail"] == "model missing"


def test_backend_filters_everything_except_person():
    class FakeModel:
        def predict(self, image, threshold, include_source_image):
            assert image.size == (10, 10)
            assert threshold == 0.55
            assert include_source_image is False
            return SimpleNamespace(
                xyxy=[(1, 2, 9, 10), (0, 0, 5, 5)],
                confidence=[0.91, 0.99],
                class_id=[1, 17],
                data={"class_name": ["person", "cat"]},
            )

    jpeg = BytesIO()
    Image.new("RGB", (10, 10), "white").save(jpeg, format="JPEG")
    detector = RFDetrBackend("cpu")
    detector.model = FakeModel()

    people, _ = detector.detect(jpeg.getvalue(), 0.55)

    assert len(people) == 1
    assert people[0].label == "person"
    assert people[0].bounding_box == (0.1, 0.2, 0.9, 1.0)


def test_detect_serializes_model_inference(monkeypatch):
    class SlowBackend:
        status = BackendStatus(True, "cpu/float32", None)
        active = 0
        max_active = 0

        def detect(self, jpeg, threshold):
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            time.sleep(0.03)
            self.active -= 1
            return [], 30.0

    class FakeRequest:
        async def body(self):
            return b"jpeg"

    slow = SlowBackend()
    monkeypatch.setattr(main, "backend", slow)

    async def exercise():
        monkeypatch.setattr(main, "inference_lock", asyncio.Lock())
        return await asyncio.gather(
            main.detect(FakeRequest(), "frame-1", "2026-07-23T00:00:00Z", 0.55),
            main.detect(FakeRequest(), "frame-2", "2026-07-23T00:00:01Z", 0.55),
        )

    responses = asyncio.run(exercise())

    assert [response.frame_id for response in responses] == ["frame-1", "frame-2"]
    assert slow.max_active == 1
