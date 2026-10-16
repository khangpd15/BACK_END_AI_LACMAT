"""Unit tests for Hirschberg session persistence and retrieval."""

import io
import json
import os
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.config import PROJECT_ROOT


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_hirschberg_predict_saves_session(client):
    image_io = io.BytesIO()
    Image.new("RGB", (320, 160), (128, 128, 128)).save(image_io, format="PNG")
    raw_bytes = image_io.getvalue()

    custom_session_id = "test-hirschberg-session-001"
    response = client.post(
        "/api/v1/hirschberg/predict",
        files={"image": ("test_eye.png", raw_bytes, "image/png")},
        data={
            "session_id": custom_session_id,
            "orientation": "auto",
            "kind": "auto",
            "fixation": "unknown",
            "save_session": "true",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["saved"] is True
    assert data["sessionId"] == custom_session_id
    assert "sessionPath" in data

    # Verify local files on filesystem
    session_dir = os.path.join(str(PROJECT_ROOT), "data", "hirschberg", "sessions", custom_session_id)
    assert os.path.exists(session_dir)
    assert os.path.exists(os.path.join(session_dir, "metadata.json"))
    assert os.path.exists(os.path.join(session_dir, "result.json"))

    # Test GET /sessions/{session_id}
    get_res = client.get(f"/api/v1/hirschberg/sessions/{custom_session_id}")
    assert get_res.status_code == 200
    session_details = get_res.json()
    assert session_details["sessionId"] == custom_session_id
    assert session_details["status"] == data["status"]

    # Test GET /sessions (list)
    list_res = client.get("/api/v1/hirschberg/sessions")
    assert list_res.status_code == 200
    list_data = list_res.json()
    assert list_data["count"] >= 1
    session_ids = [s["sessionId"] for s in list_data["sessions"]]
    assert custom_session_id in session_ids
