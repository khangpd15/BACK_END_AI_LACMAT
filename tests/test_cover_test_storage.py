"""Unit and integration test suite for Cover Test Cloud Storage and Persistence API."""

import io
import json
import os
import uuid
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from app.main import app
from app.db.database import init_db
from app.services.cover_test.storage_service import get_storage_service

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_test_db():
    """Ensure in-memory SQLite tables are created before each test and Cloudinary is mocked."""
    from app.services.cover_test.cloudinary_service import get_cloudinary_service
    c_svc = get_cloudinary_service()
    c_svc.mock_mode = True
    c_svc._configured = True
    import anyio
    anyio.run(init_db)


def _create_valid_sample(idx: int, t_ms: float, left_x: float = 0.45, right_x: float = 0.58) -> dict:
    return {
        "index": idx,
        "timestamp": 1727500000000 + int(t_ms),
        "t": t_ms,
        "phase": "BASELINE" if t_ms < 4000 else "COVER_LEFT",
        "coveredEye": None if t_ms < 4000 else "LEFT",
        "leftX": left_x,
        "leftY": 0.50,
        "leftValid": True,
        "rightX": right_x,
        "rightY": 0.50,
        "rightValid": True,
        "trackingQuality": 0.95,
        "relativeX": round(right_x - left_x, 4),
        "relativeY": 0.0,
        "signedDx": 0.0,
        "signedDy": 0.0,
    }


def _create_valid_cycle(cycle_num: int, sample_count: int = 10) -> dict:
    samples = [_create_valid_sample(i, float(i * 66.7)) for i in range(sample_count)]
    return {
        "cycle": cycle_num,
        "coveredEye": "LEFT" if cycle_num % 2 == 1 else "RIGHT",
        "trackedEye": "RIGHT" if cycle_num % 2 == 1 else "LEFT",
        "durationMs": int(sample_count * 66.7),
        "samples": samples,
    }


def _dummy_jpeg_bytes() -> bytes:
    # Minimal JPEG header bytes for unit testing
    return b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00\x00\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\xff\xc0\x00\x0b\x08\x01\x00\x01\x00\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xbf\x00\xff\xd9"


# =============================================================================
# 1. UUID V4 VALIDATION TESTS
# =============================================================================

def test_valid_uuid_v4():
    valid_uuid = str(uuid.uuid4())
    cycle_1 = _create_valid_cycle(1)
    metadata = {
        "sessionId": valid_uuid,
        "cycleCount": 1,
        "samplingRateHz": 15.0,
        "sourceDevice": "WEBCAM",
        "tracker": "MEDIAPIPE_IRIS",
        "rawSchemaVersion": "1.0.0",
        "clientMetadata": {},
    }
    files = {
        "session_metadata": (None, json.dumps(metadata)),
        "cycle_1_raw": ("raw.json", json.dumps(cycle_1), "application/json"),
        "cycle_1_left_eye": ("left.jpg", _dummy_jpeg_bytes(), "image/jpeg"),
    }
    resp = client.post("/api/v1/cover-test/sessions?run_inference=false", files=files)
    assert resp.status_code == 201
    data = resp.json()
    assert data["success"] is True
    assert data["sessionId"] == valid_uuid
    assert data["saved"] is True


def test_invalid_uuid_rejected():
    invalid_uuid = "cover_1727500000000_not_uuid"
    cycle_1 = _create_valid_cycle(1)
    metadata = {
        "sessionId": invalid_uuid,
        "cycleCount": 1,
        "samplingRateHz": 15.0,
        "sourceDevice": "WEBCAM",
        "tracker": "MEDIAPIPE_IRIS",
        "rawSchemaVersion": "1.0.0",
    }
    files = {
        "session_metadata": (None, json.dumps(metadata)),
        "cycle_1_raw": ("raw.json", json.dumps(cycle_1), "application/json"),
    }
    resp = client.post("/api/v1/cover-test/sessions?run_inference=false", files=files)
    assert resp.status_code == 422
    assert "not a valid UUID v4" in resp.json()["detail"]


# =============================================================================
# 2. COORDINATE BOUNDS & FINITENESS TESTS
# =============================================================================

@pytest.mark.parametrize("invalid_coord", [1.1, -0.05, float("inf"), float("-inf")])
def test_out_of_bounds_coordinate_rejected(invalid_coord):
    valid_uuid = str(uuid.uuid4())
    cycle_1 = _create_valid_cycle(1, 5)
    cycle_1["samples"][2]["leftX"] = invalid_coord
    metadata = {"sessionId": valid_uuid}
    files = {
        "session_metadata": (None, json.dumps(metadata)),
        "cycle_1_raw": ("raw.json", json.dumps(cycle_1), "application/json"),
    }
    resp = client.post("/api/v1/cover-test/sessions?run_inference=false", files=files)
    assert resp.status_code == 422


def test_top_level_xy_out_of_bounds_rejected():
    """Verify top-level x or y coordinates > 1.0 or < 0.0 are strictly rejected."""
    valid_uuid = str(uuid.uuid4())
    cycle_1 = _create_valid_cycle(1, 5)
    cycle_1["samples"][2]["x"] = 1.65
    metadata = {"sessionId": valid_uuid}
    files = {
        "session_metadata": (None, json.dumps(metadata)),
        "cycle_1_raw": ("raw.json", json.dumps(cycle_1), "application/json"),
    }
    resp = client.post("/api/v1/cover-test/sessions?run_inference=false", files=files)
    assert resp.status_code == 422
    assert "out of bounds" in resp.text


def test_nan_coordinate_rejected():
    valid_uuid = str(uuid.uuid4())
    cycle_1 = _create_valid_cycle(1, 5)
    cycle_1["samples"][1]["leftX"] = float("nan")
    metadata = {"sessionId": valid_uuid}
    files = {
        "session_metadata": (None, json.dumps(metadata)),
        "cycle_1_raw": ("raw.json", json.dumps(cycle_1), "application/json"),
    }
    resp = client.post("/api/v1/cover-test/sessions?run_inference=false", files=files)
    assert resp.status_code == 422


# =============================================================================
# 3. TIMESTAMP MONOTONICITY TESTS
# =============================================================================

def test_decreasing_timestamp_rejected():
    valid_uuid = str(uuid.uuid4())
    cycle_1 = _create_valid_cycle(1, 5)
    cycle_1["samples"][2]["t"] = 10.0
    cycle_1["samples"][3]["t"] = 5.0  # Decreasing timestamp
    metadata = {"sessionId": valid_uuid}
    files = {
        "session_metadata": (None, json.dumps(metadata)),
        "cycle_1_raw": ("raw.json", json.dumps(cycle_1), "application/json"),
    }
    resp = client.post("/api/v1/cover-test/sessions?run_inference=false", files=files)
    assert resp.status_code == 422
    assert "non-monotonic timestamp" in resp.json()["detail"]


# =============================================================================
# 4. TRACKING QUALITY BOUNDS
# =============================================================================

@pytest.mark.parametrize("invalid_q", [-0.1, 1.05])
def test_invalid_tracking_quality_rejected(invalid_q):
    valid_uuid = str(uuid.uuid4())
    cycle_1 = _create_valid_cycle(1, 5)
    cycle_1["samples"][1]["trackingQuality"] = invalid_q
    metadata = {"sessionId": valid_uuid}
    files = {
        "session_metadata": (None, json.dumps(metadata)),
        "cycle_1_raw": ("raw.json", json.dumps(cycle_1), "application/json"),
    }
    resp = client.post("/api/v1/cover-test/sessions?run_inference=false", files=files)
    assert resp.status_code == 422


# =============================================================================
# 5. FORBIDDEN PII REJECTION TESTS
# =============================================================================

@pytest.mark.parametrize("pii_key", ["name", "phone", "email", "cccd", "dob", "address"])
def test_pii_in_metadata_rejected(pii_key):
    valid_uuid = str(uuid.uuid4())
    cycle_1 = _create_valid_cycle(1, 5)
    metadata = {
        "sessionId": valid_uuid,
        "clientMetadata": {pii_key: "confidential_value"},
    }
    files = {
        "session_metadata": (None, json.dumps(metadata)),
        "cycle_1_raw": ("raw.json", json.dumps(cycle_1), "application/json"),
    }
    resp = client.post("/api/v1/cover-test/sessions?run_inference=false", files=files)
    assert resp.status_code == 422
    assert "PII" in resp.json()["detail"]


# =============================================================================
# 6. IMAGE VALIDATION TESTS
# =============================================================================

def test_empty_image_rejected():
    valid_uuid = str(uuid.uuid4())
    cycle_1 = _create_valid_cycle(1, 5)
    metadata = {"sessionId": valid_uuid}
    files = {
        "session_metadata": (None, json.dumps(metadata)),
        "cycle_1_raw": ("raw.json", json.dumps(cycle_1), "application/json"),
        "cycle_1_left_eye": ("left.jpg", b"", "image/jpeg"),  # 0 bytes
    }
    resp = client.post("/api/v1/cover-test/sessions?run_inference=false", files=files)
    assert resp.status_code == 422
    assert "empty" in resp.json()["detail"]


def test_invalid_image_mime_rejected():
    valid_uuid = str(uuid.uuid4())
    cycle_1 = _create_valid_cycle(1, 5)
    metadata = {"sessionId": valid_uuid}
    files = {
        "session_metadata": (None, json.dumps(metadata)),
        "cycle_1_raw": ("raw.json", json.dumps(cycle_1), "application/json"),
        "cycle_1_left_eye": ("left.png", b"fake_png_data", "image/png"),  # Invalid MIME
    }
    resp = client.post("/api/v1/cover-test/sessions?run_inference=false", files=files)
    assert resp.status_code == 422
    assert "invalid MIME" in resp.json()["detail"]


# =============================================================================
# 7. IDEMPOTENCY TEST
# =============================================================================

def test_idempotent_session_resubmission():
    session_uuid = str(uuid.uuid4())
    cycle_1 = _create_valid_cycle(1, 5)
    metadata = {
        "sessionId": session_uuid,
        "cycleCount": 1,
        "samplingRateHz": 15.0,
    }
    files = {
        "session_metadata": (None, json.dumps(metadata)),
        "cycle_1_raw": ("raw.json", json.dumps(cycle_1), "application/json"),
        "cycle_1_left_eye": ("left.jpg", _dummy_jpeg_bytes(), "image/jpeg"),
    }

    # First submission
    resp1 = client.post("/api/v1/cover-test/sessions?run_inference=false", files=files)
    assert resp1.status_code == 201

    # Second submission with same sessionId (network retry)
    files2 = {
        "session_metadata": (None, json.dumps(metadata)),
        "cycle_1_raw": ("raw.json", json.dumps(cycle_1), "application/json"),
        "cycle_1_left_eye": ("left.jpg", _dummy_jpeg_bytes(), "image/jpeg"),
    }
    resp2 = client.post("/api/v1/cover-test/sessions?run_inference=false", files=files2)
    assert resp2.status_code == 201
    assert resp2.json()["sessionId"] == session_uuid
    assert resp2.json()["saved"] is True


# =============================================================================
# 8. AI FAILURE RESILIENCE TEST
# =============================================================================

def test_raw_data_survives_ai_failure():
    session_uuid = str(uuid.uuid4())
    cycle_1 = _create_valid_cycle(1, 10)
    metadata = {
        "sessionId": session_uuid,
        "cycleCount": 1,
        "samplingRateHz": 15.0,
    }
    files = {
        "session_metadata": (None, json.dumps(metadata)),
        "cycle_1_raw": ("raw.json", json.dumps(cycle_1), "application/json"),
        "cycle_1_left_eye": ("left.jpg", _dummy_jpeg_bytes(), "image/jpeg"),
    }

    # Mock FpsModelService.aggregate_and_predict raising an exception
    with patch("app.services.fps_model_service.FpsModelService.aggregate_and_predict", side_effect=RuntimeError("AI Model OOM")):
        resp = client.post("/api/v1/cover-test/sessions?run_inference=true", files=files)
        # Raw data must still be persisted with HTTP 201 and PARTIAL_SUCCESS
        assert resp.status_code == 201
        data = resp.json()
        assert data["success"] is True
        assert data["saved"] is True
        assert data["processingStatus"] == "PARTIAL_SUCCESS"
        assert data["aiResult"] is None


# =============================================================================
# 9. HAPPY PATH FULL 3-CYCLE PERSISTENCE TEST
# =============================================================================

def test_happy_path_3_cycles_6_images():
    session_uuid = str(uuid.uuid4())
    cycle_1 = _create_valid_cycle(1, 15)
    cycle_2 = _create_valid_cycle(2, 15)
    cycle_3 = _create_valid_cycle(3, 15)

    metadata = {
        "sessionId": session_uuid,
        "cycleCount": 3,
        "samplingRateHz": 15.0,
        "sourceDevice": "WEBCAM",
        "tracker": "MEDIAPIPE_IRIS",
        "rawSchemaVersion": "1.0.0",
        "clientMetadata": {"screen": "1920x1080"},
    }

    files = {
        "session_metadata": (None, json.dumps(metadata)),
        "cycle_1_raw": ("raw.json", json.dumps(cycle_1), "application/json"),
        "cycle_1_left_eye": ("c1_left.jpg", _dummy_jpeg_bytes(), "image/jpeg"),
        "cycle_1_right_eye": ("c1_right.jpg", _dummy_jpeg_bytes(), "image/jpeg"),
        "cycle_2_raw": ("raw.json", json.dumps(cycle_2), "application/json"),
        "cycle_2_left_eye": ("c2_left.jpg", _dummy_jpeg_bytes(), "image/jpeg"),
        "cycle_2_right_eye": ("c2_right.jpg", _dummy_jpeg_bytes(), "image/jpeg"),
        "cycle_3_raw": ("raw.json", json.dumps(cycle_3), "application/json"),
        "cycle_3_left_eye": ("c3_left.jpg", _dummy_jpeg_bytes(), "image/jpeg"),
        "cycle_3_right_eye": ("c3_right.jpg", _dummy_jpeg_bytes(), "image/jpeg"),
    }

    storage = get_storage_service()

    resp = client.post("/api/v1/cover-test/sessions?run_inference=true", files=files)
    assert resp.status_code == 201
    data = resp.json()
    assert data["success"] is True
    assert data["saved"] is True
    assert data["cyclesSaved"] == 3
    assert data["imagesSaved"] == 6
    assert data["processingStatus"] == "COMPLETED"
    assert data["aiResult"] is not None
    assert "notice" in data["aiResult"]
    assert "not a medical diagnosis" in data["aiResult"]["notice"]

    # Verify Storage Objects: 1 manifest.json + 3 raw.json + 6 eye images = exactly 10 Storage objects!
    year_str = resp.json()["storageRoot"].split("/")[1]
    month_str = resp.json()["storageRoot"].split("/")[2]
    prefix = f"{year_str}/{month_str}/{session_uuid}"

    expected_storage_keys = [
        f"{prefix}/manifest.json",
        f"{prefix}/cycle_01/raw.json",
        f"{prefix}/cycle_01/left_eye.jpg",
        f"{prefix}/cycle_01/right_eye.jpg",
        f"{prefix}/cycle_02/raw.json",
        f"{prefix}/cycle_02/left_eye.jpg",
        f"{prefix}/cycle_02/right_eye.jpg",
        f"{prefix}/cycle_03/raw.json",
        f"{prefix}/cycle_03/left_eye.jpg",
        f"{prefix}/cycle_03/right_eye.jpg",
    ]
    assert len(expected_storage_keys) == 10

    # In mock offline storage, verify that all 10 keys were stored
    for key in expected_storage_keys:
        assert key in storage._in_memory_store, f"Storage object '{key}' missing from Storage!"


# =============================================================================
# 10. DATABASE & PAYLOAD FIELD REGRESSION TESTS
# =============================================================================

def test_float_duration_and_eye_normalization():
    """Verify float durationMs (e.g. from JS performance.now) and varied eye naming are accepted."""
    session_uuid = str(uuid.uuid4())
    cycle_1 = _create_valid_cycle(1, 5)
    cycle_1["durationMs"] = 3450.789  # float duration
    cycle_1["coveredEye"] = "cover_left"
    cycle_1["trackedEye"] = "track_right"

    metadata = {
        "sessionId": session_uuid,
        "cycleCount": 1,
        "samplingRateHz": 15.0,
    }
    files = {
        "session_metadata": (None, json.dumps(metadata)),
        "cycle_1_raw": ("raw.json", json.dumps(cycle_1), "application/json"),
    }
    resp = client.post("/api/v1/cover-test/sessions?run_inference=false", files=files)
    assert resp.status_code == 201
    assert resp.json()["success"] is True


def test_orphaned_image_without_trajectory_does_not_break_fk():
    """Verify uploading an image for a cycle without raw trajectory does not violate FK."""
    session_uuid = str(uuid.uuid4())
    cycle_1 = _create_valid_cycle(1, 5)

    metadata = {
        "sessionId": session_uuid,
        "cycleCount": 1,
    }
    # Upload cycle 1 raw, but upload cycle 2 left eye image
    files = {
        "session_metadata": (None, json.dumps(metadata)),
        "cycle_1_raw": ("raw.json", json.dumps(cycle_1), "application/json"),
        "cycle_2_left_eye": ("c2_left.jpg", _dummy_jpeg_bytes(), "image/jpeg"),
    }
    resp = client.post("/api/v1/cover-test/sessions?run_inference=false", files=files)
    assert resp.status_code == 201
    assert resp.json()["success"] is True
    assert resp.json()["imagesSaved"] == 1


def test_session_id_alias_resolution():
    """Verify session_id / sampleId in metadata is properly recognized."""
    session_uuid = str(uuid.uuid4())
    cycle_1 = _create_valid_cycle(1, 5)

    metadata = {
        "session_id": session_uuid,  # snake_case alias
        "cycleCount": 1,
    }
    files = {
        "session_metadata": (None, json.dumps(metadata)),
        "cycle_1_raw": ("raw.json", json.dumps(cycle_1), "application/json"),
    }
    resp = client.post("/api/v1/cover-test/sessions?run_inference=false", files=files)
    assert resp.status_code == 201
    assert resp.json()["sessionId"] == session_uuid


def test_format_async_db_url():
    """Verify DATABASE_URL formatting converts postgres:// and handles sslmode for asyncpg."""
    from app.db.database import format_async_db_url

    assert "sqlite" in format_async_db_url("")
    assert format_async_db_url("postgres://u:p@h:5432/d?sslmode=require") == "postgresql+asyncpg://u:p@h:5432/d?ssl=require"
    assert format_async_db_url("postgresql://u:p@h:5432/d") == "postgresql+asyncpg://u:p@h:5432/d"


# =============================================================================
# 11. 10-15 FPS MODEL & CLOUDINARY INTEGRATION TESTS
# =============================================================================

def test_10_15_fps_model_result_fields_and_db_persistence():
    """Verify 10-15 FPS model is saved to database with Cloudinary URL, and Korean model is not the primary result."""
    session_uuid = str(uuid.uuid4())
    cycle_1 = _create_valid_cycle(1, 20)
    cycle_2 = _create_valid_cycle(2, 20)

    metadata = {
        "sessionId": session_uuid,
        "cycleCount": 2,
        "samplingRateHz": 15.0,
    }
    files = {
        "session_metadata": (None, json.dumps(metadata)),
        "cycle_1_raw": ("raw.json", json.dumps(cycle_1), "application/json"),
        "cycle_1_left_eye": ("c1_left.jpg", _dummy_jpeg_bytes(), "image/jpeg"),
        "cycle_2_raw": ("raw.json", json.dumps(cycle_2), "application/json"),
        "cycle_2_right_eye": ("c2_right.jpg", _dummy_jpeg_bytes(), "image/jpeg"),
    }

    resp = client.post("/api/v1/cover-test/sessions?run_inference=true", files=files)
    assert resp.status_code == 201
    data = resp.json()
    assert data["success"] is True
    assert data["processingStatus"] == "COMPLETED"
    assert data["aiResult"] is not None

    ai = data["aiResult"]
    # Model source must be 10-15 FPS model
    assert ai["modelSource"] == "fps_10_15_model"
    assert ai["prediction"] in ("NORMAL", "STRABISMUS")
    assert "NORMAL" in ai["classProbabilities"]
    assert "STRABISMUS" in ai["classProbabilities"]
    assert ai["imageUrl"].startswith("https://res.cloudinary.com/")
    assert ai["cloudinaryPublicId"] is not None
    assert "notice" in ai
    assert "10-15 FPS" in ai["notice"]

    # Verify Database record directly in cover_test_results
    import anyio
    from sqlalchemy import select
    from app.db.database import get_session_factory
    from app.db.models.cover_test_result import CoverTestResultModel

    async def _verify_db():
        factory = get_session_factory()
        async with factory() as db:
            stmt = select(CoverTestResultModel).where(CoverTestResultModel.session_id == uuid.UUID(session_uuid))
            res = await db.execute(stmt)
            record = res.scalars().first()
            assert record is not None
            assert record.model_source == "fps_10_15_model"
            assert record.prediction == ai["prediction"]
            assert record.class_probabilities == ai["classProbabilities"]
            assert record.image_url == ai["imageUrl"]
            assert record.cloudinary_public_id == ai["cloudinaryPublicId"]
            assert record.model_name in ("remicare-fps-10-15", "Korean 10-15 FPS robust transfer candidate")

    anyio.run(_verify_db)


def test_cloudinary_upload_failure_blocks_db_insert():
    """Verify that when Cloudinary upload fails, NO record is inserted into cover_test_results."""
    session_uuid = str(uuid.uuid4())
    cycle_1 = _create_valid_cycle(1, 15)

    metadata = {
        "sessionId": session_uuid,
        "cycleCount": 1,
        "samplingRateHz": 15.0,
    }
    files = {
        "session_metadata": (None, json.dumps(metadata)),
        "cycle_1_raw": ("raw.json", json.dumps(cycle_1), "application/json"),
        "cycle_1_left_eye": ("c1_left.jpg", _dummy_jpeg_bytes(), "image/jpeg"),
    }

    from app.services.cover_test.cloudinary_service import CloudinaryService

    with patch.object(CloudinaryService, "upload_image", side_effect=RuntimeError("Cloudinary connection reset")):
        resp = client.post("/api/v1/cover-test/sessions?run_inference=true", files=files)
        assert resp.status_code == 201
        data = resp.json()
        assert data["success"] is True
        assert data["saved"] is True
        # Processing status must be PARTIAL_SUCCESS (raw data saved, but no AI result)
        assert data["processingStatus"] == "PARTIAL_SUCCESS"
        assert data["aiResult"] is None
        assert "IMAGE_UPLOAD_FAILED" in data["message"]

    # Verify no record was inserted in cover_test_results
    import anyio
    from sqlalchemy import select
    from app.db.database import get_session_factory
    from app.db.models.cover_test_result import CoverTestResultModel

    async def _verify_no_record():
        factory = get_session_factory()
        async with factory() as db:
            stmt = select(CoverTestResultModel).where(CoverTestResultModel.session_id == uuid.UUID(session_uuid))
            res = await db.execute(stmt)
            record = res.scalars().first()
            assert record is None, "Record must NOT be inserted if Cloudinary upload failed!"

    anyio.run(_verify_no_record)


def test_db_insert_failure_cleans_up_cloudinary_image():
    """Verify that if DB insert fails after Cloudinary upload, the image is cleaned up to prevent orphan image."""
    session_uuid = str(uuid.uuid4())
    cycle_1 = _create_valid_cycle(1, 15)

    metadata = {
        "sessionId": session_uuid,
        "cycleCount": 1,
        "samplingRateHz": 15.0,
    }
    files = {
        "session_metadata": (None, json.dumps(metadata)),
        "cycle_1_raw": ("raw.json", json.dumps(cycle_1), "application/json"),
        "cycle_1_left_eye": ("c1_left.jpg", _dummy_jpeg_bytes(), "image/jpeg"),
    }

    from app.services.cover_test.cloudinary_service import CloudinaryService
    from app.db.repositories.cover_test_repository import CoverTestRepository

    deleted_pids = []

    def mock_delete(self, public_id):
        deleted_pids.append(public_id)
        return True

    with patch.object(CoverTestRepository, "upsert_result", side_effect=RuntimeError("Simulated DB Disk Full")):
        with patch.object(CloudinaryService, "delete_image", new=mock_delete):
            resp = client.post("/api/v1/cover-test/sessions?run_inference=true", files=files)
            # Should result in 500 error and exception must NOT be swallowed silently
            assert resp.status_code == 500
            # Image cleanup must have been called
            assert len(deleted_pids) == 1
            assert session_uuid in deleted_pids[0]


def test_korean_model_separate_endpoint_unaffected():
    """Verify Korean model endpoint /api/v1/transfer/strabismus works for research without inserting to DB."""
    sample_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw", "sample.json")
    with open(sample_file, "r", encoding="utf-8") as f:
        valid_payload = json.load(f)

    resp = client.post("/api/v1/transfer/strabismus", json=valid_payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "TRANSFER_EXPERIMENT"
    assert data["prediction"] in ("NORMAL", "STRABISMUS")
    assert "classProbability" in data
    assert data["domainShiftWarning"] is True

