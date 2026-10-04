import asyncio
from pathlib import Path
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from app.schemas.cover_test import CoverTestSessionMetadataInput
from app.schemas.research_measurement import ResearchMeasurementRequest


def request():
    return ResearchMeasurementRequest(schemaVersion="remicare-research-quality-v0.1",
        featureVersion="research-geometry-v0.1", testType="COVER", sessionId="test",
        distance_bucket="UNKNOWN", eligibility={"consent": True})


def test_measurement_does_not_block_event_loop_and_releases_slot(monkeypatch):
    from app.api import research_measurement as api
    main_thread = threading.get_ident()
    worker_threads = []

    def slow_measurement(req):
        worker_threads.append(threading.get_ident())
        time.sleep(0.15)
        return {"status": "INCONCLUSIVE"}

    monkeypatch.setattr(api, "measure_research_request", slow_measurement)

    async def run():
        lock = asyncio.Lock()
        monkeypatch.setattr(api, "_measurement_lock", lock)
        measurement = asyncio.create_task(api.create_research_measurement(request()))
        await asyncio.sleep(0.03)
        assert not measurement.done()
        assert await measurement == {"status": "INCONCLUSIVE"}
        assert not lock.locked()
        await api.create_research_measurement(request())

    asyncio.run(run())
    assert all(t != main_thread for t in worker_threads)


def test_busy_measurement_returns_retryable_503(monkeypatch):
    from app.api import research_measurement as api

    async def run():
        lock = asyncio.Lock()
        await lock.acquire()
        monkeypatch.setattr(api, "_measurement_lock", lock)
        try:
            with pytest.raises(HTTPException) as error:
                await api.create_research_measurement(request())
            assert error.value.status_code == 503
            assert error.value.detail["code"] == "SERVER_BUSY"
            assert error.value.headers["Retry-After"] == "2"
            assert lock.locked()
        finally:
            lock.release()

    asyncio.run(run())


def test_failed_measurement_releases_slot(monkeypatch):
    from app.api import research_measurement as api

    def fail(req):
        raise RuntimeError("test failure")

    monkeypatch.setattr(api, "measure_research_request", fail)

    async def run():
        lock = asyncio.Lock()
        monkeypatch.setattr(api, "_measurement_lock", lock)
        with pytest.raises(HTTPException) as error:
            await api.create_research_measurement(request())
        assert error.value.status_code == 500
        assert not lock.locked()

    asyncio.run(run())


def test_shared_onnx_session_uses_bounded_threads(monkeypatch, tmp_path):
    from app.services import onnx_runtime as runtime
    sessions = []

    def fake_session(path, sess_options, providers):
        sessions.append((sess_options, providers))
        return object()

    monkeypatch.setattr(runtime, "_sessions", {})
    monkeypatch.setattr(runtime.ort, "InferenceSession", fake_session)
    monkeypatch.setenv("ONNX_CPU_THREADS", "200")
    first = runtime.get_cpu_session(tmp_path / "test.onnx")
    assert first is runtime.get_cpu_session(tmp_path / "test.onnx")
    assert len(sessions) == 1
    assert sessions[0][0].intra_op_num_threads == 4
    assert sessions[0][0].inter_op_num_threads == 1
    assert sessions[0][1] == ["CPUExecutionProvider"]


def test_cover_data_saved_with_explicit_missing_model(monkeypatch):
    from app.services.cover_test import session_service as module
    from app.services import fps_model_service
    monkeypatch.setattr(module, "ENABLE_AI_INFERENCE_ON_SAVE", True)
    monkeypatch.setattr(fps_model_service, "get_fps_model_service", lambda: SimpleNamespace(model=None))
    service = module.CoverTestSessionService.__new__(module.CoverTestSessionService)
    service.db_session = AsyncMock()
    service.repo = AsyncMock()
    service.storage = AsyncMock()
    metadata = CoverTestSessionMetadataInput(sessionId="11111111-1111-4111-8111-111111111111")
    samples = [{"t": i * 16.7, "phase": "BASELINE", "leftX": .4, "leftY": .5,
                "rightX": .6, "rightY": .5, "leftValid": True, "rightValid": True}
               for i in range(3)]
    result = asyncio.run(service.persist_session(metadata, {1: {"samples": samples}}))
    assert result.saved is True
    assert result.success is True
    assert result.aiResult["reason"] == "MODEL_NOT_LOADED"
    assert result.aiResult["prediction"] == "INCONCLUSIVE"
    assert result.aiResult["confidence"] is None
    assert result.aiResult["classProbabilities"] is None
    service.repo.upsert_result.assert_not_called()
    assert service.storage.upload_raw_json.await_count == 2
