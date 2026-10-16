import asyncio
from pathlib import Path
import threading
import time
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from app.schemas.research_measurement import ResearchMeasurementRequest


def request():
    return ResearchMeasurementRequest(
        schemaVersion="remicare-research-quality-v0.1",
        featureVersion="research-geometry-v0.1",
        testType="HIRSCHBERG",
        sessionId="test",
        distance_bucket="UNKNOWN",
        eligibility={"consent": True},
    )


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
