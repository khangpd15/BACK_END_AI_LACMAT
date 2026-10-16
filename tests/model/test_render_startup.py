import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx

from app import main


def test_ping_available_while_hirschberg_is_still_loading(monkeypatch):
    started = threading.Event()
    release = threading.Event()

    def slow_preload():
        started.set()
        assert release.wait(timeout=5)

    monkeypatch.setattr(main, "init_db", AsyncMock())
    monkeypatch.setattr(main, "get_strabismus_inference_service",
                        lambda: SimpleNamespace(is_loaded=False))
    monkeypatch.setattr(main, "preload_trained_hirschberg", slow_preload)
    monkeypatch.setattr(main, "start_keep_alive", lambda: None)
    stop = AsyncMock()
    monkeypatch.setattr(main, "stop_keep_alive", stop)

    async def run():
        async with main.lifespan(main.app):
            try:
                assert await asyncio.to_thread(started.wait, 2)
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=main.app),
                    base_url="http://test",
                ) as client:
                    response = await asyncio.wait_for(client.get("/ping"), timeout=1)
                assert response.status_code == 200
                assert response.json()["status"] == "alive"
                assert not release.is_set()
            finally:
                release.set()

    asyncio.run(run())
    stop.assert_awaited_once()
