import asyncio
from types import SimpleNamespace

import httpx
from fastapi import HTTPException
import pytest

from app import main
from app.api import hirschberg as api
from app.hirschberg_runtime.research.hirschberg_localization.input_contract import validate_input
from app.hirschberg_runtime.scripts.hirschberg_engine import screen_with_guards


def test_readiness_waits_for_model(monkeypatch):
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url='http://test') as client:
            for loaded, code in [(False, 503), (True, 200)]:
                monkeypatch.setattr(main, 'trained_hirschberg_status', lambda: {'loaded': loaded})
                response = await client.get('/ready')
                assert response.status_code == code
                assert response.json()['ready'] is loaded
    asyncio.run(run())


def test_busy_upload_retries_and_closes_file(monkeypatch):
    closed = []
    async def read(_limit):
        return b'valid bytes need not reach model'
    async def close():
        closed.append(True)
    async def run():
        lock = asyncio.Lock()
        await lock.acquire()
        monkeypatch.setattr(api, '_prediction_lock', lock)
        upload = SimpleNamespace(filename='eye.jpg', read=read, close=close)
        try:
            with pytest.raises(HTTPException) as error:
                await api.predict(image=upload, save_session=False, db=None)
            assert error.value.status_code == 503
            assert error.value.headers['Retry-After'] == '2'
            assert closed
        finally:
            lock.release()
    asyncio.run(run())


def test_invalid_input_never_calls_classifier():
    record = {'width': 100, 'height': 50, 'orientation': 'unknown', 'eyes': {}}
    record['input_contract'] = validate_input(record)
    def forbidden():
        raise AssertionError('classifier called on invalid input')
    result = screen_with_guards(forbidden, record, 'unknown', 'confirmed')
    assert result['status'] == 'INCONCLUSIVE'
    assert result['classifier_called'] is False
    assert 'orientation_unverified' in result['input_reasons']
