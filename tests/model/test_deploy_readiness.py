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
                await api.predict(image=upload, save_session=False)
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


@pytest.mark.parametrize('proposal', ['normal', 'esotropia', 'exotropia'])
def test_same_direction_allows_classifier_without_forcing_normal(proposal):
    record = {
        'input_contract': {'eligible': True},
        'binocular_comparison': {'available': True, 'same_half_plane': True},
        'eyes': {side: {'reflex_qc': {'eligible_for_score': True}}
                 for side in ('image_left', 'image_right')},
    }
    calls = []
    def classify():
        calls.append(True)
        return {'status': 'READY', 'class_proposal': proposal, 'classifier_called': True}
    result = screen_with_guards(classify, record, 'unmirrored', 'confirmed')
    assert calls == [True]
    assert result['status'] == 'HUMAN_REVIEW'
    assert result['class_proposal'] == proposal


@pytest.mark.parametrize('fixation,available', [('unknown', True), ('offaxis', True), ('confirmed', False)])
def test_capture_still_requires_fixation_and_bilateral_geometry(fixation, available):
    from app.hirschberg_runtime.research.hirschberg_localization.binocular_guard import capture_guard
    result = capture_guard({'status': 'READY'}, {'available': available, 'same_half_plane': True}, fixation)
    assert result['status'] == 'INCONCLUSIVE'
    assert result['class_proposal'] is None


def test_inconclusive_message_hides_technical_reason():
    from app.services.trained_hirschberg_service import screening_message
    for reason in ('common_direction_ambiguous', 'invalid_reflex_landmark', 'invalid_input_contract'):
        message = screening_message({'status': 'INCONCLUSIVE', 'reason': reason})
        assert 'Chưa thể đánh giá' in message
        assert 'vector' not in message
        assert reason not in message


def test_same_direction_does_not_bypass_landmark_qc():
    record = {
        'input_contract': {'eligible': True},
        'binocular_comparison': {'available': True, 'same_half_plane': True},
        'eyes': {side: {'reflex_qc': {'eligible_for_score': False, 'reasons': ['missing_reflex']}}
                 for side in ('image_left', 'image_right')},
    }
    def forbidden():
        raise AssertionError('Classifier must not run with invalid landmarks')
    result = screen_with_guards(forbidden, record, 'unmirrored', 'confirmed')
    assert result['status'] == 'INCONCLUSIVE'
    assert result['classifier_called'] is False
    assert result['reason'] == 'invalid_reflex_landmark'
