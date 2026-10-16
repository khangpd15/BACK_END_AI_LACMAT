import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from app.services.hirschberg_session_service import HirschbergSessionService
from app.services.keep_alive import KeepAliveService


def save(service, session_id, **result):
    return asyncio.run(service.persist_session(session_id, 'eye.jpg', 'unmirrored',
                                               'auto', 'confirmed', result))


def test_local_session_round_trip_without_database(tmp_path):
    service = HirschbergSessionService(tmp_path)
    saved = save(service, 'case-1', status='HUMAN_REVIEW', confidence=.8,
                 uncalibratedModelScores={'normal': .8}, measurements={'x': float('nan')})
    record = service.get_session_local(saved['sessionId'])
    assert record['status'] == 'HUMAN_REVIEW'
    assert record['sessionId'] == saved['sessionId']
    assert record['saved'] is True
    assert record['source'] == 'filesystem'
    assert record['measurements']['x'] is None
    assert set(p.name for p in (tmp_path / saved['sessionId']).iterdir()) == {'metadata.json', 'result.json'}
    assert service.list_sessions_local()['sessions'][0]['sessionId'] == saved['sessionId']


def test_duplicate_ids_remain_retrievable(tmp_path):
    service = HirschbergSessionService(tmp_path)
    first = save(service, 'duplicate', status='INCONCLUSIVE')
    second = save(service, 'duplicate', status='HUMAN_REVIEW')
    assert first['sessionId'] != second['sessionId']
    assert service.get_session_local(first['sessionId'])['status'] == 'INCONCLUSIVE'
    assert service.get_session_local(second['sessionId'])['status'] == 'HUMAN_REVIEW'
    page = service.list_sessions_local(limit=1, offset=1)
    assert page['count'] == 2
    assert len(page['sessions']) == 1


@pytest.mark.parametrize('session_id', ['..', '.', '../outside', 'nested/path'])
def test_local_lookup_stays_inside_session_directory(tmp_path, session_id):
    with pytest.raises(ValueError):
        HirschbergSessionService(tmp_path).get_session_local(session_id)


def test_keep_alive_cycle_has_no_database_ping():
    worker = KeepAliveService(enabled=False, custom_url='http://example.invalid')
    calls = []
    async def fake_http(url):
        calls.append(url)
        return {'success': True}
    worker.ping_http = fake_http
    result = asyncio.run(worker.ping_cycle())
    assert calls == ['http://example.invalid']
    assert set(result) == {'timestamp', 'http'}
    assert 'databasePings' not in worker.get_status()


def test_startup_and_health_work_when_database_libraries_are_blocked():
    code = '''
import importlib.abc, sys
class BlockDatabase(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'supabase', 'sqlalchemy', 'asyncpg', 'aiosqlite', 'greenlet'}:
            raise AssertionError('Unexpected database import: ' + fullname)
sys.meta_path.insert(0, BlockDatabase())
from app import main
from fastapi.testclient import TestClient
main.preload_trained_hirschberg = lambda: None
with TestClient(main.app) as client:
    assert client.get('/ping').status_code == 200
    assert client.get('/health/db').status_code == 404
    response = client.get('/health?check_db=true')
    assert response.status_code == 200
    data = response.json()
    assert data['sessionStorage'] == 'local'
    assert 'database' not in data
    assert 'databasePings' not in data['keepAlive']
assert not any(name.split('.')[0] in {'supabase', 'sqlalchemy', 'asyncpg', 'aiosqlite', 'greenlet'} for name in sys.modules)
'''
    env = {**os.environ, 'KEEP_ALIVE_ENABLED': 'false',
           'KEEP_ALIVE_PING_DB': 'true', 'DATABASE_URL': 'postgresql://unused.invalid/no_connection',
           'SUPABASE_URL': 'https://unused.invalid', 'PYTHONDONTWRITEBYTECODE': '1'}
    result = subprocess.run([sys.executable, '-B', '-c', code], cwd=Path(__file__).resolve().parents[2],
                            env=env, capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
