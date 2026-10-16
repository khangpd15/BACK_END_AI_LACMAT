import asyncio
import sys

import httpx

from app import main


def test_only_hirschberg_pipeline_is_exposed():
    paths = main.app.openapi()['paths']
    assert '/api/v1/hirschberg/predict' in paths
    assert not any(path.startswith('/api/v1/strabismus') for path in paths)

    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url='http://test') as client:
            assert (await client.get('/api/v1/strabismus/health')).status_code == 404
            assert (await client.post('/api/v1/strabismus/predict')).status_code == 404
            health = await client.get('/health')
            assert health.status_code == 200
            assert set(health.json()['models']) == {'hirschberg'}
    asyncio.run(run())
    assert not any(name == 'onnxruntime' or name.startswith('onnxruntime.') for name in sys.modules)
