import io
import json

import numpy as np
from PIL import Image
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.hirschberg import router
from app.services import trained_hirschberg_service as service


def test_real_checkpoints_load_and_reject_blank_image_without_persistence():
    before = set(service.ROOT.rglob('*'))
    image = io.BytesIO()
    Image.new('RGB', (320, 160), (128, 128, 128)).save(image, format='PNG')
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    response = client.post('/api/v1/hirschberg/predict', files={'image': ('blank.png', image.getvalue(), 'image/png')})
    assert response.status_code == 200
    result = response.json()
    assert result['status'] == 'INCONCLUSIVE'
    assert result['predictedClass'] == 'INCONCLUSIVE'
    assert result['screening']['classifier_called'] is False
    assert result['probabilities'] is None
    assert result['uncalibratedModelScores'] is None
    assert result['model']['pupil_version'] == 'pupil_boundary_200ep_strict122_20261031_v1'
    assert result['model']['pupil_branch'] == 'pupil_boundary_segmentation'
    assert result['clinicalDiagnosis'] is None
    assert client.get('/api/v1/hirschberg/health').json()['loaded'] is True
    # Python imports may create cache files; uploads and output assets must not be saved.
    added = set(service.ROOT.rglob('*')) - before
    assert all('__pycache__' in p.parts for p in added)


def test_tampered_artifact_fails_closed(monkeypatch, tmp_path):
    (tmp_path / 'bad.pt').write_bytes(b'bad')
    (tmp_path / 'manifest.json').write_text(json.dumps({'files': [{'path': 'bad.pt', 'sha256': 'wrong'}]}))
    monkeypatch.setattr(service, 'ROOT', tmp_path)
    monkeypatch.setattr(service, '_engine', None)
    monkeypatch.setattr(service, '_load_error', None)
    with pytest.raises(RuntimeError):
        service.load_trained_hirschberg()
    assert service.trained_hirschberg_status()['loaded'] is False
    assert 'hash mismatch' in service.trained_hirschberg_status()['error']


def test_loaded_classifier_accepts_image_geometry_and_opening_features():
    engine = service.load_trained_hirschberg()
    torch = engine.torch
    with torch.no_grad():
        logits = engine.fusion(torch.zeros(1, 3, 224, 224), torch.zeros(1, 11), torch.zeros(1, 10))
    assert tuple(logits.shape) == (1, 3)
    assert torch.isfinite(logits).all()


def test_existing_hirschberg_prediction_uses_imported_runtime(monkeypatch):
    from app.services.hirschberg_ai_service import predict_hirschberg
    calls = []
    def fake(raw, **kwargs):
        calls.append(raw)
        return {'modelId': service.MODEL_ID}
    monkeypatch.setattr(service, 'predict_trained_hirschberg', fake)
    assert predict_hirschberg(np.zeros((32, 64, 3), np.uint8), [{'x': 999}])['modelId'] == service.MODEL_ID
    assert Image.open(io.BytesIO(calls[0])).size == (64, 32)


@pytest.mark.parametrize('raw,code', [(b'', 400), (b'not an image', 400), (b'x' * (10 * 1024 * 1024 + 1), 413)], ids=['empty', 'invalid', 'oversize'])
def test_upload_validation(raw, code):
    app = FastAPI()
    app.include_router(router)
    response = TestClient(app).post('/api/v1/hirschberg/predict', files={'image': ('bad.jpg', raw, 'image/jpeg')})
    assert response.status_code == code


def test_current_bundle_matches_classifier_feature_provenance():
    engine = service.load_trained_hirschberg()
    manifest = json.loads((service.ROOT / 'manifest.json').read_text(encoding='utf-8'))
    bundle = json.loads((service.ROOT / 'model_bundle.json').read_text(encoding='utf-8'))
    assert service.MODEL_ID == manifest['modelId'] == bundle['model_id'] == engine.model_info['version']
    classifier = engine.torch.load(service.ROOT / bundle['artifacts']['classifier']['path'], map_location='cpu', weights_only=True)
    assert classifier['classes'] == ['normal', 'esotropia', 'exotropia']
    for role in ('pupil', 'reflex', 'iris'):
        assert classifier['config']['localizers'][role]['sha256'] == bundle['artifacts'][role]['sha256']
    assert engine.feature_mean == classifier['geometry_mean']
    assert engine.opening_mean == classifier['opening_mean']
    assert engine.frozen_rois == {}
    assert engine.model_info['independent_evaluation'] is False
    assert engine.model_info['visibility_validated'] is False
