"""Self-contained frozen Hirschberg CNN runtime, with serialized inference."""
import hashlib
import json
import logging
from pathlib import Path
from threading import RLock
import time

ROOT = Path(__file__).resolve().parents[1] / 'hirschberg_runtime'
MODEL_ID = 'matched_bundle_20261101_v1'
FEATURE_CONTRACT = 'resnet50_hirschberg_geometry_eyelid_v1'
_lock = RLock()
_engine = None
_load_error = None
logger = logging.getLogger(__name__)


def screening_message(screening):
    """Public copy stays simple; technical reasons remain in screening/quality."""
    if screening.get('status') == 'INCONCLUSIVE':
        return 'Chưa đủ thông tin về hướng ảnh hoặc hướng nhìn; hãy xác nhận điều kiện chụp.' if screening.get('reason') == 'invalid_input_contract' and any(x in screening.get('input_reasons', []) for x in ('orientation_unverified',)) else 'Chưa thể đánh giá: điểm mắt hoặc vùng mắt chưa đạt kiểm tra; cần xem lại ảnh đánh dấu.'
    proposal = str(screening.get('class_proposal') or '').upper()
    if proposal == 'NORMAL':
        return 'Chưa ghi nhận dấu hiệu lác rõ trong ảnh này.'
    if proposal in ('ESOTROPIA', 'EXOTROPIA'):
        return 'Có dấu hiệu cần kiểm tra thêm. Bạn nên khám chuyên khoa mắt để được đánh giá.'
    return 'Chưa thể đánh giá từ ảnh này. Bạn hãy chụp lại theo hướng dẫn.'


def load_trained_hirschberg():
    global _engine, _load_error
    with _lock:
        if _engine is not None:
            return _engine
        try:
            manifest = json.loads((ROOT / 'manifest.json').read_text(encoding='utf-8'))
            for item in manifest['files']:
                path = ROOT / item['path']
                with path.open('rb') as stream:
                    actual_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
                if actual_hash != item['sha256']:
                    raise RuntimeError(f'Hirschberg artifact hash mismatch: {item["path"]}')
            from app.hirschberg_runtime.scripts.hirschberg_engine import Engine
            _engine = Engine()
            _load_error = None
        except Exception as exc:
            _load_error = str(exc)
            logger.exception('Trained Hirschberg runtime unavailable')
            raise RuntimeError('Trained Hirschberg runtime unavailable') from exc
        return _engine


def preload_trained_hirschberg():
    try:
        load_trained_hirschberg()
    except RuntimeError:
        pass


def trained_hirschberg_status():
    return {'loaded': _engine is not None, 'configuredModelId': MODEL_ID,
            'researchOnly': True, 'error': _load_error,
            'model': _engine.model_info if _engine is not None else None}


def predict_trained_hirschberg(raw, filename='image.png', orientation='auto', kind='auto', fixation='unknown'):
    started = time.perf_counter()
    with _lock:
        engine = load_trained_hirschberg()
        record = engine.process(raw, filename, orientation, kind, fixation)
    screening = record['screening']
    scores = screening.get('uncalibrated_model_scores')
    return {
        'status': screening['status'],
        'predictedClass': (screening.get('class_proposal') or 'INCONCLUSIVE').upper(),
        'reason': screening.get('reason'), 'message': screening_message(screening),
        'modelId': MODEL_ID, 'modelVersion': MODEL_ID, 'featureContract': FEATURE_CONTRACT,
        'researchOnly': True, 'clinicalDiagnosis': None,
        'probabilities': None, 'uncalibratedModelScores': scores,
        'confidence': max(scores.values()) if scores else None,
        'screening': screening, 'model': record['model'],
        'measurements': record['eyes'],
        'quality': {key: record.get(key) for key in (
            'automatic_capture', 'binocular_iris_radius_qc', 'binocular_comparison', 'reflex_qc_config')},
        'orientation': record['orientation'], 'inputKind': record['input_kind'],
        'roiSource': record.get('roi_source'),
        'inputContract': record.get('input_contract'),
        'fixationSource': record.get('fixation_source'),
        'latencyMs': round((time.perf_counter() - started) * 1000, 3),
        'nonClinicalDeclaration': 'Sàng lọc nghiên cứu - Không thay thế chẩn đoán bác sĩ chuyên khoa.',
    }
