"""Compare live BE results with the three previously recorded training-runtime cases."""
import hashlib
import json
from pathlib import Path
import httpx

SOURCE = Path('D:/REMICARE_TRAIN_AI/HIRSCHBERG_METHOD')
OUT = Path(__file__).resolve().parents[1] / 'reports/production_integration/hirschberg_20261012'


def main():
    reference = json.loads((SOURCE / 'research/hirschberg_localization/runs/real_pupil_color_control_integration_20261012_v1/cases.json').read_text(encoding='utf-8'))
    results = []
    with httpx.Client(base_url='http://127.0.0.1:8000', timeout=90) as client:
        health = client.get('/api/v1/hirschberg/health').json()
        assert health['loaded']
        for case in reference:
            path = next((SOURCE / 'real_screening_experiments' / case['case_id']).glob('original.*'))
            raw = path.read_bytes()
            response = client.post('/api/v1/hirschberg/predict', files={'image': (path.name, raw, 'image/jpeg')})
            assert response.status_code == 200, response.text[:500]
            actual = response.json()
            errors = {side: max(abs(a-b) for a,b in zip(actual['measurements'][side]['pupil'], case['pupils'][side]['new'])) for side in ('image_left', 'image_right')}
            assert max(errors.values()) < .01, errors
            assert actual['status'] == case['screening']['status']
            assert actual['model']['classifier_sha256'] == health['model']['classifier_sha256']
            results.append({'source_sha256': hashlib.sha256(raw).hexdigest(), 'status': actual['status'],
                            'reason': actual['reason'], 'pupil_max_coordinate_difference_px': errors,
                            'latencyMs': actual['latencyMs'], 'modelId': actual['modelId']})
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / 'http_verification.json').write_text(json.dumps({'health': health, 'cases': results, 'passed': True}, indent=2), encoding='utf-8')
    print(json.dumps({'passed': True, 'cases': len(results), 'max_pupil_difference_px': max(max(r['pupil_max_coordinate_difference_px'].values()) for r in results)}))


if __name__ == '__main__':
    main()
