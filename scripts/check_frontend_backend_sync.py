"""Read-only deployment contract/CORS/bundle audit; never writes sessions."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re

import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', default='https://remicare-strabismus-ai.onrender.com')
    parser.add_argument('--frontend', default='https://aidetecteye.vercel.app')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    backend, frontend = args.backend.rstrip('/'), args.frontend.rstrip('/')

    def probe(path):
        try:
            with httpx.Client(timeout=25, follow_redirects=True) as client:
                if path.startswith('cors:'):
                    response = client.options(backend + path[5:], headers={
                        'Origin': frontend, 'Access-Control-Request-Method': 'POST',
                        'Access-Control-Request-Headers': 'content-type'})
                    return path, {'httpStatus': response.status_code,
                                  'allowOrigin': response.headers.get('access-control-allow-origin')}
                if path == 'frontend':
                    response = client.get(frontend)
                    result = {'httpStatus': response.status_code}
                    asset = re.search(r'src="(/assets/[^\"]+\.js)"', response.text)
                    if asset:
                        javascript = client.get(frontend + asset.group(1)).text
                        result.update(asset=asset.group(1), backendUrlPresent=backend in javascript,
                            localhostFallbackPresent='localhost:8000' in javascript,
                            retiredTransferRoutePresent='/api/v1/transfer/strabismus' in javascript,
                            missingModelReasonHandled='MODEL_NOT_LOADED' in javascript,
                            hirschbergTimeout25Seconds=bool(re.search(r'timeoutMs:(?:25e3|25000)', javascript)),
                            uncalibratedScoresHandled='uncalibratedModelScores' in javascript)
                    return path, result
                response = client.get(backend + path)
                result = {'httpStatus': response.status_code}
                if response.is_success:
                    body = response.json()
                    result['body'] = sorted(body.get('paths', {})) if path == '/openapi.json' else body
                else:
                    result['body'] = response.text[:300]
                return path, result
        except Exception as exc:
            return path, {'error': f'{type(exc).__name__}: {exc}'}

    paths = ['/ping', '/health', '/api/v1/hirschberg/health', '/openapi.json', 'frontend']
    paths += ['cors:' + path for path in ['/api/v1/research/measurements',
              '/api/v1/cover-test/sessions', '/api/v1/hirschberg/predict']]
    with ThreadPoolExecutor(max_workers=8) as executor:
        report = dict(executor.map(probe, paths))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
