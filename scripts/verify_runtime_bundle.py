"""Verify all shipped inference files before deployment, without loading torch."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify_runtime(root=ROOT):
    runtime = root / 'app/hirschberg_runtime'
    manifest = json.loads((runtime / 'manifest.json').read_text(encoding='utf-8'))
    total = 0
    for entry in manifest['files']:
        path = (runtime / entry['path']).resolve()
        if not path.is_relative_to(runtime.resolve()):
            raise RuntimeError('Artifact path escapes runtime')
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != entry['sha256'] or path.stat().st_size != entry['bytes']:
            raise RuntimeError(f'Invalid runtime artifact: {entry["path"]}')
        total += entry['bytes']
    onnx = root / 'app/models/best_model.onnx'
    if not onnx.is_file() or onnx.stat().st_size < 1024:
        raise RuntimeError('Missing ONNX model or Git LFS pointer in place of model')
    return {'verified': True, 'modelId': manifest['modelId'],
            'pupilVersion': manifest['pupilVersion'], 'files': len(manifest['files']),
            'runtimeBytes': total, 'onnxBytes': onnx.stat().st_size}


if __name__ == '__main__':
    print(json.dumps(verify_runtime(), indent=2))
