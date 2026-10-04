"""Read existing assets; write audit evidence only beside this new script."""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
IMAGE_EXT = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}
SKIP = {'.git', '.venv', '.venv312', 'venv', '__pycache__', '.pytest_cache', 'audits'}
sys.path.insert(0, str(ROOT))
from app.services.eye_crop_geometry_service import segment_iris_pupil, detect_corneal_reflex


def summary(values):
    return dict(zip(('min', 'median', 'max'), map(float, np.percentile(values, [0, 50, 100])))) if values else None


def image_record(path):
    with Image.open(path) as im:
        w, h = im.size
        fmt = im.format
        small = np.array(im.convert('L').resize((9, 8)), dtype=np.int16)
        dh = sum(int(v) << i for i, v in enumerate((small[:, 1:] > small[:, :-1]).ravel()))
        gray = np.array(im.convert('L'))
    return {'path': path.relative_to(ROOT).as_posix(), 'w': w, 'h': h, 'format': fmt,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'dhash': dh,
            'blur': float(cv2.Laplacian(gray, cv2.CV_64F).var())}


def collect():
    inventory = defaultdict(Counter)
    image_paths = []
    artifacts = []
    notebooks = []
    for folder, dirs, files in os.walk(ROOT):
        dirs[:] = sorted(d for d in dirs if d not in SKIP)
        for name in sorted(files):
            p = Path(folder) / name
            rel = p.relative_to(ROOT)
            inventory[rel.parts[0]][p.suffix.lower() or '(no extension)'] += 1
            if p.suffix.lower() in IMAGE_EXT:
                image_paths.append(p)
            if p.suffix.lower() in {'.onnx', '.pt', '.pth', '.joblib', '.npz', '.ckpt'}:
                artifacts.append({'path': rel.as_posix(), 'bytes': p.stat().st_size})
            if p.suffix.lower() == '.ipynb':
                notebooks.append(rel.as_posix())
    groups = defaultdict(list)
    errors = Counter()
    primary = []
    records_by_path = {}
    for p in image_paths:
        rel = p.relative_to(ROOT)
        if rel.parts[0] not in {'harschberg_data_detect', 'dataset_by_pedseye_manifest'}:
            continue
        key = '/'.join(rel.parts[:2])
        try:
            rec = image_record(p)
        except Exception as exc:
            errors[key + ':' + type(exc).__name__] += 1
            continue
        groups[key].append(rec)
        records_by_path[rec['path']] = rec
        if rel.parts[0] == 'dataset_by_pedseye_manifest' or rel.parts[1] in {
            'esotropia_harschberg', 'exotropia_harschberg', 'normal_harschberg'}:
            primary.append(rec)
    image_summary = {}
    for key, rows in sorted(groups.items()):
        image_summary[key] = {'count': len(rows), 'formats': dict(Counter(r['format'] for r in rows)),
                              'resolutions': dict(Counter(f"{r['w']}x{r['h']}" for r in rows)),
                              'blur_laplacian': summary([r['blur'] for r in rows]),
                              'blur_below_15': sum(r['blur'] < 15 for r in rows)}

    manifest = [json.loads(line) for line in (ROOT / 'datasets/pedseye_hirschberg_manifest_v0.1.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    pid_splits = defaultdict(set)
    for row in manifest:
        pid_splits[row['participant_id']].add(row['split'])
    manifest_summary = {'rows': len(manifest), 'classes': dict(Counter(r['label'] for r in manifest)),
                        'splits': dict(Counter(r['split'] for r in manifest)),
                        'participants_inferred': len(pid_splits),
                        'cross_split_inferred_groups': sum(len(v) > 1 for v in pid_splits.values()),
                        'clinician_confirmed': sum(r.get('clinician_confirmed') is True for r in manifest),
                        'missing_files': sum(not (ROOT / r['file_path']).is_file() for r in manifest),
                        'quality_flags': {k: dict(Counter(str(r.get('quality_flags', {}).get(k)) for r in manifest)) for k in ('blur', 'eyes_closed', 'face_off_axis', 'no_corneal_reflex', 'too_dark')},
                        'pupil_detected_both': sum(len(r.get('detector_summary', [])) == 2 and all(e.get('pupil_found') for e in r['detector_summary']) for r in manifest),
                        'reflex_detected_both': sum(len(r.get('detector_summary', [])) == 2 and all(e.get('reflex_status') == 'DETECTED' for e in r['detector_summary']) for r in manifest)}
    with (ROOT / 'processed/hirschberg_manual_annotations.csv').open(encoding='utf-8-sig', newline='') as stream:
        annotations = list(csv.DictReader(stream))
    annotation_issues = Counter()
    annotation_valid = 0
    for row in annotations:
        rec = records_by_path.get(row['relative_path'].replace('\\', '/'))
        bad = False
        if rec is None:
            annotation_issues['missing_image'] += 1
            continue
        folder_class = Path(row['relative_path']).parent.name.replace('_harschberg', '')
        if row['class_label'] != folder_class:
            annotation_issues['folder_label_mismatch'] += 1
            bad = True
        for eye in ('od', 'os'):
            for kind in ('pupil', 'reflex'):
                for axis, bound in [('x', rec['w']), ('y', rec['h'])]:
                    try:
                        coord = float(row[f'{eye}_{kind}_{axis}'])
                        if not np.isfinite(coord) or not 0 <= coord < bound:
                            annotation_issues['out_of_bounds_or_nonfinite'] += 1
                            bad = True
                    except (ValueError, TypeError):
                        annotation_issues['missing_or_invalid_coordinate'] += 1
                        bad = True
        annotation_valid += not bad
    ann_summary = {'rows': len(annotations), 'classes': dict(Counter(r['class_label'] for r in annotations)),
                   'duplicate_path_rows': len(annotations) - len({r['relative_path'] for r in annotations}),
                   'valid_coordinate_and_folder_rows': annotation_valid, 'issues': dict(annotation_issues),
                   'note_rows': sum(bool(r.get('notes')) for r in annotations)}

    accepted = [r for r in primary if r['path'].startswith('harschberg_data_detect/')]
    labels = np.array([0 if 'esotropia_harschberg' in r['path'] else 1 if 'exotropia_harschberg' in r['path'] else 2 for r in accepted])
    rng = np.random.default_rng(20261004)
    val = set()
    for c in np.unique(labels):
        idx = rng.permutation(np.where(labels == c)[0])
        val.update(int(i) for i in idx[:max(1, int(len(idx) * .2))])
    accepted_index = {r['path']: i for i, r in enumerate(accepted)}
    duplicates = Counter()
    examples = []
    for i, a in enumerate(primary):
        for j in range(i):
            b = primary[j]
            same = a['sha256'] == b['sha256']
            distance = (a['dhash'] ^ b['dhash']).bit_count()
            if same:
                duplicates['exact_pairs'] += 1
            if distance <= 4:
                duplicates['dhash_le4_candidate_pairs'] += 1
                if a['path'].split('/')[0] != b['path'].split('/')[0]:
                    duplicates['cross_dataset_candidate_pairs'] += 1
                ai, bi = accepted_index.get(a['path']), accepted_index.get(b['path'])
                if ai is not None and bi is not None and (ai in val) != (bi in val):
                    duplicates['v06_train_val_candidate_pairs'] += 1
                if len(examples) < 15:
                    examples.append({'a': a['path'], 'b': b['path'], 'dhash_distance': distance, 'exact': same,
                                     'v06_a_split': ('val' if ai in val else 'train') if ai is not None else None,
                                     'v06_b_split': ('val' if bi in val else 'train') if bi is not None else None})
    pupil_errors, reflex_errors = [], []
    detector_status = Counter()
    for row in annotations:
        img = cv2.imread(str(ROOT / row['relative_path']))
        if img is None:
            continue
        half = img.shape[1] // 2
        for eye, crop, offset in [('od', img[:, :half], 0), ('os', img[:, half:], half)]:
            cx, cy, rad, method = segment_iris_pupil(crop)
            rx, ry, area, status = detect_corneal_reflex(crop, cx, cy, rad)
            detector_status['iris_' + method] += 1
            detector_status['reflex_' + status] += 1
            pupil_errors.append(float(np.hypot(cx + offset - float(row[eye + '_pupil_x']), cy - float(row[eye + '_pupil_y']))))
            if rx is not None:
                reflex_errors.append(float(np.hypot(rx + offset - float(row[eye + '_reflex_x']), ry - float(row[eye + '_reflex_y']))))
    def error_stats(values):
        return {'n': len(values), 'mean_px': float(np.mean(values)), 'median_px': float(np.median(values)),
                'p90_px': float(np.percentile(values, 90))} if values else None
    detector_comparison = {'method': 'Current v2 eye-crop iris center compared to manual pupil center; anatomical targets differ.',
                           'status_counts': dict(detector_status), 'iris_vs_manual_pupil': error_stats(pupil_errors),
                           'reflex_vs_manual_reflex': error_stats(reflex_errors)}
    with (ROOT / 'harschberg_data_detect/report.csv').open(encoding='utf-8-sig', newline='') as stream:
        filter_rows = list(csv.DictReader(stream))
    filter_summary = {'rows': len(filter_rows), 'status': dict(Counter(r['status'] for r in filter_rows)),
                      'source_labels': dict(Counter(r['source_label'] for r in filter_rows)),
                      'detection_method': dict(Counter(r['detection_method'] for r in filter_rows)),
                      'rejection_reasons': dict(Counter(r['rejection_reason'] for r in filter_rows if r['status'] == 'REJECT'))}
    report_metrics = {}
    def metric_nodes(value, prefix=''):
        found = {}
        if isinstance(value, dict):
            cm = value.get('confusion_matrix')
            if isinstance(cm, list) and cm and isinstance(cm[0], list):
                arr = np.array(cm, dtype=float)
                if arr.shape[0] == arr.shape[1]:
                    tp = np.diag(arr)
                    precision = tp / np.maximum(arr.sum(axis=0), 1)
                    recall = tp / np.maximum(arr.sum(axis=1), 1)
                    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-15)
                    found[prefix] = {'n': int(arr.sum()), 'labels': value.get('confusion_matrix_labels') or value.get('labels'),
                                     'confusion_matrix': cm, 'precision': precision.tolist(), 'recall': recall.tolist(), 'f1': f1.tolist(),
                                     'coverage': value.get('coverage'), 'n_total': value.get('n_total'),
                                     'balanced_accuracy_reported': value.get('balanced_accuracy'), 'macro_f1_reported': value.get('macro_f1')}
            for k, v in value.items():
                if isinstance(v, dict):
                    found.update(metric_nodes(v, prefix + '/' + k))
        return found
    for p in sorted((ROOT / 'reports').glob('*eval.json')):
        value = json.loads(p.read_text(encoding='utf-8'))
        report_metrics[p.name] = {'top_keys': list(value), 'metric_nodes': metric_nodes(value)}
    versions = {}
    for name in ('numpy', 'scikit-learn', 'opencv-python', 'onnxruntime', 'timm', 'torch', 'torchvision', 'lightgbm', 'mediapipe'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {'notice': 'Research screening support only; not medical diagnosis. No training, downloads or source edits.',
            'inventory': dict(inventory), 'notebooks': notebooks, 'artifacts': artifacts,
            'image_summary': image_summary, 'image_read_errors': dict(errors),
            'manifest': manifest_summary, 'annotations': ann_summary, 'duplicate_screen': dict(duplicates),
            'current_eye_crop_detector_comparison': detector_comparison, 'filter_report': filter_summary,
            'duplicate_candidate_examples': examples, 'v06_split_reconstructed': {'train': len(accepted) - len(val), 'val': len(val)},
            'saved_report_metrics': report_metrics, 'environment_versions': versions}


if __name__ == '__main__':
    evidence = collect()
    (OUT / 'evidence.json').write_text(json.dumps(evidence, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: evidence[k] for k in ('manifest', 'annotations', 'duplicate_screen', 'current_eye_crop_detector_comparison', 'filter_report', 'environment_versions')}, indent=2))
