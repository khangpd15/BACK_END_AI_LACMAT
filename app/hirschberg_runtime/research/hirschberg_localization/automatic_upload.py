"""Frozen inference subset imported from the training workspace."""
import copy
import math

def paired_eye_rois(image):
    """Independent eye-cascade fallback on source pixels; no pupil/GT-seeded crops."""
    import cv2
    import numpy as np
    scale = min(1.0, 1280 / max(image.size))
    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    if scale < 1:
        gray = cv2.resize(gray, (round(image.width * scale), round(image.height * scale)))
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
    if detector.empty():
        return ({}, {'status': 'unavailable', 'reason': 'bundled eye cascade missing'})
    found = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
    boxes = sorted([list(map(int, b)) for b in found], key=lambda b: b[2] * b[3], reverse=True)[:20]
    pairs = []
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            left, right = sorted([a, b], key=lambda c: c[0] + c[2] / 2)
            ax, ay = (left[0] + left[2] / 2, left[1] + left[3] / 2)
            bx, by = (right[0] + right[2] / 2, right[1] + right[3] / 2)
            if bx - ax <= max(left[2], right[2]) * 0.5 or abs(ay - by) > 0.6 * max(left[3], right[3]):
                continue
            if min(left[2], right[2]) / max(left[2], right[2]) < 0.4:
                continue
            pairs.append((left[2] * left[3] + right[2] * right[3], left, right))
    if not pairs:
        return ({}, {'status': 'not_found', 'detections': len(boxes), 'input_scale': scale})
    _, left, right = max(pairs, key=lambda r: r[0])
    rois = {}
    for side, (x, y, w, h) in zip(('image_left', 'image_right'), (left, right)):
        box = [max(0, round(x / scale)), max(0, round((y + 0.2 * h) / scale)), min(image.width, round((x + w) / scale)), min(image.height, round((y + 0.8 * h) / scale))]
        rois[side] = box
    return (rois, {'status': 'paired_candidates', 'detections': len(boxes), 'input_scale': scale, 'method': 'OpenCV eye_tree_eyeglasses; horizontal pairing; central60% height', 'human_review_required': True, 'anatomical_sides_verified': False})

def capture_assessment(image, faces, eyes):
    kind = 'fullface' if len(faces) == 1 else 'bilateral_candidate' if image.width / image.height >= 1.4 else 'unresolved'
    out = {'input_type': kind, 'face_count': len(faces), 'mirror_status': 'unknown', 'fixation_status': 'unknown', 'fixation_verified': False, 'note': 'A still image does not establish mirror history or the actual fixation target.'}
    if len(eyes) == 2:
        centers = [((e['bbox'][0] + e['bbox'][2]) / 2, (e['bbox'][1] + e['bbox'][3]) / 2) for e in eyes]
        centers.sort()
        a, b = centers
        out['eye_line_roll_deg'] = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
        out['roll_note'] = '2D eye-line inclination only; not a measurement of gaze.'
    return out

def mirrored_proposal(proposal, width):
    out = {}
    for target, source in [('image_left', 'image_right'), ('image_right', 'image_left')]:
        e = copy.deepcopy(proposal[source])
        box = e.get('bbox')
        if box:
            e['bbox'] = [width - box[2], box[1], width - box[0], box[3]]
        iris_box = e.get('iris_bbox')
        if iris_box:
            e['iris_bbox'] = [width - iris_box[2], iris_box[1], width - iris_box[0], iris_box[3]]
        for key in ('pupil', 'reflex', 'iris_center', 'iris_left', 'iris_right', 'iris_top', 'iris_bottom'):
            p = e.get(key)
            if p is not None:
                e[key] = [width - 1 - p[0], p[1]]
        for key in ('eyelid_contour', 'eyelid_branch_contour'):
            if key in e:
                e[key] = [[width - 1 - p[0], p[1]] for p in e[key]]
        e['iris_left'], e['iris_right'] = (e.get('iris_right'), e.get('iris_left'))
        if e.get('iris_edges'):
            e['iris_edges'] = [[width - 1 - p[0], p[1]] for p in reversed(e['iris_edges'])]
        if e.get('iris_four'):
            four = {k: [width - 1 - p[0], p[1]] for k, p in e['iris_four'].items()}
            four['iris_left'], four['iris_right'] = (four['iris_right'], four['iris_left'])
            e['iris_four'] = four
        out[target] = e
    return out

def mirror_preview(infer, model, image, proposal, mean, std, device):
    from PIL import Image
    first = infer(model, image, proposal, mean, std, device)
    if first['status'] == 'INCONCLUSIVE':
        return first
    flipped = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    second = infer(model, flipped, mirrored_proposal(proposal, image.width), mean, std, device)
    diagnostics = {'class_original': first.get('class_proposal'), 'class_mirrored': second.get('class_proposal'), 'scope': 'classifier TTA on reflected fixed landmarks; not a full detector equivariance test'}
    if second['status'] != 'INCONCLUSIVE' and first['uncalibrated_model_scores'].keys() == second['uncalibrated_model_scores'].keys():
        diagnostics['maximum_score_gap'] = max((abs(v - second['uncalibrated_model_scores'][k]) for k, v in first['uncalibrated_model_scores'].items()))
    if second['status'] == 'INCONCLUSIVE' or first.get('class_proposal') != second.get('class_proposal'):
        return {'status': 'INCONCLUSIVE', 'reason': 'mirror_sensitive_prediction', 'scores': None, 'mirror_diagnostics': diagnostics, 'warning': 'Model class changes under horizontal reflection; mirror history remains unknown.'}
    labels = first['uncalibrated_model_scores']
    averaged = {k: (labels[k] + second['uncalibrated_model_scores'][k]) / 2 for k in labels}
    return {**first, 'uncalibrated_model_scores': averaged, 'mirror_consistency': 'same class in two orientations', 'mirror_diagnostics': diagnostics, 'role': 'RESEARCH_PREVIEW_ONLY', 'capture_verified': False, 'warning': 'Uncalibrated mirror-averaged model preview; consistency does not verify fixation or clinical accuracy.'}
