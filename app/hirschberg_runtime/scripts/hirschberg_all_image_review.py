"""Frozen inference subset imported from the training workspace."""
import hashlib
import math
from pathlib import Path

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def mesh_eyes(lms, width, height):
    groups = [(468, (469, 470, 471, 472), (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246)), (473, (474, 475, 476, 477), (362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398))]
    point = lambda i: [float(lms[i].x * width), float(lms[i].y * height)]
    result = []
    for center_idx, ring, contour in groups:
        center, boundary, lid = (point(center_idx), [point(i) for i in ring], [point(i) for i in contour])
        edges = {'iris_left': min(boundary, key=lambda p: p[0]), 'iris_right': max(boundary, key=lambda p: p[0]), 'iris_top': min(boundary, key=lambda p: p[1]), 'iris_bottom': max(boundary, key=lambda p: p[1])}
        xs, ys = ([p[0] for p in lid], [p[1] for p in lid])
        dx, dy = ((max(xs) - min(xs)) * 0.2, max(max(ys) - min(ys), edges['iris_bottom'][1] - edges['iris_top'][1]) * 0.5)
        bbox = [max(0, int(min(xs) - dx)), max(0, int(min(ys) - dy)), min(width, math.ceil(max(xs) + dx)), min(height, math.ceil(max(ys) + dy))]
        result.append({'iris_center': center, **edges, 'eyelid_contour': lid, 'bbox': bbox, 'iris_method': 'MediaPipe_refined_iris', 'iris_is_pupil': False})
    return sorted(result, key=lambda r: r['iris_center'][0])
