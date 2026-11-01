"""Conservative cascade consensus for tightly cropped bilateral eye photographs.

No reference boxes, labels, pupil coordinates, or filename rules are used here.
Reflection padding supplies detector context only; returned ROIs use source pixels.
"""
import math

POLICY = 'bilateral_cascade_consensus_v1'


def iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1]))
    area = lambda r: max(0, r[2] - r[0]) * max(0, r[3] - r[1])
    union = area(a) + area(b) - intersection
    return intersection / union if union else 0.


def consensus_pair(candidates, width, height):
    """Require repeatable detections and exactly one anatomically plausible pair."""
    import numpy as np
    clusters = []
    for candidate in sorted(candidates, key=lambda c: (c['box'], c['view'])):
        matches = [c for c in clusters if all(iou(candidate['box'], x['box']) >= .45 for x in c)]
        if len(matches) == 1:
            matches[0].append(candidate)
        elif not matches:
            clusters.append([candidate])
    supported = []
    for cluster in clusters:
        views = {c['view'] for c in cluster}
        if len(views) < 2 or not any(v.endswith(':raw') for v in views):
            continue
        box = np.median([c['box'] for c in cluster], axis=0).tolist()
        x0, y0, x1, y1 = box
        bw, bh = x1-x0, y1-y0
        cx, cy = (x0+x1)/2, (y0+y1)/2
        core = [x0, y0+.2*bh, x1, y0+.8*bh]
        fraction = max(0, min(height, core[3])-max(0, core[1])) / (.6*bh)
        if not (.12*width <= bw <= .5*width and 0 <= cx < width and 0 <= cy < height and fraction >= .8):
            continue
        supported.append({'box': box, 'views': sorted(views)})
    pairs = []
    for left in supported:
        for right in supported:
            a, b = left['box'], right['box']
            ax, ay, bx, by = (a[0]+a[2])/2, (a[1]+a[3])/2, (b[0]+b[2])/2, (b[1]+b[3])/2
            aw, bw = a[2]-a[0], b[2]-b[0]
            if not (ax < .45*width and bx > .55*width and .3*width <= bx-ax <= .85*width):
                continue
            if min(aw, bw)/max(aw, bw) < .6 or abs(ay-by) > .35*max(aw, bw):
                continue
            if a[2] > b[0]:
                continue
            pairs.append((left, right))
    info = {'policy': POLICY, 'supported_candidates': supported, 'valid_pair_count': len(pairs),
            'status': 'paired_candidates' if len(pairs) == 1 else 'ambiguous' if pairs else 'not_found',
            'human_review_required': True, 'anatomical_sides_verified': False}
    if len(pairs) != 1:
        return {}, info
    rois = {}
    for side, candidate in zip(('image_left', 'image_right'), pairs[0]):
        x0, y0, x1, y1 = candidate['box']
        h = y1-y0
        box = [max(0, math.floor(x0)), max(0, math.floor(y0+.2*h)),
               min(width, math.ceil(x1)), min(height, math.ceil(y0+.8*h))]
        if box[0] >= box[2] or box[1] >= box[3]:
            return {}, {**info, 'status': 'invalid_source_roi'}
        rois[side] = box
    return rois, info


def cropped_eye_rois(image):
    import cv2
    import numpy as np
    width, height = image.size
    if width/height < 2 or min(width, height) < 32:
        return {}, {'policy': POLICY, 'status': 'not_bilateral_crop'}
    scale = min(1., 1280/width)
    gray = cv2.cvtColor(np.asarray(image.convert('RGB')), cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (round(width*scale), round(height*scale)))
    # Record actual scales after rounding, rather than assuming isotropic rounding.
    sx, sy = gray.shape[1]/width, gray.shape[0]/height
    pad = gray.shape[0]//2
    padded = cv2.copyMakeBorder(gray, pad, pad, 0, 0, cv2.BORDER_REFLECT_101)
    minimum = max(24, round(.12*gray.shape[1]))
    candidates, attempts = [], []
    for name in ('haarcascade_eye_tree_eyeglasses.xml', 'haarcascade_eye.xml'):
        detector = cv2.CascadeClassifier(cv2.data.haarcascades+name)
        if detector.empty():
            attempts.append({'model': name, 'status': 'unavailable'})
            continue
        for mode, pixels in (('raw', padded), ('equalized', cv2.equalizeHist(padded))):
            boxes = detector.detectMultiScale(pixels, scaleFactor=1.1, minNeighbors=5,
                                             minSize=(minimum, minimum))
            view = name+':'+mode
            attempts.append({'view': view, 'detections': len(boxes)})
            for x, y, w, h in boxes:
                candidates.append({'box': [float(x/sx), float((y-pad)/sy),
                                            float((x+w)/sx), float((y+h-pad)/sy)], 'view': view})
    boxes, info = consensus_pair(candidates, width, height)
    info.update(method=POLICY, attempts=attempts, input_scale_xy=[sx, sy], padding_y=pad,
                padding_method='reflect101; detector context only', source_pixels_only=True)
    return boxes, info
