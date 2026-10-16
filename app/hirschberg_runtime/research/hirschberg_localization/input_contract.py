"""Frozen inference subset imported from the training workspace."""
import math
SIDES = ('image_left', 'image_right')

def anatomical_sides(orientation):
    if orientation == 'unmirrored':
        return dict(zip(SIDES, ('OD', 'OS')))
    if orientation == 'mirrored':
        return dict(zip(SIDES, ('OS', 'OD')))
    return dict.fromkeys(SIDES)

def validate_input(record):
    """Fail closed; a numeric landmark alone does not establish visibility."""
    reasons = []
    width, height = (record['width'], record['height'])
    mapping = anatomical_sides(record.get('orientation'))
    if not all(mapping.values()):
        reasons.append('orientation_unverified')
    if record.get('exif_orientation', 1) != 1:
        reasons.append('exif_transform_requires_normalized_reupload')
    if record.get('detected_faces', 0) > 1:
        reasons.append('multiple_faces')
    centers = []
    for side in SIDES:
        eye = record.get('eyes', {}).get(side, {})
        box = eye.get('bbox')
        valid = isinstance(box, (list, tuple)) and len(box) == 4 and all((isinstance(v, (int, float)) and math.isfinite(v) for v in box)) and (0 <= box[0] < box[2] <= width) and (0 <= box[1] < box[3] <= height)
        if not valid:
            reasons.append(side + ':missing_or_invalid_roi')
        else:
            centers.append((box[0] + box[2]) / 2)
        for kind in ('pupil', 'reflex'):
            visibility = eye.get('visibility', {}).get(kind)
            raw = eye.get('raw_predictions', {}).get(kind, {})
            visibility = visibility or raw.get('visibility')
            if visibility in ('occluded', 'not_visible'):
                reasons.append(side + ':' + kind + '_' + visibility)
            point = eye.get(kind)
            if not (isinstance(point, (list, tuple)) and len(point) == 2 and all((isinstance(v, (int, float)) and math.isfinite(v) for v in point)) and (0 <= point[0] < width) and (0 <= point[1] < height)):
                reasons.append(side + ':' + kind + '_missing_or_invalid')
            elif valid and (not (box[0] <= point[0] < box[2] and box[1] <= point[1] < box[3])):
                reasons.append(side + ':' + kind + '_outside_eye_roi')
    if len(centers) == 2 and centers[0] >= centers[1]:
        reasons.append('image_side_order_invalid')
    return {'version': 'source_pixels_v1', 'eligible': not reasons, 'reasons': reasons, 'anatomical_sides': mapping, 'coordinate_space': 'stored_source_pixels', 'visibility_policy': 'occluded/not_visible rejected; model visibility is not clinical confirmation'}

def human_review_only(proposal):
    result = dict(proposal)
    result.update(clinical_diagnosis=None, requires_human_review=True)
    if result.get('status') != 'INCONCLUSIVE':
        result['status'] = 'HUMAN_REVIEW'
    return result
