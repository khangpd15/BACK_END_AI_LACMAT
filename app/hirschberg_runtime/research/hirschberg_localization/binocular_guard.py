"""Frozen inference subset imported from the training workspace."""
import math

def compare_eyes(record, orientation):
    if orientation not in ('unmirrored', 'mirrored'):
        return {'available': False, 'reason': 'orientation unverified'}
    vectors = {}
    flip = -1 if orientation == 'mirrored' else 1
    for side in ('image_left', 'image_right'):
        eye = record.get('eyes', {}).get(side, {})
        pupil, reflex, diameter = (eye.get('pupil'), eye.get('reflex'), eye.get('iris_diameter_px'))
        if not pupil or not reflex or (not isinstance(diameter, (int, float))) or (diameter <= 0):
            return {'available': False, 'reason': 'missing bilateral pupil/reflex/iris'}
        if not all((math.isfinite(v) for p in (pupil, reflex) for v in p)) or not math.isfinite(diameter):
            return {'available': False, 'reason': 'nonfinite geometry'}
        anatomical = ('od' if side == 'image_left' else 'os') if orientation == 'unmirrored' else 'os' if side == 'image_left' else 'od'
        vectors[anatomical] = [flip * (reflex[0] - pupil[0]) / (diameter / 2), (reflex[1] - pupil[1]) / (diameter / 2)]
    a, b = (vectors['od'], vectors['os'])
    common = [(x + y) / 2 for x, y in zip(a, b)]
    difference = [x - y for x, y in zip(a, b)]
    dot = sum((x * y for x, y in zip(a, b)))
    norm_a = math.hypot(*a)
    norm_b = math.hypot(*b)
    cosine = max(-1, min(1, dot / (norm_a * norm_b))) if norm_a and norm_b else None
    return {'available': True, 'frame': 'camera axes with mirror undone; x right,y down; NOT nasal axes', 'unit': 'iris_radius', 'vectors': vectors, 'common': common, 'difference_od_minus_os': difference, 'common_norm': math.hypot(*common), 'difference_norm': math.hypot(*difference), 'direction_cosine': cosine, 'shared_horizontal_sign': a[0] * b[0] > 0, 'shared_vertical_sign': a[1] * b[1] > 0, 'same_half_plane': dot > 0, 'clinical_thresholds_validated': False, 'note': 'Same-half-plane is a mathematical flag, not proof of oblique fixation or normal alignment. Different magnitudes can coexist.'}

def capture_guard(proposal, comparison, fixation):
    reason = None
    if fixation != 'confirmed':
        reason = 'fixation_not_confirmed'
    if not comparison.get('available'):
        reason = 'missing_binocular_geometry'
    if reason:
        return {'status': 'INCONCLUSIVE', 'reason': reason, 'class_proposal': None, 'clinical_diagnosis': None, 'capture_guard_version': 'research_capture_v2', 'warning': 'Conservative unvalidated capture guard; abstention is not a normal or disease label.'}
    return {**proposal, 'capture_guard_version': 'research_capture_v2'}
