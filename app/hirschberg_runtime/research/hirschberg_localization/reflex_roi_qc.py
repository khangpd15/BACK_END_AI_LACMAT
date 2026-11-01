"""Frozen inference subset imported from the training workspace."""
import math
import numpy as np
import cv2
from PIL import Image, ImageDraw
CONFIG = {'version': 'reflex_roi_qc_v3_geometry_warnings', 'search_radius_over_iris': 0.4, 'outlier_vector_over_iris': 0.4, 'glint_gray_threshold': 230, 'glint_open_kernel_px': 3, 'glint_component_bounds_px': [3, 10], 'clinical_validated': False, 'mask_method': 'predicted_iris_disc_intersect_eyelid_polygon_intersect_pupil_center_disc', 'candidate_replaces_cnn': False, 'segmentation_model_available': False}

def point(value):
    return isinstance(value, (list, tuple)) and len(value) == 2 and all((isinstance(v, (float, int)) and (not isinstance(v, bool)) and math.isfinite(v) for v in value))

def diagnose(image, eye, *, visualize=True):
    """Search source pixels, retain all raw landmarks, return mask and explicit QC."""
    q = {'status': 'Invalid Landmark / Low Confidence', 'reasons': [], 'warnings': [], 'eligible_for_score': False, 'config': CONFIG.copy(), 'local_intensity_candidate': None, 'candidate_requires_primary_light_review': True}
    pupil, reflex, center, diameter = (eye.get(k) for k in ('pupil', 'reflex', 'iris_center', 'iris_diameter_px'))
    if not all((point(v) for v in (pupil, reflex, center))) or not isinstance(diameter, (float, int)) or (not math.isfinite(diameter)) or (diameter <= 0):
        q['reasons'].append('missing_or_nonfinite_geometry')
        return (q, None)
    radius = diameter / 2
    ratio = math.dist(pupil, reflex) / radius
    q['vector_over_iris_radius'] = ratio
    if ratio > CONFIG['outlier_vector_over_iris']:
        q['warnings'].append('reflex_pupil_vector_exceeds_0_4R')
    box = eye.get('bbox')
    contour = eye.get('eyelid_contour', [])
    if not isinstance(box, list) or len(box) != 4 or (not all((isinstance(v, int) for v in box))) or (not (0 <= box[0] < box[2] <= image.width and 0 <= box[1] < box[3] <= image.height)):
        q['reasons'].append('invalid_crop')
        return (q, None)
    if len(contour) < 3 or not all((point(p) for p in contour)):
        q['reasons'].append('missing_visible_eye_boundary')
        return (q, None)
    x0, y0, x1, y1 = box
    visible = Image.new('L', (x1 - x0, y1 - y0))
    ImageDraw.Draw(visible).polygon([(x - x0, y - y0) for x, y in contour], fill=255)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    mask = (np.asarray(visible) > 0) & ((xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= radius ** 2) & ((xx - pupil[0]) ** 2 + (yy - pupil[1]) ** 2 <= (CONFIG['search_radius_over_iris'] * radius) ** 2)
    mask_image = Image.fromarray(mask.astype('uint8') * 255) if visualize else None
    q['roi_pixel_count'] = int(mask.sum())
    if not q['roi_pixel_count']:
        q['reasons'].append('empty_search_mask')
        return (q, mask_image)
    px, py = reflex
    ix, iy = (int(round(px)) - x0, int(round(py)) - y0)
    inside = 0 <= ix < x1 - x0 and 0 <= iy < y1 - y0 and bool(np.asarray(visible)[iy, ix])
    if math.dist(reflex, center) > radius:
        q['warnings'].append('reflex_outside_estimated_iris_disc')
    if ratio > CONFIG['search_radius_over_iris']:
        q['warnings'].append('cnn_reflex_outside_search_roi')
    if not inside:
        q['reasons'].append('cnn_reflex_outside_visible_eye')
    gray = np.asarray(image.crop(box).convert('L'))
    values = gray[mask]
    maximum = int(values.max())
    minimum = int(values.min())
    q['intensity_range'] = [minimum, maximum]
    if minimum == maximum:
        q['reasons'].append('flat_intensity_roi')
    else:
        candidates = np.argwhere(mask & (gray == maximum))
        y, x = candidates[0]
        q['local_intensity_candidate'] = [int(x) + x0, int(y) + y0]
        q['maximum_pixel_count'] = len(candidates)
        q['selection_rule'] = 'first raster-order maximum; ties retained as count; not verified reflex'
    binary = ((gray > CONFIG['glint_gray_threshold']) & mask).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CONFIG['glint_open_kernel_px'],) * 2)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(opened, 8)
    low, high = CONFIG['glint_component_bounds_px']
    glints = []
    for label in range(1, count):
        bx, by, bw, bh, area = map(int, stats[label])
        if not (low <= bw <= high and low <= bh <= high) or max(bw, bh) / min(bw, bh) > 2.0:
            continue
        component = labels[by:by + bh, bx:bx + bw] == label
        component_mask = np.zeros_like(mask)
        component_mask[by:by + bh, bx:bx + bw] = component
        ys, xs = np.nonzero(component_mask)
        weights = gray[ys, xs].astype(np.float32) - CONFIG['glint_gray_threshold']
        point_xy = [float(np.average(xs + x0, weights=weights)), float(np.average(ys + y0, weights=weights))]
        surrounding = cv2.dilate(component_mask.astype(np.uint8), kernel, iterations=2).astype(bool) & mask & ~component_mask
        local = float(gray[surrounding].mean()) if surrounding.any() else float(minimum)
        peak = int(gray[component_mask].max())
        if area / (bw * bh) < 0.35:
            continue
        glints.append({'xy': point_xy, 'bbox_px': [bx + x0, by + y0, bx + bw + x0, by + bh + y0], 'size_px': [bw, bh], 'area_px': area, 'peak_gray': peak, 'local_contrast': peak - local, 'distance_to_iris_center_over_radius': math.dist(point_xy, center) / radius})
    glints.sort(key=lambda c: (-c['local_contrast'], c['distance_to_iris_center_over_radius']))
    q['glint_candidates'] = glints
    q['glint_candidate_count'] = len(glints)
    q['glint_candidate_selection'] = 'ranked by local contrast, then proximity to iris center; human confirmation required'
    q['eligible_for_score'] = not q['reasons']
    if q['eligible_for_score']:
        q['status'] = 'ROI_QC_PASSED_RESEARCH_ONLY'
    return (q, mask_image)

def guarded_infer(bundle, record, orientation, classifier):
    """Do not invoke classifier at all if either eye fails QC."""
    failed = {side: record.get('eyes', {}).get(side, {}).get('reflex_qc', {}).get('reasons', ['qc_not_run']) for side in ('image_left', 'image_right') if record.get('eyes', {}).get(side, {}).get('reflex_qc', {}).get('eligible_for_score') is not True}
    if failed:
        return {'status': 'INCONCLUSIVE', 'reason': 'invalid_reflex_landmark', 'failed_eyes': failed, 'class_proposal': None, 'clinical_diagnosis': None, 'classifier_called': False}
    return classifier(bundle, record, orientation)
