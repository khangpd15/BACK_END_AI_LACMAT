"""Frozen inference subset imported from the training workspace."""
import math
import cv2
import numpy as np
import torch
from torch import nn
from app.hirschberg_runtime.research.hirschberg_localization.screening_fusion import ScreeningFusion

def opening_features(eyes):
    values = []
    valid = []
    details = {}
    for side in ('image_left', 'image_right'):
        e = eyes.get(side, {})
        p = e.get('pupil')
        pts = e.get('eyelid_contour', [])
        meta = e.get('eyelid_prediction_metadata', {})
        a = np.asarray(pts, dtype=np.float32).reshape(-1, 2) if pts else np.empty((0, 2), np.float32)
        ok = len(a) >= 3 and np.isfinite(a).all() and (p is not None) and (len(p) == 2) and all((math.isfinite(v) for v in p)) and (not meta.get('touches_crop_edge', False))
        reason = 'missing_nonfinite_or_truncated_opening'
        item = {'valid': False, 'endpoint_role': 'CNN mask extrema along PCA axis; NOT human-verified canthi'}
        if ok:
            centered = a - a.mean(0)
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            axis = vt[0]
            if axis[0] < 0:
                axis = -axis
            projections = a @ axis
            left = a[int(projections.argmin())]
            right = a[int(projections.argmax())]
            delta = right - left
            width = float(np.linalg.norm(delta))
            normal = np.array([-delta[1], delta[0]])
            ok = width > 1e-06 and cv2.pointPolygonTest(a, tuple(map(float, p)), False) >= 0
            reason = 'pupil_outside_opening_or_degenerate_axis'
            if ok:
                q = np.asarray(p) - left
                u = float(q @ delta / width ** 2)
                v = float(q @ normal / width ** 2)
                height = float(np.ptp(a @ (normal / width)))
                values.extend([u, v, height / width])
                valid.append(1.0)
                item.update(valid=True, left_endpoint=left.tolist(), right_endpoint=right.tolist(), pupil_u=u, pupil_v=v, height_width=height / width)
        if not ok:
            values.extend([0.0, 0.0, 0.0])
            valid.append(0.0)
            item['reason'] = reason
        details[side] = item
    pair = all(valid)
    common = (values[0] + values[3]) / 2 if pair else 0.0
    diff = values[0] - values[3] if pair else 0.0
    return (np.asarray(values + [common, diff] + valid, np.float32), details)

def normalize_opening(features, mean, std):
    f = np.asarray(features, np.float32)
    z = (f[:8] - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)
    z[:3] *= f[8]
    z[3:6] *= f[9]
    z[6:] *= f[8] * f[9]
    return np.r_[z, f[8:]].astype(np.float32)

class ScreeningEyelidBranch(nn.Module):

    def __init__(self, base=None):
        super().__init__()
        self.base = base if base is not None else ScreeningFusion()
        self.opening = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 3))
        nn.init.zeros_(self.opening[-1].weight)
        nn.init.zeros_(self.opening[-1].bias)

    def forward(self, image, geometry, opening):
        complete = (opening[:, -2:].prod(1, keepdim=True) > 0).to(image.dtype)
        return self.base(image, geometry) + self.opening(opening) * complete

def infer(model, image, proposal, geometry_mean, geometry_std, opening_mean, opening_std, device='cpu'):
    from app.hirschberg_runtime.research.hirschberg_localization.screening_fusion import geometry, image_panel, tensor, normalize_features, CLASSES
    g = geometry(proposal, *image.size)
    if not all(g[-2:]):
        return {'status': 'INCONCLUSIVE', 'reason': 'missing_or_invalid_bilateral_geometry', 'scores': None}
    try:
        panel, _ = image_panel(image, proposal)
    except ValueError as e:
        return {'status': 'INCONCLUSIVE', 'reason': str(e), 'scores': None}
    branch_eyes = {s: {**e, 'eyelid_contour': e.get('eyelid_branch_contour', e.get('eyelid_contour', []))} for s, e in proposal.items()}
    opening, details = opening_features(branch_eyes)
    with torch.no_grad():
        scores = model(tensor(panel)[None].to(device), torch.from_numpy(normalize_features(g, geometry_mean, geometry_std))[None].to(device), torch.from_numpy(normalize_opening(opening, opening_mean, opening_std))[None].to(device)).softmax(1)[0].cpu().tolist()
    label = CLASSES[int(np.argmax(scores))]
    return {'status': 'HUMAN_REVIEW', 'class_proposal': label, 'uncalibrated_model_scores': dict(zip(CLASSES, scores)), 'opening_branch_active': bool(all(opening[-2:])), 'opening_geometry': details, 'screening_proposal': 'NO_ALIGNMENT_SIGN_DETECTED' if label == 'normal' else 'POSSIBLE_ALIGNMENT_SIGN', 'clinical_diagnosis': None, 'pseudostrabismus_supported': False, 'warning': 'Research only; estimated opening endpoints are not verified canthi; scores are uncalibrated.'}
