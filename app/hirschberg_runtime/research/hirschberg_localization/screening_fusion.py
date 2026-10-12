"""Frozen inference subset imported from the training workspace."""
import math
import numpy as np
import torch
from torch import nn
from PIL import Image
from app.hirschberg_runtime.research.hirschberg_localization.resnet_reference import resnet
CLASSES = ['normal', 'esotropia', 'exotropia']

def geometry(proposal, width, height):
    vectors = []
    radii = []
    valid = []
    for side in ('image_left', 'image_right'):
        eye = proposal.get(side, {})
        points = [eye.get(k) for k in ('pupil', 'reflex', 'iris_left', 'iris_right')]
        ok = all((isinstance(p, list) and len(p) == 2 and all((isinstance(v, (int, float)) and math.isfinite(v) for v in p)) and (0 <= p[0] < width) and (0 <= p[1] < height) for p in points))
        radius = math.dist(points[2], points[3]) / 2 if ok else 0
        ok = bool(ok and radius > 0)
        vectors.append([(points[1][j] - points[0][j]) / radius for j in (0, 1)] if ok else [0.0, 0.0])
        radii.append(radius if ok else 0.0)
        valid.append(float(ok))
    left, right = vectors
    pair = all(valid)
    common = [(a + b) / 2 for a, b in zip(left, right)] if pair else [0.0, 0.0]
    difference = [a - b for a, b in zip(left, right)] if pair else [0.0, 0.0]
    return np.array(left + right + common + difference + [min(radii) / max(radii) if pair else 0.0] + valid, np.float32)

def image_panel(image, proposal):
    out = Image.new('RGB', (224, 224))
    transforms = []
    for index, side in enumerate(('image_left', 'image_right')):
        box = proposal.get(side, {}).get('bbox')
        if not box or not all((isinstance(x, (float, int)) and math.isfinite(x) for x in box)) or (not (0 <= box[0] < box[2] <= image.width and 0 <= box[1] < box[3] <= image.height)):
            raise ValueError('missing_or_invalid_eye_crop:' + side)
        crop = image.crop(box)
        w, h = crop.size
        scale = min(224 / w, 112 / h)
        nw, nh = (max(1, round(w * scale)), max(1, round(h * scale)))
        x, y = ((224 - nw) // 2, index * 112 + (112 - nh) // 2)
        out.paste(crop.resize((nw, nh), Image.Resampling.BILINEAR), (x, y))
        transforms.append({'side': side, 'bbox': box, 'scale_xy': [nw / w, nh / h], 'offset_xy': [x, y]})
    return (out, transforms)

def tensor(image):
    a = np.asarray(image, np.float32).transpose(2, 0, 1) / 255
    return torch.from_numpy((a - np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]) / np.array([0.229, 0.224, 0.225], np.float32)[:, None, None])

def normalize_features(features, mean, std):
    z = (features[:9] - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)
    z[:2] *= features[9]
    z[2:4] *= features[10]
    z[4:] *= features[9] * features[10]
    return np.r_[z, features[9:]].astype(np.float32)

class ScreeningFusion(nn.Module):

    def __init__(self):
        super().__init__()
        self.image = resnet('resnet50')
        self.image.fc = nn.Identity()
        self.biometry = nn.Sequential(nn.Linear(11, 32), nn.ReLU())
        self.fusion = nn.Sequential(nn.Linear(2080, 128), nn.ReLU(), nn.Linear(128, 3))

    def forward(self, image, features):
        return self.fusion(torch.cat((self.image(image), self.biometry(features)), dim=1))
