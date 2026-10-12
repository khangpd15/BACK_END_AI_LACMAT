"""Frozen inference subset imported from the training workspace."""
import cv2
import torch
from torch import nn
from app.hirschberg_runtime.scripts.hirschberg_reviewed_incremental_train import crop_input, prepare
from app.hirschberg_runtime.research.hirschberg_localization.eye_crops import to_source

class EyelidUNet(nn.Module):

    def __init__(self):
        super().__init__()

        def block(a, b):
            return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.GroupNorm(4, b), nn.ReLU(), nn.Conv2d(b, b, 3, padding=1), nn.GroupNorm(4, b), nn.ReLU())
        self.e1 = block(3, 16)
        self.e2 = block(16, 32)
        self.e3 = block(32, 64)
        self.d2 = block(96, 32)
        self.d1 = block(48, 16)
        self.head = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        a = self.e1(x)
        b = self.e2(nn.functional.max_pool2d(a, 2))
        c = self.e3(nn.functional.max_pool2d(b, 2))
        d = self.d2(torch.cat((nn.functional.interpolate(c, size=b.shape[-2:], mode='bilinear', align_corners=False), b), 1))
        return self.head(self.d1(torch.cat((nn.functional.interpolate(d, size=a.shape[-2:], mode='bilinear', align_corners=False), a), 1)))

def predict_mask(model, image, box, side, device='cpu', size=192):
    crop, t = crop_input(image, box, side)
    array, lb = prepare(crop, size)
    with torch.no_grad():
        prob = model(torch.from_numpy(array)[None].to(device)).sigmoid()[0, 0].cpu().numpy()
    sx, sy, left, top = lb
    w, h = (round(crop.width * sx), round(crop.height * sy))
    mask = (prob[top:top + h, left:left + w] >= 0.5).astype('uint8')
    mask = cv2.resize(mask, crop.size, interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    metadata = {'threshold': 0.5, 'source': 'eyelid_UNet_human_edited_contours', 'human_review_required': True, 'clinical_validated': False, 'crop': t, 'letterbox': lb}
    source_mask = cv2.flip(mask, 1) if t['horizontal_flip'] else mask
    metadata['mask_coordinate_space'] = 'source crop pixels, horizontal flip undone'
    if not contours:
        return ([], source_mask, metadata)
    c = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    points = cv2.approxPolyDP(c, 0.005 * cv2.arcLength(c, True), True).reshape(-1, 2)
    touches = any((x <= 0 or y <= 0 or x >= crop.width - 1 or (y >= crop.height - 1) for x, y in points))
    metadata.update(area_px=area, touches_crop_edge=bool(touches), candidate_only=True)
    pts = [list(map(float, to_source([float(x), float(y)], t))) for x, y in points] if area > 0 else []
    return (pts, source_mask, metadata)
