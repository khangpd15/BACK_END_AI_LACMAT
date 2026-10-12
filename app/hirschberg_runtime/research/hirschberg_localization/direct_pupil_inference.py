"""Frozen inference subset imported from the training workspace."""
import numpy as np
from PIL import Image
from app.hirschberg_runtime.scripts.hirschberg_reviewed_incremental_train import crop_input, prepare
from app.hirschberg_runtime.scripts.hirschberg_localizer import inverse
from app.hirschberg_runtime.research.hirschberg_localization.eye_crops import to_source
from app.hirschberg_runtime.research.hirschberg_localization.pupil_iris_prior import soft_prior
from app.hirschberg_runtime.research.hirschberg_localization.pupil_center_decoder import expected_center

def predict(model, image, box, side, iris_four, torch, device):
    if not box:
        return (None, {}, {})
    crop, transform = crop_input(image, box, side)
    array, lb = prepare(crop, 256)
    prior = soft_prior(iris_four, transform, lb, 256)
    yy, xx = np.mgrid[:256, :256]
    sx, sy, left, top = lb
    mask = (xx >= left) & (xx < left + round(crop.width * sx)) & (yy >= top) & (yy < top + round(crop.height * sy))
    with torch.no_grad():
        heat, flags = model(torch.from_numpy(np.concatenate((array, prior), axis=0))[None].to(device))
        xy = expected_center(heat[:, 0], torch.from_numpy(mask)[None].to(device), torch)[0].cpu().tolist()
        values = heat[0, 0].masked_fill(~torch.from_numpy(mask).to(device), -1000000000.0).flatten().softmax(0).reshape(256, 256).cpu().numpy()
        visibility = flags[0, 0].softmax(0).cpu().tolist()
    local = inverse(xy, lb)
    point = to_source(local, transform)
    valid = all(np.isfinite(point)) and 0 <= local[0] < crop.width and (0 <= local[1] < crop.height) and (0 <= point[0] < image.width) and (0 <= point[1] < image.height)
    gray = Image.fromarray(np.round(values / values.max() * 255).astype('uint8'))
    mapped = gray.crop((left, top, left + round(crop.width * sx), top + round(crop.height * sy))).resize(crop.size, Image.Resampling.BILINEAR)
    if transform['horizontal_flip']:
        mapped = mapped.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    metadata = {'peak_xy': list(map(float, point)), 'in_crop_and_source': bool(valid), 'decoder': 'expected_center', 'visibility': ('visible', 'occluded', 'not_visible')[int(np.argmax(visibility))], 'visibility_probabilities': visibility, 'visibility_reliable': False, 'warning': 'Visibility head unreliable on occluded eyes; coordinate proposal requires human QC', 'transform': {'crop': transform, 'letterbox': lb}}
    return (list(map(float, point)) if valid else None, {'pupil': mapped, 'pupil_model256': gray}, metadata)
