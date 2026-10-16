"""Frozen inference subset imported from the training workspace."""
from app.hirschberg_runtime.scripts.hirschberg_reviewed_incremental_train import crop_input
from app.hirschberg_runtime.scripts.hirschberg_localizer import prepare, inverse
from app.hirschberg_runtime.research.hirschberg_localization.eye_crops import to_source

def decode(pr_model, image, box, side, torch, device, size=192, *, visualize=True):
    import numpy as np
    from PIL import Image
    result = {'bbox': box, 'pupil': None, 'reflex': None, 'raw_predictions': {}, 'heatmap_transforms': {}}
    if not box:
        return (result, {})
    crop, t = crop_input(image, box, side)
    array, letterbox = prepare(crop, size)
    with torch.no_grad():
        heat, flags = pr_model(torch.tensor(array[None]).to(device))
        prob = heat.flatten(2).softmax(-1).reshape_as(heat)
    maps = {}
    for j, kind in enumerate(('pupil', 'reflex')):
        flat = prob[0, j].argmax().item()
        p = inverse([flat % size, flat // size], letterbox)
        source = to_source(p, t)
        visibility_probs = flags[0, j].softmax(0).cpu().tolist()
        visibility = ('visible', 'occluded', 'not_visible')[max(range(3), key=lambda i: visibility_probs[i])]
        valid = 0 <= p[0] < crop.width and 0 <= p[1] < crop.height and (0 <= source[0] < image.width) and (0 <= source[1] < image.height)
        result['raw_predictions'][kind] = {'peak_xy': source, 'in_crop_and_source': valid, 'visibility': visibility, 'visibility_probabilities': visibility_probs}
        result[kind] = source if valid and visibility == 'visible' else None
        if visualize:
            values = prob[0, j].cpu().numpy()
            gray = Image.fromarray(np.round(values / values.max() * 255).astype('uint8'))
            maps[kind + f'_model{size}'] = gray
            sx, sy, left, top = letterbox
            w, h = (round(crop.width * sx), round(crop.height * sy))
            source_map = gray.crop((left, top, left + w, top + h)).resize(crop.size, Image.Resampling.BILINEAR)
            if t['horizontal_flip']:
                source_map = source_map.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            maps[kind] = source_map
        result['heatmap_transforms'][kind] = {'crop': t, 'letterbox': letterbox, 'normalization': 'spatial softmax divided by max for visualization only'}
    return (result, maps)
