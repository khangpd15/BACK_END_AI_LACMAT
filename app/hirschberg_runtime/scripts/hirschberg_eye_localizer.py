"""Frozen inference subset imported from the training workspace."""
from app.hirschberg_runtime.research.hirschberg_localization.eye_crops import crop_eye, to_source

def predict_eyes(model, image, orientation, size, torch, device):
    from app.hirschberg_runtime.scripts.hirschberg_localizer import prepare, inverse
    from app.hirschberg_runtime.research.hirschberg_localization.data import VISIBILITY
    output = {}
    for eye in ('od', 'os'):
        crop, t = crop_eye(image, eye, orientation)
        array, letterbox = prepare(crop, size)
        with torch.no_grad():
            heat, flags = model(torch.tensor(array[None]).to(device))
            heat = heat.flatten(2).softmax(-1).reshape_as(heat)
        for i, kind in enumerate(('pupil', 'reflex')):
            flat = heat[0, i].argmax().item()
            local = inverse([flat % size, flat // size], letterbox)
            xy = to_source(local, t)
            probabilities = flags[0, i].softmax(0).cpu().tolist()
            output[eye + '_' + kind] = {'peak_xy': xy, 'visibility': VISIBILITY[max(range(3), key=lambda j: probabilities[j])], 'visibility_probabilities': probabilities, 'heatmap_peak': heat[0, i].max().item(), 'in_image': 0 <= local[0] < crop.width and 0 <= local[1] < crop.height, 'crop_transform': t, 'letterbox_transform': letterbox}
    return output
