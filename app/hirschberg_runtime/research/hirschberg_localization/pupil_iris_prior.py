"""Frozen inference subset imported from the training workspace."""
def make_model(torch, state):
    from app.hirschberg_runtime.scripts.hirschberg_localizer import network
    model = network(torch, spatial_logits=True, targets=2)
    model.load_state_dict(state)
    old = model.e1[0]
    new = torch.nn.Conv2d(6, old.out_channels, 3, padding=1)
    with torch.no_grad():
        new.weight.zero_()
        new.bias.copy_(old.bias)
        new.weight[:, :3].copy_(old.weight[:, :3])
        new.weight[:, 4:].copy_(old.weight[:, 3:])
    model.e1[0] = new
    return model

def soft_prior(points, transform, letterbox, size):
    import numpy as np
    from app.hirschberg_runtime.research.hirschberg_localization.eye_crops import to_crop
    if not points:
        return np.zeros((1, size, size), dtype='float32')
    local = [to_crop(points[k], transform) for k in ('iris_left', 'iris_right', 'iris_top', 'iris_bottom')]
    sx, sy, left, top = letterbox
    x = (local[0][0] + local[1][0]) / 2 * sx + left
    y = (local[2][1] + local[3][1]) / 2 * sy + top
    rx = abs(local[1][0] - local[0][0]) * sx / 2
    ry = abs(local[3][1] - local[2][1]) * sy / 2
    if not all((np.isfinite(v) for v in (x, y, rx, ry))) or min(rx, ry) <= 0:
        return np.zeros((1, size, size), dtype='float32')
    yy, xx = np.mgrid[:size, :size]
    return np.exp(-0.5 * (((xx - x) / rx) ** 2 + ((yy - y) / ry) ** 2))[None].astype('float32')
