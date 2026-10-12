"""Frozen inference subset imported from the training workspace."""
def four_point_model(torch, horizontal_model):
    nn = torch.nn

    class IrisFour(nn.Module):

        def __init__(self):
            super().__init__()
            self.e1, self.e2, self.e3 = (horizontal_model.e1, horizontal_model.e2, horizontal_model.e3)
            self.head = horizontal_model.head
            old = self.head[-1]
            new = nn.Linear(old.in_features, 8)
            with torch.no_grad():
                new.weight.zero_()
                new.bias.zero_()
                new.weight[:4].copy_(old.weight)
                new.bias[:4].copy_(old.bias)
            self.head[-1] = new

        def forward(self, x):
            h, w = x.shape[-2:]
            yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=x.device), torch.linspace(-1, 1, w, device=x.device), indexing='ij')
            x = torch.cat((x, torch.stack((xx, yy))[None].expand(x.shape[0], -1, -1, -1)), 1)
            a = self.e1(x)
            b = self.e2(nn.functional.max_pool2d(a, 2))
            c = self.e3(nn.functional.max_pool2d(b, 2))
            raw = self.head(c)
            cx, cy = (raw[:, 0].sigmoid(), raw[:, 1].sigmoid())
            rx = raw[:, 2].sigmoid() * torch.minimum(cx, 1 - cx)
            dy = raw[:, 3].tanh() * torch.minimum(cy, 1 - cy)
            vx, vy = (raw[:, 4].sigmoid(), raw[:, 5].sigmoid())
            ry = raw[:, 6].sigmoid() * torch.minimum(vy, 1 - vy)
            dx = raw[:, 7].tanh() * torch.minimum(vx, 1 - vx)
            return torch.stack((cx - rx, cy - dy, cx + rx, cy + dy, vx - dx, vy - ry, vx + dx, vy + ry), 1)
    return IrisFour()

def predict_four(model, image, box, side, torch, device):
    from app.hirschberg_runtime.scripts.hirschberg_reviewed_incremental_train import crop_input, prepare
    from app.hirschberg_runtime.research.hirschberg_localization.eye_crops import to_source
    if not box:
        return None
    crop, transform = crop_input(image, box, side)
    array, (sx, sy, left, top) = prepare(crop, 192)
    with torch.no_grad():
        raw = model(torch.tensor(array[None]).to(device))[0].cpu().tolist()
    points = [to_source([(raw[i] * 192 - left) / sx, (raw[i + 1] * 192 - top) / sy], transform) for i in range(0, 8, 2)]
    if any((not (box[0] <= p[0] < box[2] and box[1] <= p[1] < box[3]) for p in points)):
        return None
    horizontal = sorted(points[:2], key=lambda p: p[0])
    vertical = sorted(points[2:], key=lambda p: p[1])
    return dict(zip(('iris_left', 'iris_right', 'iris_top', 'iris_bottom'), horizontal + vertical))
