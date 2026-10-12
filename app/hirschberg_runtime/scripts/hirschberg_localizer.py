"""Frozen inference subset imported from the training workspace."""
def prepare(image, size):
    from PIL import Image
    import numpy as np
    width, height = image.size
    factor = min(size / width, size / height)
    w, h = (max(1, round(width * factor)), max(1, round(height * factor)))
    left, top = ((size - w) // 2, (size - h) // 2)
    canvas = Image.new('RGB', (size, size))
    canvas.paste(image.resize((w, h)), (left, top))
    return (np.asarray(canvas, dtype='float32').transpose(2, 0, 1) / 255, (w / width, h / height, left, top))

def inverse(xy, transform):
    sx, sy, left, top = transform
    return [(xy[0] - left) / sx, (xy[1] - top) / sy]

def network(torch, spatial_logits=False, targets=4):
    nn = torch.nn

    def block(a, b):
        return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.ReLU(), nn.Conv2d(b, b, 3, padding=1), nn.ReLU())

    class Localizer(nn.Module):

        def __init__(self):
            super().__init__()
            self.spatial_logits = spatial_logits
            self.e1, self.e2, self.e3 = (block(5 if spatial_logits else 3, 16), block(16, 32), block(32, 64))
            self.d2, self.d1 = (block(96, 32), block(48, 16))
            self.heat = nn.Conv2d(16, targets, 1)
            self.visible = nn.Linear(64, targets * 3)

        def forward(self, x):
            if self.spatial_logits:
                height, width = x.shape[-2:]
                yy, xx = torch.meshgrid(torch.linspace(-1, 1, height, device=x.device, dtype=x.dtype), torch.linspace(-1, 1, width, device=x.device, dtype=x.dtype), indexing='ij')
                coordinates = torch.stack((xx, yy))[None].expand(x.shape[0], -1, -1, -1)
                x = torch.cat((x, coordinates), 1)
            a = self.e1(x)
            b = self.e2(nn.functional.max_pool2d(a, 2))
            c = self.e3(nn.functional.max_pool2d(b, 2))
            d = self.d2(torch.cat((nn.functional.interpolate(c, size=b.shape[-2:], mode='bilinear', align_corners=False), b), 1))
            d = self.d1(torch.cat((nn.functional.interpolate(d, size=a.shape[-2:], mode='bilinear', align_corners=False), a), 1))
            heat = self.heat(d)
            return (heat if self.spatial_logits else heat.sigmoid(), self.visible(c.mean((2, 3))).reshape(-1, targets, 3))
    return Localizer()
