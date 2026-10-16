"""Frozen inference subset imported from the training workspace."""
from app.hirschberg_runtime.scripts.hirschberg_localizer import network

def model_factory(torch, initializer):
    nn = torch.nn

    class IrisGeometry(nn.Module):

        def __init__(self):
            super().__init__()
            original = network(torch, spatial_logits=True, targets=2)
            original.load_state_dict(initializer['state_dict'])
            self.e1, self.e2, self.e3 = (original.e1, original.e2, original.e3)
            for module in (self.e1, self.e2):
                for p in module.parameters():
                    p.requires_grad = False
            self.head = nn.Sequential(nn.AvgPool2d(8), nn.Flatten(), nn.Linear(64 * 36, 128), nn.ReLU(), nn.Linear(128, 4))

        def forward(self, x):
            h, w = x.shape[-2:]
            yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=x.device), torch.linspace(-1, 1, w, device=x.device), indexing='ij')
            x = torch.cat((x, torch.stack((xx, yy))[None].expand(x.shape[0], -1, -1, -1)), 1)
            a = self.e1(x)
            b = self.e2(nn.functional.max_pool2d(a, 2))
            c = self.e3(nn.functional.max_pool2d(b, 2))
            raw = self.head(c)
            center = raw[:, :2].sigmoid()
            rx = raw[:, 2].sigmoid() * torch.minimum(center[:, 0], 1 - center[:, 0])
            dy = raw[:, 3].tanh() * torch.minimum(center[:, 1], 1 - center[:, 1])
            return torch.stack((center[:, 0] - rx, center[:, 1] - dy, center[:, 0] + rx, center[:, 1] + dy), 1)
    return IrisGeometry()
