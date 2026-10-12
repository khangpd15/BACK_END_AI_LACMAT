"""Frozen inference subset imported from the training workspace."""
from app.hirschberg_runtime.scripts import hirschberg_iris_geometry_pilot as pilot
ORIGINAL_FACTORY = pilot.model_factory

def stable_model(torch, initializer):
    model = ORIGINAL_FACTORY(torch, initializer)
    nn = torch.nn
    model.head = nn.Sequential(nn.GroupNorm(8, 64), nn.AvgPool2d(8), nn.Flatten(), nn.Linear(64 * 36, 128), nn.ReLU(), nn.Linear(128, 4))
    nn.init.zeros_(model.head[-1].weight)
    nn.init.zeros_(model.head[-1].bias)
    return model
