"""Numerical regression checks for removing production-only visualization work."""
import numpy as np
from PIL import Image
import pytest

from app.services.trained_hirschberg_service import load_trained_hirschberg
from app.hirschberg_runtime.scripts.export_hirschberg_review163_steps import decode
from app.hirschberg_runtime.research.hirschberg_localization.boundary_pupil import predict
from app.hirschberg_runtime.research.hirschberg_localization.reflex_roi_qc import diagnose


@pytest.mark.parametrize('side', ['image_left', 'image_right'])
def test_landmarks_and_metadata_identical_without_visualizations(side):
    engine = load_trained_hirschberg()
    image = Image.fromarray(np.random.default_rng(42).integers(0, 256, (80, 160, 3), dtype=np.uint8))
    box = [10, 12, 140, 68]
    full, maps = decode(engine.pr, image, box, side, engine.torch, engine.device, size=256)
    lean, empty = decode(engine.pr, image, box, side, engine.torch, engine.device, size=256, visualize=False)
    assert full == lean
    assert maps and empty == {}
    # The boundary branch accepts the compatible signature but does not use an iris prior.
    iris = {'iris_left': [55, 40], 'iris_right': [95, 40], 'iris_top': [75, 22], 'iris_bottom': [75, 58]}
    point, maps, metadata = predict(engine.pupil, image, box, side, iris, engine.torch, engine.device)
    lean_point, empty, lean_metadata = predict(engine.pupil, image, box, side, iris, engine.torch, engine.device, visualize=False)
    assert point == lean_point
    assert metadata == lean_metadata
    assert metadata['branch'] == 'pupil_boundary'
    assert maps == {} and empty == {}


def test_reflex_quality_gate_identical_without_mask_image():
    pixels = np.full((64, 96, 3), 50, dtype=np.uint8)
    pixels[29:34, 46:51] = 255
    image = Image.fromarray(pixels)
    eye = {'pupil': [48, 32], 'reflex': [48, 31], 'iris_center': [48, 32],
           'iris_diameter_px': 40, 'bbox': [10, 10, 86, 54],
           'eyelid_contour': [[10, 10], [86, 10], [86, 54], [10, 54]]}
    quality, mask = diagnose(image, eye)
    lean_quality, no_mask = diagnose(image, eye, visualize=False)
    assert quality == lean_quality
    assert quality['roi_pixel_count'] > 0
    assert quality['glint_candidate_count'] > 0
    assert mask is not None and no_mask is None
