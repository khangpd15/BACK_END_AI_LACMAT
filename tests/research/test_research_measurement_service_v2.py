import math
import numpy as np
import pytest
from PIL import Image, ImageDraw

from app.services.research_measurement_service import (
    EyeHirschbergMeasurement,
    QualityGateMetrics,
    ResearchMeasurementOutput,
    ResearchMeasurementService,
    ROICrops,
)


def _create_synthetic_face_image(w=400, h=300, blur=False):
    """Creates a synthetic face image with eyes and corneal glints."""
    img = Image.new("RGB", (w, h), color=(210, 180, 150))
    draw = ImageDraw.Draw(img)

    # Face oval
    draw.ellipse((50, 20, 350, 280), fill=(230, 200, 175))

    # Eyes: Right eye around (140, 130), Left eye around (260, 130)
    # Sclera
    draw.ellipse((105, 115, 175, 145), fill=(250, 250, 250))
    draw.ellipse((225, 115, 295, 145), fill=(250, 250, 250))

    # Irises
    draw.ellipse((125, 115, 155, 145), fill=(70, 50, 30))
    draw.ellipse((245, 115, 275, 145), fill=(70, 50, 30))

    # Pupils
    draw.ellipse((135, 125, 145, 135), fill=(10, 10, 10))
    draw.ellipse((255, 125, 265, 135), fill=(10, 10, 10))

    # Corneal glints (specular highlight: peak brightness 255)
    draw.rectangle((139, 129, 141, 131), fill=(255, 255, 255))
    draw.rectangle((259, 129, 261, 131), fill=(255, 255, 255))

    arr = np.array(img)
    if blur:
        # Heavily smooth array to trigger blur check
        import cv2
        arr = cv2.GaussianBlur(arr, (25, 25), 10.0)

    return arr


def _create_synthetic_landmarks(w=400, h=300, tilt_angle_deg=0.0):
    """Generates 478 MediaPipe landmarks matching synthetic face coordinates."""
    lms = [{"x": 0.5, "y": 0.5, "z": 0.0} for _ in range(478)]

    # Key head pose landmarks
    lms[1] = {"x": 200 / w, "y": 160 / h, "z": 0.0}    # Nose tip
    lms[152] = {"x": 200 / w, "y": 270 / h, "z": 0.0}  # Chin
    lms[33] = {"x": 105 / w, "y": 130 / h, "z": 0.0}   # Right outer corner
    lms[133] = {"x": 175 / w, "y": 130 / h, "z": 0.0}  # Right inner corner (nasal)
    lms[362] = {"x": 225 / w, "y": 130 / h, "z": 0.0}  # Left inner corner (nasal)
    lms[263] = {"x": 295 / w, "y": 130 / h, "z": 0.0}  # Left outer corner (temporal)
    lms[57] = {"x": 160 / w, "y": 230 / h, "z": 0.0}   # Right mouth corner
    lms[287] = {"x": 240 / w, "y": 230 / h, "z": 0.0}  # Left mouth corner

    # Iris landmarks:
    # Right iris: (473 center, 474 right, 475 top, 476 left, 477 bottom)
    cx_r, cy_r, r_r = 140, 130, 15
    lms[473] = {"x": cx_r / w, "y": cy_r / h, "z": 0.0}
    lms[474] = {"x": (cx_r + r_r) / w, "y": cy_r / h, "z": 0.0}
    lms[475] = {"x": cx_r / w, "y": (cy_r - r_r) / h, "z": 0.0}
    lms[476] = {"x": (cx_r - r_r) / w, "y": cy_r / h, "z": 0.0}
    lms[477] = {"x": cx_r / w, "y": (cy_r + r_r) / h, "z": 0.0}

    # Left iris: (468 center, 469 right, 470 top, 471 left, 472 bottom)
    cx_l, cy_l, r_l = 260, 130, 15
    lms[468] = {"x": cx_l / w, "y": cy_l / h, "z": 0.0}
    lms[469] = {"x": (cx_l + r_l) / w, "y": cy_l / h, "z": 0.0}
    lms[470] = {"x": cx_l / w, "y": (cy_l - r_l) / h, "z": 0.0}
    lms[471] = {"x": (cx_l - r_l) / w, "y": cy_l / h, "z": 0.0}
    lms[472] = {"x": cx_l / w, "y": (cy_l + r_l) / h, "z": 0.0}

    # Eyelids
    lms[386] = {"x": 260 / w, "y": 115 / h, "z": 0.0}
    lms[374] = {"x": 260 / w, "y": 145 / h, "z": 0.0}
    lms[159] = {"x": 140 / w, "y": 115 / h, "z": 0.0}
    lms[145] = {"x": 140 / w, "y": 145 / h, "z": 0.0}

    return lms


def test_service_initialization():
    service = ResearchMeasurementService()
    assert service.min_blur_var == 100.0
    assert service.cornea_diameter_mm == 11.7


def test_blur_variance_detection():
    service = ResearchMeasurementService(min_blur_var=100.0)
    sharp_img = _create_synthetic_face_image(blur=False)
    blurry_img = _create_synthetic_face_image(blur=True)

    var_sharp = service.compute_blur_variance(sharp_img)
    var_blurry = service.compute_blur_variance(blurry_img)

    assert var_sharp > 100.0, f"Expected sharp variance > 100, got {var_sharp}"
    assert var_blurry < 100.0, f"Expected blurry variance < 100, got {var_blurry}"


def test_interpupillary_line_leveling():
    service = ResearchMeasurementService()
    img = _create_synthetic_face_image()
    # Left eye tilted down by 20 pixels
    p_right = (140.0, 120.0)
    p_left = (260.0, 150.0)
    lms = _create_synthetic_landmarks()

    leveled_img, leveled_lms, angle_deg, M = service.level_interpupillary_line(
        img, p_right, p_left, lms
    )

    assert abs(angle_deg) > 0.0
    # Transform right and left points with M
    xr_rot = M[0, 0] * p_right[0] + M[0, 1] * p_right[1] + M[0, 2]
    yr_rot = M[1, 0] * p_right[0] + M[1, 1] * p_right[1] + M[1, 2]
    xl_rot = M[0, 0] * p_left[0] + M[0, 1] * p_left[1] + M[0, 2]
    yl_rot = M[1, 0] * p_left[0] + M[1, 1] * p_left[1] + M[1, 2]

    # After leveling, y coordinates must be virtually identical
    assert abs(yr_rot - yl_rot) < 1e-4, f"Y coordinates not leveled: {yr_rot} vs {yl_rot}"


def test_pseudostrabismus_rule_check_classification():
    service = ResearchMeasurementService()

    # Scenario 1: Pseudostrabismus (Orthophoric reflexes |dx| < 0.40mm, narrow nasal sclera ratio 0.50)
    meas_left = EyeHirschbergMeasurement(
        eye="left",
        pupil_center_x_px=260.0,
        pupil_center_y_px=130.0,
        iris_diameter_px=30.0,
        clr_found=True,
        clr_x_px=260.5,
        clr_y_px=130.0,
        dx_px=0.5,
        dy_px=0.0,
        dx_mm=0.19,  # Within normal |dx| < 0.40mm
        dy_mm=0.0,
        nasal_scleral_area=40.0,
        temporal_scleral_area=80.0,
        nasal_to_temporal_scleral_ratio=0.50,  # Narrow nasal sclera
        status="DETECTED",
    )
    meas_right = EyeHirschbergMeasurement(
        eye="right",
        pupil_center_x_px=140.0,
        pupil_center_y_px=130.0,
        iris_diameter_px=30.0,
        clr_found=True,
        clr_x_px=140.4,
        clr_y_px=130.0,
        dx_px=0.4,
        dy_px=0.0,
        dx_mm=0.16,  # Within normal
        dy_mm=0.0,
        nasal_scleral_area=42.0,
        temporal_scleral_area=82.0,
        nasal_to_temporal_scleral_ratio=0.51,  # Narrow nasal sclera
        status="DETECTED",
    )

    is_pseudo, notes = service.check_pseudostrabismus_rule(meas_left, meas_right)
    assert is_pseudo is True
    assert "Pseudostrabismus" in notes

    # Scenario 2: Genuine Esotropia (Left eye reflex deviated temporal by dx_mm = +0.85mm)
    meas_esotropia_left = EyeHirschbergMeasurement(
        eye="left",
        pupil_center_x_px=260.0,
        pupil_center_y_px=130.0,
        iris_diameter_px=30.0,
        clr_found=True,
        clr_x_px=262.2,
        clr_y_px=130.0,
        dx_px=2.2,
        dy_px=0.0,
        dx_mm=0.85,  # Decentration >= 0.40mm (Esotropia)
        dy_mm=0.0,
        nasal_scleral_area=40.0,
        temporal_scleral_area=80.0,
        nasal_to_temporal_scleral_ratio=0.50,
        status="DETECTED",
    )

    is_pseudo_eso, notes_eso = service.check_pseudostrabismus_rule(meas_esotropia_left, meas_right)
    assert is_pseudo_eso is False
    assert "Esotropia" in notes_eso


def test_process_image_full_pipeline():
    service = ResearchMeasurementService()
    img = _create_synthetic_face_image(blur=False)
    lms = _create_synthetic_landmarks()

    result = service.process_image(img, landmarks_provided=lms)
    assert isinstance(result, ResearchMeasurementOutput)
    assert result.quality.face_detected is True
    assert result.hirschberg_left is not None
    assert result.hirschberg_right is not None
    assert result.rois.left_eye_224 is not None
    assert result.rois.left_eye_224.shape == (224, 224, 3)
    assert result.rois.right_eye_224 is not None
    assert result.rois.right_eye_224.shape == (224, 224, 3)
    assert result.intercanthal_distance_mm is not None
    assert result.intercanthal_distance_mm > 0.0
