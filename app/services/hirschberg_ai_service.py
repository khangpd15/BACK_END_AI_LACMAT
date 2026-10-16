"""Compatibility adapter for the trained Hirschberg CNN pipeline."""
from typing import Any, Dict, List, Optional

import cv2
import numpy as np


def hirschberg_runtime_status() -> Dict[str, Any]:
    from app.services.trained_hirschberg_service import trained_hirschberg_status
    return trained_hirschberg_status()


def predict_hirschberg(image_bgr: np.ndarray,
                      landmarks: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Use the trained CNNs; client landmarks do not choose model geometry."""
    from app.services.trained_hirschberg_service import predict_trained_hirschberg
    try:
        ok, encoded = cv2.imencode('.png', image_bgr)
        if not ok:
            return {'status': 'UNAVAILABLE', 'reason': 'INVALID_IMAGE',
                    'message': 'Cannot encode Hirschberg image'}
        return predict_trained_hirschberg(encoded.tobytes())
    except (RuntimeError, ValueError, OSError) as exc:
        return {'status': 'UNAVAILABLE', 'reason': 'RUNTIME_ERROR', 'message': str(exc)}
