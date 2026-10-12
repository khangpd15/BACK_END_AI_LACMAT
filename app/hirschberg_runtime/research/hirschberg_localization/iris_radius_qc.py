"""Frozen inference subset imported from the training workspace."""
import math

def compare_diameters(diameters, limit=0.15):
    if len(diameters) != 2 or not all((isinstance(v, (int, float)) and (not isinstance(v, bool)) and math.isfinite(v) and (v > 0) for v in diameters)):
        return {'status': 'NOT_ASSESSED', 'reason': 'missing iris diameters'}
    a, b = diameters
    difference = abs(a - b) / max(a, b)
    return {'relative_difference': difference, 'limit': limit, 'denominator': 'max(R_OD,R_OS)', 'formula': 'abs(R_OD-R_OS)/max(R_OD,R_OS)', 'status': 'PASS_RESEARCH_ONLY' if difference <= limit else 'Invalid Landmark / Low Confidence', 'clinical_validated': False}
