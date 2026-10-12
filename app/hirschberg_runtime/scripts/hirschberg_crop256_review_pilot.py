"""Frozen inference subset imported from the training workspace."""
import math

def padded_box(box, width, height):
    if not box or len(box) != 4 or (not all((math.isfinite(v) for v in box))):
        return None
    x0, y0, x1, y1 = box
    if x1 <= x0 or y1 <= y0:
        return None
    dx, dy = ((x1 - x0) * 0.1, (y1 - y0) * 0.1)
    result = [max(0, math.floor(x0 - dx)), max(0, math.floor(y0 - dy)), min(width, math.ceil(x1 + dx)), min(height, math.ceil(y1 + dy))]
    return result if result[0] < result[2] and result[1] < result[3] else None
