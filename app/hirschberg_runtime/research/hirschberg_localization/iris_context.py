"""Frozen inference subset imported from the training workspace."""
from app.hirschberg_runtime.research.hirschberg_localization.iris_four_points import predict_four

def square_height_context(box, width, height):
    if not box:
        return None
    x0, y0, x1, y1 = box
    target = min(height, max(y1 - y0, x1 - x0))
    top = max(0, min(height - target, round((y0 + y1 - target) / 2)))
    return [x0, top, x1, top + target]

def predict_with_context(model, image, box, side, torch, device):
    points = predict_four(model, image, box, side, torch, device)
    meta = {'original_bbox': box, 'used_bbox': box, 'retry_used': False, 'method': 'original_frozen_iris_crop', 'human_review_required': True}
    if points is not None or not box:
        return (points, meta)
    expanded = square_height_context(box, *image.size)
    if expanded == box:
        return (None, {**meta, 'status': 'no_additional_source_context'})
    retry = predict_four(model, image, expanded, side, torch, device)
    meta.update(used_bbox=expanded, retry_used=True, method='single_square_height_source_context_retry', status='proposal_requires_review' if retry else 'rejected_retry')
    return (retry, meta)
