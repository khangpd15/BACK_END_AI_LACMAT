"""Frozen inference subset imported from the training workspace."""
def to_crop(xy, transform):
    x, y = (xy[0] - transform['bbox'][0], xy[1] - transform['bbox'][1])
    if transform['horizontal_flip']:
        x = transform['crop_width'] - 1 - x
    return [x, y]

def to_source(xy, transform):
    x, y = xy
    if transform['horizontal_flip']:
        x = transform['crop_width'] - 1 - x
    return [x + transform['bbox'][0], y + transform['bbox'][1]]
