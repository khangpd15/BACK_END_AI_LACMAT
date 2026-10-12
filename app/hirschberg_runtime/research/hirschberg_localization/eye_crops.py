"""Frozen inference subset imported from the training workspace."""
from PIL import Image

def crop_eye(image, eye, orientation):
    if eye not in ('od', 'os') or orientation not in ('unmirrored', 'mirrored'):
        raise ValueError('Verified anatomical eye and orientation required')
    left = (eye == 'od') == (orientation == 'unmirrored')
    midpoint = image.width // 2
    bbox = (0, 0, midpoint, image.height) if left else (midpoint, 0, image.width, image.height)
    crop = image.crop(bbox)
    flip = not left
    if flip:
        crop = crop.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    return (crop, {'bbox': bbox, 'horizontal_flip': flip, 'crop_width': crop.width})

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
