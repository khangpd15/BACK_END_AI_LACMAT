"""Frozen inference subset imported from the training workspace."""
from app.hirschberg_runtime.scripts.hirschberg_localizer import prepare

def crop_input(image, box, side):
    from PIL import Image
    crop = image.crop(box)
    if side == 'image_right':
        crop = crop.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    return (crop, {'bbox': box, 'horizontal_flip': side == 'image_right', 'crop_width': crop.width})
