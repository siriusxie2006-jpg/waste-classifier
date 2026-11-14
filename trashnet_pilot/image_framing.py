"""Keep the full image extent, or crop only an explicitly selected rectangle."""
import numpy as np
from PIL import Image, ImageOps


def crop_image(image, box=None):
    if box is None:
        return image.copy()
    values = np.asarray(box, dtype=float)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError('Invalid crop rectangle')
    left, top, right, bottom = values.tolist()
    if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
        raise ValueError('Crop rectangle is outside the image')
    return image.crop((int(np.floor(left*image.width)), int(np.floor(top*image.height)),
                       int(np.ceil(right*image.width)), int(np.ceil(bottom*image.height))))


def fit_image(image):
    resized = ImageOps.contain(image.convert('RGB'), (224, 224), method=Image.Resampling.BILINEAR)
    framed = Image.new('RGB', (224, 224), 'white')
    framed.paste(resized, ((224-resized.width)//2, (224-resized.height)//2))
    return framed
