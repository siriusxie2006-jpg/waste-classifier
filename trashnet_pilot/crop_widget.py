"""Local-only manual rectangle selection; no third-party JavaScript or CDN."""
import base64
import io
from pathlib import Path
from PIL import Image, ImageOps
from streamlit.components.v1 import declare_component

_rectangle = declare_component('manual_image_crop', path=str(Path(__file__).parent/'components'/'manual_crop'))


def manual_crop(image, image_id, applied_box=None, reset_token=0):
    preview = ImageOps.contain(image.convert('RGB'), (760, 360), method=Image.Resampling.BILINEAR)
    buffer = io.BytesIO()
    preview.save(buffer, format='PNG')
    return _rectangle(image='data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii'),
                      image_id=image_id, applied_box=applied_box, reset_token=reset_token,
                      default=None, key='manual_crop_'+image_id[:16])
