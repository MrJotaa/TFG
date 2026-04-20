import base64
from io import BytesIO
from PIL import Image

def decode_base64_image(image_b64: str) -> Image.Image:
    """
    Decodifica una imagen en base64 y la convierte en un objeto PIL Image.
    """
    image_bytes = base64.b64decode(image_b64)
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    return image