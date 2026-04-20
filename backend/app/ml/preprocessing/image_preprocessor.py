from PIL import Image


def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Preprocesado básico de la imagen.
    Por ahora solo redimensiona la imagen.
    """
    return image.resize((224, 224))