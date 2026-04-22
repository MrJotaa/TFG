from PIL import Image
from torchvision import transforms


def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Preprocesa una imagen para que sea compatible con modelos de torchvision
    como ResNet18.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # Redimensiona la imagen a 224x224 píxeles
        transforms.ToTensor(), # Convierte la imagen a un tensor de PyTorch
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], # Normaliza la imagen con los valores de media y desviación estándar utilizados durante el entrenamiento de ResNet
            std=[0.229, 0.224, 0.225]
        ),
    ])

    tensor = transform(image)
    tensor = tensor.unsqueeze(0)  # añade dimensión batch
    return tensor