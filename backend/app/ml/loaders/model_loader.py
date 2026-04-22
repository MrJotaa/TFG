from torchvision import models


def load_resnet18():
    """
    Carga un modelo ResNet18 preentrenado y lo prepara para inferencia.
    """
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights) # Pesos preentrenados por defecto
    model.eval() # Configura el modelo para inferencia (desactiva dropout, batchnorm, etc.)
    categories = weights.meta["categories"] # Lista de categorías del modelo
    return model, categories