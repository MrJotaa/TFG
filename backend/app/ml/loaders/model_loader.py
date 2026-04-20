from torchvision import models


def load_resnet18():
    """
    Carga un modelo ResNet18 preentrenado y lo prepara para inferencia.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT) # Pesos preentrenados por defecto
    model.eval() # Configura el modelo para inferencia (desactiva dropout, batchnorm, etc.)
    return model