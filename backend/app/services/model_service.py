from ml.models.resnet18.model import ResNet18Model
from schemas.models import ModelInfo


resnet18 = ResNet18Model()
resnet18.load()

_loaded_models = {
    resnet18.id: {
        "info": ModelInfo(
            id=resnet18.id,
            name=resnet18.name,
            active=True,
        ),
        "instance": resnet18,
    }
}


def list_models() -> list[ModelInfo]:
    return [model_data["info"] for model_data in _loaded_models.values()]


def get_active_model() -> ModelInfo:
    for model_data in _loaded_models.values():
        if model_data["info"].active:
            return model_data["info"]

    raise ValueError("No active model found")

def get_active_model_instance():
    for model_data in _loaded_models.values():
        if model_data["info"].active:
            return model_data["instance"]

    raise ValueError("No active model instance found")

"""
No se necesita get_active_model_categories()
porque cada modelo ya tiene su propia lógica para manejar las categorías.
"""

def select_model(model_id: str) -> ModelInfo:
    if model_id not in _loaded_models:
        raise ValueError(f"Model '{model_id}' not found")

    for current_id, model_data in _loaded_models.items():
        model_data["info"].active = current_id == model_id

    return _loaded_models[model_id]["info"]