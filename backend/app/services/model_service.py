from ml.loaders.model_loader import load_resnet18
from schemas.models import ModelInfo


resnet_model, resnet_categories = load_resnet18()

_loaded_models = {
    "resnet18": {
        "info": ModelInfo(id="resnet18", name="ResNet18 pretrained", active=True),
        "instance": resnet_model,
        "categories": resnet_categories,
    }
}


def list_models() -> list[ModelInfo]:
    return [model["info"] for model in _loaded_models.values()]


def get_active_model() -> ModelInfo:
    for model_data in _loaded_models.values():
        if model_data["info"].active:
            return model_data["info"]
    raise ValueError("No active model found")

def get_active_model_instance():
    for model_data in _loaded_models.values():
        if model_data["info"].active:
            return model_data["instance"]
    raise ValueError("No active model found")

def get_active_model_categories():
    for model_data in _loaded_models.values():
        if model_data["info"].active:
            return model_data["categories"]
    raise ValueError("No active model found")


def select_model(model_id: str) -> ModelInfo:
    if model_id not in _loaded_models:
        raise ValueError(f"Model with id '{model_id}' not found")

    for current_id, model_data in _loaded_models.items():
        model_data["info"].active = (current_id == model_id)

    return _loaded_models[model_id]["info"]