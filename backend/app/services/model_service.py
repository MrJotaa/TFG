from ml.models.resnet18.model import ResNet18Model
from schemas.models import ModelInfo


resnet18 = ResNet18Model()
resnet18.load()

_models = {
    resnet18.id: {
        "instance": resnet18,
        "active": True,
    }
}


def _to_model_info(model_id: str, model_data: dict) -> ModelInfo:
    instance = model_data["instance"]

    return ModelInfo(
        id=instance.id,
        name=instance.name,
        active=model_data["active"],
        loaded=instance.is_loaded(),
    )


def list_models() -> list[ModelInfo]:
    return [
        _to_model_info(model_id, model_data)
        for model_id, model_data in _models.items()
    ]


def get_active_model() -> ModelInfo:
    for model_id, model_data in _models.items():
        if model_data["active"]:
            return _to_model_info(model_id, model_data)

    raise ValueError("No active model found")


def get_active_model_instance():
    for model_data in _models.values():
        if model_data["active"]:
            instance = model_data["instance"]

            if not instance.is_loaded():
                raise ValueError("Active model is not loaded")

            return instance

    raise ValueError("No active model instance found")


def select_model(model_id: str) -> ModelInfo:
    if model_id not in _models:
        raise ValueError(f"Model '{model_id}' not found")

    # Si ya es el modelo activo, aseguramos que esté cargado y devolvemos estado.
    if _models[model_id]["active"]:
        instance = _models[model_id]["instance"]
        instance.load()
        return _to_model_info(model_id, _models[model_id])

    # Descarga cualquier modelo actualmente activo y lo desactiva.
    for current_id, model_data in _models.items():
        if model_data["active"]:
            model_data["instance"].unload()
            model_data["active"] = False

    # Carga y activa el modelo seleccionado.
    selected_model = _models[model_id]
    selected_model["instance"].load()
    selected_model["active"] = True

    return _to_model_info(model_id, selected_model)