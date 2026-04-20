from schemas.models import ModelInfo


_models = [
    ModelInfo(id="resnet18", name="ResNet18 pretrained", active=True),
    ModelInfo(id="mobilenet_v2", name="MobileNetV2 pretrained", active=False),
]


def list_models() -> list[ModelInfo]:
    return _models


def get_active_model() -> ModelInfo:
    for model in _models:
        if model.active:
            return model
    raise ValueError("No active model found")


def select_model(model_id: str) -> ModelInfo:
    selected_model = None

    for model in _models:
        if model.id == model_id:
            selected_model = model
            model.active = True
        else:
            model.active = False

    if selected_model is None:
        raise ValueError(f"Model '{model_id}' not found")

    return selected_model