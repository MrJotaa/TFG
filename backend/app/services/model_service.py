from core.errors import NotFoundError
from ml.models.resnet18.model import ResNet18Model
from ml.models.yolo.model import YOLODetectionModel
from schemas.models import ModelInfo

resnet18 = ResNet18Model()
yolov8n = YOLODetectionModel()

_models = {
    resnet18.id: {
        "instance": resnet18,
        "active": True,
    },
    yolov8n.id: {
        "instance": yolov8n,
        "active": False,
    },
}


def _to_model_info(model_data: dict) -> ModelInfo:
    instance = model_data["instance"]

    return ModelInfo(
        id=instance.id,
        name=instance.name,
        active=model_data["active"],
        loaded=instance.is_loaded(),
        task_type=instance.task_type,
    )


def list_models() -> list[ModelInfo]:
    return [_to_model_info(model_data) for model_data in _models.values()]


def get_active_model() -> ModelInfo:
    for model_data in _models.values():
        if model_data["active"]:
            return _to_model_info(model_data)

    raise NotFoundError("No active model found")


def get_active_model_instance():
    for model_data in _models.values():
        if model_data["active"]:
            instance = model_data["instance"]

            if not instance.is_loaded():
                instance.load()

            return instance

    raise NotFoundError("No active model found")


def select_model(model_id: str) -> ModelInfo:
    if model_id not in _models:
        raise NotFoundError("Model not found")

    selected_model = _models[model_id]

    if selected_model["active"]:
        selected_model["instance"].load()
        return _to_model_info(selected_model)

    for model_data in _models.values():
        if model_data["active"]:
            model_data["instance"].unload()
            model_data["active"] = False

    selected_model["instance"].load()
    selected_model["active"] = True

    return _to_model_info(selected_model)
