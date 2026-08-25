from core.errors import NotFoundError
from ml.models.base import BaseModel
from ml.models.definitions import build_model_registry
from ml.models.registry import ModelNotRegisteredError
from schemas.models import ModelInfo

_model_registry = build_model_registry()


def _to_model_info(instance: BaseModel) -> ModelInfo:
    return ModelInfo(
        id=instance.id,
        name=instance.name,
        active=instance.id == _model_registry.active_model_id,
        loaded=instance.is_loaded(),
        task_type=instance.task_type,
    )


def list_models() -> list[ModelInfo]:
    return [_to_model_info(instance) for instance in _model_registry.list_models()]


def get_active_model() -> ModelInfo:
    return _to_model_info(_model_registry.get_active_model())


def get_active_model_revision() -> str:
    return _model_registry.get_active_model().revision


def get_active_model_instance() -> BaseModel:
    instance = _model_registry.get_active_model()

    if not instance.is_loaded():
        instance.load()

    return instance


def select_model(model_id: str) -> ModelInfo:
    try:
        selected_model = _model_registry.select_model(model_id)
    except ModelNotRegisteredError as error:
        raise NotFoundError("Modelo no encontrado") from error

    return _to_model_info(selected_model)
