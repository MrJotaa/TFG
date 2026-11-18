from collections.abc import Iterable

from ml.models.base import BaseModel


class ModelRegistryError(ValueError):
    """Base error for invalid model registry configuration."""


class ModelNotRegisteredError(LookupError):
    """Raised when a requested model ID is not registered."""


class ModelRegistry:
    """Owns registered model instances and the single active model selection."""

    def __init__(
        self,
        *,
        models: Iterable[BaseModel],
        default_model_id: str,
    ) -> None:
        registered_models: dict[str, BaseModel] = {}

        for model in models:
            model_id = getattr(model, "id", None)
            if not isinstance(model_id, str) or not model_id.strip():
                raise ModelRegistryError("Registered models must define a non-empty id")
            if model_id in registered_models:
                raise ModelRegistryError(f"Duplicate model id: {model_id!r}")

            registered_models[model_id] = model

        if not registered_models:
            raise ModelRegistryError("At least one model must be registered")
        if default_model_id not in registered_models:
            raise ModelRegistryError(
                f"Default model {default_model_id!r} is not registered"
            )

        self._models = registered_models
        self._active_model_id = default_model_id

    @property
    def active_model_id(self) -> str:
        return self._active_model_id

    def list_models(self) -> tuple[BaseModel, ...]:
        return tuple(self._models.values())

    def get_model(self, model_id: str) -> BaseModel:
        try:
            return self._models[model_id]
        except KeyError as error:
            raise ModelNotRegisteredError(model_id) from error

    def get_active_model(self) -> BaseModel:
        return self._models[self._active_model_id]

    def select_model(self, model_id: str) -> BaseModel:
        selected_model = self.get_model(model_id)

        if model_id == self._active_model_id:
            selected_model.load()
            return selected_model

        previous_model = self.get_active_model()
        previous_model.unload()
        selected_model.load()
        self._active_model_id = model_id

        return selected_model
