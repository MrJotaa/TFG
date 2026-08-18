import pytest

from ml.models.base import BaseModel
from ml.models.registry import (
    ModelNotRegisteredError,
    ModelRegistry,
    ModelRegistryError,
)


class _TestModel(BaseModel):
    task_type = "test"

    def __init__(self, model_id: str, *, fail_on_load: bool = False):
        self.id = model_id
        self.name = f"Model {model_id}"
        self._loaded = False
        self.fail_on_load = fail_on_load
        self.load_calls = 0
        self.unload_calls = 0

    def load(self) -> None:
        self.load_calls += 1
        if self.fail_on_load:
            raise RuntimeError("load failed")
        self._loaded = True

    def unload(self) -> None:
        self.unload_calls += 1
        self._loaded = False

    def is_loaded(self) -> bool:
        return self._loaded


def test_model_registry_rejects_duplicate_model_ids():
    with pytest.raises(ModelRegistryError, match="Duplicate model id"):
        ModelRegistry(
            models=(_TestModel("detector"), _TestModel("detector")),
            default_model_id="detector",
        )


def test_model_registry_rejects_unknown_default_model():
    with pytest.raises(ModelRegistryError, match="is not registered"):
        ModelRegistry(
            models=(_TestModel("detector"),),
            default_model_id="missing",
        )


def test_model_registry_tracks_exactly_one_active_model():
    first = _TestModel("first")
    second = _TestModel("second")
    registry = ModelRegistry(
        models=(first, second),
        default_model_id="first",
    )

    assert registry.active_model_id == "first"
    assert registry.get_active_model() is first

    registry.select_model("second")

    assert registry.active_model_id == "second"
    assert registry.get_active_model() is second
    assert first.unload_calls == 1
    assert second.load_calls == 1


def test_model_registry_selecting_active_model_loads_without_unloading_it():
    model = _TestModel("model")
    registry = ModelRegistry(models=(model,), default_model_id="model")

    selected = registry.select_model("model")

    assert selected is model
    assert model.load_calls == 1
    assert model.unload_calls == 0
    assert registry.active_model_id == "model"


def test_model_registry_does_not_switch_active_id_when_new_model_fails_to_load():
    first = _TestModel("first")
    second = _TestModel("second", fail_on_load=True)
    registry = ModelRegistry(
        models=(first, second),
        default_model_id="first",
    )

    with pytest.raises(RuntimeError, match="load failed"):
        registry.select_model("second")

    assert registry.active_model_id == "first"
    assert registry.get_active_model() is first
    assert first.unload_calls == 1
    assert second.load_calls == 1


def test_model_registry_rejects_unknown_model_lookup():
    registry = ModelRegistry(
        models=(_TestModel("detector"),),
        default_model_id="detector",
    )

    with pytest.raises(ModelNotRegisteredError):
        registry.get_model("missing")
