import pytest

import ml.models.ultralytics.model as ultralytics_model_module
from ml.models.ultralytics.config import UltralyticsDetectionModelConfig
from ml.models.ultralytics.model import UltralyticsDetectionModel


def test_ultralytics_model_uses_declarative_identity_and_weights(monkeypatch):
    loaded_weights = []
    provider_model = object()

    def fake_yolo(weights):
        loaded_weights.append(weights)
        return provider_model

    monkeypatch.setattr(ultralytics_model_module, "YOLO", fake_yolo)

    config = UltralyticsDetectionModelConfig(
        id="detector-a",
        name="Detector A",
        revision="detector-a-revision",
        weights="detector-a.pt",
    )
    model = UltralyticsDetectionModel(config)

    assert model.id == "detector-a"
    assert model.name == "Detector A"
    assert model.revision == "detector-a-revision"
    assert model.task_type == "detection"
    assert model.is_loaded() is False

    model.load()
    model.load()

    assert loaded_weights == ["detector-a.pt"]
    assert model.model is provider_model


def test_ultralytics_model_config_normalizes_required_text():
    config = UltralyticsDetectionModelConfig(
        id="  detector-a  ",
        name="  Detector A  ",
        revision="  detector-a-revision  ",
        weights="  detector-a.pt  ",
    )

    assert config.id == "detector-a"
    assert config.name == "Detector A"
    assert config.revision == "detector-a-revision"
    assert config.weights == "detector-a.pt"


@pytest.mark.parametrize("field_name", ["id", "name", "revision", "weights"])
def test_ultralytics_model_config_rejects_empty_required_text(field_name):
    values = {
        "id": "detector-a",
        "name": "Detector A",
        "revision": "detector-a-revision",
        "weights": "detector-a.pt",
    }
    values[field_name] = "   "

    with pytest.raises(ValueError, match=f"{field_name} must not be empty"):
        UltralyticsDetectionModelConfig(**values)


def test_ultralytics_model_requires_configured_explainability_layer():
    config = UltralyticsDetectionModelConfig(
        id="detector-a",
        name="Detector A",
        revision="detector-a-revision",
        weights="detector-a.pt",
    )
    model = UltralyticsDetectionModel(config)
    model.model = object()

    with pytest.raises(NotImplementedError, match="does not define"):
        model.get_explainability_target_layer()


def test_ultralytics_model_resolves_configured_explainability_layer():
    class _TorchModel:
        model = ["early", "target", "head"]

    class _ProviderModel:
        model = _TorchModel()

    config = UltralyticsDetectionModelConfig(
        id="detector-a",
        name="Detector A",
        revision="detector-a-revision",
        weights="detector-a.pt",
        explainability_target_layer_index=-2,
    )
    model = UltralyticsDetectionModel(config)
    model.model = _ProviderModel()

    assert model.get_explainability_target_layer() == "target"
