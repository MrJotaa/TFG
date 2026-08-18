import numpy as np
import pytest
import torch
from PIL import Image

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


def test_ultralytics_model_config_requires_boolean_end_to_end_mode():
    with pytest.raises(TypeError, match="end_to_end must be a boolean"):
        UltralyticsDetectionModelConfig(
            id="detector-a",
            name="Detector A",
            revision="detector-a-revision",
            weights="detector-a.pt",
            end_to_end=1,
        )


@pytest.mark.parametrize("image_size", [0, -1])
def test_ultralytics_model_config_requires_positive_image_size(image_size):
    with pytest.raises(ValueError, match="image_size must be greater than zero"):
        UltralyticsDetectionModelConfig(
            id="detector-a",
            name="Detector A",
            revision="detector-a-revision",
            weights="detector-a.pt",
            image_size=image_size,
        )


@pytest.mark.parametrize("image_size", [640.0, True])
def test_ultralytics_model_config_requires_integer_image_size(image_size):
    with pytest.raises(TypeError, match="image_size must be an integer"):
        UltralyticsDetectionModelConfig(
            id="detector-a",
            name="Detector A",
            revision="detector-a-revision",
            weights="detector-a.pt",
            image_size=image_size,
        )


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
        model.get_explainability_target_layers()


def test_ultralytics_model_resolves_configured_explainability_layers():
    class _TorchModel:
        model = ["early", "target", "head"]

    class _ProviderModel:
        model = _TorchModel()

    config = UltralyticsDetectionModelConfig(
        id="detector-a",
        name="Detector A",
        revision="detector-a-revision",
        weights="detector-a.pt",
        explainability_target_layer_indices=(-2,),
    )
    model = UltralyticsDetectionModel(config)
    model.model = _ProviderModel()

    assert model.get_explainability_target_layers() == ["target"]


@pytest.mark.parametrize(
    "layer_indices",
    [(-2, True), (-2, 1.5)],
)
def test_ultralytics_model_config_requires_integer_explainability_layers(layer_indices):
    with pytest.raises(TypeError, match="contain only integers"):
        UltralyticsDetectionModelConfig(
            id="detector-a",
            name="Detector A",
            revision="detector-a-revision",
            weights="detector-a.pt",
            explainability_target_layer_indices=layer_indices,
        )


def test_ultralytics_model_config_requires_immutable_explainability_layers():
    with pytest.raises(TypeError, match="must be a tuple"):
        UltralyticsDetectionModelConfig(
            id="detector-a",
            name="Detector A",
            revision="detector-a-revision",
            weights="detector-a.pt",
            explainability_target_layer_indices=[16, 19, 22],
        )


def test_ultralytics_model_config_rejects_duplicate_explainability_layers():
    with pytest.raises(ValueError, match="must not contain duplicates"):
        UltralyticsDetectionModelConfig(
            id="detector-a",
            name="Detector A",
            revision="detector-a-revision",
            weights="detector-a.pt",
            explainability_target_layer_indices=(16, 16),
        )


def test_ultralytics_explainability_uses_rect_letterbox_and_source_projection():
    class _TorchModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.stride = torch.tensor([8.0, 16.0, 32.0])
            self.model = torch.nn.ModuleList([torch.nn.Identity() for _ in range(23)])

    class _ProviderModel:
        def __init__(self):
            self.model = _TorchModel()

    model = UltralyticsDetectionModel(
        UltralyticsDetectionModelConfig(
            id="detector-a",
            name="Detector A",
            revision="detector-a-revision",
            weights="detector-a.pt",
            image_size=640,
            explainability_target_layer_indices=(16, 19, 22),
        )
    )
    model.model = _ProviderModel()
    source_image = Image.new("RGB", (23, 17), color=(20, 30, 40))

    prepared = model.prepare_detection_explainability_input(source_image)

    assert prepared.shape == (1, 3, 480, 640)
    assert model.get_explainability_target_layers() == [
        model.model.model.model[16],
        model.model.model.model[19],
        model.model.model.model[22],
    ]

    cam = np.zeros((480, 640), dtype=np.float32)
    cam[3:476, :] = 1.0
    projected = model.project_detection_explainability_map(cam, source_image)

    assert projected.shape == (17, 23)
    assert float(projected.min()) == pytest.approx(1.0, abs=1 / 255)
    assert float(projected.max()) == pytest.approx(1.0, abs=1 / 255)
