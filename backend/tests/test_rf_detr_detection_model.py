from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

import ml.models.rf_detr.model as rf_detr_model_module
from ml.attacks.detection_confidence import DetectionConfidenceAttack
from ml.detection_categories import DetectionClass
from ml.models.definitions import build_model_registry
from ml.models.detection import DetectionResult
from ml.models.detection_inference import DetectionEvaluationInferenceConfiguration
from ml.models.rf_detr.adapter import (
    RfDetrDetectionAdapter,
    RfDetrDetectionAdapterError,
)
from ml.models.rf_detr.catalog import (
    RfDetrClassCatalogError,
    build_rf_detr_class_catalog,
    normalize_rf_detr_id2label,
)
from ml.models.rf_detr.config import RfDetrDetectionModelConfig
from ml.models.rf_detr.model import RfDetrDetectionModel

TRANSFORMERS_VERSION = "5.15.1"
RF_DETR_MODEL_ID = "rf_detr_nano"
RF_DETR_MODEL_NAME = "RF-DETR Nano pretrained"
RF_DETR_CHECKPOINT = "Roboflow/rf-detr-nano"
RF_DETR_CHECKPOINT_REVISION = "9f201577d9415f0378084748d27e68586e6b3600"
RF_DETR_REVISION = "transformers-5.15.1-rf-detr-nano-coco-9f201577"


def _build_model() -> RfDetrDetectionModel:
    return RfDetrDetectionModel(
        RfDetrDetectionModelConfig(
            id="detector-a",
            name="Detector A",
            revision="detector-a-revision",
            checkpoint="provider/detector-a",
            checkpoint_revision="checkpoint-revision-a",
        )
    )


class _FakeProcessor:
    def __init__(self, native_result=None):
        self.do_resize = True
        self.do_rescale = True
        self.do_normalize = True
        self.do_pad = True
        self.size = {"height": 384, "width": 384}
        self.image_mean = [0.485, 0.456, 0.406]
        self.image_std = [0.229, 0.224, 0.225]
        self.native_result = native_result or {
            "scores": torch.tensor([], dtype=torch.float32),
            "labels": torch.tensor([], dtype=torch.int64),
            "boxes": torch.empty((0, 4), dtype=torch.float32),
        }
        self.preprocess_calls = []
        self.postprocess_calls = []

    def __call__(self, *, images, return_tensors):
        self.preprocess_calls.append((images, return_tensors))
        return {"pixel_values": torch.ones((1, 3, 384, 384))}

    def post_process_object_detection(
        self,
        outputs,
        *,
        threshold,
        target_sizes,
        top_k,
    ):
        self.postprocess_calls.append(
            {
                "outputs": outputs,
                "threshold": threshold,
                "target_sizes": target_sizes.detach().cpu().tolist(),
                "top_k": top_k,
            }
        )
        return [self.native_result]


class _FakeProviderModel(torch.nn.Module):
    def __init__(self, id2label=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.config = SimpleNamespace(
            id2label=id2label
            or {
                0: "N/A",
                1: "person",
                3: "car",
                12: "N/A",
            }
        )
        self.forward_calls = []

    def forward(self, **inputs):
        self.forward_calls.append(inputs)
        pixel_values = inputs["pixel_values"]
        batch_size = pixel_values.shape[0]
        signal = pixel_values.mean(dim=(1, 2, 3)) * self.weight
        logits = signal[:, None, None].expand(batch_size, 3, 13)
        return SimpleNamespace(logits=logits)


def _install_fake_provider(monkeypatch, *, processor=None, provider_model=None):
    processor = processor or _FakeProcessor()
    provider_model = provider_model or _FakeProviderModel()
    processor_loads = []
    model_loads = []

    def load_processor(checkpoint, *, revision):
        processor_loads.append((checkpoint, revision))
        return processor

    def load_model(checkpoint, *, revision):
        model_loads.append((checkpoint, revision))
        return provider_model

    monkeypatch.setattr(
        rf_detr_model_module,
        "AutoImageProcessor",
        SimpleNamespace(from_pretrained=load_processor),
    )
    monkeypatch.setattr(
        rf_detr_model_module,
        "AutoModelForObjectDetection",
        SimpleNamespace(from_pretrained=load_model),
    )
    return processor, provider_model, processor_loads, model_loads


def test_rf_detr_config_normalizes_required_text_and_threshold():
    config = RfDetrDetectionModelConfig(
        id="  detector-a  ",
        name="  Detector A  ",
        revision="  revision-a  ",
        checkpoint="  provider/detector-a  ",
        checkpoint_revision="  commit-a  ",
        confidence_threshold=0.3,
    )

    assert config.id == "detector-a"
    assert config.name == "Detector A"
    assert config.revision == "revision-a"
    assert config.checkpoint == "provider/detector-a"
    assert config.checkpoint_revision == "commit-a"
    assert config.confidence_threshold == 0.3


@pytest.mark.parametrize(
    "field_name",
    ["id", "name", "revision", "checkpoint", "checkpoint_revision"],
)
def test_rf_detr_config_rejects_empty_required_text(field_name):
    values = {
        "id": "detector-a",
        "name": "Detector A",
        "revision": "revision-a",
        "checkpoint": "provider/detector-a",
        "checkpoint_revision": "commit-a",
    }
    values[field_name] = "   "

    with pytest.raises(ValueError, match=f"{field_name} must not be empty"):
        RfDetrDetectionModelConfig(**values)


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_rf_detr_config_rejects_out_of_range_threshold(threshold):
    with pytest.raises(ValueError, match="between 0 and 1"):
        RfDetrDetectionModelConfig(
            id="detector-a",
            name="Detector A",
            revision="revision-a",
            checkpoint="provider/detector-a",
            checkpoint_revision="commit-a",
            confidence_threshold=threshold,
        )


def test_rf_detr_catalog_preserves_sparse_coco_ids_and_filters_placeholders():
    catalog = build_rf_detr_class_catalog(
        model_id="rf-detr",
        model_name="RF-DETR",
        id2label={"0": "N/A", "1": "person", 3: "car", 12: "N/A"},
    )

    assert catalog.classes == (
        DetectionClass(class_id=1, name="person"),
        DetectionClass(class_id=3, name="car"),
    )
    assert set(catalog.class_by_id) == {1, 3}


def test_rf_detr_catalog_rejects_ids_that_collide_after_normalization():
    with pytest.raises(RfDetrClassCatalogError, match="duplicate"):
        normalize_rf_detr_id2label({1: "person", "1": "car"})


def test_rf_detr_adapter_preserves_ids_precision_and_clips_source_geometry():
    result = RfDetrDetectionAdapter().adapt(
        {
            "scores": torch.tensor([0.91234567], dtype=torch.float64),
            "labels": torch.tensor([3]),
            "boxes": torch.tensor(
                [[-0.25, 10.125, 640.75, 319.875]],
                dtype=torch.float64,
            ),
        },
        source_width=640,
        source_height=320,
        id2label={1: "person", 3: "car"},
    )

    assert len(result.detections) == 1
    detection = result.detections[0]
    assert detection.class_id == 3
    assert detection.class_name == "car"
    assert detection.confidence == 0.91234567
    assert (
        detection.box.x_min,
        detection.box.y_min,
        detection.box.x_max,
        detection.box.y_max,
    ) == (0.0, 10.125, 640.0, 319.875)


def test_rf_detr_adapter_filters_placeholder_and_fully_outside_boxes():
    result = RfDetrDetectionAdapter().adapt(
        {
            "scores": torch.tensor([0.9, 0.8, 0.7]),
            "labels": torch.tensor([0, 1, 3]),
            "boxes": torch.tensor(
                [
                    [1.0, 2.0, 20.0, 30.0],
                    [700.0, 10.0, 720.0, 30.0],
                    [10.0, 20.0, 50.0, 60.0],
                ]
            ),
        },
        source_width=640,
        source_height=320,
        id2label={0: "N/A", 1: "person", 3: "car"},
    )

    assert [(item.class_id, item.class_name) for item in result.detections] == [
        (3, "car")
    ]


def test_rf_detr_adapter_rejects_unknown_prediction_class():
    with pytest.raises(RfDetrDetectionAdapterError, match="unknown class id 7"):
        RfDetrDetectionAdapter().adapt(
            {
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([7]),
                "boxes": torch.tensor([[1.0, 2.0, 20.0, 30.0]]),
            },
            source_width=640,
            source_height=320,
            id2label={1: "person"},
        )


@pytest.mark.parametrize(
    ("native_result", "message"),
    [
        (
            {
                "scores": torch.tensor([1], dtype=torch.int64),
                "labels": torch.tensor([1], dtype=torch.int64),
                "boxes": torch.tensor([[1.0, 2.0, 20.0, 30.0]]),
            },
            "scores must use a floating-point dtype",
        ),
        (
            {
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([1.0]),
                "boxes": torch.tensor([[1.0, 2.0, 20.0, 30.0]]),
            },
            "labels must use an integer dtype",
        ),
        (
            {
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([1], dtype=torch.int64),
                "boxes": torch.tensor([[1, 2, 20, 30]], dtype=torch.int64),
            },
            "boxes must use a floating-point dtype",
        ),
    ],
)
def test_rf_detr_adapter_rejects_provider_dtype_contract_violations(
    native_result,
    message,
):
    with pytest.raises(RfDetrDetectionAdapterError, match=message):
        RfDetrDetectionAdapter().adapt(
            native_result,
            source_width=640,
            source_height=320,
            id2label={1: "person"},
        )


def test_rf_detr_model_loads_pinned_provider_resources_once(monkeypatch):
    processor, provider_model, processor_loads, model_loads = _install_fake_provider(
        monkeypatch
    )
    model = _build_model()

    model.load()
    model.load()

    assert processor_loads == [("provider/detector-a", "checkpoint-revision-a")]
    assert model_loads == [("provider/detector-a", "checkpoint-revision-a")]
    assert model.image_processor is processor
    assert model.model is provider_model
    assert provider_model.training is False
    assert model.is_loaded() is True

    catalog = model.get_detection_class_catalog()
    assert catalog.classes == (
        DetectionClass(class_id=1, name="person"),
        DetectionClass(class_id=3, name="car"),
    )
    assert provider_model.forward_calls == []


def test_rf_detr_model_unload_releases_provider_state(monkeypatch):
    _install_fake_provider(monkeypatch)
    model = _build_model()
    model.load()

    model.unload()

    assert model.is_loaded() is False
    assert model.model is None
    assert model.image_processor is None


def test_rf_detr_detect_uses_official_processor_and_source_geometry(monkeypatch):
    native_result = {
        "scores": torch.tensor([0.91]),
        "labels": torch.tensor([3]),
        "boxes": torch.tensor([[10.25, 20.5, 200.75, 300.25]]),
    }
    processor = _FakeProcessor(native_result)
    provider_model = _FakeProviderModel()
    _install_fake_provider(
        monkeypatch,
        processor=processor,
        provider_model=provider_model,
    )
    model = _build_model()
    source_image = Image.new("RGBA", (640, 320), color=(20, 30, 40, 255))

    result = model.detect(source_image)

    assert isinstance(result, DetectionResult)
    assert result.source_width == 640
    assert result.source_height == 320
    assert [(item.class_id, item.class_name) for item in result.detections] == [
        (3, "car")
    ]
    assert len(provider_model.forward_calls) == 1
    assert set(provider_model.forward_calls[0]) == {"pixel_values"}
    assert provider_model.forward_calls[0]["pixel_values"].device == model.device

    processed_image, return_tensors = processor.preprocess_calls[0]
    assert processed_image.mode == "RGB"
    assert processed_image.size == source_image.size
    assert return_tensors == "pt"
    assert len(processor.postprocess_calls) == 1
    postprocess_call = processor.postprocess_calls[0]
    assert postprocess_call["outputs"] is not None
    assert postprocess_call["threshold"] == 0.25
    assert postprocess_call["target_sizes"] == [[320, 640]]
    assert postprocess_call["top_k"] == 300


def test_rf_detr_evaluation_uses_explicit_scientific_configuration(monkeypatch):
    processor = _FakeProcessor()
    provider_model = _FakeProviderModel()
    _install_fake_provider(
        monkeypatch,
        processor=processor,
        provider_model=provider_model,
    )
    model = _build_model()
    configuration = DetectionEvaluationInferenceConfiguration(score_floor=0.005)

    result = model.detect_for_evaluation(
        Image.new("RGB", (640, 320)),
        configuration,
    )

    assert result == DetectionResult(source_width=640, source_height=320)
    assert processor.postprocess_calls[0]["threshold"] == 0.005
    assert processor.postprocess_calls[0]["top_k"] == 300
    assert processor.postprocess_calls[0]["target_sizes"] == [[320, 640]]


def test_rf_detr_visualization_reuses_single_inference(monkeypatch):
    processor = _FakeProcessor(
        {
            "scores": torch.tensor([0.91]),
            "labels": torch.tensor([3]),
            "boxes": torch.tensor([[10.0, 20.0, 100.0, 120.0]]),
        }
    )
    provider_model = _FakeProviderModel()
    _install_fake_provider(
        monkeypatch,
        processor=processor,
        provider_model=provider_model,
    )
    model = _build_model()
    source_image = Image.new("RGB", (200, 160), color=(10, 20, 30))

    result, annotated = model.detect_with_visualization(source_image)

    assert len(provider_model.forward_calls) == 1
    assert len(result.detections) == 1
    assert annotated.size == source_image.size
    assert annotated.mode == "RGB"
    assert annotated.tobytes() != source_image.tobytes()


def test_rf_detr_nano_registry_configuration_is_reproducible():
    registry = build_model_registry()
    model = registry.get_model(RF_DETR_MODEL_ID)

    assert isinstance(model, RfDetrDetectionModel)
    assert model.id == RF_DETR_MODEL_ID
    assert model.name == RF_DETR_MODEL_NAME
    assert model.revision == RF_DETR_REVISION
    assert model.config.checkpoint == RF_DETR_CHECKPOINT
    assert model.config.checkpoint_revision == RF_DETR_CHECKPOINT_REVISION
    assert model.config.confidence_threshold == 0.25
    assert version("transformers") == TRANSFORMERS_VERSION

    requirements_path = Path(__file__).resolve().parents[1] / "requirements.txt"
    requirements = requirements_path.read_text(encoding="utf-8").splitlines()
    assert f"transformers=={TRANSFORMERS_VERSION}" in requirements


def test_registry_exposes_both_detection_architectures():
    detection_models = [
        model
        for model in build_model_registry().list_models()
        if model.task_type == "detection"
    ]

    assert [model.id for model in detection_models] == ["yolo26n", RF_DETR_MODEL_ID]


def test_rf_detr_attack_input_preserves_source_geometry_and_model_device(monkeypatch):
    _install_fake_provider(monkeypatch)
    model = _build_model()
    model.load()
    image = Image.new("RGB", (640, 320), color=(128, 64, 32))

    prepared = model.prepare_detection_attack_input(image)

    assert prepared.shape == (1, 3, 320, 640)
    assert prepared.device == model.device
    assert prepared.dtype == model.get_torch_model().weight.dtype
    assert prepared.min().item() >= 0.0
    assert prepared.max().item() <= 1.0


def test_rf_detr_differentiable_attack_preprocessing_matches_checkpoint_contract(
    monkeypatch,
):
    _install_fake_provider(monkeypatch)
    model = _build_model()
    model.load()
    image = Image.new("RGB", (32, 20), color=(128, 128, 128))
    prepared = model.prepare_detection_attack_input(image)

    pixel_values, pixel_mask = model._differentiable_attack_model_input(prepared)

    assert pixel_values.shape == (1, 3, 384, 384)
    assert pixel_mask.shape == (1, 384, 384)
    assert torch.all(pixel_mask == 1)

    source_value = 128 / 255
    expected_channels = torch.tensor(
        [
            (source_value - 0.485) / 0.229,
            (source_value - 0.456) / 0.224,
            (source_value - 0.406) / 0.225,
        ],
        device=pixel_values.device,
        dtype=pixel_values.dtype,
    )
    observed_channels = pixel_values[0, :, 100, 100]
    assert torch.allclose(observed_channels, expected_channels, atol=1e-6)


def test_rf_detr_confidence_objective_matches_query_semantics_and_ignores_na():
    probabilities = torch.tensor(
        [
            [
                [0.99, 0.20, 0.95, 0.60],
                [0.98, 0.40, 0.96, 0.10],
            ]
        ],
        dtype=torch.float32,
    )
    logits = torch.logit(probabilities).requires_grad_(True)

    objective = RfDetrDetectionModel._query_confidence_objective(
        logits,
        valid_class_ids=(1, 3),
    )

    assert objective.item() == pytest.approx(0.5)
    assert objective.requires_grad is True


def test_rf_detr_confidence_objective_propagates_gradient_to_source_image(
    monkeypatch,
):
    processor, provider_model, *_ = _install_fake_provider(monkeypatch)
    model = _build_model()
    model.load()
    image = Image.new("RGB", (64, 40), color=(192, 192, 192))
    prepared = model.prepare_detection_attack_input(image).requires_grad_(True)

    objective = model.detection_confidence_objective(prepared)
    (gradient,) = torch.autograd.grad(objective, prepared)

    assert objective.ndim == 0
    assert objective.requires_grad is True
    assert gradient.shape == prepared.shape
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0
    attack_forward = provider_model.forward_calls[-1]
    assert attack_forward["pixel_values"].shape == (1, 3, 384, 384)
    assert attack_forward["pixel_mask"].shape == (1, 384, 384)
    assert processor.size == {"height": 384, "width": 384}
    assert all(parameter.grad is None for parameter in provider_model.parameters())


def test_detection_confidence_attack_runs_against_rf_detr_contract(monkeypatch):
    _install_fake_provider(monkeypatch)
    model = _build_model()
    model.load()
    attack = DetectionConfidenceAttack()
    image = Image.new("RGB", (32, 24), color=(192, 192, 192))

    original_prepared = model.prepare_detection_attack_input(image)
    original_objective = model.detection_confidence_objective(original_prepared).item()

    result = attack.apply(
        model_wrapper=model,
        image=image,
        parameters=SimpleNamespace(epsilon=0.03, alpha=0.01, steps=2),
    )

    adversarial_prepared = model.prepare_detection_attack_input(
        result["adversarial_image"]
    )
    adversarial_objective = model.detection_confidence_objective(
        adversarial_prepared
    ).item()

    assert adversarial_objective < original_objective
    assert result["perturbation_linf"] <= 0.03
    assert result["adversarial_image"].size == image.size
    assert result["perturbation_image"].size == image.size


class _FakeRfDetrProjector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm = torch.nn.Identity()


class _FakeRfDetrExplainabilityProvider(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.config = SimpleNamespace(id2label={1: "person", 3: "car"})
        self.model = SimpleNamespace(
            backbone=SimpleNamespace(projector=_FakeRfDetrProjector())
        )

    def forward(self, pixel_values=None, pixel_mask=None, **kwargs):
        if pixel_values is None:
            raise ValueError("pixel_values are required")
        return SimpleNamespace(logits=pixel_values.mean((2, 3))[:, None, :])


def test_rf_detr_explainability_uses_spatial_projector_and_checkpoint_geometry():
    model = _build_model()
    provider = _FakeRfDetrExplainabilityProvider()
    model.model = provider
    model.image_processor = _FakeProcessor()
    model._id2label = {1: "person", 3: "car"}
    model._class_catalog = build_rf_detr_class_catalog(
        model_id=model.id,
        model_name=model.name,
        id2label=model._id2label,
    )
    source_image = Image.new("RGB", (23, 17), color=(20, 30, 40))

    prepared = model.prepare_detection_explainability_input(source_image)

    assert model.supports_detection_explainability() is True
    assert model.get_detection_explainability_model() is provider
    assert model.get_explainability_target_layers() == [
        provider.model.backbone.projector.layer_norm
    ]
    assert prepared.shape == (1, 3, 384, 384)
    assert torch.isfinite(prepared).all()

    cam = torch.linspace(0.0, 1.0, steps=384 * 384).reshape(384, 384).numpy()
    projected = model.project_detection_explainability_map(cam, source_image)

    assert projected.shape == (17, 23)
    assert projected.dtype.name == "float32"
    assert float(projected.min()) >= 0.0
    assert float(projected.max()) <= 1.0


def test_rf_detr_explainability_rejects_unexpected_transformers_structure():
    model = _build_model()
    provider = _FakeProviderModel()
    model.model = provider
    model.image_processor = _FakeProcessor()
    model._id2label = {1: "person", 3: "car"}
    model._class_catalog = build_rf_detr_class_catalog(
        model_id=model.id,
        model_name=model.name,
        id2label=model._id2label,
    )

    with pytest.raises(RuntimeError, match="pinned Transformers architecture"):
        model.get_explainability_target_layers()
