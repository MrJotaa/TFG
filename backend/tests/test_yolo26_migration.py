import os
import re
from importlib.metadata import version
from pathlib import Path

import pytest
import torch
from PIL import Image

from ml.attacks.detection_confidence import DetectionConfidenceAttack
from ml.attacks.yolo_confidence import YOLOConfidenceAttack
from ml.models.definitions import build_model_registry
from ml.models.detection import DetectionResult
from ml.models.detection_inference import DetectionEvaluationInferenceConfiguration
from ml.models.ultralytics.model import UltralyticsDetectionModel
from services.evaluation_history_service import DETECTION_PIPELINE_VERSION
from utils.detection_evaluation_artifacts import (
    DETECTION_EIGENCAM_PIPELINE_VERSION,
)

ULTRALYTICS_VERSION = "8.4.121"
YOLO26_MODEL_ID = "yolo26n"
YOLO26_MODEL_NAME = "YOLO26n · preentrenado en COCO"
YOLO26_WEIGHTS = "yolo26n.pt"
YOLO26_REVISION = f"ultralytics-yolo26n-coco-{ULTRALYTICS_VERSION}"
YOLO26_EXPLAINABILITY_LAYERS = (16, 19, 22)
LEGACY_YOLO_PATTERN = re.compile(r"yolo(?:v)?8", re.IGNORECASE)
REAL_SMOKE_ENABLED = os.getenv("RUN_YOLO26_REAL_SMOKE") == "1"


def _registered_yolo26_model() -> UltralyticsDetectionModel:
    model = build_model_registry().get_model(YOLO26_MODEL_ID)
    assert isinstance(model, UltralyticsDetectionModel)
    return model


def test_yolo26_remains_registered_as_an_ultralytics_detector():
    registry = build_model_registry()
    detection_models = [
        model for model in registry.list_models() if model.task_type == "detection"
    ]

    yolo = next(model for model in detection_models if model.id == YOLO26_MODEL_ID)

    assert isinstance(yolo, UltralyticsDetectionModel)


def test_yolo26_runtime_configuration_matches_reproducible_contract():
    model = _registered_yolo26_model()

    assert model.id == YOLO26_MODEL_ID
    assert model.name == YOLO26_MODEL_NAME
    assert model.revision == YOLO26_REVISION
    assert model.config.weights == YOLO26_WEIGHTS
    assert model.config.image_size == 640
    assert model.config.end_to_end is True
    assert (
        model.config.explainability_target_layer_indices == YOLO26_EXPLAINABILITY_LAYERS
    )
    assert version("ultralytics") == ULTRALYTICS_VERSION

    requirements_path = Path(__file__).resolve().parents[1] / "requirements.txt"
    requirements = requirements_path.read_text(encoding="utf-8").splitlines()
    assert f"ultralytics=={ULTRALYTICS_VERSION}" in requirements


def test_yolo26_scientific_pipeline_revisions_remain_explicit():
    assert "yolo26" in DETECTION_PIPELINE_VERSION
    assert "end2end" in DETECTION_PIPELINE_VERSION
    assert DetectionConfidenceAttack.revision == "detection-confidence-linf-v2"
    assert YOLOConfidenceAttack is DetectionConfidenceAttack
    assert DetectionConfidenceAttack.id == "yolo_confidence"
    assert "model-contract" in DETECTION_EIGENCAM_PIPELINE_VERSION
    assert "yolo26" not in DETECTION_EIGENCAM_PIPELINE_VERSION


def test_production_backend_contains_no_legacy_yolov8_references():
    backend_root = Path(__file__).resolve().parents[1]
    source_files = sorted((backend_root / "app").rglob("*.py"))
    checked_files = [backend_root / "requirements.txt", *source_files]
    violations = []

    for path in checked_files:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if LEGACY_YOLO_PATTERN.search(line):
                violations.append(
                    f"{path.relative_to(backend_root)}:{line_number}: {line.strip()}"
                )

    assert violations == [], f"Legacy YOLOv8 references remain: {violations}"


@pytest.mark.real_model
@pytest.mark.skipif(
    not REAL_SMOKE_ENABLED,
    reason=(
        "Set RUN_YOLO26_REAL_SMOKE=1 to execute the real YOLO26 checkpoint smoke test"
    ),
)
def test_real_yolo26_checkpoint_supports_application_detection_contracts():
    model = _registered_yolo26_model()
    source_image = Image.new("RGB", (96, 64), color=(127, 127, 127))

    try:
        model.load()
        torch_model = model.get_torch_model()
        detection_head = torch_model.model[-1]

        assert model.is_loaded() is True
        assert detection_head.end2end is True
        assert tuple(detection_head.f) == YOLO26_EXPLAINABILITY_LAYERS

        catalog = model.get_detection_class_catalog()
        assert catalog.model_id == YOLO26_MODEL_ID
        assert len(catalog.classes) == 80

        interactive_result = model.detect(source_image)
        assert isinstance(interactive_result, DetectionResult)
        assert interactive_result.source_width == source_image.width
        assert interactive_result.source_height == source_image.height

        evaluation_result = model.detect_for_evaluation(
            source_image,
            DetectionEvaluationInferenceConfiguration(),
        )
        assert isinstance(evaluation_result, DetectionResult)
        assert evaluation_result.source_width == source_image.width
        assert evaluation_result.source_height == source_image.height

        target_layers = model.get_explainability_target_layers()
        assert tuple(target_layers) == tuple(
            torch_model.model[index] for index in YOLO26_EXPLAINABILITY_LAYERS
        )

        prepared_input = model.prepare_detection_attack_input(source_image)
        prepared_input.requires_grad_(True)
        confidence_objective = model.detection_confidence_objective(prepared_input)
        (input_gradient,) = torch.autograd.grad(
            confidence_objective,
            prepared_input,
        )

        assert confidence_objective.ndim == 0
        assert torch.isfinite(confidence_objective)
        assert input_gradient.shape == prepared_input.shape
        assert torch.isfinite(input_gradient).all()
        assert all(parameter.grad is None for parameter in torch_model.parameters())
    finally:
        model.unload()
