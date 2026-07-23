from types import SimpleNamespace

import pytest
import torch
from PIL import Image

import services.attack_service as attack_service
import services.detection_attack_service as detection_attack_service
import services.detection_evaluation_service as detection_evaluation_service
import services.evaluation_service as evaluation_service
import services.explainability_service as explainability_service
from ml.models.detection import Detection, DetectionBox, DetectionResult
from schemas.attack import AttackRequest, AttackType
from schemas.detection_attack import DetectionAttackRequest
from schemas.detection_evaluation import (
    DetectionEvaluationRequest,
)
from schemas.evaluation import EvaluationRequest
from schemas.explainability import DetectionExplainRequest

INTERNAL_ERROR = "Sensitive internal model failure"
VALID_DATASET_ID = "507f1f77bcf86cd799439011"


def _create_image() -> Image.Image:
    return Image.new(
        "RGB",
        (8, 8),
        color=(128, 128, 128),
    )


async def _fake_get_image_from_dataset(
    dataset_id: str,
    image_id: str,
) -> Image.Image:
    return _create_image()


class FakeClassificationModel:
    id = "fake-classifier"
    name = "Fake Classifier"
    task_type = "classification"

    def prepare_attack_input(self, image: Image.Image):
        return torch.full(
            (1, 3, image.height, image.width),
            0.5,
            dtype=torch.float32,
            device="cpu",
        )

    def preprocess(self, image: Image.Image):
        return "prepared-input"

    def predict(self, image: Image.Image):
        return SimpleNamespace(
            prediction="test-class",
            confidence=0.9,
        )

    def get_class_labels(self):
        return ("test-class",)

    def predict_detailed(
        self,
        image,
        top_k=5,
    ):
        return SimpleNamespace(
            class_index=0,
            class_label="test-class",
            confidence=0.9,
            probabilities=(1.0,),
            top_k=(
                SimpleNamespace(
                    index=0,
                    label="test-class",
                    probability=1.0,
                ),
            ),
            prediction_margin=None,
        )

    def predict_detailed_from_prepared_input(
        self,
        prepared_input,
        top_k=5,
    ):
        assert prepared_input.ndim == 4
        assert prepared_input.shape[0] == 1
        assert prepared_input.shape[1] == 3
        assert prepared_input.is_floating_point()
        assert torch.isfinite(prepared_input).all()
        assert prepared_input.device.type == "cpu"
        return self.predict_detailed(None, top_k=top_k)


class FakeDetectionModel:
    id = "fake-detector"
    name = "Fake Detector"
    task_type = "detection"

    def detect(self, image: Image.Image):
        return DetectionResult(
            source_width=image.width,
            source_height=image.height,
            detections=(
                Detection(
                    class_id=0,
                    class_name="object",
                    confidence=0.9,
                    box=DetectionBox(
                        x_min=0,
                        y_min=0,
                        x_max=image.width,
                        y_max=image.height,
                    ),
                ),
            ),
        )

    def detect_with_visualization(
        self,
        image: Image.Image,
    ):
        return self.detect(image), image


class FailingAttack:
    id = "random_noise"

    def __init__(self):
        self.apply_calls = 0

    def supports_task_type(self, task_type: str) -> bool:
        return task_type in {
            "classification",
            "detection",
        }

    def apply(
        self,
        model_wrapper,
        image,
        parameters,
        original_class_index=None,
    ):
        self.apply_calls += 1
        raise ValueError(INTERNAL_ERROR)


@pytest.mark.asyncio
async def test_classification_attack_preserves_unexpected_error(
    monkeypatch,
):
    monkeypatch.setattr(
        attack_service,
        "get_image_from_dataset",
        _fake_get_image_from_dataset,
    )
    monkeypatch.setattr(
        attack_service,
        "get_active_model_instance",
        lambda: FakeClassificationModel(),
    )
    monkeypatch.setattr(
        attack_service,
        "get_attack",
        lambda attack_type: FailingAttack(),
    )

    request = AttackRequest(
        dataset_id="dataset-1",
        image_id="image-1",
        attack_type=AttackType.RANDOM_NOISE,
    )

    with pytest.raises(
        ValueError,
        match=INTERNAL_ERROR,
    ):
        await attack_service.apply_adversarial_attack(request)


@pytest.mark.asyncio
async def test_classification_evaluation_preserves_unexpected_error(
    monkeypatch,
):
    failing_attack = FailingAttack()

    async def fake_get_dataset(dataset_id: str):
        return SimpleNamespace(
            id=dataset_id,
            name="Test Dataset",
            owner_id="user-1",
            image_count=1,
            images=[
                SimpleNamespace(
                    id="image-1",
                    filename="image.png",
                    label=None,
                )
            ],
        )

    monkeypatch.setattr(
        evaluation_service,
        "get_dataset",
        fake_get_dataset,
    )
    monkeypatch.setattr(
        evaluation_service,
        "get_image_from_dataset",
        _fake_get_image_from_dataset,
    )
    monkeypatch.setattr(
        evaluation_service,
        "get_active_model_instance",
        lambda: FakeClassificationModel(),
    )
    monkeypatch.setattr(
        evaluation_service,
        "get_attack",
        lambda attack_type: failing_attack,
    )
    monkeypatch.setattr(
        evaluation_service,
        "create_result_directory",
        lambda prefix: ("result-id", "/tmp"),
    )

    request = EvaluationRequest(
        dataset_id=VALID_DATASET_ID,
        attack_type=AttackType.RANDOM_NOISE,
    )

    with pytest.raises(
        ValueError,
        match=INTERNAL_ERROR,
    ):
        await evaluation_service.evaluate_dataset(request)

    assert failing_attack.apply_calls == 1


@pytest.mark.asyncio
async def test_detection_attack_preserves_unexpected_error(
    monkeypatch,
):
    monkeypatch.setattr(
        detection_attack_service,
        "get_image_from_dataset",
        _fake_get_image_from_dataset,
    )
    monkeypatch.setattr(
        detection_attack_service,
        "get_active_model_instance",
        lambda: FakeDetectionModel(),
    )
    monkeypatch.setattr(
        detection_attack_service,
        "get_attack",
        lambda attack_type: FailingAttack(),
    )

    request = DetectionAttackRequest(
        dataset_id="dataset-1",
        image_id="image-1",
        attack_type=AttackType.RANDOM_NOISE,
    )

    with pytest.raises(
        ValueError,
        match=INTERNAL_ERROR,
    ):
        await detection_attack_service.apply_detection_attack(request)


@pytest.mark.asyncio
async def test_detection_evaluation_preserves_unexpected_error(
    monkeypatch,
):
    dataset = SimpleNamespace(
        images=[
            SimpleNamespace(
                id="image-1",
                filename="image.png",
            )
        ]
    )

    async def fake_get_dataset(dataset_id: str):
        return dataset

    monkeypatch.setattr(
        detection_evaluation_service,
        "get_dataset",
        fake_get_dataset,
    )
    monkeypatch.setattr(
        detection_evaluation_service,
        "get_image_from_dataset",
        _fake_get_image_from_dataset,
    )
    monkeypatch.setattr(
        detection_evaluation_service,
        "get_active_model_instance",
        lambda: FakeDetectionModel(),
    )
    monkeypatch.setattr(
        detection_evaluation_service,
        "get_attack",
        lambda attack_type: FailingAttack(),
    )

    request = DetectionEvaluationRequest(
        dataset_id="dataset-1",
        attack_type=AttackType.RANDOM_NOISE,
    )

    with pytest.raises(
        ValueError,
        match=INTERNAL_ERROR,
    ):
        await detection_evaluation_service.evaluate_detection_dataset(request)


@pytest.mark.asyncio
async def test_gradcam_preserves_unexpected_error(
    monkeypatch,
):
    def fake_generate_clean_gradcam(model_wrapper, model_input, source_image):
        raise ValueError(INTERNAL_ERROR)

    monkeypatch.setattr(
        explainability_service,
        "get_image_from_dataset",
        _fake_get_image_from_dataset,
    )
    monkeypatch.setattr(
        explainability_service,
        "get_active_model_instance",
        lambda: FakeClassificationModel(),
    )
    monkeypatch.setattr(
        explainability_service,
        "generate_clean_gradcam",
        fake_generate_clean_gradcam,
    )

    with pytest.raises(
        ValueError,
        match=INTERNAL_ERROR,
    ):
        await explainability_service.generate_gradcam_for_image(
            dataset_id="dataset-1",
            image_id="image-1",
        )


@pytest.mark.asyncio
async def test_detection_explainability_preserves_unexpected_error(
    monkeypatch,
):
    def fake_generate_detection_eigencam(
        model,
        image,
    ):
        raise ValueError(INTERNAL_ERROR)

    monkeypatch.setattr(
        explainability_service,
        "get_image_from_dataset",
        _fake_get_image_from_dataset,
    )
    monkeypatch.setattr(
        explainability_service,
        "get_active_model_instance",
        lambda: FakeDetectionModel(),
    )
    monkeypatch.setattr(
        explainability_service,
        "generate_detection_eigencam",
        fake_generate_detection_eigencam,
    )

    request = DetectionExplainRequest(
        dataset_id="dataset-1",
        image_id="image-1",
    )

    with pytest.raises(
        ValueError,
        match=INTERNAL_ERROR,
    ):
        await explainability_service.generate_detection_explainability(request)
