from types import SimpleNamespace

import pytest

import services.detection_evaluation_service as detection_evaluation_service
from ml.models.detection import Detection, DetectionBox, DetectionResult
from schemas.attack import AttackType
from schemas.detection_evaluation import DetectionEvaluationRequest


def _detection(confidence: float) -> Detection:
    return Detection(
        class_id=0,
        class_name="object",
        confidence=confidence,
        box=DetectionBox(x_min=0, y_min=0, x_max=1, y_max=1),
    )


class _FakeAttack:
    id = "random_noise"

    def supports_task_type(self, task_type: str) -> bool:
        return task_type == "detection"

    def apply(self, model_wrapper, image, parameters):
        return {
            "adversarial_image": f"adversarial-{image}",
            "perturbation_linf": 0.03,
            "perturbation_l2": 0.5,
        }


class _FakeDetectionModel:
    id = "detector-1"
    name = "Detector"
    task_type = "detection"

    def detect(self, image):
        detections_by_image = {
            "image-1": [_detection(0.9), _detection(0.8)],
            "adversarial-image-1": [_detection(0.5)],
            "image-2": [],
            "adversarial-image-2": [_detection(0.6)],
        }
        return DetectionResult(
            source_width=1,
            source_height=1,
            detections=detections_by_image[image],
        )


@pytest.mark.asyncio
async def test_detection_evaluation_uses_canonical_deltas_and_legacy_success(
    monkeypatch,
):
    dataset = SimpleNamespace(
        images=[
            SimpleNamespace(id="image-1", filename="image-1.png"),
            SimpleNamespace(id="image-2", filename="image-2.png"),
        ]
    )

    async def fake_get_dataset(dataset_id: str):
        assert dataset_id == "dataset-1"
        return dataset

    async def fake_get_image_from_dataset(dataset_id: str, image_id: str):
        assert dataset_id == "dataset-1"
        return image_id

    monkeypatch.setattr(
        detection_evaluation_service,
        "get_dataset",
        fake_get_dataset,
    )
    monkeypatch.setattr(
        detection_evaluation_service,
        "get_image_from_dataset",
        fake_get_image_from_dataset,
    )
    monkeypatch.setattr(
        detection_evaluation_service,
        "get_active_model_instance",
        _FakeDetectionModel,
    )
    monkeypatch.setattr(
        detection_evaluation_service,
        "get_attack",
        lambda attack_type: _FakeAttack(),
    )

    response = await detection_evaluation_service.evaluate_detection_dataset(
        DetectionEvaluationRequest(
            dataset_id="dataset-1",
            attack_type=AttackType.RANDOM_NOISE,
        )
    )

    assert response.metrics_schema_version == "1.0"
    assert response.delta_convention == "original_minus_adversarial"

    assert response.results[0].detections_difference == 1
    assert response.results[0].average_confidence_difference == 0.35
    assert response.results[0].attack_success is True

    assert response.results[1].detections_difference == -1
    assert response.results[1].average_confidence_difference == -0.6
    assert response.results[1].attack_success is False

    assert response.summary.total_original_detections == 2
    assert response.summary.total_adversarial_detections == 2
    assert response.summary.detections_difference == 0
    assert response.summary.average_confidence_difference == -0.125
    assert response.summary.attack_success_rate == 0.5
