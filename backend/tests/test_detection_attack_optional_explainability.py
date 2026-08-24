import pytest
from PIL import Image

import services.detection_attack_service as detection_attack_service
from ml.models.detection import Detection, DetectionBox, DetectionResult
from schemas.attack import AttackType
from schemas.detection_attack import DetectionAttackRequest


def _result(image: Image.Image) -> DetectionResult:
    return DetectionResult(
        source_width=image.width,
        source_height=image.height,
        detections=(
            Detection(
                class_id=1,
                class_name="person",
                confidence=0.9,
                box=DetectionBox(1, 1, image.width - 1, image.height - 1),
            ),
        ),
    )


class _DetectionModelWithoutExplainability:
    id = "rf-detr-like"
    name = "RF-DETR-like"
    task_type = "detection"

    def supports_detection_explainability(self) -> bool:
        return False

    def detect_with_visualization(self, image):
        return _result(image), image.copy()


class _RandomNoiseLikeAttack:
    id = "random_noise"

    def supports_task_type(self, task_type):
        return task_type == "detection"

    def apply(self, *, image, **kwargs):
        return {
            "adversarial_image": image.copy(),
            "perturbation_image": Image.new("RGB", image.size),
            "perturbation_linf": 0.03,
            "perturbation_l2": 0.2,
        }


@pytest.mark.asyncio
async def test_detection_attack_succeeds_without_optional_explainability(monkeypatch):
    source_image = Image.new("RGB", (16, 12), color=(30, 40, 50))
    saved_filenames = []
    saved_metadata = {}

    async def fake_get_image_from_dataset(**kwargs):
        return source_image

    monkeypatch.setattr(
        detection_attack_service,
        "get_image_from_dataset",
        fake_get_image_from_dataset,
    )
    monkeypatch.setattr(
        detection_attack_service,
        "get_active_model_instance",
        _DetectionModelWithoutExplainability,
    )
    monkeypatch.setattr(
        detection_attack_service,
        "get_attack",
        lambda attack_type: _RandomNoiseLikeAttack(),
    )
    monkeypatch.setattr(
        detection_attack_service,
        "generate_detection_eigencam",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("EigenCAM must not be requested")
        ),
    )
    monkeypatch.setattr(
        detection_attack_service,
        "create_result_directory",
        lambda prefix: ("result-1", "/tmp/result-1"),
    )

    def fake_save_result_image(**kwargs):
        saved_filenames.append(kwargs["filename"])
        return f"/results/result-1/{kwargs['filename']}"

    def fake_save_result_metadata(**kwargs):
        saved_metadata.update(kwargs["metadata"])

    monkeypatch.setattr(
        detection_attack_service,
        "save_result_image",
        fake_save_result_image,
    )
    monkeypatch.setattr(
        detection_attack_service,
        "save_result_metadata",
        fake_save_result_metadata,
    )

    response = await detection_attack_service.apply_detection_attack(
        DetectionAttackRequest(
            dataset_id="dataset-1",
            image_id="image-1",
            attack_type=AttackType.RANDOM_NOISE,
        )
    )

    assert response.original_explanation_image_url is None
    assert response.adversarial_explanation_image_url is None
    assert "original_explanation.png" not in saved_filenames
    assert "adversarial_explanation.png" not in saved_filenames
    assert "explanation_type" not in saved_metadata
    assert "original_explanation_image" not in saved_metadata
    assert "adversarial_explanation_image" not in saved_metadata


def test_rf_detr_claims_explainability_after_model_contract_integration():
    from ml.models.rf_detr.config import RfDetrDetectionModelConfig
    from ml.models.rf_detr.model import RfDetrDetectionModel

    model = RfDetrDetectionModel(
        RfDetrDetectionModelConfig(
            id="rf-detr-test",
            name="RF-DETR Test",
            revision="test",
            checkpoint="provider/checkpoint",
            checkpoint_revision="commit",
        )
    )

    assert model.supports_detection_explainability() is True
