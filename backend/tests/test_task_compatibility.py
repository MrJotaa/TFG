from types import SimpleNamespace

import pytest
from PIL import Image

import services.attack_service as attack_service
import services.detection_attack_service as detection_attack_service
import services.detection_evaluation_service as detection_evaluation_service
import services.detection_service as detection_service
import services.evaluation_service as evaluation_service
import services.explainability_service as explainability_service
import services.inference_service as inference_service
from core.errors import BadRequestError
from core.task_compatibility import (
    CLASSIFICATION_TASK_TYPE,
    DETECTION_TASK_TYPE,
    require_attack_task_type,
    require_model_task_type,
)


class FakeModel:
    def __init__(self, task_type: str):
        self.id = "fake-model"
        self.name = "Fake Model"
        self.task_type = task_type

    def predict(self, image):
        raise AssertionError("Prediction must not run for an incompatible model")


class FakeAttack:
    def __init__(
        self,
        attack_id: str,
        supported_task_types: list[str],
    ):
        self.id = attack_id
        self.supported_task_types = supported_task_types

    def supports_task_type(self, task_type: str) -> bool:
        return task_type in self.supported_task_types

    def apply(
        self,
        model_wrapper,
        image,
        parameters,
        original_class_index=None,
    ):
        raise AssertionError("Attack must not run when it is incompatible")


def _create_image() -> Image.Image:
    return Image.new(
        "RGB",
        (8, 8),
        color=(128, 128, 128),
    )


@pytest.mark.parametrize(
    ("actual_task_type", "expected_task_type"),
    [
        (
            CLASSIFICATION_TASK_TYPE,
            CLASSIFICATION_TASK_TYPE,
        ),
        (
            DETECTION_TASK_TYPE,
            DETECTION_TASK_TYPE,
        ),
    ],
)
def test_require_model_task_type_accepts_compatible_model(
    actual_task_type,
    expected_task_type,
):
    model = FakeModel(actual_task_type)

    require_model_task_type(
        model=model,
        expected_task_type=expected_task_type,
        endpoint="/test",
    )


def test_require_model_task_type_rejects_incompatible_model():
    model = FakeModel(DETECTION_TASK_TYPE)

    with pytest.raises(
        BadRequestError,
        match=("Endpoint '/predict' requires a classification model"),
    ):
        require_model_task_type(
            model=model,
            expected_task_type=CLASSIFICATION_TASK_TYPE,
            endpoint="/predict",
        )


def test_require_attack_task_type_accepts_compatible_attack():
    attack = FakeAttack(
        attack_id="random_noise",
        supported_task_types=[
            CLASSIFICATION_TASK_TYPE,
            DETECTION_TASK_TYPE,
        ],
    )

    require_attack_task_type(
        attack=attack,
        task_type=CLASSIFICATION_TASK_TYPE,
    )


def test_require_attack_task_type_rejects_incompatible_attack():
    attack = FakeAttack(
        attack_id="yolo_confidence",
        supported_task_types=[DETECTION_TASK_TYPE],
    )

    with pytest.raises(
        BadRequestError,
        match=("Attack 'yolo_confidence' does not support task type 'classification'"),
    ):
        require_attack_task_type(
            attack=attack,
            task_type=CLASSIFICATION_TASK_TYPE,
        )


def test_predict_rejects_detection_model(
    client,
    monkeypatch,
):
    async def fake_get_image_from_dataset(
        dataset_id: str,
        image_id: str,
    ):
        return _create_image()

    monkeypatch.setattr(
        inference_service,
        "get_image_from_dataset",
        fake_get_image_from_dataset,
    )
    monkeypatch.setattr(
        inference_service,
        "get_active_model_instance",
        lambda: FakeModel(DETECTION_TASK_TYPE),
    )

    response = client.post(
        "/predict",
        json={
            "dataset_id": "dataset-1",
            "image_id": "image-1",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Endpoint '/predict' requires a classification model"
    )


def test_attack_rejects_detection_model(
    client,
    monkeypatch,
):
    async def fake_get_image_from_dataset(
        dataset_id: str,
        image_id: str,
    ):
        return _create_image()

    monkeypatch.setattr(
        attack_service,
        "get_image_from_dataset",
        fake_get_image_from_dataset,
    )
    monkeypatch.setattr(
        attack_service,
        "get_active_model_instance",
        lambda: FakeModel(DETECTION_TASK_TYPE),
    )

    response = client.post(
        "/attack",
        json={
            "dataset_id": "dataset-1",
            "image_id": "image-1",
            "attack_type": "random_noise",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Endpoint '/attack' requires a classification model"
    )


def test_attack_rejects_detection_only_attack(
    client,
    monkeypatch,
):
    async def fake_get_image_from_dataset(
        dataset_id: str,
        image_id: str,
    ):
        return _create_image()

    incompatible_attack = FakeAttack(
        attack_id="yolo_confidence",
        supported_task_types=[DETECTION_TASK_TYPE],
    )

    monkeypatch.setattr(
        attack_service,
        "get_image_from_dataset",
        fake_get_image_from_dataset,
    )
    monkeypatch.setattr(
        attack_service,
        "get_active_model_instance",
        lambda: FakeModel(CLASSIFICATION_TASK_TYPE),
    )
    monkeypatch.setattr(
        attack_service,
        "get_attack",
        lambda attack_type: incompatible_attack,
    )

    response = client.post(
        "/attack",
        json={
            "dataset_id": "dataset-1",
            "image_id": "image-1",
            "attack_type": "yolo_confidence",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Attack 'yolo_confidence' does not support task type 'classification'"
    )


def test_evaluate_rejects_detection_model(
    client,
    created_dataset,
    monkeypatch,
):
    monkeypatch.setattr(
        evaluation_service,
        "get_active_model_instance",
        lambda: FakeModel(DETECTION_TASK_TYPE),
    )

    response = client.post(
        "/evaluate",
        json={
            "dataset_id": created_dataset["id"],
            "attack_type": "random_noise",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Endpoint '/evaluate' requires a classification model"
    )


def test_evaluate_rejects_detection_only_attack(
    client,
    created_dataset,
    monkeypatch,
):
    incompatible_attack = FakeAttack(
        attack_id="yolo_confidence",
        supported_task_types=[DETECTION_TASK_TYPE],
    )

    monkeypatch.setattr(
        evaluation_service,
        "get_active_model_instance",
        lambda: FakeModel(CLASSIFICATION_TASK_TYPE),
    )
    monkeypatch.setattr(
        evaluation_service,
        "get_attack",
        lambda attack_type: incompatible_attack,
    )

    response = client.post(
        "/evaluate",
        json={
            "dataset_id": created_dataset["id"],
            "attack_type": "yolo_confidence",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Attack 'yolo_confidence' does not support task type 'classification'"
    )


def test_detect_rejects_classification_model(
    client,
    monkeypatch,
):
    async def fake_get_image_from_dataset(
        dataset_id: str,
        image_id: str,
    ):
        return _create_image()

    monkeypatch.setattr(
        detection_service,
        "get_image_from_dataset",
        fake_get_image_from_dataset,
    )
    monkeypatch.setattr(
        detection_service,
        "get_active_model_instance",
        lambda: FakeModel(CLASSIFICATION_TASK_TYPE),
    )

    response = client.post(
        "/detect",
        json={
            "dataset_id": "dataset-1",
            "image_id": "image-1",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Endpoint '/detect' requires a detection model"
    )


def test_detection_attack_rejects_classification_model(
    client,
    monkeypatch,
):
    async def fake_get_image_from_dataset(
        dataset_id: str,
        image_id: str,
    ):
        return _create_image()

    monkeypatch.setattr(
        detection_attack_service,
        "get_image_from_dataset",
        fake_get_image_from_dataset,
    )
    monkeypatch.setattr(
        detection_attack_service,
        "get_active_model_instance",
        lambda: FakeModel(CLASSIFICATION_TASK_TYPE),
    )

    response = client.post(
        "/detect/attack",
        json={
            "dataset_id": "dataset-1",
            "image_id": "image-1",
            "attack_type": "random_noise",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Endpoint '/detect/attack' requires a detection model"
    )


def test_detection_attack_rejects_classification_only_attack(
    client,
    monkeypatch,
):
    async def fake_get_image_from_dataset(
        dataset_id: str,
        image_id: str,
    ):
        return _create_image()

    incompatible_attack = FakeAttack(
        attack_id="fgsm",
        supported_task_types=[CLASSIFICATION_TASK_TYPE],
    )

    monkeypatch.setattr(
        detection_attack_service,
        "get_image_from_dataset",
        fake_get_image_from_dataset,
    )
    monkeypatch.setattr(
        detection_attack_service,
        "get_active_model_instance",
        lambda: FakeModel(DETECTION_TASK_TYPE),
    )
    monkeypatch.setattr(
        detection_attack_service,
        "get_attack",
        lambda attack_type: incompatible_attack,
    )

    response = client.post(
        "/detect/attack",
        json={
            "dataset_id": "dataset-1",
            "image_id": "image-1",
            "attack_type": "fgsm",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Attack 'fgsm' does not support task type 'detection'"
    )


def test_detection_evaluate_rejects_classification_model(
    client,
    monkeypatch,
):
    async def fake_get_dataset(dataset_id: str):
        return SimpleNamespace(images=[])

    monkeypatch.setattr(
        detection_evaluation_service,
        "get_dataset",
        fake_get_dataset,
    )
    monkeypatch.setattr(
        detection_evaluation_service,
        "get_active_model_instance",
        lambda: FakeModel(CLASSIFICATION_TASK_TYPE),
    )

    response = client.post(
        "/detect/evaluate",
        json={
            "dataset_id": "dataset-1",
            "attack_type": "random_noise",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Endpoint '/detect/evaluate' requires a detection model"
    )


def test_detection_evaluate_rejects_classification_only_attack(
    client,
    monkeypatch,
):
    async def fake_get_dataset(dataset_id: str):
        return SimpleNamespace(images=[])

    incompatible_attack = FakeAttack(
        attack_id="fgsm",
        supported_task_types=[CLASSIFICATION_TASK_TYPE],
    )

    monkeypatch.setattr(
        detection_evaluation_service,
        "get_dataset",
        fake_get_dataset,
    )
    monkeypatch.setattr(
        detection_evaluation_service,
        "get_active_model_instance",
        lambda: FakeModel(DETECTION_TASK_TYPE),
    )
    monkeypatch.setattr(
        detection_evaluation_service,
        "get_attack",
        lambda attack_type: incompatible_attack,
    )

    response = client.post(
        "/detect/evaluate",
        json={
            "dataset_id": "dataset-1",
            "attack_type": "fgsm",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Attack 'fgsm' does not support task type 'detection'"
    )


def test_gradcam_rejects_detection_model(
    client,
    monkeypatch,
):
    async def fake_get_image_from_dataset(
        dataset_id: str,
        image_id: str,
    ):
        return _create_image()

    monkeypatch.setattr(
        explainability_service,
        "get_image_from_dataset",
        fake_get_image_from_dataset,
    )
    monkeypatch.setattr(
        explainability_service,
        "get_active_model_instance",
        lambda: FakeModel(DETECTION_TASK_TYPE),
    )

    response = client.post(
        "/explain/gradcam",
        json={
            "dataset_id": "dataset-1",
            "image_id": "image-1",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Endpoint '/explain/gradcam' requires a classification model"
    )


def test_detection_explainability_rejects_classification_model(
    client,
    monkeypatch,
):
    async def fake_get_image_from_dataset(
        dataset_id: str,
        image_id: str,
    ):
        return _create_image()

    monkeypatch.setattr(
        explainability_service,
        "get_image_from_dataset",
        fake_get_image_from_dataset,
    )
    monkeypatch.setattr(
        explainability_service,
        "get_active_model_instance",
        lambda: FakeModel(CLASSIFICATION_TASK_TYPE),
    )

    response = client.post(
        "/explain/detection",
        json={
            "dataset_id": "dataset-1",
            "image_id": "image-1",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Endpoint '/explain/detection' requires a detection model"
    )
