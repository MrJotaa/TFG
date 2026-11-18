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
    require_attack_model_compatibility,
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
        supported_model_ids: tuple[str, ...] | None = None,
    ):
        self.id = attack_id
        self.supported_task_types = supported_task_types
        if supported_model_ids is not None:
            self.supported_model_ids = supported_model_ids

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
        match=("El endpoint '/predict' requiere un modelo de clasificación"),
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
        match=(
            "El ataque 'yolo_confidence' no es compatible con la tarea de clasificación"
        ),
    ):
        require_attack_task_type(
            attack=attack,
            task_type=CLASSIFICATION_TASK_TYPE,
        )


@pytest.mark.parametrize("task_type", [CLASSIFICATION_TASK_TYPE, DETECTION_TASK_TYPE])
def test_attack_model_compatibility_keeps_unrestricted_legacy_attacks(task_type):
    attack = FakeAttack("random_noise", [task_type])

    require_attack_model_compatibility(attack, FakeModel(task_type))


def test_attack_model_compatibility_accepts_declared_model():
    attack = FakeAttack(
        "localization_attack",
        [DETECTION_TASK_TYPE],
        supported_model_ids=("rf_detr_nano",),
    )
    model = FakeModel(DETECTION_TASK_TYPE)
    model.id = "rf_detr_nano"

    require_attack_model_compatibility(attack, model)


def test_attack_model_compatibility_rejects_undeclared_model():
    attack = FakeAttack(
        "localization_attack",
        [DETECTION_TASK_TYPE],
        supported_model_ids=("rf_detr_nano",),
    )
    model = FakeModel(DETECTION_TASK_TYPE)
    model.id = "yolo26n"

    with pytest.raises(
        BadRequestError,
        match=(
            "El ataque 'localization_attack' no es compatible "
            "con el modelo 'yolo26n'"
        ),
    ):
        require_attack_model_compatibility(attack, model)


def test_attack_model_compatibility_keeps_task_error_priority():
    attack = FakeAttack(
        "localization_attack",
        [DETECTION_TASK_TYPE],
        supported_model_ids=("rf_detr_nano",),
    )

    with pytest.raises(BadRequestError, match="tarea de clasificación"):
        require_attack_model_compatibility(
            attack,
            FakeModel(CLASSIFICATION_TASK_TYPE),
        )


def test_predict_rejects_detection_model(
    client,
    monkeypatch,
):
    async def fake_get_image_from_dataset(
        dataset_id: str,
        image_id: str,
        **kwargs,
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
        "El endpoint '/predict' requiere un modelo de clasificación"
    )


def test_attack_rejects_detection_model(
    client,
    monkeypatch,
):
    async def fake_get_image_from_dataset(
        dataset_id: str,
        image_id: str,
        **kwargs,
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
        "El endpoint '/attack' requiere un modelo de clasificación"
    )


def test_attack_rejects_detection_only_attack(
    client,
    monkeypatch,
):
    async def fake_get_image_from_dataset(
        dataset_id: str,
        image_id: str,
        **kwargs,
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
        "El ataque 'yolo_confidence' no es compatible con la tarea de clasificación"
    )


def test_evaluate_rejects_detection_model(
    client,
    created_dataset,
    auth_headers,
    monkeypatch,
):
    monkeypatch.setattr(
        evaluation_service,
        "get_active_model_instance",
        lambda: FakeModel(DETECTION_TASK_TYPE),
    )

    response = client.post(
        "/evaluate",
        headers=auth_headers,
        json={
            "dataset_id": created_dataset["id"],
            "attack_type": "random_noise",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "El endpoint '/evaluate' requiere un modelo de clasificación"
    )


def test_evaluate_rejects_detection_only_attack(
    client,
    created_dataset,
    auth_headers,
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
        headers=auth_headers,
        json={
            "dataset_id": created_dataset["id"],
            "attack_type": "yolo_confidence",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "El ataque 'yolo_confidence' no es compatible con la tarea de clasificación"
    )


def test_detect_rejects_classification_model(
    client,
    monkeypatch,
):
    async def fake_get_image_from_dataset(
        dataset_id: str,
        image_id: str,
        **kwargs,
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
        "El endpoint '/detect' requiere un modelo de detección"
    )


def test_detection_attack_rejects_classification_model(
    client,
    monkeypatch,
):
    async def fake_get_image_from_dataset(
        dataset_id: str,
        image_id: str,
        **kwargs,
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
        "El endpoint '/detect/attack' requiere un modelo de detección"
    )


def test_detection_attack_rejects_classification_only_attack(
    client,
    monkeypatch,
):
    async def fake_get_image_from_dataset(
        dataset_id: str,
        image_id: str,
        **kwargs,
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
        "El ataque 'fgsm' no es compatible con la tarea de detección"
    )


def test_detection_evaluate_rejects_classification_model(
    client,
    created_dataset,
    auth_headers,
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
        headers=auth_headers,
        json={
            "dataset_id": created_dataset["id"],
            "attack_type": "random_noise",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "El endpoint '/detect/evaluate' requiere un modelo de detección"
    )


def test_detection_evaluate_rejects_classification_only_attack(
    client,
    created_dataset,
    auth_headers,
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
        headers=auth_headers,
        json={
            "dataset_id": created_dataset["id"],
            "attack_type": "fgsm",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "El ataque 'fgsm' no es compatible con la tarea de detección"
    )


@pytest.mark.parametrize(
    ("service", "route", "task_type", "attack_type", "allowed_model_id"),
    [
        (attack_service, "/attack", "classification", "fgsm", "resnet18"),
        (
            detection_attack_service,
            "/detect/attack",
            "detection",
            "yolo_confidence",
            "rf_detr_nano",
        ),
    ],
)
def test_individual_attack_rejects_model_restriction_before_inference(
    client,
    monkeypatch,
    service,
    route,
    task_type,
    attack_type,
    allowed_model_id,
):
    async def fake_get_image_from_dataset(dataset_id, image_id, **kwargs):
        return _create_image()

    restricted_attack = FakeAttack(
        attack_type,
        [task_type],
        supported_model_ids=(allowed_model_id,),
    )
    monkeypatch.setattr(
        service,
        "get_image_from_dataset",
        fake_get_image_from_dataset,
    )
    monkeypatch.setattr(
        service,
        "get_active_model_instance",
        lambda: FakeModel(task_type),
    )
    monkeypatch.setattr(service, "get_attack", lambda attack_type: restricted_attack)

    response = client.post(
        route,
        json={
            "dataset_id": "dataset-1",
            "image_id": "image-1",
            "attack_type": attack_type,
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        f"El ataque '{attack_type}' no es compatible con el modelo 'fake-model'"
    )


@pytest.mark.parametrize(
    ("service", "route", "task_type", "attack_type", "allowed_model_id"),
    [
        (evaluation_service, "/evaluate", "classification", "fgsm", "resnet18"),
        (
            detection_evaluation_service,
            "/detect/evaluate",
            "detection",
            "yolo_confidence",
            "rf_detr_nano",
        ),
    ],
)
def test_evaluation_rejects_model_restriction_before_running_attack(
    client,
    created_dataset,
    auth_headers,
    monkeypatch,
    service,
    route,
    task_type,
    attack_type,
    allowed_model_id,
):
    restricted_attack = FakeAttack(
        attack_type,
        [task_type],
        supported_model_ids=(allowed_model_id,),
    )
    if task_type == DETECTION_TASK_TYPE:

        async def fake_get_dataset(dataset_id):
            return SimpleNamespace(images=[])

        monkeypatch.setattr(service, "get_dataset", fake_get_dataset)

    monkeypatch.setattr(
        service,
        "get_active_model_instance",
        lambda: FakeModel(task_type),
    )
    monkeypatch.setattr(service, "get_attack", lambda attack_type: restricted_attack)

    response = client.post(
        route,
        headers=auth_headers,
        json={
            "dataset_id": created_dataset["id"],
            "attack_type": attack_type,
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        f"El ataque '{attack_type}' no es compatible con el modelo 'fake-model'"
    )


def test_gradcam_rejects_detection_model(
    client,
    monkeypatch,
):
    async def fake_get_image_from_dataset(
        dataset_id: str,
        image_id: str,
        **kwargs,
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
        "El endpoint '/explain/gradcam' requiere un modelo de clasificación"
    )


def test_detection_explainability_rejects_classification_model(
    client,
    monkeypatch,
):
    async def fake_get_image_from_dataset(
        dataset_id: str,
        image_id: str,
        **kwargs,
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
        "El endpoint '/explain/detection' requiere un modelo de detección"
    )


pytestmark = pytest.mark.usefixtures("authenticated_client")
