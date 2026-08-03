import pytest
from fastapi import HTTPException

import api.routes.detection_evaluate as detection_evaluate_route


def _mock_detection(label: str = "car", confidence: float = 0.91):
    return {
        "label": label,
        "confidence": confidence,
        "bbox": {
            "x1": 10.0,
            "y1": 20.0,
            "x2": 120.0,
            "y2": 160.0,
        },
    }


def _empty_evaluation_response(result_id: str):
    return {
        "result_id": result_id,
        "metrics_schema_version": "1.0",
        "delta_convention": "original_minus_adversarial",
        "dataset_id": "dataset-1",
        "dataset_revision": 0,
        "annotation_snapshot_id": None,
        "model_id": "yolov8n",
        "model_name": "YOLOv8n pretrained",
        "attack_type": "random_noise",
        "parameters": {
            "epsilon": 0.03,
            "alpha": 0.01,
            "steps": 10,
        },
        "summary": {
            "total_images": 0,
            "total_original_detections": 0,
            "total_adversarial_detections": 0,
            "detections_difference": 0,
            "average_original_detections_per_image": 0.0,
            "average_adversarial_detections_per_image": 0.0,
            "average_original_confidence": 0.0,
            "average_adversarial_confidence": 0.0,
            "average_confidence_difference": 0.0,
            "images_with_reduced_detections": 0,
            "images_with_no_adversarial_detections": 0,
            "attack_success_rate": 0.0,
            "average_perturbation_linf": 0.0,
            "average_perturbation_l2": 0.0,
        },
        "results": [],
        "supervised": None,
    }


def test_detect_evaluate_returns_detection_evaluation_response(client, monkeypatch):
    async def fake_evaluate_detection_dataset(request):
        assert request.dataset_id == "dataset-1"
        assert request.attack_type.value == "yolo_confidence"
        assert request.parameters.epsilon == 0.05
        assert request.parameters.alpha == 0.01
        assert request.parameters.steps == 10
        assert request.supervised_configuration.score_floor == 0.001
        assert request.supervised_configuration.confidence_threshold == 0.25
        assert request.supervised_configuration.iou_threshold == 0.5

        return {
            "result_id": ("detection_evaluation_00000000-0000-4000-8000-000000000001"),
            "metrics_schema_version": "1.0",
            "delta_convention": "original_minus_adversarial",
            "dataset_id": "dataset-1",
            "dataset_revision": 0,
            "annotation_snapshot_id": None,
            "model_id": "yolov8n",
            "model_name": "YOLOv8n pretrained",
            "attack_type": "yolo_confidence",
            "parameters": {
                "epsilon": 0.05,
                "alpha": 0.01,
                "steps": 10,
            },
            "summary": {
                "total_images": 2,
                "total_original_detections": 3,
                "total_adversarial_detections": 1,
                "detections_difference": 2,
                "average_original_detections_per_image": 1.5,
                "average_adversarial_detections_per_image": 0.5,
                "average_original_confidence": 0.88,
                "average_adversarial_confidence": 0.42,
                "average_confidence_difference": 0.46,
                "images_with_reduced_detections": 2,
                "images_with_no_adversarial_detections": 1,
                "attack_success_rate": 1.0,
                "average_perturbation_linf": 0.05,
                "average_perturbation_l2": 1.64,
            },
            "results": [
                {
                    "image_id": "image-1",
                    "filename": "image-1.png",
                    "original_detection_count": 2,
                    "adversarial_detection_count": 1,
                    "detections_difference": 1,
                    "average_original_confidence": 0.9,
                    "average_adversarial_confidence": 0.42,
                    "average_confidence_difference": 0.48,
                    "attack_success": True,
                    "perturbation_linf": 0.05,
                    "perturbation_l2": 1.7,
                    "original_detections": [
                        _mock_detection("car", 0.91),
                        _mock_detection("person", 0.89),
                    ],
                    "adversarial_detections": [
                        _mock_detection("car", 0.42),
                    ],
                },
                {
                    "image_id": "image-2",
                    "filename": "image-2.png",
                    "original_detection_count": 1,
                    "adversarial_detection_count": 0,
                    "detections_difference": 1,
                    "average_original_confidence": 0.86,
                    "average_adversarial_confidence": 0.0,
                    "average_confidence_difference": 0.86,
                    "attack_success": True,
                    "perturbation_linf": 0.05,
                    "perturbation_l2": 1.58,
                    "original_detections": [
                        _mock_detection("truck", 0.86),
                    ],
                    "adversarial_detections": [],
                },
            ],
        }

    monkeypatch.setattr(
        detection_evaluate_route,
        "evaluate_detection_dataset",
        fake_evaluate_detection_dataset,
    )

    response = client.post(
        "/detect/evaluate",
        json={
            "dataset_id": "dataset-1",
            "attack_type": "yolo_confidence",
            "parameters": {
                "epsilon": 0.05,
                "alpha": 0.01,
                "steps": 10,
            },
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["dataset_id"] == "dataset-1"
    assert data["result_id"].startswith("detection_evaluation_")
    assert data["model_id"] == "yolov8n"
    assert data["model_name"] == "YOLOv8n pretrained"
    assert data["attack_type"] == "yolo_confidence"
    assert data["metrics_schema_version"] == "1.0"
    assert data["delta_convention"] == "original_minus_adversarial"

    summary = data["summary"]

    assert summary["total_images"] == 2
    assert summary["total_original_detections"] == 3
    assert summary["total_adversarial_detections"] == 1
    assert summary["detections_difference"] == 2
    assert summary["average_confidence_difference"] == 0.46
    assert summary["images_with_reduced_detections"] == 2
    assert summary["images_with_no_adversarial_detections"] == 1
    assert summary["attack_success_rate"] == 1.0

    assert len(data["results"]) == 2
    assert data["results"][0]["attack_success"] is True
    assert data["results"][1]["adversarial_detections"] == []


def test_detect_evaluate_uses_default_parameters(client, monkeypatch):
    async def fake_evaluate_detection_dataset(request):
        assert request.attack_type.value == "random_noise"
        assert request.parameters.epsilon == 0.03
        assert request.parameters.alpha == 0.01
        assert request.parameters.steps == 10

        return {
            "result_id": ("detection_evaluation_00000000-0000-4000-8000-000000000002"),
            "metrics_schema_version": "1.0",
            "delta_convention": "original_minus_adversarial",
            "dataset_id": request.dataset_id,
            "dataset_revision": 0,
            "annotation_snapshot_id": None,
            "model_id": "yolov8n",
            "model_name": "YOLOv8n pretrained",
            "attack_type": "random_noise",
            "parameters": {
                "epsilon": request.parameters.epsilon,
                "alpha": request.parameters.alpha,
                "steps": request.parameters.steps,
            },
            "summary": {
                "total_images": 1,
                "total_original_detections": 1,
                "total_adversarial_detections": 1,
                "detections_difference": 0,
                "average_original_detections_per_image": 1.0,
                "average_adversarial_detections_per_image": 1.0,
                "average_original_confidence": 0.88,
                "average_adversarial_confidence": 0.81,
                "average_confidence_difference": 0.07,
                "images_with_reduced_detections": 0,
                "images_with_no_adversarial_detections": 0,
                "attack_success_rate": 0.0,
                "average_perturbation_linf": 0.03,
                "average_perturbation_l2": 0.75,
            },
            "results": [
                {
                    "image_id": "image-1",
                    "filename": "image-1.png",
                    "original_detection_count": 1,
                    "adversarial_detection_count": 1,
                    "detections_difference": 0,
                    "average_original_confidence": 0.88,
                    "average_adversarial_confidence": 0.81,
                    "average_confidence_difference": 0.07,
                    "attack_success": False,
                    "perturbation_linf": 0.03,
                    "perturbation_l2": 0.75,
                    "original_detections": [
                        _mock_detection("car", 0.88),
                    ],
                    "adversarial_detections": [
                        _mock_detection("car", 0.81),
                    ],
                },
            ],
        }

    monkeypatch.setattr(
        detection_evaluate_route,
        "evaluate_detection_dataset",
        fake_evaluate_detection_dataset,
    )

    response = client.post(
        "/detect/evaluate",
        json={
            "dataset_id": "dataset-1",
            "attack_type": "random_noise",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["dataset_id"] == "dataset-1"
    assert data["model_id"] == "yolov8n"
    assert data["model_name"] == "YOLOv8n pretrained"
    assert data["attack_type"] == "random_noise"
    assert data["metrics_schema_version"] == "1.0"
    assert data["delta_convention"] == "original_minus_adversarial"
    assert data["parameters"]["epsilon"] == 0.03
    assert data["parameters"]["alpha"] == 0.01
    assert data["parameters"]["steps"] == 10


def test_detect_evaluate_with_missing_dataset_id_returns_422(client):
    response = client.post(
        "/detect/evaluate",
        json={
            "attack_type": "random_noise",
        },
    )

    assert response.status_code == 422


def test_detect_evaluate_with_invalid_attack_type_returns_422(client):
    response = client.post(
        "/detect/evaluate",
        json={
            "dataset_id": "dataset-1",
            "attack_type": "invalid_attack",
        },
    )

    assert response.status_code == 422


@pytest.mark.parametrize(
    "supervised_configuration",
    [
        {
            "score_floor": 0.3,
            "confidence_threshold": 0.3,
        },
        {
            "score_floor": 0.2,
            "confidence_threshold": 0.1,
        },
        {
            "iou_threshold": 0.51,
        },
        {
            "max_detections_per_image": 99,
        },
        {
            "nms_iou_threshold": 0.6,
        },
    ],
)
def test_detect_evaluate_rejects_invalid_supervised_thresholds(
    client,
    supervised_configuration,
):
    response = client.post(
        "/detect/evaluate",
        json={
            "dataset_id": "dataset-1",
            "attack_type": "random_noise",
            "supervised_configuration": supervised_configuration,
        },
    )

    assert response.status_code == 422


def test_detect_evaluate_propagates_incompatible_model_error(
    client,
    monkeypatch,
):
    async def fake_evaluate_detection_dataset(request):
        raise HTTPException(
            status_code=400,
            detail=("Endpoint '/detect/evaluate' requires a detection model"),
        )

    monkeypatch.setattr(
        detection_evaluate_route,
        "evaluate_detection_dataset",
        fake_evaluate_detection_dataset,
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


def test_detect_evaluate_propagates_dataset_error(client, monkeypatch):
    async def fake_evaluate_detection_dataset(request):
        raise HTTPException(status_code=404, detail="Dataset not found")

    monkeypatch.setattr(
        detection_evaluate_route,
        "evaluate_detection_dataset",
        fake_evaluate_detection_dataset,
    )

    response = client.post(
        "/detect/evaluate",
        json={
            "dataset_id": "missing-dataset",
            "attack_type": "random_noise",
        },
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Dataset not found"


def test_detection_evaluation_history_requires_authentication(client):
    response = client.get("/detect/evaluations")

    assert response.status_code == 401


def test_detection_evaluation_history_rejects_invalid_pagination(
    client,
    auth_headers,
):
    response = client.get(
        "/detect/evaluations?limit=101&offset=-1",
        headers=auth_headers,
    )

    assert response.status_code == 422


def test_detection_evaluation_history_uses_authenticated_owner(
    client,
    auth_headers,
    registered_user,
    monkeypatch,
):
    captured = {}

    async def fake_list_detection_evaluations(*, owner_id, limit, offset):
        captured.update(owner_id=owner_id, limit=limit, offset=offset)
        return []

    monkeypatch.setattr(
        detection_evaluate_route,
        "list_detection_evaluations",
        fake_list_detection_evaluations,
    )

    response = client.get(
        "/detect/evaluations?limit=17&offset=3",
        headers=auth_headers,
    )

    assert response.status_code == 200
    assert response.json() == []
    assert captured == {
        "owner_id": registered_user["user"]["id"],
        "limit": 17,
        "offset": 3,
    }


def test_detection_evaluation_detail_uses_authenticated_owner(
    client,
    auth_headers,
    registered_user,
    monkeypatch,
):
    result_id = "detection_evaluation_00000000-0000-4000-8000-000000000010"
    captured = {}

    async def fake_get_detection_evaluation(*, evaluation_id, owner_id):
        captured.update(evaluation_id=evaluation_id, owner_id=owner_id)
        return _empty_evaluation_response(evaluation_id)

    monkeypatch.setattr(
        detection_evaluate_route,
        "get_detection_evaluation",
        fake_get_detection_evaluation,
    )

    response = client.get(
        f"/detect/evaluations/{result_id}",
        headers=auth_headers,
    )

    assert response.status_code == 200
    assert response.json()["result_id"] == result_id
    assert captured == {
        "evaluation_id": result_id,
        "owner_id": registered_user["user"]["id"],
    }
