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


def test_detect_evaluate_returns_detection_evaluation_response(client, monkeypatch):
    async def fake_evaluate_detection_dataset(request):
        assert request.dataset_id == "dataset-1"
        assert request.attack_type.value == "yolo_confidence"
        assert request.parameters.epsilon == 0.05
        assert request.parameters.alpha == 0.01
        assert request.parameters.steps == 10

        return {
            "dataset_id": "dataset-1",
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
    assert data["model_id"] == "yolov8n"
    assert data["model_name"] == "YOLOv8n pretrained"
    assert data["attack_type"] == "yolo_confidence"

    summary = data["summary"]

    assert summary["total_images"] == 2
    assert summary["total_original_detections"] == 3
    assert summary["total_adversarial_detections"] == 1
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
            "dataset_id": request.dataset_id,
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


def test_detect_evaluate_propagates_incompatible_model_error(
    client,
    monkeypatch,
):
    async def fake_evaluate_detection_dataset(request):
        raise HTTPException(
            status_code=400,
            detail=(
                "Endpoint '/detect/evaluate' "
                "requires a detection model"
            ),
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