from fastapi import HTTPException

import api.routes.detection_attack as detection_attack_route


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


def test_detect_attack_returns_detection_attack_response(client, monkeypatch):
    async def fake_apply_detection_attack(request):
        assert request.dataset_id == "dataset-1"
        assert request.image_id == "image-1"
        assert request.attack_type.value == "yolo_confidence"
        assert request.parameters.epsilon == 0.05
        assert request.parameters.alpha == 0.01
        assert request.parameters.steps == 10

        return {
            "metrics_schema_version": "1.0",
            "delta_convention": "original_minus_adversarial",
            "result_id": "detection-attack-result-1",
            "attack_type": "yolo_confidence",
            "parameters": {
                "epsilon": 0.05,
                "alpha": 0.01,
                "steps": 10,
            },
            "original_annotated_image_url": "/results/detection-attack-result-1/original_annotated.png",
            "adversarial_image_url": "/results/detection-attack-result-1/adversarial.png",
            "adversarial_annotated_image_url": "/results/detection-attack-result-1/adversarial_annotated.png",
            "perturbation_image_url": "/results/detection-attack-result-1/perturbation.png",
            "original_explanation_image_url": "/results/detection-attack-result-1/original_explanation.png",
            "adversarial_explanation_image_url": "/results/detection-attack-result-1/adversarial_explanation.png",
            "original_detections": [
                _mock_detection("car", 0.91),
            ],
            "adversarial_detections": [
                _mock_detection("car", 0.43),
            ],
            "comparison": {
                "original_detection_count": 1,
                "adversarial_detection_count": 1,
                "detections_difference": 0,
                "average_original_confidence": 0.91,
                "average_adversarial_confidence": 0.43,
                "average_confidence_difference": 0.48,
                "attack_success": True,
                "perturbation_linf": 0.05,
                "perturbation_l2": 1.72,
            },
        }

    monkeypatch.setattr(
        detection_attack_route,
        "apply_detection_attack",
        fake_apply_detection_attack,
    )

    response = client.post(
        "/detect/attack",
        json={
            "dataset_id": "dataset-1",
            "image_id": "image-1",
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

    assert data["result_id"] == "detection-attack-result-1"
    assert data["attack_type"] == "yolo_confidence"
    assert data["parameters"]["epsilon"] == 0.05
    assert data["metrics_schema_version"] == "1.0"
    assert data["delta_convention"] == "original_minus_adversarial"

    assert data["original_annotated_image_url"]
    assert data["adversarial_image_url"]
    assert data["adversarial_annotated_image_url"]
    assert data["perturbation_image_url"]
    assert data["original_explanation_image_url"]
    assert data["adversarial_explanation_image_url"]

    assert len(data["original_detections"]) == 1
    assert len(data["adversarial_detections"]) == 1

    assert data["comparison"]["attack_success"] is True
    assert data["comparison"]["average_confidence_difference"] == 0.48


def test_detect_attack_uses_default_parameters(client, monkeypatch):
    async def fake_apply_detection_attack(request):
        assert request.attack_type.value == "random_noise"
        assert request.parameters.epsilon == 0.03
        assert request.parameters.alpha == 0.01
        assert request.parameters.steps == 10

        return {
            "metrics_schema_version": "1.0",
            "delta_convention": "original_minus_adversarial",
            "result_id": "detection-attack-defaults",
            "attack_type": "random_noise",
            "parameters": {
                "epsilon": request.parameters.epsilon,
                "alpha": request.parameters.alpha,
                "steps": request.parameters.steps,
            },
            "original_annotated_image_url": "/results/detection-attack-defaults/original_annotated.png",
            "adversarial_image_url": "/results/detection-attack-defaults/adversarial.png",
            "adversarial_annotated_image_url": "/results/detection-attack-defaults/adversarial_annotated.png",
            "perturbation_image_url": "/results/detection-attack-defaults/perturbation.png",
            "original_explanation_image_url": "/results/detection-attack-defaults/original_explanation.png",
            "adversarial_explanation_image_url": "/results/detection-attack-defaults/adversarial_explanation.png",
            "original_detections": [
                _mock_detection("car", 0.91),
            ],
            "adversarial_detections": [
                _mock_detection("car", 0.86),
            ],
            "comparison": {
                "original_detection_count": 1,
                "adversarial_detection_count": 1,
                "detections_difference": 0,
                "average_original_confidence": 0.91,
                "average_adversarial_confidence": 0.86,
                "average_confidence_difference": 0.05,
                "attack_success": False,
                "perturbation_linf": 0.03,
                "perturbation_l2": 0.8,
            },
        }

    monkeypatch.setattr(
        detection_attack_route,
        "apply_detection_attack",
        fake_apply_detection_attack,
    )

    response = client.post(
        "/detect/attack",
        json={
            "dataset_id": "dataset-1",
            "image_id": "image-1",
            "attack_type": "random_noise",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["attack_type"] == "random_noise"
    assert data["metrics_schema_version"] == "1.0"
    assert data["delta_convention"] == "original_minus_adversarial"
    assert data["parameters"]["epsilon"] == 0.03
    assert data["parameters"]["alpha"] == 0.01
    assert data["parameters"]["steps"] == 10


def test_detect_attack_with_invalid_attack_type_returns_422(client):
    response = client.post(
        "/detect/attack",
        json={
            "dataset_id": "dataset-1",
            "image_id": "image-1",
            "attack_type": "invalid_attack",
        },
    )

    assert response.status_code == 422


def test_detect_attack_with_invalid_epsilon_returns_422(client):
    response = client.post(
        "/detect/attack",
        json={
            "dataset_id": "dataset-1",
            "image_id": "image-1",
            "attack_type": "yolo_confidence",
            "parameters": {
                "epsilon": 1.0,
                "alpha": 0.01,
                "steps": 10,
            },
        },
    )

    assert response.status_code == 422


def test_detect_attack_propagates_incompatible_attack_error(
    client,
    monkeypatch,
):
    async def fake_apply_detection_attack(request):
        raise HTTPException(
            status_code=400,
            detail=("Attack 'fgsm' does not support task type 'detection'"),
        )

    monkeypatch.setattr(
        detection_attack_route,
        "apply_detection_attack",
        fake_apply_detection_attack,
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


def test_detect_attack_propagates_dataset_error(client, monkeypatch):
    async def fake_apply_detection_attack(request):
        raise HTTPException(status_code=404, detail="Image not found")

    monkeypatch.setattr(
        detection_attack_route,
        "apply_detection_attack",
        fake_apply_detection_attack,
    )

    response = client.post(
        "/detect/attack",
        json={
            "dataset_id": "dataset-1",
            "image_id": "missing-image",
            "attack_type": "random_noise",
        },
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Image not found"
