from fastapi import HTTPException

import api.routes.attack as attack_route


def _mock_prediction(label: str, confidence: float):
    return {
        "prediction": label,
        "confidence": confidence,
        "top_k": [
            {
                "label": label,
                "probability": confidence,
            },
            {
                "label": "other class",
                "probability": round(1 - confidence, 4),
            },
        ],
    }


def test_attack_returns_attack_response(client, monkeypatch):
    async def fake_apply_adversarial_attack(request, **kwargs):
        assert request.dataset_id == "dataset-1"
        assert request.image_id == "image-1"
        assert request.attack_type.value == "fgsm"
        assert request.parameters.epsilon == 0.03

        return {
            "result_id": "attack-result-1",
            "attack_type": "fgsm",
            "parameters": {
                "epsilon": 0.03,
                "alpha": 0.01,
                "steps": 10,
            },
            "adversarial_image_url": "/results/attack-result-1/adversarial.png",
            "perturbation_image_url": "/results/attack-result-1/perturbation.png",
            "original_gradcam_image_url": "/results/attack-result-1/original_gradcam.png",
            "adversarial_gradcam_image_url": "/results/attack-result-1/adversarial_gradcam.png",
            "original_prediction": _mock_prediction("tabby cat", 0.92),
            "adversarial_prediction": _mock_prediction("tiger cat", 0.81),
            "comparison": {
                "original_label": "tabby cat",
                "adversarial_label": "tiger cat",
                "original_confidence": 0.92,
                "adversarial_confidence": 0.81,
                "confidence_difference": 0.11,
                "prediction_changed": True,
                "attack_success": True,
                "perturbation_linf": 0.03,
                "perturbation_l2": 1.245,
            },
        }

    monkeypatch.setattr(
        attack_route,
        "apply_adversarial_attack",
        fake_apply_adversarial_attack,
    )

    response = client.post(
        "/attack",
        json={
            "dataset_id": "dataset-1",
            "image_id": "image-1",
            "attack_type": "fgsm",
            "parameters": {
                "epsilon": 0.03,
                "alpha": 0.01,
                "steps": 10,
            },
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["result_id"] == "attack-result-1"
    assert data["attack_type"] == "fgsm"
    assert data["parameters"]["epsilon"] == 0.03
    assert data["original_prediction"]["prediction"] == "tabby cat"
    assert data["adversarial_prediction"]["prediction"] == "tiger cat"
    assert data["comparison"]["attack_success"] is True
    assert data["comparison"]["prediction_changed"] is True


def test_attack_uses_default_attack_parameters(client, monkeypatch):
    async def fake_apply_adversarial_attack(request, **kwargs):
        assert request.attack_type.value == "fgsm"
        assert request.parameters.epsilon == 0.03
        assert request.parameters.alpha == 0.01
        assert request.parameters.steps == 10

        return {
            "result_id": "attack-result-defaults",
            "attack_type": "fgsm",
            "parameters": {
                "epsilon": request.parameters.epsilon,
                "alpha": request.parameters.alpha,
                "steps": request.parameters.steps,
            },
            "adversarial_image_url": "/results/attack-result-defaults/adversarial.png",
            "perturbation_image_url": "/results/attack-result-defaults/perturbation.png",
            "original_gradcam_image_url": "/results/attack-result-defaults/original_gradcam.png",
            "adversarial_gradcam_image_url": "/results/attack-result-defaults/adversarial_gradcam.png",
            "original_prediction": _mock_prediction("tabby cat", 0.92),
            "adversarial_prediction": _mock_prediction("tabby cat", 0.72),
            "comparison": {
                "original_label": "tabby cat",
                "adversarial_label": "tabby cat",
                "original_confidence": 0.92,
                "adversarial_confidence": 0.72,
                "confidence_difference": 0.2,
                "prediction_changed": False,
                "attack_success": False,
                "perturbation_linf": 0.03,
                "perturbation_l2": 1.0,
            },
        }

    monkeypatch.setattr(
        attack_route,
        "apply_adversarial_attack",
        fake_apply_adversarial_attack,
    )

    response = client.post(
        "/attack",
        json={
            "dataset_id": "dataset-1",
            "image_id": "image-1",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["attack_type"] == "fgsm"
    assert data["parameters"]["epsilon"] == 0.03
    assert data["parameters"]["alpha"] == 0.01
    assert data["parameters"]["steps"] == 10


def test_attack_with_invalid_attack_type_returns_422(client):
    response = client.post(
        "/attack",
        json={
            "dataset_id": "dataset-1",
            "image_id": "image-1",
            "attack_type": "invalid_attack",
            "parameters": {
                "epsilon": 0.03,
                "alpha": 0.01,
                "steps": 10,
            },
        },
    )

    assert response.status_code == 422


def test_attack_with_invalid_epsilon_returns_422(client):
    response = client.post(
        "/attack",
        json={
            "dataset_id": "dataset-1",
            "image_id": "image-1",
            "attack_type": "fgsm",
            "parameters": {
                "epsilon": 1.0,
                "alpha": 0.01,
                "steps": 10,
            },
        },
    )

    assert response.status_code == 422


def test_attack_propagates_service_error(client, monkeypatch):
    async def fake_apply_adversarial_attack(request, **kwargs):
        raise HTTPException(
            status_code=400,
            detail="Endpoint /attack currently supports classification models only.",
        )

    monkeypatch.setattr(
        attack_route,
        "apply_adversarial_attack",
        fake_apply_adversarial_attack,
    )

    response = client.post(
        "/attack",
        json={
            "dataset_id": "dataset-1",
            "image_id": "image-1",
            "attack_type": "fgsm",
        },
    )

    assert response.status_code == 400
    assert "classification models only" in response.json()["detail"]


def test_attack_catalog_endpoint_exposes_registered_metadata(client):
    response = client.get("/attacks")

    assert response.status_code == 200

    catalog = {item["id"]: item for item in response.json()}
    assert set(catalog) == {
        "fgsm",
        "random_noise",
        "pgd",
        "carlini_wagner",
        "deepfool",
        "yolo_confidence",
    }
    assert catalog["fgsm"]["supported_task_types"] == ["classification"]
    assert catalog["yolo_confidence"]["supported_task_types"] == ["detection"]
    assert catalog["pgd"]["parameters"][0]["key"] == "epsilon"
