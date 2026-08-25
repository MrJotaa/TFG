import json

from fastapi import HTTPException

import api.routes.evaluate as evaluate_route


def test_evaluate_returns_evaluation_response(client, monkeypatch, auth_headers):
    async def fake_evaluate_dataset(request, **kwargs):
        assert request.dataset_id == "dataset-1"
        assert request.attack_type.value == "pgd"
        assert request.parameters.epsilon == 0.05
        assert request.parameters.alpha == 0.01
        assert request.parameters.steps == 5

        return {
            "result_id": "evaluation-result-1",
            "dataset_id": "dataset-1",
            "model_id": "resnet18",
            "attack_type": "pgd",
            "parameters": {
                "epsilon": 0.05,
                "alpha": 0.01,
                "steps": 5,
            },
            "summary": {
                "total_images": 2,
                "changed_predictions": 1,
                "unchanged_predictions": 1,
                "attack_success_rate": 0.5,
                "average_original_confidence": 0.9,
                "average_adversarial_confidence": 0.65,
                "average_confidence_difference": 0.25,
                "average_perturbation_linf": 0.05,
                "average_perturbation_l2": 1.35,
                "labeled_images": 2,
                "original_accuracy": 1.0,
                "adversarial_accuracy": 0.5,
                "accuracy_drop": 0.5,
                "correct_before_attack": 2,
                "correct_after_attack": 1,
                "correct_before_wrong_after": 1,
            },
            "results": [
                {
                    "image_id": "image-1",
                    "filename": "image-1.png",
                    "original_image_url": "/datasets/dataset-1/images/image-1",
                    "adversarial_image_url": "/results/evaluation-result-1/image-1-adversarial.png",
                    "perturbation_image_url": "/results/evaluation-result-1/image-1-perturbation.png",
                    "ground_truth_label": "tabby cat",
                    "original_label": "tabby cat",
                    "adversarial_label": "tiger cat",
                    "original_confidence": 0.92,
                    "adversarial_confidence": 0.66,
                    "confidence_difference": 0.26,
                    "prediction_changed": True,
                    "attack_success": True,
                    "original_correct": True,
                    "adversarial_correct": False,
                    "perturbation_linf": 0.05,
                    "perturbation_l2": 1.4,
                },
                {
                    "image_id": "image-2",
                    "filename": "image-2.png",
                    "original_image_url": "/datasets/dataset-1/images/image-2",
                    "adversarial_image_url": "/results/evaluation-result-1/image-2-adversarial.png",
                    "perturbation_image_url": "/results/evaluation-result-1/image-2-perturbation.png",
                    "ground_truth_label": "golden retriever",
                    "original_label": "golden retriever",
                    "adversarial_label": "golden retriever",
                    "original_confidence": 0.88,
                    "adversarial_confidence": 0.64,
                    "confidence_difference": 0.24,
                    "prediction_changed": False,
                    "attack_success": False,
                    "original_correct": True,
                    "adversarial_correct": True,
                    "perturbation_linf": 0.05,
                    "perturbation_l2": 1.3,
                },
            ],
        }

    monkeypatch.setattr(evaluate_route, "evaluate_dataset", fake_evaluate_dataset)

    response = client.post(
        "/evaluate",
        headers=auth_headers,
        json={
            "dataset_id": "dataset-1",
            "attack_type": "pgd",
            "parameters": {
                "epsilon": 0.05,
                "alpha": 0.01,
                "steps": 5,
            },
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["result_id"] == "evaluation-result-1"
    assert data["dataset_id"] == "dataset-1"
    assert data["model_id"] == "resnet18"
    assert data["attack_type"] == "pgd"

    summary = data["summary"]

    assert summary["total_images"] == 2
    assert summary["changed_predictions"] == 1
    assert summary["unchanged_predictions"] == 1
    assert summary["attack_success_rate"] == 0.5
    assert summary["original_accuracy"] == 1.0
    assert summary["adversarial_accuracy"] == 0.5
    assert summary["accuracy_drop"] == 0.5

    assert len(data["results"]) == 2
    assert data["results"][0]["attack_success"] is True
    assert data["results"][1]["attack_success"] is False


def test_evaluate_uses_default_attack_parameters(client, monkeypatch, auth_headers):
    async def fake_evaluate_dataset(request, **kwargs):
        assert request.attack_type.value == "fgsm"
        assert request.parameters.epsilon == 0.03
        assert request.parameters.alpha == 0.01
        assert request.parameters.steps == 10

        return {
            "result_id": "evaluation-defaults",
            "dataset_id": request.dataset_id,
            "model_id": "resnet18",
            "attack_type": "fgsm",
            "parameters": {
                "epsilon": request.parameters.epsilon,
                "alpha": request.parameters.alpha,
                "steps": request.parameters.steps,
            },
            "summary": {
                "total_images": 1,
                "changed_predictions": 0,
                "unchanged_predictions": 1,
                "attack_success_rate": 0.0,
                "average_original_confidence": 0.9,
                "average_adversarial_confidence": 0.8,
                "average_confidence_difference": 0.1,
                "average_perturbation_linf": 0.03,
                "average_perturbation_l2": 1.0,
                "labeled_images": 0,
                "original_accuracy": None,
                "adversarial_accuracy": None,
                "accuracy_drop": None,
                "correct_before_attack": None,
                "correct_after_attack": None,
                "correct_before_wrong_after": None,
            },
            "results": [
                {
                    "image_id": "image-1",
                    "filename": "image-1.png",
                    "original_image_url": None,
                    "adversarial_image_url": None,
                    "perturbation_image_url": None,
                    "ground_truth_label": None,
                    "original_label": "tabby cat",
                    "adversarial_label": "tabby cat",
                    "original_confidence": 0.9,
                    "adversarial_confidence": 0.8,
                    "confidence_difference": 0.1,
                    "prediction_changed": False,
                    "attack_success": False,
                    "original_correct": None,
                    "adversarial_correct": None,
                    "perturbation_linf": 0.03,
                    "perturbation_l2": 1.0,
                },
            ],
        }

    monkeypatch.setattr(evaluate_route, "evaluate_dataset", fake_evaluate_dataset)

    response = client.post(
        "/evaluate",
        headers=auth_headers,
        json={
            "dataset_id": "dataset-1",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["attack_type"] == "fgsm"
    assert data["parameters"]["epsilon"] == 0.03
    assert data["parameters"]["alpha"] == 0.01
    assert data["parameters"]["steps"] == 10


def test_evaluate_with_missing_dataset_id_returns_422(client, auth_headers):
    response = client.post(
        "/evaluate",
        headers=auth_headers,
        json={
            "attack_type": "fgsm",
        },
    )

    assert response.status_code == 422


def test_evaluate_with_invalid_attack_type_returns_422(client, auth_headers):
    response = client.post(
        "/evaluate",
        headers=auth_headers,
        json={
            "dataset_id": "dataset-1",
            "attack_type": "invalid_attack",
        },
    )

    assert response.status_code == 422


def test_evaluate_propagates_service_error(client, monkeypatch, auth_headers):
    async def fake_evaluate_dataset(request, **kwargs):
        raise HTTPException(
            status_code=400,
            detail="Endpoint /evaluate currently supports classification models only.",
        )

    monkeypatch.setattr(evaluate_route, "evaluate_dataset", fake_evaluate_dataset)

    response = client.post(
        "/evaluate",
        headers=auth_headers,
        json={
            "dataset_id": "dataset-1",
            "attack_type": "fgsm",
        },
    )

    assert response.status_code == 400
    assert "classification models only" in response.json()["detail"]


def test_evaluate_stream_emits_progress_and_final_result(
    client,
    monkeypatch,
    auth_headers,
):
    from schemas.evaluation import EvaluationResponse, EvaluationSummary
    from services.evaluation_progress import EvaluationProgress

    async def fake_evaluate_dataset(request, **kwargs):
        report = kwargs["progress_callback"]
        report(EvaluationProgress(processed=0, total=1))
        report(EvaluationProgress(processed=1, total=1, phase="metrics"))
        return EvaluationResponse(
            result_id="evaluation-stream-1",
            dataset_id=request.dataset_id,
            model_id="resnet18",
            model_name="ResNet-18 · preentrenado en ImageNet-1K",
            attack_type=request.attack_type,
            parameters=request.parameters,
            summary=EvaluationSummary(
                total_images=0,
                changed_predictions=0,
                unchanged_predictions=0,
                attack_success_rate=0.0,
                average_original_confidence=0.0,
                average_adversarial_confidence=0.0,
                average_confidence_difference=0.0,
                average_perturbation_linf=0.0,
                average_perturbation_l2=0.0,
            ),
            results=[],
        )

    monkeypatch.setattr(evaluate_route, "evaluate_dataset", fake_evaluate_dataset)

    response = client.post(
        "/evaluate/stream",
        headers=auth_headers,
        json={"dataset_id": "dataset-1", "attack_type": "fgsm"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    events = [json.loads(line) for line in response.text.splitlines()]
    assert [event["type"] for event in events] == ["progress", "progress", "result"]
    assert events[0]["processed"] == 0
    assert events[1]["phase"] == "metrics"
    assert events[2]["data"]["result_id"] == "evaluation-stream-1"
