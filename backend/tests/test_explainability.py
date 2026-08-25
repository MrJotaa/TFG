from fastapi import HTTPException

import api.routes.explainability as explainability_route


def _mock_detection(
    label: str = "car",
    confidence: float = 0.91,
):
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


def test_gradcam_returns_explainability_response(
    client,
    monkeypatch,
):
    async def fake_generate_gradcam_for_image(
        dataset_id: str,
        image_id: str,
        **kwargs,
    ):
        assert dataset_id == "dataset-1"
        assert image_id == "image-1"

        return {
            "result_id": "gradcam-result-1",
            "prediction": "tabby cat",
            "confidence": 0.9342,
            "gradcam_image_url": ("/results/gradcam-result-1/gradcam.png"),
        }

    monkeypatch.setattr(
        explainability_route,
        "generate_gradcam_for_image",
        fake_generate_gradcam_for_image,
    )

    response = client.post(
        "/explain/gradcam",
        json={
            "dataset_id": "dataset-1",
            "image_id": "image-1",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["result_id"] == "gradcam-result-1"
    assert data["prediction"] == "tabby cat"
    assert data["confidence"] == 0.9342
    assert data["gradcam_image_url"] == ("/results/gradcam-result-1/gradcam.png")


def test_gradcam_with_missing_image_id_returns_422(client):
    response = client.post(
        "/explain/gradcam",
        json={
            "dataset_id": "dataset-1",
        },
    )

    assert response.status_code == 422


def test_gradcam_propagates_dataset_error(
    client,
    monkeypatch,
):
    async def fake_generate_gradcam_for_image(
        dataset_id: str,
        image_id: str,
        **kwargs,
    ):
        assert dataset_id == "dataset-1"
        assert image_id == "missing-image"

        raise HTTPException(
            status_code=404,
            detail="Image not found",
        )

    monkeypatch.setattr(
        explainability_route,
        "generate_gradcam_for_image",
        fake_generate_gradcam_for_image,
    )

    response = client.post(
        "/explain/gradcam",
        json={
            "dataset_id": "dataset-1",
            "image_id": "missing-image",
        },
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Image not found"


def test_gradcam_propagates_incompatible_model_error(
    client,
    monkeypatch,
):
    async def fake_generate_gradcam_for_image(
        dataset_id: str,
        image_id: str,
        **kwargs,
    ):
        assert dataset_id == "dataset-1"
        assert image_id == "image-1"

        raise HTTPException(
            status_code=400,
            detail=("Endpoint '/explain/gradcam' requires a classification model"),
        )

    monkeypatch.setattr(
        explainability_route,
        "generate_gradcam_for_image",
        fake_generate_gradcam_for_image,
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


def test_detection_explainability_returns_response(
    client,
    monkeypatch,
):
    async def fake_generate_detection_explainability(request, **kwargs):
        assert request.dataset_id == "dataset-1"
        assert request.image_id == "image-1"

        return {
            "result_id": "detection-explain-result-1",
            "model_id": "yolo26n",
            "model_name": "YOLO26n · preentrenado en COCO",
            "explanation_type": "eigencam",
            "explanation_image_url": (
                "/results/detection-explain-result-1/detection_explanation.png"
            ),
            "detections": [
                _mock_detection("car", 0.91),
                _mock_detection("person", 0.84),
            ],
        }

    monkeypatch.setattr(
        explainability_route,
        "generate_detection_explainability",
        fake_generate_detection_explainability,
    )

    response = client.post(
        "/explain/detection",
        json={
            "dataset_id": "dataset-1",
            "image_id": "image-1",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["result_id"] == "detection-explain-result-1"
    assert data["model_id"] == "yolo26n"
    assert data["model_name"] == "YOLO26n · preentrenado en COCO"
    assert data["explanation_type"] == "eigencam"
    assert data["explanation_image_url"] == (
        "/results/detection-explain-result-1/detection_explanation.png"
    )

    assert len(data["detections"]) == 2
    assert data["detections"][0]["label"] == "car"
    assert data["detections"][0]["confidence"] == 0.91
    assert data["detections"][0]["bbox"]["x1"] == 10.0


def test_detection_explainability_with_missing_image_id_returns_422(
    client,
):
    response = client.post(
        "/explain/detection",
        json={
            "dataset_id": "dataset-1",
        },
    )

    assert response.status_code == 422


def test_detection_explainability_propagates_dataset_error(
    client,
    monkeypatch,
):
    async def fake_generate_detection_explainability(request, **kwargs):
        raise HTTPException(
            status_code=404,
            detail="Image not found",
        )

    monkeypatch.setattr(
        explainability_route,
        "generate_detection_explainability",
        fake_generate_detection_explainability,
    )

    response = client.post(
        "/explain/detection",
        json={
            "dataset_id": "dataset-1",
            "image_id": "missing-image",
        },
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Image not found"


def test_detection_explainability_propagates_incompatible_model_error(
    client,
    monkeypatch,
):
    async def fake_generate_detection_explainability(request, **kwargs):
        raise HTTPException(
            status_code=400,
            detail=("Endpoint '/explain/detection' requires a detection model"),
        )

    monkeypatch.setattr(
        explainability_route,
        "generate_detection_explainability",
        fake_generate_detection_explainability,
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


def test_detection_explainability_propagates_no_detections_error(
    client,
    monkeypatch,
):
    async def fake_generate_detection_explainability(request, **kwargs):
        raise HTTPException(
            status_code=400,
            detail="No detections found for this image.",
        )

    monkeypatch.setattr(
        explainability_route,
        "generate_detection_explainability",
        fake_generate_detection_explainability,
    )

    response = client.post(
        "/explain/detection",
        json={
            "dataset_id": "dataset-1",
            "image_id": "image-1",
        },
    )

    assert response.status_code == 400
    assert "no detections" in response.json()["detail"].lower()
