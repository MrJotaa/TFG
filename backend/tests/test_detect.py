from fastapi import HTTPException
import pytest

import api.routes.detect as detect_route


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


def test_detect_returns_detection_response(client, monkeypatch):
    async def fake_detect_objects(request, **kwargs):
        assert request.dataset_id == "dataset-1"
        assert request.image_id == "image-1"

        return {
            "model_id": "yolo26n",
            "model_name": "YOLO26n · preentrenado en COCO",
            "detections": [
                _mock_detection("car", 0.91),
                _mock_detection("person", 0.84),
            ],
            "annotated_image_url": "/results/detection-result-1/annotated.png",
        }

    monkeypatch.setattr(detect_route, "detect_objects", fake_detect_objects)

    response = client.post(
        "/detect",
        json={
            "dataset_id": "dataset-1",
            "image_id": "image-1",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["model_id"] == "yolo26n"
    assert data["model_name"] == "YOLO26n · preentrenado en COCO"
    assert data["annotated_image_url"] == "/results/detection-result-1/annotated.png"
    assert len(data["detections"]) == 2

    first_detection = data["detections"][0]

    assert first_detection["label"] == "car"
    assert first_detection["confidence"] == 0.91
    assert first_detection["bbox"]["x1"] == 10.0
    assert first_detection["bbox"]["y1"] == 20.0
    assert first_detection["bbox"]["x2"] == 120.0
    assert first_detection["bbox"]["y2"] == 160.0


def test_detect_with_missing_image_id_returns_422(client):
    response = client.post(
        "/detect",
        json={
            "dataset_id": "dataset-1",
        },
    )

    assert response.status_code == 422


def test_detect_propagates_dataset_error(client, monkeypatch):
    async def fake_detect_objects(request, **kwargs):
        raise HTTPException(status_code=404, detail="Image not found")

    monkeypatch.setattr(detect_route, "detect_objects", fake_detect_objects)

    response = client.post(
        "/detect",
        json={
            "dataset_id": "dataset-1",
            "image_id": "missing-image",
        },
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Image not found"


def test_detect_propagates_incompatible_model_error(
    client,
    monkeypatch,
):
    async def fake_detect_objects(request, **kwargs):
        raise HTTPException(
            status_code=400,
            detail=("Endpoint '/detect' requires a detection model"),
        )

    monkeypatch.setattr(
        detect_route,
        "detect_objects",
        fake_detect_objects,
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


pytestmark = pytest.mark.usefixtures("authenticated_client")
