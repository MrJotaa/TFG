from fastapi import HTTPException

import api.routes.predict as predict_route


def test_predict_returns_prediction_response(client, monkeypatch):
    async def fake_predict_image(dataset_id: str, image_id: str, **kwargs):
        assert dataset_id == "dataset-1"
        assert image_id == "image-1"

        return {
            "prediction": "tabby cat",
            "confidence": 0.9342,
            "top_k": [
                {
                    "label": "tabby cat",
                    "probability": 0.9342,
                },
                {
                    "label": "tiger cat",
                    "probability": 0.0411,
                },
            ],
        }

    monkeypatch.setattr(predict_route, "predict_image", fake_predict_image)

    response = client.post(
        "/predict",
        json={
            "dataset_id": "dataset-1",
            "image_id": "image-1",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] == "tabby cat"
    assert data["confidence"] == 0.9342
    assert len(data["top_k"]) == 2
    assert data["top_k"][0]["label"] == "tabby cat"
    assert data["top_k"][0]["probability"] == 0.9342


def test_predict_with_missing_image_id_returns_422(client):
    response = client.post(
        "/predict",
        json={
            "dataset_id": "dataset-1",
        },
    )

    assert response.status_code == 422


def test_predict_propagates_service_error(client, monkeypatch):
    async def fake_predict_image(dataset_id: str, image_id: str, **kwargs):
        raise HTTPException(status_code=404, detail="Image not found")

    monkeypatch.setattr(predict_route, "predict_image", fake_predict_image)

    response = client.post(
        "/predict",
        json={
            "dataset_id": "dataset-1",
            "image_id": "missing-image",
        },
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Image not found"
