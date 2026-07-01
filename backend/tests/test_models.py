def test_get_models_returns_registered_models(client):
    response = client.get("/models")

    assert response.status_code == 200

    models = response.json()
    model_ids = {model["id"] for model in models}

    assert "resnet18" in model_ids
    assert "yolov8n" in model_ids

    for model in models:
        assert "id" in model
        assert "name" in model
        assert "active" in model
        assert "loaded" in model
        assert "task_type" in model


def test_get_active_model_returns_one_model(client):
    response = client.get("/models/active")

    assert response.status_code == 200

    active_model = response.json()

    assert active_model["active"] is True
    assert active_model["id"] in {"resnet18", "yolov8n"}
    assert active_model["task_type"] in {"classification", "detection"}


def test_select_unknown_model_returns_404(client):
    response = client.post("/models/unknown-model/select")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"]