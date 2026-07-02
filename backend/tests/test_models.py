import pytest

from services import model_service


@pytest.fixture(autouse=True)
def reset_model_registry(monkeypatch):
    """
    Evita cargar pesos reales durante los tests y deja el registro de modelos
    en un estado conocido antes de cada test.
    """
    for model_data in model_service._models.values():
        instance = model_data["instance"]

        instance._test_loaded = False

        def fake_load(instance=instance):
            instance._test_loaded = True

        def fake_unload(instance=instance):
            instance._test_loaded = False

        def fake_is_loaded(instance=instance):
            return instance._test_loaded

        monkeypatch.setattr(instance, "load", fake_load)
        monkeypatch.setattr(instance, "unload", fake_unload)
        monkeypatch.setattr(instance, "is_loaded", fake_is_loaded)

        model_data["active"] = False

    model_service._models["resnet18"]["active"] = True

    yield

    for model_data in model_service._models.values():
        model_data["active"] = False
        model_data["instance"]._test_loaded = False

    model_service._models["resnet18"]["active"] = True


def test_get_models_returns_registered_models(client):
    response = client.get("/models")

    assert response.status_code == 200

    models = response.json()
    model_ids = {model["id"] for model in models}

    assert model_ids == {"resnet18", "yolov8n"}

    for model in models:
        assert "id" in model
        assert "name" in model
        assert "active" in model
        assert "loaded" in model
        assert "task_type" in model

    resnet = next(model for model in models if model["id"] == "resnet18")
    yolo = next(model for model in models if model["id"] == "yolov8n")

    assert resnet["task_type"] == "classification"
    assert yolo["task_type"] == "detection"


def test_get_models_marks_resnet18_as_active_by_default(client):
    response = client.get("/models")

    assert response.status_code == 200

    models = response.json()

    active_models = [model for model in models if model["active"]]

    assert len(active_models) == 1
    assert active_models[0]["id"] == "resnet18"


def test_get_active_model_returns_default_active_model(client):
    response = client.get("/models/active")

    assert response.status_code == 200

    active_model = response.json()

    assert active_model["id"] == "resnet18"
    assert active_model["active"] is True
    assert active_model["task_type"] == "classification"


def test_select_yolo_model_sets_yolo_as_active(client):
    response = client.post("/models/yolov8n/select")

    assert response.status_code == 200

    selected_model = response.json()

    assert selected_model["id"] == "yolov8n"
    assert selected_model["active"] is True
    assert selected_model["task_type"] == "detection"

    models_response = client.get("/models")
    assert models_response.status_code == 200

    models = models_response.json()

    resnet = next(model for model in models if model["id"] == "resnet18")
    yolo = next(model for model in models if model["id"] == "yolov8n")

    assert resnet["active"] is False
    assert yolo["active"] is True


def test_select_resnet18_model_sets_resnet_as_active(client):
    yolo_response = client.post("/models/yolov8n/select")
    assert yolo_response.status_code == 200

    response = client.post("/models/resnet18/select")

    assert response.status_code == 200

    selected_model = response.json()

    assert selected_model["id"] == "resnet18"
    assert selected_model["active"] is True
    assert selected_model["task_type"] == "classification"

    models_response = client.get("/models")
    assert models_response.status_code == 200

    models = models_response.json()

    resnet = next(model for model in models if model["id"] == "resnet18")
    yolo = next(model for model in models if model["id"] == "yolov8n")

    assert resnet["active"] is True
    assert yolo["active"] is False


def test_select_model_loads_selected_model(client):
    response = client.post("/models/yolov8n/select")

    assert response.status_code == 200

    yolo_instance = model_service._models["yolov8n"]["instance"]

    assert yolo_instance.is_loaded() is True


def test_select_model_unloads_previous_model(client):
    resnet_instance = model_service._models["resnet18"]["instance"]
    resnet_instance.load()

    response = client.post("/models/yolov8n/select")

    assert response.status_code == 200

    assert resnet_instance.is_loaded() is False


def test_get_active_model_after_selecting_yolo_returns_yolo(client):
    select_response = client.post("/models/yolov8n/select")
    assert select_response.status_code == 200

    response = client.get("/models/active")

    assert response.status_code == 200

    active_model = response.json()

    assert active_model["id"] == "yolov8n"
    assert active_model["active"] is True
    assert active_model["task_type"] == "detection"


def test_select_unknown_model_returns_404(client):
    response = client.post("/models/unknown-model/select")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()

def test_select_already_active_model_loads_it(client):
    resnet_instance = model_service._models["resnet18"]["instance"]

    assert resnet_instance.is_loaded() is False

    response = client.post("/models/resnet18/select")

    assert response.status_code == 200

    selected_model = response.json()

    assert selected_model["id"] == "resnet18"
    assert selected_model["active"] is True
    assert selected_model["loaded"] is True
    assert resnet_instance.is_loaded() is True