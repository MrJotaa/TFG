from datetime import datetime, timezone

import api.routes.individual_analysis_history as history_route


def _item():
    now = datetime.now(timezone.utc)
    return {
        "analysis_id": "individual_analysis_00000000-0000-4000-8000-000000000001",
        "source_result_id": "attack_00000000-0000-4000-8000-000000000001",
        "task_type": "classification",
        "dataset_id": "dataset-1",
        "dataset_name": "Imagenette",
        "dataset_revision": 3,
        "image_id": "image-1",
        "image_filename": "tench.jpg",
        "model_id": "resnet18",
        "model_name": "ResNet-18",
        "attack_type": "fgsm",
        "parameters": {"epsilon": 0.03},
        "summary": {"original_label": "tench", "adversarial_label": "church"},
        "explanation_available": True,
        "created_at": now,
        "saved_at": now,
    }


def _identity():
    return {
        "task_type": "classification",
        "dataset_id": "dataset-1",
        "dataset_name": "Imagenette",
        "dataset_revision": 3,
        "image_id": "image-1",
        "image_filename": "tench.jpg",
        "model_id": "resnet18",
        "model_name": "ResNet-18",
        "model_revision": "resnet18-v1",
        "attack_type": "fgsm",
        "attack_revision": "fgsm-v1",
        "parameters": {"epsilon": 0.03},
        "pipeline_version": "classification-individual-analysis-v1",
    }


def test_individual_analysis_history_requires_authentication(client):
    assert client.get("/individual-analyses").status_code == 401


def test_individual_analysis_save_uses_authenticated_owner(
    client, auth_headers, registered_user, monkeypatch
):
    captured = {}

    async def fake_save(**kwargs):
        captured.update(kwargs)
        return _item()

    monkeypatch.setattr(history_route, "save_individual_analysis", fake_save)
    response = client.post(
        "/individual-analyses",
        headers=auth_headers,
        json={
            "task_type": "classification",
            "result_id": "attack_00000000-0000-4000-8000-000000000001",
        },
    )

    assert response.status_code == 201
    assert captured["owner_id"] == registered_user["user"]["id"]


def test_individual_analysis_compatibility_uses_authenticated_owner(
    client, auth_headers, registered_user, monkeypatch
):
    captured = {}

    async def fake_find(**kwargs):
        captured.update(kwargs)
        return {"identity": _identity(), "match": _item()}

    monkeypatch.setattr(history_route, "find_compatible_individual_analysis", fake_find)
    response = client.post(
        "/individual-analyses/compatibility",
        headers=auth_headers,
        json={
            "task_type": "classification",
            "dataset_id": "dataset-1",
            "image_id": "image-1",
            "attack_type": "fgsm",
            "parameters": {"epsilon": 0.03},
        },
    )

    assert response.status_code == 200
    assert response.json()["identity"]["dataset_revision"] == 3
    assert response.json()["match"]["source_result_id"].startswith("attack_")
    assert captured["owner_id"] == registered_user["user"]["id"]
