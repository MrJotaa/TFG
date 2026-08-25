from datetime import datetime, timedelta, timezone

import api.routes.evaluation_history as history_route


def _history_item(owner_dataset_id="dataset-1"):
    now = datetime.now(timezone.utc)
    return {
        "result_id": "evaluation_00000000-0000-4000-8000-000000000001",
        "task_type": "classification",
        "status": "completed",
        "dataset_id": owner_dataset_id,
        "dataset_name": "Validation set",
        "dataset_revision": 3,
        "model_id": "resnet18",
        "model_name": "ResNet-18 · preentrenado en ImageNet-1K",
        "attack_type": "fgsm",
        "parameters": {
            "epsilon": 0.03,
            "alpha": 0.01,
            "steps": 10,
            "seed": 0,
        },
        "summary": {"total_images": 2},
        "created_at": now,
        "completed_at": now,
        "expires_at": now + timedelta(hours=24),
        "retention_seconds": 86400,
        "exact_reuse_available": True,
    }


def test_history_requires_authentication(client):
    assert client.get("/evaluations").status_code == 401


def test_history_uses_authenticated_owner(
    client, auth_headers, registered_user, monkeypatch
):
    captured = {}

    async def fake_list(**kwargs):
        captured.update(kwargs)
        return {"items": [_history_item()], "total": 1, "limit": 10, "offset": 0}

    monkeypatch.setattr(history_route, "list_evaluation_history", fake_list)
    response = client.get("/evaluations?limit=10&offset=0", headers=auth_headers)

    assert response.status_code == 200
    assert response.json()["items"][0]["exact_reuse_available"] is True
    assert captured["owner_id"] == registered_user["user"]["id"]


def test_compatibility_uses_authenticated_owner(
    client, auth_headers, registered_user, monkeypatch
):
    captured = {}

    async def fake_find(**kwargs):
        captured.update(kwargs)
        return {"match": _history_item()}

    monkeypatch.setattr(history_route, "find_compatible_evaluation", fake_find)
    response = client.post(
        "/evaluations/compatibility",
        headers=auth_headers,
        json={
            "task_type": "classification",
            "dataset_id": "dataset-1",
            "attack_type": "fgsm",
            "parameters": {"epsilon": 0.03, "alpha": 0.01, "steps": 10},
        },
    )

    assert response.status_code == 200
    assert captured["owner_id"] == registered_user["user"]["id"]
    assert captured["request"].parameters.seed == 0


def test_retention_preference_is_validated_and_returned(client, auth_headers):
    response = client.patch(
        "/users/me/retention",
        headers=auth_headers,
        json={"result_retention_seconds": 604800},
    )
    assert response.status_code == 200
    assert response.json()["result_retention_seconds"] == 604800

    invalid = client.patch(
        "/users/me/retention",
        headers=auth_headers,
        json={"result_retention_seconds": 604801},
    )
    assert invalid.status_code == 422
