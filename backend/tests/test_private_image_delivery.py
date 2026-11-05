"""Authenticated image delivery must not expose files from another account."""

import json
import os
import shutil
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pymongo import MongoClient

RESULTS = Path("uploads/results")


def _other_headers(client):
    signup = client.post(
        "/auth/signup",
        json={
            "name": "Other Viewer",
            "email": "other-viewer@example.com",
            "password": "another-password-123",
        },
    )
    assert signup.status_code == 200
    return {"Authorization": f"Bearer {signup.json()['access_token']}"}


def _temporary_artifact(metadata, *, prefix="gradcam"):
    result_id = f"{prefix}_{uuid.uuid4()}"
    result_path = RESULTS / result_id
    result_path.mkdir(parents=True, exist_ok=False)
    (result_path / "gradcam.png").write_bytes(b"artifact")
    (result_path / "metadata.json").write_text(
        json.dumps(
            {
                "result_id": result_id,
                "result_type": prefix,
                "created_at": datetime.now(timezone.utc).isoformat(),
                **metadata,
            }
        ),
        encoding="utf-8",
    )
    return result_id, result_path


def test_dataset_image_is_private(client, created_dataset, auth_headers):
    dataset_id = created_dataset["id"]
    image_id = created_dataset["images"][0]["id"]
    url = f"/datasets/{dataset_id}/images/{image_id}"

    assert client.get(url).status_code == 401
    assert (
        client.get(url, headers={"Authorization": "Bearer broken"}).status_code
        == 401
    )
    own = client.get(url, headers=auth_headers)
    assert own.status_code == 200
    assert own.content
    assert own.headers["cache-control"] == "private, no-store, max-age=0"
    assert client.get(url, headers=_other_headers(client)).status_code == 403


def test_temporary_artifact_requires_owner_and_never_exposes_json(
    client, auth_headers, registered_user
):
    result_id, result_path = _temporary_artifact(
        {"owner_id": registered_user["user"]["id"]}
    )
    url = f"/results/{result_id}/gradcam.png"
    try:
        assert client.get(url).status_code == 401
        assert (
            client.get(url, headers={"Authorization": "Bearer broken"}).status_code
            == 401
        )
        assert client.get(url, headers=_other_headers(client)).status_code == 404
        own = client.get(url, headers=auth_headers)
        assert own.status_code == 200
        assert own.content == b"artifact"
        assert own.headers["cache-control"] == "private, no-store, max-age=0"
        assert (
            client.get(
                f"/results/{result_id}/metadata.json", headers=auth_headers
            ).status_code
            == 404
        )
    finally:
        shutil.rmtree(result_path, ignore_errors=True)


def test_legacy_temporary_artifact_requires_ownership_of_source_dataset(
    client, created_dataset, auth_headers
):
    result_id, result_path = _temporary_artifact(
        {"dataset_id": created_dataset["id"]}
    )
    try:
        url = f"/results/{result_id}/gradcam.png"
        assert client.get(url, headers=auth_headers).status_code == 200
        assert client.get(url, headers=_other_headers(client)).status_code == 404
    finally:
        shutil.rmtree(result_path, ignore_errors=True)


def test_expired_private_artifact_is_gone_for_owner(
    client, auth_headers, registered_user
):
    result_id, result_path = _temporary_artifact(
        {
            "owner_id": registered_user["user"]["id"],
            "expires_at": (
                datetime.now(timezone.utc) - timedelta(minutes=1)
            ).isoformat(),
        }
    )
    try:
        url = f"/results/{result_id}/gradcam.png"
        assert client.get(url, headers=auth_headers).status_code == 410
        assert client.get(url, headers=_other_headers(client)).status_code == 404
    finally:
        shutil.rmtree(result_path, ignore_errors=True)


def _insert_document(collection, document):
    mongo_url = (
        os.environ.get("MONGO_URI")
        or os.environ.get("MONGODB_URI")
        or os.environ.get("MONGO_URL")
        or "mongodb://mongo:27017"
    )
    with MongoClient(mongo_url) as mongo:
        mongo[os.environ["MONGO_DB_NAME"]][collection].insert_one(document)


def test_preserved_evaluation_uses_database_owner_not_temporary_ttl(
    client, auth_headers, registered_user
):
    result_id, result_path = _temporary_artifact(
        {"expires_at": (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()},
        prefix="evaluation",
    )
    _insert_document(
        "detection_evaluations",
        {
            "_id": result_id,
            "owner_id": registered_user["user"]["id"],
            "status": "completed",
            "preserved": True,
        },
    )
    try:
        url = f"/results/{result_id}/gradcam.png"
        assert client.get(url, headers=auth_headers).status_code == 200
        assert client.get(url, headers=_other_headers(client)).status_code == 404
    finally:
        shutil.rmtree(result_path, ignore_errors=True)


def test_saved_individual_analysis_uses_database_owner_not_temporary_ttl(
    client, auth_headers, registered_user
):
    result_id, result_path = _temporary_artifact(
        {"expires_at": (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()},
        prefix="attack",
    )
    _insert_document(
        "individual_analyses",
        {
            "_id": f"individual_analysis_{uuid.uuid4()}",
            "source_result_id": result_id,
            "owner_id": registered_user["user"]["id"],
        },
    )
    try:
        url = f"/results/{result_id}/gradcam.png"
        assert client.get(url, headers=auth_headers).status_code == 200
        assert client.get(url, headers=_other_headers(client)).status_code == 404
    finally:
        shutil.rmtree(result_path, ignore_errors=True)
