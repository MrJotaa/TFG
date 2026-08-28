import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path


def _temporary_result(created_dataset):
    result_id = f"gradcam_{uuid.uuid4()}"
    result_path = Path("uploads/results") / result_id
    result_path.mkdir(parents=True, exist_ok=False)
    (result_path / "gradcam.png").write_bytes(b"artifact")
    (result_path / "metadata.json").write_text(
        json.dumps(
            {
                "result_id": result_id,
                "result_type": "gradcam",
                "dataset_id": created_dataset["id"],
                "image_id": created_dataset["images"][0]["id"],
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        ),
        encoding="utf-8",
    )
    return result_id, result_path


def test_result_artifact_requires_artifact_session_cookie(
    client,
    registered_user,
    created_dataset,
):
    result_id, result_path = _temporary_result(created_dataset)
    try:
        client.cookies.clear()
        response = client.get(f"/results/{result_id}/gradcam.png")
        assert response.status_code == 401
    finally:
        shutil.rmtree(result_path, ignore_errors=True)


def test_result_artifact_is_available_to_dataset_owner(
    client,
    registered_user,
    created_dataset,
):
    result_id, result_path = _temporary_result(created_dataset)
    try:
        response = client.get(f"/results/{result_id}/gradcam.png")
        assert response.status_code == 200
        assert response.content == b"artifact"
        assert response.headers["cache-control"] == "private, no-store, max-age=0"
    finally:
        shutil.rmtree(result_path, ignore_errors=True)


def test_result_artifact_is_not_available_to_another_user(
    client,
    registered_user,
    created_dataset,
):
    result_id, result_path = _temporary_result(created_dataset)
    try:
        second_signup = client.post(
            "/auth/signup",
            json={
                "name": "Other User",
                "email": "other@example.com",
                "password": "secure-password-456",
            },
        )
        assert second_signup.status_code == 200

        response = client.get(f"/results/{result_id}/gradcam.png")
        assert response.status_code == 410
    finally:
        shutil.rmtree(result_path, ignore_errors=True)
