"""End-to-end authorization checks with two real users and one real dataset.

No inference model is loaded: access to another user's dataset must be
rejected while the input image is being retrieved, before model execution.
"""

import pytest


@pytest.fixture
def other_user_headers(client):
    signup = client.post(
        "/auth/signup",
        json={
            "name": "Other User",
            "email": "other-jwt@example.com",
            "password": "other-password-123",
        },
    )
    assert signup.status_code == 200
    return {"Authorization": f"Bearer {signup.json()['access_token']}"}


@pytest.mark.parametrize(
    ("method", "path", "payload"),
    [
        ("GET", "/datasets", None),
        ("GET", "/datasets/507f1f77bcf86cd799439011", None),
        ("PATCH", "/datasets/507f1f77bcf86cd799439011", {"name": "X"}),
        ("DELETE", "/datasets/507f1f77bcf86cd799439011", None),
        ("GET", "/datasets/507f1f77bcf86cd799439011/images", None),
        ("GET", "/models", None),
        ("GET", "/models/active", None),
        ("GET", "/models/active/classes", None),
        ("POST", "/models/resnet18/select", None),
        ("GET", "/attacks", None),
        ("POST", "/predict", {"dataset_id": "not-an-id", "image_id": "image"}),
        (
            "POST",
            "/attack",
            {
                "dataset_id": "not-an-id",
                "image_id": "image",
                "attack_type": "random_noise",
            },
        ),
        ("POST", "/detect", {"dataset_id": "not-an-id", "image_id": "image"}),
        (
            "POST",
            "/detect/attack",
            {
                "dataset_id": "not-an-id",
                "image_id": "image",
                "attack_type": "random_noise",
            },
        ),
        ("POST", "/explain/gradcam", {"dataset_id": "not-an-id", "image_id": "image"}),
        (
            "POST",
            "/explain/detection",
            {"dataset_id": "not-an-id", "image_id": "image"},
        ),
    ],
)
def test_private_routes_reject_anonymous_requests(client, method, path, payload):
    response = client.request(method, path, json=payload)
    assert response.status_code == 401


def test_dataset_upload_requires_authentication(client, image_file_factory):
    response = client.post(
        "/datasets",
        data={"name": "Anonymous"},
        files=[image_file_factory("anonymous.png")],
    )
    assert response.status_code == 401


@pytest.mark.parametrize("path", ["/datasets", "/models", "/attacks"])
def test_private_routes_reject_invalid_jwt(client, path):
    response = client.get(path, headers={"Authorization": "Bearer invalid-token"})
    assert response.status_code == 401


def test_dataset_listing_is_limited_to_the_token_owner(
    client, created_dataset, auth_headers, other_user_headers
):
    mine = client.get("/datasets", headers=auth_headers)
    others = client.get("/datasets", headers=other_user_headers)
    assert mine.status_code == 200
    assert [item["id"] for item in mine.json()] == [created_dataset["id"]]
    assert others.status_code == 200
    assert others.json() == []


@pytest.mark.parametrize(
    ("method", "path_template", "payload", "upload"),
    [
        ("GET", "/datasets/{dataset_id}", None, False),
        ("PATCH", "/datasets/{dataset_id}", {"name": "Taken over"}, False),
        ("DELETE", "/datasets/{dataset_id}", None, False),
        ("GET", "/datasets/{dataset_id}/images", None, False),
        ("POST", "/datasets/{dataset_id}/images", None, True),
        ("DELETE", "/datasets/{dataset_id}/images/{image_id}", None, False),
        (
            "PATCH",
            "/datasets/{dataset_id}/images/{image_id}/label",
            {"label": "hijacked"},
            False,
        ),
        ("GET", "/datasets/{dataset_id}/label-resolution", None, False),
        ("GET", "/datasets/{dataset_id}/annotations/coverage", None, False),
    ],
)
def test_foreign_dataset_cannot_be_read_or_modified(
    client,
    created_dataset,
    auth_headers,
    other_user_headers,
    image_file_factory,
    method,
    path_template,
    payload,
    upload,
):
    path = path_template.format(
        dataset_id=created_dataset["id"],
        image_id=created_dataset["images"][0]["id"],
    )
    kwargs = {"headers": other_user_headers}
    if upload:
        kwargs["files"] = [image_file_factory("other.png")]
    elif payload is not None:
        kwargs["json"] = payload

    denied = client.request(method, path, **kwargs)
    assert denied.status_code == 403

    # A request from the foreign account never changes the original dataset.
    own = client.get(f"/datasets/{created_dataset['id']}", headers=auth_headers)
    assert own.status_code == 200
    assert own.json()["name"] == created_dataset["name"]
    assert own.json()["image_count"] == 1
    assert own.json()["images"][0]["label"] is None


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        ("/predict", {"attack_type": None}),
        ("/detect", {}),
        ("/attack", {"attack_type": "random_noise"}),
        ("/detect/attack", {"attack_type": "random_noise"}),
        ("/explain/gradcam", {}),
        ("/explain/detection", {}),
    ],
)
def test_individual_execution_rejects_foreign_images(
    client, created_dataset, other_user_headers, path, payload
):
    request = {
        "dataset_id": created_dataset["id"],
        "image_id": created_dataset["images"][0]["id"],
        **{key: value for key, value in payload.items() if value is not None},
    }
    response = client.post(path, headers=other_user_headers, json=request)
    assert response.status_code == 403


def test_upload_dataset_uses_token_owner_not_form_owner(
    client, registered_user, auth_headers, other_user_headers, image_file_factory
):
    # A foreign user id may not be used to create datasets on behalf of others.
    denied = client.post(
        "/datasets",
        headers=other_user_headers,
        data={"user_id": registered_user["user"]["id"], "name": "Not Mine"},
        files=[image_file_factory("rejected.png")],
    )
    assert denied.status_code == 403

    # The old request body is still accepted for legitimate clients.
    success = client.post(
        "/datasets",
        headers=auth_headers,
        data={"user_id": registered_user["user"]["id"], "name": "Mine"},
        files=[image_file_factory("mine.png")],
    )
    assert success.status_code == 200
    assert success.json()["owner_id"] == registered_user["user"]["id"]

    # New consumers may omit the old untrusted user_id field.
    without_user_id = client.post(
        "/datasets",
        headers=auth_headers,
        data={"name": "Mine too"},
        files=[image_file_factory("mine-again.png")],
    )
    assert without_user_id.status_code == 200
    assert without_user_id.json()["owner_id"] == registered_user["user"]["id"]
