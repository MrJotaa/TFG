import pytest

INVALID_OBJECT_ID = "invalid-id"
UNKNOWN_OBJECT_ID = "507f1f77bcf86cd799439011"


@pytest.mark.parametrize(
    (
        "method",
        "path",
        "payload",
        "expected_detail",
    ),
    [
        (
            "GET",
            f"/users/{INVALID_OBJECT_ID}",
            None,
            "Invalid user id",
        ),
        (
            "GET",
            f"/users/{INVALID_OBJECT_ID}/datasets",
            None,
            "Invalid user id",
        ),
        (
            "GET",
            f"/datasets/{INVALID_OBJECT_ID}",
            None,
            "Invalid dataset id",
        ),
    ],
)
def test_invalid_resource_ids_return_standardized_400(
    client,
    method,
    path,
    payload,
    expected_detail,
):
    request_arguments = {}

    if payload is not None:
        request_arguments["json"] = payload

    response = client.request(
        method,
        path,
        **request_arguments,
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": expected_detail,
    }


@pytest.mark.parametrize(
    (
        "method",
        "path",
        "payload",
        "expected_detail",
    ),
    [
        (
            "GET",
            f"/users/{UNKNOWN_OBJECT_ID}",
            None,
            "User not found",
        ),
        (
            "GET",
            f"/users/{UNKNOWN_OBJECT_ID}/datasets",
            None,
            "User not found",
        ),
        (
            "GET",
            f"/datasets/{UNKNOWN_OBJECT_ID}",
            None,
            "Dataset not found",
        ),
        (
            "POST",
            "/models/unknown-model/select",
            None,
            "Model not found",
        ),
    ],
)
def test_missing_resources_return_standardized_404(
    client,
    method,
    path,
    payload,
    expected_detail,
):
    request_arguments = {}

    if payload is not None:
        request_arguments["json"] = payload

    response = client.request(
        method,
        path,
        **request_arguments,
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": expected_detail,
    }


@pytest.mark.parametrize(
    (
        "path",
        "payload",
    ),
    [
        (
            "/predict",
            {
                "dataset_id": INVALID_OBJECT_ID,
                "image_id": "image-1",
            },
        ),
        (
            "/attack",
            {
                "dataset_id": INVALID_OBJECT_ID,
                "image_id": "image-1",
                "attack_type": "random_noise",
            },
        ),
        (
            "/evaluate",
            {
                "dataset_id": INVALID_OBJECT_ID,
                "attack_type": "random_noise",
            },
        ),
        (
            "/detect",
            {
                "dataset_id": INVALID_OBJECT_ID,
                "image_id": "image-1",
            },
        ),
        (
            "/detect/attack",
            {
                "dataset_id": INVALID_OBJECT_ID,
                "image_id": "image-1",
                "attack_type": "random_noise",
            },
        ),
        (
            "/detect/evaluate",
            {
                "dataset_id": INVALID_OBJECT_ID,
                "attack_type": "random_noise",
            },
        ),
        (
            "/explain/gradcam",
            {
                "dataset_id": INVALID_OBJECT_ID,
                "image_id": "image-1",
            },
        ),
        (
            "/explain/detection",
            {
                "dataset_id": INVALID_OBJECT_ID,
                "image_id": "image-1",
            },
        ),
    ],
)
def test_dataset_endpoints_return_400_for_invalid_dataset_id(
    client,
    path,
    payload,
):
    response = client.post(
        path,
        json=payload,
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Invalid dataset id",
    }


@pytest.mark.parametrize(
    (
        "path",
        "payload",
    ),
    [
        (
            "/predict",
            {
                "dataset_id": UNKNOWN_OBJECT_ID,
                "image_id": "image-1",
            },
        ),
        (
            "/attack",
            {
                "dataset_id": UNKNOWN_OBJECT_ID,
                "image_id": "image-1",
                "attack_type": "random_noise",
            },
        ),
        (
            "/evaluate",
            {
                "dataset_id": UNKNOWN_OBJECT_ID,
                "attack_type": "random_noise",
            },
        ),
        (
            "/detect",
            {
                "dataset_id": UNKNOWN_OBJECT_ID,
                "image_id": "image-1",
            },
        ),
        (
            "/detect/attack",
            {
                "dataset_id": UNKNOWN_OBJECT_ID,
                "image_id": "image-1",
                "attack_type": "random_noise",
            },
        ),
        (
            "/detect/evaluate",
            {
                "dataset_id": UNKNOWN_OBJECT_ID,
                "attack_type": "random_noise",
            },
        ),
        (
            "/explain/gradcam",
            {
                "dataset_id": UNKNOWN_OBJECT_ID,
                "image_id": "image-1",
            },
        ),
        (
            "/explain/detection",
            {
                "dataset_id": UNKNOWN_OBJECT_ID,
                "image_id": "image-1",
            },
        ),
    ],
)
def test_dataset_endpoints_return_404_for_missing_dataset(
    client,
    path,
    payload,
):
    response = client.post(
        path,
        json=payload,
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Dataset not found",
    }


def test_login_with_unknown_user_returns_standardized_401(
    client,
):
    response = client.post(
        "/auth/login",
        json={
            "email": "unknown@example.com",
            "password": "invalid-password",
        },
    )

    assert response.status_code == 401
    assert response.json() == {
        "detail": "Invalid email or password",
    }


def test_invalid_token_returns_standardized_401(client):
    response = client.get(
        "/auth/me",
        headers={
            "Authorization": "Bearer invalid-token",
        },
    )

    assert response.status_code == 401
    assert response.json() == {
        "detail": "Invalid or expired token",
    }
