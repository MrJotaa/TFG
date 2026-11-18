import pytest


@pytest.mark.parametrize(
    ("method", "path", "payload"),
    [
        ("GET", "/users", None),
        ("POST", "/users", {"name": "A", "email": "a@example.com", "password": "x"}),
        ("GET", "/users/507f1f77bcf86cd799439011", None),
        ("PATCH", "/users/me", {"name": "A"}),
        ("PATCH", "/users/507f1f77bcf86cd799439011", {"name": "A"}),
        ("DELETE", "/users/me", {"password": "x"}),
        ("DELETE", "/users/507f1f77bcf86cd799439011", {"password": "x"}),
        ("GET", "/users/507f1f77bcf86cd799439011/datasets", None),
        ("PATCH", "/users/me/retention", {"result_retention_seconds": 86400}),
    ],
)
def test_all_user_routes_require_jwt(client, method, path, payload):
    response = client.request(method, path, json=payload)

    assert response.status_code == 401


def test_create_user_requires_authentication(client, test_user_payload):
    response = client.post("/users", json=test_user_payload)

    assert response.status_code == 401


def test_create_user_returns_user_without_password(client, auth_headers):
    response = client.post(
        "/users",
        headers=auth_headers,
        json={
            "name": "Created User",
            "email": "created@example.com",
            "password": "secure-password-123",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["id"]
    assert data["name"] == "Created User"
    assert data["email"] == "created@example.com"
    assert "password" not in data


def test_create_user_with_existing_email_returns_400(
    client,
    auth_headers,
    test_user_payload,
):
    response = client.post(
        "/users",
        headers=auth_headers,
        json=test_user_payload,
    )

    assert response.status_code == 400
    assert "ya existe" in response.json()["detail"].lower()


def test_create_user_with_invalid_email_returns_422(client, auth_headers):
    response = client.post(
        "/users",
        headers=auth_headers,
        json={
            "name": "Invalid Email User",
            "email": "not-an-email",
            "password": "secure-password-123",
        },
    )

    assert response.status_code == 422


def test_list_users_requires_authentication(client):
    assert client.get("/users").status_code == 401


def test_list_users_only_returns_current_user(client, auth_headers, registered_user):
    another_user = client.post(
        "/auth/signup",
        json={
            "name": "Another User",
            "email": "another@example.com",
            "password": "another-password-123",
        },
    )
    assert another_user.status_code == 200

    response = client.get("/users", headers=auth_headers)

    assert response.status_code == 200
    assert response.json() == [registered_user["user"]]
    assert another_user.json()["user"]["id"] not in {
        user["id"] for user in response.json()
    }


def test_get_user_by_id_requires_authentication(client, registered_user):
    user_id = registered_user["user"]["id"]
    response = client.get(f"/users/{user_id}")

    assert response.status_code == 401


def test_get_user_by_id_returns_own_user(client, registered_user, auth_headers):
    user_id = registered_user["user"]["id"]
    response = client.get(f"/users/{user_id}", headers=auth_headers)

    assert response.status_code == 200
    data = response.json()
    assert data["id"] == user_id
    assert data["name"] == registered_user["user"]["name"]
    assert data["email"] == registered_user["user"]["email"]


def test_get_user_with_invalid_id_returns_400(client, auth_headers):
    response = client.get("/users/invalid-id", headers=auth_headers)

    assert response.status_code == 400
    assert response.json()["detail"] == "Identificador de usuario no válido"


def test_get_user_with_unknown_id_returns_404(client, auth_headers):
    response = client.get(
        "/users/507f1f77bcf86cd799439011",
        headers=auth_headers,
    )

    assert response.status_code == 404
    assert "no encontrado" in response.json()["detail"].lower()


@pytest.mark.parametrize("method", ["GET", "PATCH", "DELETE"])
def test_other_user_account_cannot_be_accessed(
    client,
    auth_headers,
    method,
):
    another_user = client.post(
        "/auth/signup",
        json={
            "name": "Another User",
            "email": "another@example.com",
            "password": "another-password-123",
        },
    )
    assert another_user.status_code == 200
    another_id = another_user.json()["user"]["id"]
    payload = (
        {"name": "Unauthorized Change"}
        if method == "PATCH"
        else {"password": "secure-password-123"}
        if method == "DELETE"
        else None
    )

    response = client.request(
        method,
        f"/users/{another_id}",
        headers=auth_headers,
        json=payload,
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Usuario no encontrado"
    unchanged_user = client.get(
        "/auth/me",
        headers={
            "Authorization": (f"Bearer {another_user.json()['access_token']}"),
        },
    )
    assert unchanged_user.status_code == 200
    assert unchanged_user.json()["user"]["name"] == "Another User"


def test_update_user_name_returns_updated_user(client, registered_user, auth_headers):
    user_id = registered_user["user"]["id"]
    response = client.patch(
        f"/users/{user_id}",
        headers=auth_headers,
        json={"name": "Updated Name"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["id"] == user_id
    assert data["name"] == "Updated Name"
    assert data["email"] == registered_user["user"]["email"]


def test_update_current_user_name_uses_token_identity(
    client,
    auth_headers,
    registered_user,
):
    response = client.patch(
        "/users/me",
        headers=auth_headers,
        json={"name": "My New Name"},
    )

    assert response.status_code == 200
    assert response.json()["id"] == registered_user["user"]["id"]
    assert response.json()["name"] == "My New Name"
    me = client.get("/auth/me", headers=auth_headers)
    assert me.json()["user"]["name"] == "My New Name"


def test_update_current_user_requires_authentication(client):
    response = client.patch("/users/me", json={"name": "Unauthorized"})

    assert response.status_code == 401


def test_update_user_email_returns_updated_user(client, registered_user, auth_headers):
    user_id = registered_user["user"]["id"]
    response = client.patch(
        f"/users/{user_id}",
        headers=auth_headers,
        json={"email": "updated@example.com"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["id"] == user_id
    assert data["email"] == "updated@example.com"


def test_update_user_password_allows_login_with_new_password(
    client,
    registered_user,
    auth_headers,
):
    response = client.patch(
        "/users/me",
        headers=auth_headers,
        json={
            "password": "new-secure-password-123",
            "current_password": "secure-password-123",
        },
    )

    assert response.status_code == 200
    login_response = client.post(
        "/auth/login",
        json={
            "email": registered_user["user"]["email"],
            "password": "new-secure-password-123",
        },
    )
    assert login_response.status_code == 200
    assert login_response.json()["access_token"]


def test_update_user_password_requires_current_password(client, auth_headers):
    response = client.patch(
        "/users/me",
        headers=auth_headers,
        json={"password": "new-secure-password-123"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Debes introducir tu contraseña actual para cambiarla"
    )


def test_update_user_password_rejects_wrong_current_password(
    client,
    auth_headers,
):
    response = client.patch(
        "/users/me",
        headers=auth_headers,
        json={
            "password": "new-secure-password-123",
            "current_password": "wrong-password",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Contraseña actual no válida"


def test_update_user_with_invalid_email_returns_422(client, auth_headers):
    response = client.patch(
        "/users/me",
        headers=auth_headers,
        json={"email": "invalid-email"},
    )

    assert response.status_code == 422


def test_update_user_with_invalid_id_returns_400(client, auth_headers):
    response = client.patch(
        "/users/invalid-id",
        headers=auth_headers,
        json={"name": "Updated"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Identificador de usuario no válido"


def test_update_unknown_user_returns_404(client, auth_headers):
    response = client.patch(
        "/users/507f1f77bcf86cd799439011",
        headers=auth_headers,
        json={"name": "Updated"},
    )

    assert response.status_code == 404
    assert "no encontrado" in response.json()["detail"].lower()


def test_delete_user_by_id_requires_authentication(client, registered_user):
    user_id = registered_user["user"]["id"]
    response = client.request(
        "DELETE",
        f"/users/{user_id}",
        json={"password": "secure-password-123"},
    )

    assert response.status_code == 401


def test_delete_user_by_id_requires_correct_password(
    client,
    auth_headers,
    registered_user,
):
    user_id = registered_user["user"]["id"]
    response = client.request(
        "DELETE",
        f"/users/{user_id}",
        headers=auth_headers,
        json={"password": "wrong-password"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Contraseña no válida"
    assert client.get("/auth/me", headers=auth_headers).status_code == 200


def test_delete_user_removes_user(client, auth_headers, registered_user):
    user_id = registered_user["user"]["id"]
    response = client.request(
        "DELETE",
        f"/users/{user_id}",
        headers=auth_headers,
        json={"password": "secure-password-123"},
    )

    assert response.status_code == 200
    assert response.json() == {"message": "User deleted successfully"}
    assert client.get("/auth/me", headers=auth_headers).status_code == 401


def test_delete_user_with_invalid_id_returns_400(client, auth_headers):
    response = client.request(
        "DELETE",
        "/users/invalid-id",
        headers=auth_headers,
        json={"password": "secure-password-123"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Identificador de usuario no válido"


def test_delete_unknown_user_returns_404(client, auth_headers):
    response = client.request(
        "DELETE",
        "/users/507f1f77bcf86cd799439011",
        headers=auth_headers,
        json={"password": "secure-password-123"},
    )

    assert response.status_code == 404
    assert "no encontrado" in response.json()["detail"].lower()


def test_get_user_datasets_requires_authentication(client, registered_user):
    user_id = registered_user["user"]["id"]
    response = client.get(f"/users/{user_id}/datasets")

    assert response.status_code == 401


def test_get_user_datasets_returns_empty_list_for_user_without_datasets(
    client,
    registered_user,
    auth_headers,
):
    user_id = registered_user["user"]["id"]
    response = client.get(f"/users/{user_id}/datasets", headers=auth_headers)

    assert response.status_code == 200
    assert response.json() == []


def test_get_other_user_datasets_returns_404(client, auth_headers):
    another_user = client.post(
        "/auth/signup",
        json={
            "name": "Another User",
            "email": "another@example.com",
            "password": "another-password-123",
        },
    )
    assert another_user.status_code == 200
    another_id = another_user.json()["user"]["id"]

    response = client.get(
        f"/users/{another_id}/datasets",
        headers=auth_headers,
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Usuario no encontrado"


def test_get_user_datasets_with_invalid_user_id_returns_400(client, auth_headers):
    response = client.get(
        "/users/invalid-id/datasets",
        headers=auth_headers,
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Identificador de usuario no válido"


def test_get_user_datasets_for_unknown_user_returns_404(client, auth_headers):
    response = client.get(
        "/users/507f1f77bcf86cd799439011/datasets",
        headers=auth_headers,
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Usuario no encontrado"


def test_invalid_token_cannot_read_user(client, registered_user):
    response = client.get(
        f"/users/{registered_user['user']['id']}",
        headers={"Authorization": "Bearer invalid-token"},
    )

    assert response.status_code == 401
