def test_create_user_returns_user_without_password(client):
    response = client.post(
        "/users",
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


def test_create_user_with_existing_email_returns_400(client, test_user_payload):
    first_response = client.post("/users", json=test_user_payload)
    assert first_response.status_code == 200

    second_response = client.post("/users", json=test_user_payload)

    assert second_response.status_code == 400
    assert "already exists" in second_response.json()["detail"].lower()


def test_create_user_with_invalid_email_returns_422(client):
    response = client.post(
        "/users",
        json={
            "name": "Invalid Email User",
            "email": "not-an-email",
            "password": "secure-password-123",
        },
    )

    assert response.status_code == 422


def test_list_users_returns_created_users(client):
    first_response = client.post(
        "/users",
        json={
            "name": "User One",
            "email": "one@example.com",
            "password": "secure-password-123",
        },
    )
    assert first_response.status_code == 200

    second_response = client.post(
        "/users",
        json={
            "name": "User Two",
            "email": "two@example.com",
            "password": "secure-password-123",
        },
    )
    assert second_response.status_code == 200

    response = client.get("/users")

    assert response.status_code == 200

    users = response.json()
    emails = {user["email"] for user in users}

    assert emails == {"one@example.com", "two@example.com"}

    for user in users:
        assert "password" not in user


def test_get_user_by_id_returns_user(client, registered_user):
    user_id = registered_user["user"]["id"]

    response = client.get(f"/users/{user_id}")

    assert response.status_code == 200

    data = response.json()

    assert data["id"] == user_id
    assert data["name"] == registered_user["user"]["name"]
    assert data["email"] == registered_user["user"]["email"]


def test_get_user_with_invalid_id_returns_400(client):
    response = client.get("/users/invalid-id")

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid user id"


def test_get_user_with_unknown_id_returns_404(client):
    response = client.get("/users/507f1f77bcf86cd799439011")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_update_user_name_returns_updated_user(client, registered_user):
    user_id = registered_user["user"]["id"]

    response = client.patch(
        f"/users/{user_id}",
        json={
            "name": "Updated Name",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["id"] == user_id
    assert data["name"] == "Updated Name"
    assert data["email"] == registered_user["user"]["email"]


def test_update_user_email_returns_updated_user(client, registered_user):
    user_id = registered_user["user"]["id"]

    response = client.patch(
        f"/users/{user_id}",
        json={
            "email": "updated@example.com",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["id"] == user_id
    assert data["email"] == "updated@example.com"


def test_update_user_password_allows_login_with_new_password(
    client,
    registered_user,
):
    user_id = registered_user["user"]["id"]

    response = client.patch(
        f"/users/{user_id}",
        json={
            "password": "new-secure-password-123",
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


def test_update_user_with_invalid_email_returns_422(client, registered_user):
    user_id = registered_user["user"]["id"]

    response = client.patch(
        f"/users/{user_id}",
        json={
            "email": "invalid-email",
        },
    )

    assert response.status_code == 422


def test_update_user_with_invalid_id_returns_400(client):
    response = client.patch(
        "/users/invalid-id",
        json={
            "name": "Updated",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid user id"


def test_update_unknown_user_returns_404(client):
    response = client.patch(
        "/users/507f1f77bcf86cd799439011",
        json={
            "name": "Updated",
        },
    )

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_delete_user_removes_user(client, registered_user):
    user_id = registered_user["user"]["id"]

    response = client.delete(f"/users/{user_id}")

    assert response.status_code == 200
    assert response.json() == {"message": "User deleted successfully"}

    get_response = client.get(f"/users/{user_id}")

    assert get_response.status_code == 404


def test_delete_user_with_invalid_id_returns_400(client):
    response = client.delete("/users/invalid-id")

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid user id"


def test_delete_unknown_user_returns_404(client):
    response = client.delete("/users/507f1f77bcf86cd799439011")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_get_user_datasets_returns_empty_list_for_user_without_datasets(
    client,
    registered_user,
):
    user_id = registered_user["user"]["id"]

    response = client.get(f"/users/{user_id}/datasets")

    assert response.status_code == 200
    assert response.json() == []


def test_get_user_datasets_with_invalid_user_id_returns_400(client):
    response = client.get("/users/invalid-id/datasets")

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid user id"

def test_get_user_datasets_for_unknown_user_returns_404(client):
    response = client.get(
        "/users/507f1f77bcf86cd799439011/datasets"
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "User not found"