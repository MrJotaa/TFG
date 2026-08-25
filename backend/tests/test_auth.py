def test_signup_creates_user_and_returns_token(client, test_user_payload):
    response = client.post("/auth/signup", json=test_user_payload)

    assert response.status_code == 200

    data = response.json()

    assert data["access_token"]
    assert data["token_type"] == "bearer"

    assert data["user"]["name"] == test_user_payload["name"]
    assert data["user"]["email"] == test_user_payload["email"]
    assert "password" not in data["user"]


def test_signup_with_existing_email_returns_400(client, test_user_payload):
    first_response = client.post("/auth/signup", json=test_user_payload)
    assert first_response.status_code == 200

    second_response = client.post("/auth/signup", json=test_user_payload)

    assert second_response.status_code == 400
    assert "already exists" in second_response.json()["detail"]


def test_login_returns_token_for_valid_credentials(client, test_user_payload):
    signup_response = client.post("/auth/signup", json=test_user_payload)
    assert signup_response.status_code == 200

    response = client.post(
        "/auth/login",
        json={
            "email": test_user_payload["email"],
            "password": test_user_payload["password"],
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["access_token"]
    assert data["token_type"] == "bearer"
    assert data["user"]["email"] == test_user_payload["email"]


def test_login_with_invalid_password_returns_401(client, test_user_payload):
    signup_response = client.post("/auth/signup", json=test_user_payload)
    assert signup_response.status_code == 200

    response = client.post(
        "/auth/login",
        json={
            "email": test_user_payload["email"],
            "password": "wrong-password",
        },
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Correo electrónico o contraseña no válidos"


def test_auth_me_returns_current_user(client, registered_user, auth_headers):
    response = client.get("/auth/me", headers=auth_headers)

    assert response.status_code == 200

    data = response.json()

    assert data["user"]["id"] == registered_user["user"]["id"]
    assert data["user"]["name"] == registered_user["user"]["name"]
    assert data["user"]["email"] == registered_user["user"]["email"]


def test_auth_me_without_token_returns_401(client):
    response = client.get("/auth/me")

    assert response.status_code == 401


def test_auth_me_with_invalid_token_returns_401(client):
    response = client.get(
        "/auth/me",
        headers={
            "Authorization": "Bearer invalid-token",
        },
    )

    assert response.status_code == 401


def test_delete_current_user_with_wrong_password_returns_400(
    client,
    auth_headers,
):
    response = client.request(
        "DELETE",
        "/users/me",
        headers=auth_headers,
        json={
            "password": "wrong-password",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Contraseña no válida"


def test_delete_current_user_with_valid_password_deletes_account(
    client,
    test_user_payload,
    auth_headers,
):
    response = client.request(
        "DELETE",
        "/users/me",
        headers=auth_headers,
        json={
            "password": test_user_payload["password"],
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "message": "User account deleted successfully",
    }

    me_response = client.get("/auth/me", headers=auth_headers)

    assert me_response.status_code == 401
    assert me_response.json()["detail"] == "Usuario no encontrado"
