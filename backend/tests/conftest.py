import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pymongo import MongoClient


TEST_DB_NAME = "tfg_adversarial_ai_test"

os.environ["JWT_SECRET_KEY"] = "test-secret-key"
os.environ["MONGO_DB_NAME"] = TEST_DB_NAME

Path("uploads/results").mkdir(parents=True, exist_ok=True)


from main import app  # noqa: E402


def _get_sync_mongo_client() -> MongoClient:
    mongo_uri = (
        os.environ.get("MONGO_URI")
        or os.environ.get("MONGODB_URI")
        or os.environ.get("MONGO_URL")
        or "mongodb://mongo:27017"
    )

    return MongoClient(mongo_uri)


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def clean_test_database():
    if os.environ["MONGO_DB_NAME"] != TEST_DB_NAME:
        raise RuntimeError("Tests must run against the test database")

    mongo_client = _get_sync_mongo_client()

    mongo_client.drop_database(TEST_DB_NAME)

    yield

    mongo_client.drop_database(TEST_DB_NAME)
    mongo_client.close()


@pytest.fixture
def test_user_payload():
    return {
        "name": "Test User",
        "email": "test@example.com",
        "password": "secure-password-123",
    }


@pytest.fixture
def registered_user(client, test_user_payload):
    response = client.post("/auth/signup", json=test_user_payload)

    assert response.status_code == 200

    return response.json()


@pytest.fixture
def auth_headers(registered_user):
    token = registered_user["access_token"]

    return {
        "Authorization": f"Bearer {token}",
    }