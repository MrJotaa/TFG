import os
import shutil
from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pymongo import MongoClient

TEST_DB_NAME = "tfg_adversarial_ai_test"
TEST_UPLOADS_DATASETS_PATH = "uploads/test_datasets"

os.environ["JWT_SECRET_KEY"] = "test-secret-key"
os.environ["MONGO_DB_NAME"] = TEST_DB_NAME
os.environ["UPLOADS_DATASETS_PATH"] = TEST_UPLOADS_DATASETS_PATH

Path("uploads/results").mkdir(parents=True, exist_ok=True)
Path(TEST_UPLOADS_DATASETS_PATH).mkdir(parents=True, exist_ok=True)


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

    shutil.rmtree(TEST_UPLOADS_DATASETS_PATH, ignore_errors=True)
    Path(TEST_UPLOADS_DATASETS_PATH).mkdir(parents=True, exist_ok=True)

    yield

    mongo_client.drop_database(TEST_DB_NAME)
    mongo_client.close()

    shutil.rmtree(TEST_UPLOADS_DATASETS_PATH, ignore_errors=True)
    Path(TEST_UPLOADS_DATASETS_PATH).mkdir(parents=True, exist_ok=True)


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


@pytest.fixture
def image_file_factory():
    def _create_image_file(
        filename: str = "test-image.png",
        color: tuple[int, int, int] = (255, 0, 0),
    ):
        image = Image.new("RGB", (32, 32), color=color)
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)

        return (
            "files",
            (filename, buffer, "image/png"),
        )

    return _create_image_file


@pytest.fixture
def created_dataset(client, registered_user, image_file_factory):
    user_id = registered_user["user"]["id"]

    response = client.post(
        "/datasets",
        data={
            "user_id": user_id,
            "name": "Test Dataset",
        },
        files=[
            image_file_factory("image-1.png"),
        ],
    )

    assert response.status_code == 200

    return response.json()
