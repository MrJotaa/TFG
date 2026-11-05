import os
import shutil
import sys
import tempfile
from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pymongo import MongoClient

TEST_DB_NAME = "tfg_adversarial_ai_test"

# Configure every writable storage path before importing the FastAPI app.
# A TestClient starts the result cleanup task, which must never inspect or
# delete the real application files mounted at /app/uploads in Docker.
_TEST_STORAGE = tempfile.TemporaryDirectory(prefix="tfg-pytest-uploads-")
_TEST_STORAGE_ROOT = Path(_TEST_STORAGE.name).resolve()
_APPLICATION_UPLOADS = (Path(__file__).resolve().parents[1] / "uploads").resolve()
if (
    _TEST_STORAGE_ROOT == _APPLICATION_UPLOADS
    or _APPLICATION_UPLOADS in _TEST_STORAGE_ROOT.parents
):
    _TEST_STORAGE.cleanup()
    raise RuntimeError(
        "Test storage must be outside the application uploads directory"
    )

TEST_UPLOADS_DATASETS_PATH = str(_TEST_STORAGE_ROOT / "datasets")
TEST_RESULTS_BASE_PATH = str(_TEST_STORAGE_ROOT / "results")
TEST_DETECTION_EVALUATIONS_BASE_PATH = str(_TEST_STORAGE_ROOT / "evaluations")

os.environ["JWT_SECRET_KEY"] = "test-secret-key"
os.environ["MONGO_DB_NAME"] = TEST_DB_NAME
os.environ["UPLOADS_DATASETS_PATH"] = TEST_UPLOADS_DATASETS_PATH
os.environ["RESULTS_BASE_PATH"] = TEST_RESULTS_BASE_PATH
os.environ["DETECTION_EVALUATIONS_BASE_PATH"] = TEST_DETECTION_EVALUATIONS_BASE_PATH

for directory in (
    TEST_UPLOADS_DATASETS_PATH,
    TEST_RESULTS_BASE_PATH,
    TEST_DETECTION_EVALUATIONS_BASE_PATH,
):
    Path(directory).mkdir(parents=True, exist_ok=True)


from main import app  # noqa: E402

# Fail before TestClient can run startup cleanup if a module was imported with
# a production path before pytest configured its temporary storage.
for module_name, attribute, expected in (
    ("services.dataset_service", "UPLOADS_DATASETS_PATH", TEST_UPLOADS_DATASETS_PATH),
    ("utils.result_files", "RESULTS_BASE_PATH", TEST_RESULTS_BASE_PATH),
    (
        "services.evaluation_history_service",
        "DETECTION_EVALUATIONS_BASE_PATH",
        TEST_DETECTION_EVALUATIONS_BASE_PATH,
    ),
    (
        "services.detection_evaluation_persistence",
        "DETECTION_EVALUATIONS_BASE_PATH",
        TEST_DETECTION_EVALUATIONS_BASE_PATH,
    ),
):
    loaded_module = sys.modules.get(module_name)
    if loaded_module is None:
        continue
    actual_path = Path(getattr(loaded_module, attribute)).resolve()
    if actual_path != Path(expected).resolve():
        raise RuntimeError(f"Unsafe test storage configuration: {module_name}")


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
def created_dataset(client, registered_user, image_file_factory, auth_headers):
    user_id = registered_user["user"]["id"]

    response = client.post(
        "/datasets",
        headers=auth_headers,
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


@pytest.fixture
def authenticated_client(client, auth_headers):
    """Authenticate legacy route tests without changing the anonymous client fixture."""
    original = client.headers.get("Authorization")
    client.headers["Authorization"] = auth_headers["Authorization"]
    try:
        yield client
    finally:
        if original is None:
            client.headers.pop("Authorization", None)
        else:
            client.headers["Authorization"] = original
