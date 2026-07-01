import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


os.environ["JWT_SECRET_KEY"] = "test-secret-key"
os.environ["MONGO_DB_NAME"] = "tfg_adversarial_ai_test"

Path("uploads/results").mkdir(parents=True, exist_ok=True)


from main import app  # noqa: E402


@pytest.fixture
def client():
    return TestClient(app)