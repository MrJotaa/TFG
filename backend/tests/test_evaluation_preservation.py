from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

import services.evaluation_history_service as history
from schemas.evaluation_history import EvaluationPreservationUpdate


class _Collection:
    def __init__(self, document):
        self.document = document

    async def find_one(self, query):
        if query.get("_id") == self.document["_id"]:
            return self.document
        return None

    async def update_one(self, query, update):
        if query.get("_id") == self.document["_id"]:
            self.document.update(update.get("$set", {}))
            return SimpleNamespace(matched_count=1)
        return SimpleNamespace(matched_count=0)


def _document():
    now = datetime.now(timezone.utc)
    return {
        "_id": "evaluation_00000000-0000-4000-8000-000000000001",
        "task_type": "classification",
        "storage_schema_version": history.EVALUATION_STORAGE_SCHEMA_VERSION,
        "status": "completed",
        "owner_id": "user-1",
        "signature": "signature",
        "dataset_id": "dataset-1",
        "dataset_name": "Dataset",
        "dataset_revision": 2,
        "model_id": "resnet18",
        "model_name": "ResNet-18",
        "attack_type": "fgsm",
        "parameters": {"epsilon": 0.03},
        "created_at": now - timedelta(hours=1),
        "completed_at": now - timedelta(minutes=50),
        "expires_at": now + timedelta(hours=1),
        "retention_seconds": 3600,
        "preserved": False,
    }


@pytest.mark.asyncio
async def test_preserve_evaluation_removes_expiration(monkeypatch, tmp_path):
    document = _document()
    collection = _Collection(document)
    artifact = tmp_path / "evaluation.json"
    artifact.write_text("{}")
    monkeypatch.setattr(history, "evaluations_collection", collection)
    monkeypatch.setattr(history, "_artifact_path", lambda *_args: Path(artifact))

    item = await history.update_evaluation_preservation(
        evaluation_id=document["_id"],
        owner_id="user-1",
        update=EvaluationPreservationUpdate(preserved=True),
        retention_seconds=86400,
    )

    assert item.preserved is True
    assert item.expires_at is None
    assert item.exact_reuse_available is True


@pytest.mark.asyncio
async def test_unpreserve_evaluation_restarts_retention_from_now(monkeypatch, tmp_path):
    document = _document()
    document["preserved"] = True
    document["expires_at"] = None
    collection = _Collection(document)
    artifact = tmp_path / "evaluation.json"
    artifact.write_text("{}")
    monkeypatch.setattr(history, "evaluations_collection", collection)
    monkeypatch.setattr(history, "_artifact_path", lambda *_args: Path(artifact))

    before = datetime.now(timezone.utc)
    item = await history.update_evaluation_preservation(
        evaluation_id=document["_id"],
        owner_id="user-1",
        update=EvaluationPreservationUpdate(preserved=False),
        retention_seconds=7200,
    )

    assert item.preserved is False
    assert item.retention_seconds == 7200
    assert item.expires_at is not None
    assert item.expires_at >= before + timedelta(seconds=7199)


@pytest.mark.asyncio
async def test_preserved_evaluation_artifact_access_is_owner_scoped(monkeypatch):
    document = _document()
    document["preserved"] = True
    document["expires_at"] = None
    collection = _Collection(document)
    monkeypatch.setattr(history, "evaluations_collection", collection)

    assert (
        await history.evaluation_artifact_is_accessible(
            document["_id"],
            owner_id="user-1",
        )
        is True
    )
    assert (
        await history.evaluation_artifact_is_accessible(
            document["_id"],
            owner_id="user-2",
        )
        is False
    )
