"""History records stay visible when their files have been lost.

Uses only synthetic documents and pytest's isolated temporary filesystem.
"""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from core.errors import GoneError
from schemas.evaluation_history import EvaluationPreservationUpdate
from services import evaluation_history_service as evaluation_history
from services import individual_analysis_history_service as individual_history


def _evaluation_document(task_type):
    now = datetime.now(timezone.utc)
    prefix = "evaluation" if task_type == "classification" else "detection_evaluation"
    return {
        "_id": f"{prefix}_00000000-0000-4000-8000-000000000001",
        "status": "completed",
        "task_type": task_type,
        "owner_id": "owner-1",
        "dataset_id": "dataset-1",
        "dataset_revision": 1,
        "model_id": "resnet18",
        "model_name": "ResNet-18",
        "attack_type": "fgsm",
        "parameters": {"epsilon": 0.03},
        "created_at": now,
        "completed_at": now,
        "expires_at": None,
        "preserved": True,
        "signature": "synthetic-signature",
        "storage_schema_version": evaluation_history.EVALUATION_STORAGE_SCHEMA_VERSION,
    }


@pytest.mark.parametrize("task_type", ["classification", "detection"])
def test_missing_evaluation_artifact_is_reported_without_losing_record(
    monkeypatch, tmp_path, task_type
):
    monkeypatch.setattr(
        evaluation_history, "RESULTS_BASE_PATH", str(tmp_path / "results")
    )
    monkeypatch.setattr(
        evaluation_history,
        "DETECTION_EVALUATIONS_BASE_PATH",
        tmp_path / "evaluations",
    )
    document = _evaluation_document(task_type)

    missing = evaluation_history._history_item(document)
    assert missing.preserved is True
    assert missing.artifact_available is False
    assert missing.exact_reuse_available is False

    path = evaluation_history._artifact_path(document["_id"], task_type)
    path.parent.mkdir(parents=True)
    path.write_text("{}", encoding="utf-8")

    available = evaluation_history._history_item(document)
    assert available.artifact_available is True
    assert available.exact_reuse_available is True


@pytest.mark.asyncio
async def test_cannot_preserve_evaluation_without_its_artifact(monkeypatch, tmp_path):
    document = _evaluation_document("classification")
    document["preserved"] = False
    document["expires_at"] = datetime.now(timezone.utc) + timedelta(days=1)
    monkeypatch.setattr(
        evaluation_history, "RESULTS_BASE_PATH", str(tmp_path / "results")
    )

    async def get_owned(evaluation_id, owner_id):
        assert (evaluation_id, owner_id) == (document["_id"], "owner-1")
        return document

    updates = []

    async def update_one(*args):
        updates.append(args)
        return SimpleNamespace(modified_count=1)

    monkeypatch.setattr(evaluation_history, "_get_owned_completed_document", get_owned)
    monkeypatch.setattr(
        evaluation_history,
        "evaluations_collection",
        SimpleNamespace(update_one=update_one),
    )

    with pytest.raises(GoneError, match="ya no está disponible"):
        await evaluation_history.update_evaluation_preservation(
            evaluation_id=document["_id"],
            owner_id="owner-1",
            update=EvaluationPreservationUpdate(preserved=True),
            retention_seconds=86400,
        )
    assert updates == []
    assert document["preserved"] is False

    path = evaluation_history._artifact_path(document["_id"], "classification")
    path.parent.mkdir(parents=True)
    path.write_text("{}", encoding="utf-8")

    updated = await evaluation_history.update_evaluation_preservation(
        evaluation_id=document["_id"],
        owner_id="owner-1",
        update=EvaluationPreservationUpdate(preserved=True),
        retention_seconds=86400,
    )
    assert updated.preserved is True
    assert updated.artifact_available is True
    assert len(updates) == 1


def test_missing_individual_analysis_artifact_is_reported(monkeypatch, tmp_path):
    monkeypatch.setattr(individual_history, "RESULTS_BASE_PATH", str(tmp_path))
    now = datetime.now(timezone.utc)
    document = {
        "_id": "individual_analysis_00000000-0000-4000-8000-000000000001",
        "source_result_id": "attack_00000000-0000-4000-8000-000000000001",
        "task_type": "classification",
        "dataset_id": "dataset-1",
        "dataset_revision": 1,
        "image_id": "image-1",
        "model_id": "resnet18",
        "attack_type": "fgsm",
        "parameters": {"epsilon": 0.03},
        "created_at": now,
        "saved_at": now,
    }
    assert individual_history._history_item(document).artifact_available is False
    path = tmp_path / document["source_result_id"] / "analysis.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}", encoding="utf-8")
    assert individual_history._history_item(document).artifact_available is True
