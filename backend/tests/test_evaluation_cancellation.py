from types import SimpleNamespace

import pytest

from core.cancellation import CancellationToken
from core.errors import BadRequestError
from services import evaluation_history_service as history
from services.evaluation_cancellation import (
    register_evaluation_cancellation,
    unregister_evaluation_cancellation,
)

EVALUATION_ID = "evaluation_00000000-0000-4000-8000-000000000001"


class _RunningCollection:
    def __init__(self, *, owner_id: str = "user-1"):
        self.document = {
            "_id": EVALUATION_ID,
            "owner_id": owner_id,
            "status": "running",
        }

    async def find_one(self, query):
        if self.document is not None and query.get("_id") == self.document["_id"]:
            return self.document
        return None

    async def delete_one(self, query):
        matches = (
            self.document is not None
            and query.get("_id") == self.document["_id"]
            and query.get("owner_id") == self.document["owner_id"]
            and query.get("status") == self.document["status"]
        )
        if matches:
            self.document = None
            return SimpleNamespace(deleted_count=1)
        return SimpleNamespace(deleted_count=0)


@pytest.mark.asyncio
async def test_cancel_running_evaluation_signals_active_execution(monkeypatch):
    collection = _RunningCollection()
    cancellation_token = CancellationToken()
    cleaned = []
    monkeypatch.setattr(history, "evaluations_collection", collection)
    monkeypatch.setattr(
        history,
        "_remove_evaluation_artifacts",
        lambda evaluation_id: cleaned.append(evaluation_id),
    )
    register_evaluation_cancellation(EVALUATION_ID, cancellation_token)

    try:
        await history.cancel_running_evaluation(
            evaluation_id=EVALUATION_ID,
            owner_id="user-1",
        )
    finally:
        unregister_evaluation_cancellation(EVALUATION_ID, cancellation_token)

    assert cancellation_token.is_cancelled is True
    assert collection.document is None
    assert cleaned == []


@pytest.mark.asyncio
async def test_cancel_stale_running_evaluation_removes_orphaned_artifacts(monkeypatch):
    collection = _RunningCollection()
    cleaned = []
    monkeypatch.setattr(history, "evaluations_collection", collection)
    monkeypatch.setattr(
        history,
        "_remove_evaluation_artifacts",
        lambda evaluation_id: cleaned.append(evaluation_id),
    )

    await history.cancel_running_evaluation(
        evaluation_id=EVALUATION_ID,
        owner_id="user-1",
    )

    assert collection.document is None
    assert cleaned == [EVALUATION_ID]


@pytest.mark.asyncio
async def test_cancel_rejects_completed_evaluation(monkeypatch):
    collection = _RunningCollection()
    collection.document["status"] = "completed"
    monkeypatch.setattr(history, "evaluations_collection", collection)

    with pytest.raises(BadRequestError, match="La evaluación no está en curso"):
        await history.cancel_running_evaluation(
            evaluation_id=EVALUATION_ID,
            owner_id="user-1",
        )
