import os
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

import services.detection_evaluation_persistence as persistence
from core.errors import BadRequestError, ForbiddenError, GoneError, NotFoundError
from db.migrations import upgrade_detection_evaluations_schema
from schemas.attack import AttackType
from schemas.detection_evaluation import (
    DetectionEvaluationRequest,
    DetectionEvaluationResponse,
    DetectionEvaluationSummary,
)


class _FakeCursor:
    def __init__(self, documents):
        self.documents = list(documents)

    def sort(self, specification):
        for field_name, direction in reversed(specification):
            self.documents.sort(
                key=lambda document: document[field_name],
                reverse=direction < 0,
            )
        return self

    def limit(self, limit):
        self.documents = self.documents[:limit]
        return self

    def skip(self, offset):
        self.documents = self.documents[offset:]
        return self

    def __aiter__(self):
        self._iterator = iter(self.documents)
        return self

    async def __anext__(self):
        try:
            return next(self._iterator)
        except StopIteration as error:
            raise StopAsyncIteration from error


class _FakeEvaluationCollection:
    def __init__(self):
        self.documents = {}
        self.insert_error = None

    async def insert_one(self, document):
        if self.insert_error is not None:
            raise self.insert_error
        self.documents[document["_id"]] = dict(document)
        return SimpleNamespace(inserted_id=document["_id"])

    async def find_one(self, query):
        return self.documents.get(query["_id"])

    def find(self, query):
        now = query["expires_at"]["$gt"]
        documents = [
            document
            for document in self.documents.values()
            if document["owner_id"] == query["owner_id"]
            and document["status"] == query["status"]
            and document["expires_at"] > now
        ]
        return _FakeCursor(documents)


class _FakeIndexCollection:
    def __init__(self):
        self.indexes = None

    async def create_indexes(self, indexes):
        self.indexes = indexes


def _request() -> DetectionEvaluationRequest:
    return DetectionEvaluationRequest(
        dataset_id="dataset-1",
        attack_type=AttackType.RANDOM_NOISE,
    )


def _response(result_id: str) -> DetectionEvaluationResponse:
    return DetectionEvaluationResponse(
        result_id=result_id,
        model_id="detector-1",
        model_name="Detector",
        dataset_id="dataset-1",
        dataset_revision=3,
        annotation_snapshot_id="snapshot-1",
        attack_type=AttackType.RANDOM_NOISE,
        parameters=_request().parameters,
        summary=DetectionEvaluationSummary(
            total_images=0,
            total_original_detections=0,
            total_adversarial_detections=0,
            detections_difference=0,
            average_original_detections_per_image=0.0,
            average_adversarial_detections_per_image=0.0,
            average_original_confidence=0.0,
            average_adversarial_confidence=0.0,
            average_confidence_difference=0.0,
            images_with_reduced_detections=0,
            images_with_no_adversarial_detections=0,
            attack_success_rate=0.0,
            average_perturbation_linf=0.0,
            average_perturbation_l2=0.0,
        ),
        results=[],
        supervised=None,
    )


@pytest.fixture
def persistence_environment(tmp_path, monkeypatch):
    collection = _FakeEvaluationCollection()
    monkeypatch.setattr(
        persistence,
        "DETECTION_EVALUATIONS_BASE_PATH",
        str(tmp_path),
    )
    monkeypatch.setattr(
        persistence,
        "detection_evaluations_collection",
        collection,
    )
    return collection


@pytest.mark.asyncio
async def test_detection_evaluation_round_trip_and_history(
    tmp_path,
    persistence_environment,
):
    result_id = "detection_evaluation_00000000-0000-4000-8000-000000000001"
    response = _response(result_id)

    await persistence.persist_detection_evaluation(
        response=response,
        request=_request(),
        owner_id="user-1",
    )

    artifact_path = (
        tmp_path / result_id / persistence.DETECTION_EVALUATION_ARTIFACT_FILENAME
    )
    assert artifact_path.is_file()
    assert persistence_environment.documents[result_id]["summary"] == (
        response.summary.model_dump(mode="json")
    )

    restored = await persistence.get_detection_evaluation(
        evaluation_id=result_id,
        owner_id="user-1",
    )
    assert restored == response

    history = await persistence.list_detection_evaluations(
        owner_id="user-1",
        limit=50,
        offset=0,
    )
    assert len(history) == 1
    assert history[0].result_id == result_id
    assert history[0].summary == response.summary
    assert history[0].supervised_available is False


@pytest.mark.asyncio
async def test_detection_evaluation_detail_enforces_ownership(
    persistence_environment,
):
    result_id = "detection_evaluation_00000000-0000-4000-8000-000000000002"
    await persistence.persist_detection_evaluation(
        response=_response(result_id),
        request=_request(),
        owner_id="user-1",
    )

    with pytest.raises(ForbiddenError, match="otro usuario"):
        await persistence.get_detection_evaluation(
            evaluation_id=result_id,
            owner_id="user-2",
        )


@pytest.mark.asyncio
async def test_detection_evaluation_expiration_and_missing_artifact_are_typed(
    tmp_path,
    persistence_environment,
):
    result_id = "detection_evaluation_00000000-0000-4000-8000-000000000003"
    await persistence.persist_detection_evaluation(
        response=_response(result_id),
        request=_request(),
        owner_id="user-1",
    )
    document = persistence_environment.documents[result_id]
    document["expires_at"] = datetime.now(timezone.utc) - timedelta(seconds=1)

    with pytest.raises(GoneError, match="ha caducado"):
        await persistence.get_detection_evaluation(
            evaluation_id=result_id,
            owner_id="user-1",
        )
    assert (
        await persistence.list_detection_evaluations(
            owner_id="user-1",
            limit=50,
            offset=0,
        )
        == []
    )

    document["expires_at"] = datetime.now(timezone.utc) + timedelta(hours=1)
    artifact_path = (
        tmp_path / result_id / persistence.DETECTION_EVALUATION_ARTIFACT_FILENAME
    )
    artifact_path.unlink()
    with pytest.raises(GoneError, match="ya no está disponible"):
        await persistence.get_detection_evaluation(
            evaluation_id=result_id,
            owner_id="user-1",
        )


@pytest.mark.asyncio
async def test_detection_evaluation_rejects_invalid_identity_and_artifact(
    tmp_path,
    persistence_environment,
):
    with pytest.raises(NotFoundError, match="no encontrada"):
        await persistence.get_detection_evaluation(
            evaluation_id="../evaluation.json",
            owner_id="user-1",
        )

    result_id = "detection_evaluation_00000000-0000-4000-8000-000000000004"
    await persistence.persist_detection_evaluation(
        response=_response(result_id),
        request=_request(),
        owner_id="user-1",
    )
    artifact_path = (
        tmp_path / result_id / persistence.DETECTION_EVALUATION_ARTIFACT_FILENAME
    )
    artifact_path.write_text("{}", encoding="utf-8")

    with pytest.raises(BadRequestError, match="artefacto de la evaluación de detección no es válido"):
        await persistence.get_detection_evaluation(
            evaluation_id=result_id,
            owner_id="user-1",
        )


@pytest.mark.asyncio
async def test_detection_evaluation_cleans_artifact_when_index_insert_fails(
    tmp_path,
    persistence_environment,
):
    result_id = "detection_evaluation_00000000-0000-4000-8000-000000000005"
    persistence_environment.insert_error = RuntimeError("mongo unavailable")

    with pytest.raises(RuntimeError, match="mongo unavailable"):
        await persistence.persist_detection_evaluation(
            response=_response(result_id),
            request=_request(),
            owner_id="user-1",
        )

    assert not (tmp_path / result_id).exists()


@pytest.mark.asyncio
async def test_evaluation_indexes_include_history_expiry_and_concurrency():
    collection = _FakeIndexCollection()

    await upgrade_detection_evaluations_schema({"detection_evaluations": collection})

    index_documents = [index.document for index in collection.indexes]
    assert index_documents[0] == {
        "name": "detection_evaluations_by_owner",
        "key": {"owner_id": 1, "created_at": -1},
    }
    assert index_documents[1] == {
        "name": "detection_evaluations_expiration",
        "sparse": True,
        "key": {"expires_at": 1},
    }
    assert index_documents[2]["name"] == "evaluations_compatibility_lookup"
    assert index_documents[3]["name"] == "one_running_evaluation_per_signature"
    assert index_documents[3]["unique"] is True
    assert index_documents[3]["partialFilterExpression"] == {
        "status": "running",
        "signature": {"$type": "string"},
    }


@pytest.mark.asyncio
async def test_persistence_removes_only_expired_evaluation_artifact_directories(
    tmp_path,
    persistence_environment,
):
    expired_id = "detection_evaluation_00000000-0000-4000-8000-000000000020"
    active_id = "detection_evaluation_00000000-0000-4000-8000-000000000021"
    expired_path = tmp_path / expired_id
    unrelated_path = tmp_path / "unrelated"
    expired_path.mkdir()
    unrelated_path.mkdir()
    expired_timestamp = datetime.now(timezone.utc).timestamp() - (
        persistence.RESULTS_TTL_SECONDS + 1
    )
    os.utime(expired_path, (expired_timestamp, expired_timestamp))

    await persistence.persist_detection_evaluation(
        response=_response(active_id),
        request=_request(),
        owner_id="user-1",
    )

    assert not expired_path.exists()
    assert unrelated_path.is_dir()
    assert (tmp_path / active_id).is_dir()
