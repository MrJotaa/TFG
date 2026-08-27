from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from bson import ObjectId

import services.individual_analysis_history_service as history
from schemas.individual_analysis_history import (
    IndividualAnalysisCompatibilityRequest,
    IndividualAnalysisSaveRequest,
)


class _Cursor:
    def __init__(self, documents):
        self.documents = list(documents)

    def sort(self, *_args):
        return self

    def skip(self, offset):
        self.documents = self.documents[offset:]
        return self

    def limit(self, limit):
        self.documents = self.documents[:limit]
        return self

    def __aiter__(self):
        self.iterator = iter(self.documents)
        return self

    async def __anext__(self):
        try:
            return next(self.iterator)
        except StopIteration as error:
            raise StopAsyncIteration from error


class _Collection:
    def __init__(self, documents=None):
        self.documents = list(documents or [])

    async def find_one(self, query):
        for document in self.documents:
            if all(document.get(key) == value for key, value in query.items()):
                return document
        return None

    async def insert_one(self, document):
        self.documents.append(dict(document))
        return SimpleNamespace(inserted_id=document["_id"])

    async def count_documents(self, query):
        return len(
            [
                document
                for document in self.documents
                if all(document.get(key) == value for key, value in query.items())
            ]
        )

    def find(self, query, *_args):
        return _Cursor(
            document
            for document in self.documents
            if all(document.get(key) == value for key, value in query.items())
        )

    async def delete_one(self, query):
        before = len(self.documents)
        self.documents = [
            document
            for document in self.documents
            if not all(document.get(key) == value for key, value in query.items())
        ]
        return SimpleNamespace(deleted_count=before - len(self.documents))


def _classification_payload(result_id, dataset_id, image_id):
    return {
        "result_id": result_id,
        "dataset_id": dataset_id,
        "dataset_revision": 7,
        "image_id": image_id,
        "model_id": "resnet18",
        "model_name": "ResNet-18",
        "attack_type": "fgsm",
        "parameters": {"epsilon": 0.03},
        "adversarial_image_url": f"/results/{result_id}/adversarial.png",
        "perturbation_image_url": f"/results/{result_id}/perturbation.png",
        "original_gradcam_image_url": f"/results/{result_id}/original_gradcam.png",
        "adversarial_gradcam_image_url": (
            f"/results/{result_id}/adversarial_gradcam.png"
        ),
        "original_prediction": {
            "prediction": "tench",
            "confidence": 0.9,
            "top_k": [{"label": "tench", "probability": 0.9}],
        },
        "adversarial_prediction": {
            "prediction": "church",
            "confidence": 0.6,
            "top_k": [{"label": "church", "probability": 0.6}],
        },
        "comparison": {
            "original_label": "tench",
            "adversarial_label": "church",
            "original_confidence": 0.9,
            "adversarial_confidence": 0.6,
            "confidence_difference": -0.3,
            "prediction_changed": True,
            "attack_success": True,
            "perturbation_linf": 0.03,
            "perturbation_l2": 4.2,
        },
    }


@pytest.fixture
def analysis_environment(monkeypatch):
    owner_id = str(ObjectId())
    dataset_id = str(ObjectId())
    image_id = "image-1"
    result_id = "attack_00000000-0000-4000-8000-000000000001"
    datasets = _Collection(
        [
            {
                "_id": ObjectId(dataset_id),
                "owner_id": ObjectId(owner_id),
                "name": "Imagenette",
                "dataset_revision": 7,
                "images": [{"id": image_id, "filename": "tench.jpg"}],
            }
        ]
    )
    analyses = _Collection()
    metadata = {
        "result_id": result_id,
        "task_type": "classification",
        "dataset_id": dataset_id,
        "dataset_name": "Imagenette",
        "dataset_revision": 7,
        "image_id": image_id,
        "model_id": "resnet18",
        "model_name": "ResNet-18",
        "model_revision": "resnet18-v1",
        "attack_type": "fgsm",
        "attack_revision": "fgsm-v1",
        "parameters": {"epsilon": 0.03},
        "pipeline_version": "classification-individual-analysis-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    payload = _classification_payload(result_id, dataset_id, image_id)
    monkeypatch.setattr(history, "datasets_collection", datasets)
    monkeypatch.setattr(history, "individual_analyses_collection", analyses)
    monkeypatch.setattr(history, "read_result_metadata", lambda _result_id: metadata)
    monkeypatch.setattr(history, "read_result_payload", lambda _result_id: payload)
    return owner_id, result_id, analyses


@pytest.mark.asyncio
async def test_saved_individual_analysis_is_permanent_and_idempotent(
    analysis_environment,
):
    owner_id, result_id, analyses = analysis_environment
    request = IndividualAnalysisSaveRequest(
        task_type="classification",
        result_id=result_id,
    )

    first = await history.save_individual_analysis(request=request, owner_id=owner_id)
    second = await history.save_individual_analysis(request=request, owner_id=owner_id)

    assert first.analysis_id == second.analysis_id
    assert first.image_filename == "tench.jpg"
    assert first.explanation_available is True
    assert first.summary["original_label"] == "tench"
    assert len(analyses.documents) == 1
    assert "expires_at" not in analyses.documents[0]


@pytest.mark.asyncio
async def test_compatibility_returns_saved_identical_analysis(
    analysis_environment,
    monkeypatch,
):
    owner_id, result_id, analyses = analysis_environment
    saved = await history.save_individual_analysis(
        request=IndividualAnalysisSaveRequest(
            task_type="classification",
            result_id=result_id,
        ),
        owner_id=owner_id,
    )
    signature = analyses.documents[0]["signature"]
    async def fake_identity(**_kwargs):
        return {"signature": signature}

    monkeypatch.setattr(history, "_build_current_identity", fake_identity)
    monkeypatch.setattr(history, "_source_artifact_exists", lambda _document: True)

    response = await history.find_compatible_individual_analysis(
        request=IndividualAnalysisCompatibilityRequest(
            task_type="classification",
            dataset_id="unused",
            image_id="image-1",
            attack_type="fgsm",
            parameters={"epsilon": 0.03},
        ),
        owner_id=owner_id,
    )

    assert response.match is not None
    assert response.match.analysis_id == saved.analysis_id


@pytest.mark.asyncio
async def test_saved_analysis_source_is_protected_from_temporary_cleanup(
    analysis_environment,
):
    owner_id, result_id, _analyses = analysis_environment
    await history.save_individual_analysis(
        request=IndividualAnalysisSaveRequest(
            task_type="classification",
            result_id=result_id,
        ),
        owner_id=owner_id,
    )

    assert await history.protected_individual_result_ids() == {result_id}
