from bson import ObjectId

from schemas.attack import AttackType
from schemas.evaluation_history import EvaluationCompatibilityRequest
from services import evaluation_history_service


class FakeDatasetsCollection:
    def __init__(self, document: dict):
        self.document = document

    async def find_one(self, query: dict):
        assert query == {"_id": self.document["_id"]}
        return self.document


class FakeModel:
    id = "yolo26n"
    name = "YOLO26n pretrained"


def _dataset(owner_id: str) -> dict:
    dataset_id = ObjectId()
    return {
        "_id": dataset_id,
        "owner_id": owner_id,
        "name": "Detection dataset",
        "dataset_revision": 1,
        "images": [],
    }


async def _identity(monkeypatch, attack_type: AttackType) -> dict:
    owner_id = str(ObjectId())
    dataset = _dataset(owner_id)
    monkeypatch.setattr(
        evaluation_history_service,
        "datasets_collection",
        FakeDatasetsCollection(dataset),
    )
    monkeypatch.setattr(
        evaluation_history_service,
        "get_active_model",
        lambda: FakeModel(),
    )
    monkeypatch.setattr(
        evaluation_history_service,
        "get_active_model_revision",
        lambda: "ultralytics-yolo26n-coco-8.4.121",
    )

    request = EvaluationCompatibilityRequest(
        task_type="detection",
        dataset_id=str(dataset["_id"]),
        attack_type=attack_type,
    )
    return await evaluation_history_service._build_identity(
        request=request,
        owner_id=owner_id,
        task_type="detection",
    )


async def test_yolo_confidence_revision_participates_in_evaluation_identity(
    monkeypatch,
):
    identity = await _identity(monkeypatch, AttackType.YOLO_CONFIDENCE)

    assert identity["attack_revision"] == "yolo26-end2end-confidence-v1"


async def test_unversioned_attacks_keep_legacy_evaluation_identity_shape(monkeypatch):
    identity = await _identity(monkeypatch, AttackType.RANDOM_NOISE)

    assert "attack_revision" not in identity
