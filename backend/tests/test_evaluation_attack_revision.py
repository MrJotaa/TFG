import pytest
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
    name = "YOLO26n · preentrenado en COCO"


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


async def test_detection_confidence_revision_participates_in_evaluation_identity(
    monkeypatch,
):
    identity = await _identity(monkeypatch, AttackType.YOLO_CONFIDENCE)

    assert identity["attack_revision"] == "detection-confidence-linf-v2"


async def test_unversioned_attacks_keep_legacy_evaluation_identity_shape(monkeypatch):
    identity = await _identity(monkeypatch, AttackType.RANDOM_NOISE)

    assert "attack_revision" not in identity


@pytest.mark.parametrize(
    ("attack_type", "expected_revision"),
    [
        (
            AttackType.CARLINI_WAGNER,
            "art-1.20.1-cw-l2-linf-v2",
        ),
        (
            AttackType.DEEPFOOL,
            "art-1.20.1-deepfool-v1",
        ),
        (
            AttackType.HOP_SKIP_JUMP,
            "art-1.20.1-hop-skip-jump-v1",
        ),
        (
            AttackType.ZOO,
            "art-1.20.1-zoo-noise-v2",
        ),
    ],
)
async def test_art_attack_revision_participates_in_classification_identity(
    monkeypatch,
    attack_type,
    expected_revision,
):
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
        lambda: "torchvision-resnet18-imagenet-v1",
    )

    request = EvaluationCompatibilityRequest(
        task_type="classification",
        dataset_id=str(dataset["_id"]),
        attack_type=attack_type,
    )
    identity = await evaluation_history_service._build_identity(
        request=request,
        owner_id=owner_id,
        task_type="classification",
    )

    assert identity["attack_revision"] == expected_revision
