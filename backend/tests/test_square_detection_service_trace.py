"""C4: trazas del presupuesto Square en análisis, evaluaciones e historial."""

from types import SimpleNamespace

import pytest
from PIL import Image

import services.detection_attack_service as individual_service
import services.detection_evaluation_service as evaluation_service
from ml.attacks.registry import get_attack
from ml.models.detection import Detection, DetectionBox, DetectionResult
from schemas.attack import AttackType
from schemas.detection_attack import DetectionAttackRequest
from schemas.detection_evaluation import DetectionEvaluationRequest
from services.detection_annotation_snapshot import (
    DetectionSnapshotImage,
    MaterializedDetectionAnnotationSnapshot,
)


class _Detector:
    id = "yolo26n"
    name = "YOLO sintético"
    task_type = "detection"
    revision = "synthetic-v1"

    @staticmethod
    def _result(image):
        return DetectionResult(
            image.width, image.height,
            (Detection(0, 0.9, DetectionBox(1, 1, 10, 10)),),
        )

    def detect_with_visualization(self, image):
        return self._result(image), image.copy()

    def detect(self, image):
        return self._result(image)

    def supports_detection_explainability(self):
        return False


def _image():
    return Image.new("RGB", (20, 20), (128, 128, 128))


def _execution_trace():
    return {
        "status": "budget_exhausted",
        "queries": 5,
        "max_queries": 5,
        "cache_hits": 2,
        "seed": 123,
        "attack_revision": get_attack(AttackType.SQUARE_DETECTION).revision,
        "reference_count": 1,
        "clean_score": 1.0,
        "best_score": 0.75,
        "objective_reached": True,
    }


async def _attack_result(*args, **kwargs):
    image = kwargs["image"]
    return {
        "adversarial_image": image.copy(),
        "perturbation_image": image.copy(),
        "perturbation_linf": 0.0,
        "perturbation_l2": 0.0,
        "execution_trace": _execution_trace(),
    }


@pytest.mark.asyncio
async def test_individual_response_and_saved_payload_include_query_trace(monkeypatch, tmp_path):
    model = _Detector()
    persisted_metadata = []
    persisted_payloads = []

    async def get_image(**kwargs):
        return _image()

    monkeypatch.setattr(individual_service, "get_image_from_dataset", get_image)
    monkeypatch.setattr(individual_service, "get_active_model_instance", lambda: model)
    monkeypatch.setattr(individual_service, "run_detection_attack_operation", _attack_result)
    monkeypatch.setattr(individual_service, "create_result_directory", lambda **kw: ("detection_attack_a", tmp_path))
    monkeypatch.setattr(individual_service, "save_result_image", lambda **kw: "/results/" + kw["filename"])
    monkeypatch.setattr(individual_service, "save_result_metadata", lambda **kw: persisted_metadata.append(kw["metadata"]))
    monkeypatch.setattr(individual_service, "save_result_payload", lambda **kw: persisted_payloads.append(kw["payload"]))
    response = await individual_service.apply_detection_attack(
        DetectionAttackRequest(dataset_id="dataset-1", image_id="image-1", attack_type=AttackType.SQUARE_DETECTION)
    )
    assert response.execution_trace.queries == 5
    assert response.execution_trace.status == "budget_exhausted"
    assert response.comparison.attack_success is False  # misma clase y caja, sin pérdida de correspondencia
    assert persisted_metadata[0]["execution_trace"] == _execution_trace()
    assert persisted_metadata[0]["attack_revision"] == _execution_trace()["attack_revision"]
    assert persisted_payloads[0]["execution_trace"] == _execution_trace()
    assert persisted_payloads[0] == response.model_dump(mode="json")


@pytest.mark.asyncio
async def test_dataset_response_persists_trace_per_image_without_changing_legacy(monkeypatch):
    model = _Detector()
    persisted = []

    async def get_dataset(dataset_id):
        return SimpleNamespace(owner_id="user-1", images=[SimpleNamespace(id="image-1", filename="image-1.png")])

    async def get_image(**kwargs):
        return _image()

    async def read_snapshot(dataset_id):
        return MaterializedDetectionAnnotationSnapshot(
            dataset_id=dataset_id,
            dataset_revision=0,
            snapshot_id=None,
            images=(DetectionSnapshotImage(
                image_id="image-1", filename="image-1.png", width=20, height=20,
                review_status="unreviewed",
            ),),
            categories=(), annotations=(),
        )

    async def persist(**kwargs):
        persisted.append(kwargs)

    monkeypatch.setattr(evaluation_service, "get_dataset", get_dataset)
    monkeypatch.setattr(evaluation_service, "get_image_from_dataset", get_image)
    monkeypatch.setattr(evaluation_service, "read_detection_annotation_snapshot", read_snapshot)
    monkeypatch.setattr(evaluation_service, "get_active_model_instance", lambda: model)
    monkeypatch.setattr(evaluation_service, "run_detection_attack_operation", _attack_result)
    monkeypatch.setattr(evaluation_service, "new_detection_evaluation_id", lambda: "evaluation-1")
    monkeypatch.setattr(evaluation_service, "save_detection_evaluation_adversarial_image", lambda **kw: "/results/adv.png")
    monkeypatch.setattr(evaluation_service, "persist_detection_evaluation", persist)
    monkeypatch.setattr(evaluation_service, "log_evaluation_profile", lambda *a: None)

    response = await evaluation_service.evaluate_detection_dataset(
        DetectionEvaluationRequest(dataset_id="dataset-1", attack_type=AttackType.SQUARE_DETECTION)
    )
    assert len(response.results) == 1
    assert response.results[0].execution_trace.queries == 5
    assert response.results[0].execution_trace.status == "budget_exhausted"
    assert response.results[0].attack_success is False
    assert response.summary.attack_success_rate == 0
    assert response.model_dump(mode="json")["results"][0]["execution_trace"] == _execution_trace()
    assert persisted[0]["response"] is response
