"""C5: regresión transversal de Square a través de servicios y API.

Las pruebas de contrato usan detectores sintéticos, nunca cambian los pesos ni
requieren descargar checkpoints. ART se utiliza de verdad cuando está instalado.
Las pruebas opcionales con checkpoints reales continúan en
``test_square_detection_attack.py``.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import api.routes.detection_attack as detection_attack_route
import services.detection_attack_service as individual_service
import services.detection_evaluation_service as evaluation_service
from core.cancellation import CancellationToken, OperationCancelled
from ml.attacks.registry import (
    create_attack_runner, get_attack, list_attack_catalog,
    list_available_attacks, resolve_attack_parameters,
)
from ml.models.detection import Detection, DetectionBox, DetectionResult
from schemas.attack import AttackParameters, AttackType
from schemas.detection_attack import (
    DetectionAttackComparison, DetectionAttackRequest, DetectionAttackResponse,
)
from schemas.detection_evaluation import DetectionEvaluationRequest
from services.detection_annotation_snapshot import (
    DetectionSnapshotImage, MaterializedDetectionAnnotationSnapshot,
)
from utils.reproducibility import attack_parameters_for_image


class _Detector:
    name = "Detector sintético C5"
    task_type = "detection"
    revision = "c5-synthetic-v1"

    def __init__(self, model_id):
        self.id = model_id
        self.queries = 0

    def is_loaded(self):
        return True

    @staticmethod
    def _result(image, confidence):
        return DetectionResult(
            image.width, image.height,
            (Detection(0, confidence, DetectionBox(2, 2, image.width - 2, image.height - 2)),),
        )

    def detect_for_evaluation(self, image, configuration):
        assert configuration.score_floor == 0.001
        self.queries += 1
        pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
        variation = float(np.abs(pixels - 128).mean() / 255.0)
        return self._result(image, max(0.01, 0.9 - 10 * variation))

    def detect(self, image):
        return self._result(image, 0.9)

    def detect_with_visualization(self, image):
        return self.detect(image), image.copy()

    def supports_detection_explainability(self):
        return False


def _image(width=28, height=24):
    return Image.new("RGB", (width, height), (128, 128, 128))


def _params(**values):
    return resolve_attack_parameters(
        get_attack(AttackType.SQUARE_DETECTION),
        AttackParameters(epsilon=0.03, max_queries=8, steps=5, seed=21, **values),
    )


def _pixel_linf(original, adversarial):
    a = np.asarray(original.convert("RGB"), dtype=np.int16)
    b = np.asarray(adversarial.convert("RGB"), dtype=np.int16)
    return np.abs(a - b).max() / 255.0


def test_shared_attack_registration_contract_remains_consistent():
    """Ningún ataque registrado utiliza un catálogo o registro paralelo."""
    catalog = {item["id"]: item for item in list_attack_catalog()}
    assert set(catalog) == {attack_type.value for attack_type in list_available_attacks()}
    for attack_type in list_available_attacks():
        attack = get_attack(attack_type)
        assert attack.id == attack_type.value
        assert attack.knowledge_type in {"white_box", "black_box", "baseline"}
        assert len(attack.supported_task_types) >= 1
        assert attack.catalog_entry()["supported_task_types"] == list(attack.supported_task_types)
        assert catalog[attack_type.value]["supported_task_types"] == list(attack.supported_task_types)
        assert resolve_attack_parameters(attack, AttackParameters()).model_dump() == {
            spec.key: spec.default for spec in attack.parameter_specs
        }
    square = get_attack(AttackType.SQUARE_DETECTION)
    assert square.knowledge_type == "black_box"
    assert square.supported_model_ids == ("yolo26n", "rf_detr_nano")


@pytest.mark.parametrize("model_id", ["yolo26n", "rf_detr_nano"])
def test_real_art_runner_is_independent_and_budgeted_for_both_models(model_id):
    pytest.importorskip("art")
    detector = _Detector(model_id)
    attack = get_attack(AttackType.SQUARE_DETECTION)
    parameters = _params()
    runner = create_attack_runner(attack, detector, parameters)
    assert runner is not attack
    assert runner.requires_exclusive_model_execution
    image = _image()
    first = runner.apply(detector, image, parameters)
    assert 1 <= detector.queries <= parameters.max_queries
    assert first["execution_trace"]["queries"] == detector.queries
    assert first["execution_trace"]["seed"] == parameters.seed
    assert _pixel_linf(image, first["adversarial_image"]) <= parameters.epsilon + 1e-7
    first_calls = detector.queries
    second = create_attack_runner(attack, detector, parameters).apply(detector, image, parameters)
    assert 1 <= detector.queries - first_calls <= parameters.max_queries
    assert second["execution_trace"]["queries"] == detector.queries - first_calls
    np.testing.assert_array_equal(np.asarray(first["adversarial_image"]), np.asarray(second["adversarial_image"]))
    assert np.asarray(image).min() == np.asarray(image).max() == 128
    token = CancellationToken()
    token.cancel()
    with pytest.raises(OperationCancelled):
        create_attack_runner(attack, detector, parameters, cancellation_token=token).apply(
            detector, image, parameters,
        )
    assert detector.queries <= 2 * parameters.max_queries


@pytest.mark.asyncio
@pytest.mark.parametrize("model_id", ["yolo26n", "rf_detr_nano"])
async def test_square_individual_uses_shared_persistence_and_metrics(monkeypatch, tmp_path, model_id):
    pytest.importorskip("art")
    model = _Detector(model_id)
    recorded = {"metadata": [], "payload": []}

    async def get_image(**kwargs):
        return _image()

    monkeypatch.setattr(individual_service, "get_image_from_dataset", get_image)
    monkeypatch.setattr(individual_service, "get_active_model_instance", lambda: model)
    monkeypatch.setattr(individual_service, "create_result_directory", lambda **kw: ("c5-individual", tmp_path))
    monkeypatch.setattr(individual_service, "save_result_image", lambda **kw: "/results/" + kw["filename"])
    monkeypatch.setattr(individual_service, "save_result_metadata", lambda **kw: recorded["metadata"].append(kw["metadata"]))
    monkeypatch.setattr(individual_service, "save_result_payload", lambda **kw: recorded["payload"].append(kw["payload"]))

    response = await individual_service.apply_detection_attack(
        DetectionAttackRequest(
            dataset_id="dataset-1", image_id="image-1",
            attack_type=AttackType.SQUARE_DETECTION, parameters=_params(),
        )
    )
    assert 1 <= response.execution_trace.queries <= 8
    assert response.execution_trace.attack_revision == get_attack(AttackType.SQUARE_DETECTION).revision
    assert response.comparison.perturbation_linf <= 0.03 + 1e-4
    assert recorded["metadata"][0]["attack_revision"] == response.execution_trace.attack_revision
    assert recorded["payload"][0] == response.model_dump(mode="json")
    assert response.metrics_schema_version == "1.1"
    assert response.attack_success_criterion == "matched_detections_iou50_v1"
    assert response.comparison.attack_success is False  # Misma clase y caja.


@pytest.mark.asyncio
@pytest.mark.parametrize("model_id", ["yolo26n", "rf_detr_nano"])
async def test_square_dataset_preserves_per_image_trace_seed_and_history(monkeypatch, model_id):
    pytest.importorskip("art")
    model = _Detector(model_id)
    images = {"first": _image(28, 24), "second": _image(32, 25)}
    persisted = []

    async def get_dataset(dataset_id):
        return SimpleNamespace(
            owner_id="owner", images=[
                SimpleNamespace(id=image_id, filename=image_id + ".png") for image_id in images
            ],
        )

    async def get_image_from_dataset(**kwargs):
        return images[kwargs["image_id"]]

    async def read_snapshot(dataset_id):
        return MaterializedDetectionAnnotationSnapshot(
            dataset_id=dataset_id, dataset_revision=0, snapshot_id=None,
            images=tuple(
                DetectionSnapshotImage(
                    image_id=image_id, filename=image_id + ".png",
                    width=image.width, height=image.height, review_status="unreviewed",
                ) for image_id, image in images.items()
            ),
            categories=(), annotations=(),
        )

    async def persist(**kwargs):
        persisted.append(kwargs)

    monkeypatch.setattr(evaluation_service, "get_dataset", get_dataset)
    monkeypatch.setattr(evaluation_service, "get_image_from_dataset", get_image_from_dataset)
    monkeypatch.setattr(evaluation_service, "read_detection_annotation_snapshot", read_snapshot)
    monkeypatch.setattr(evaluation_service, "get_active_model_instance", lambda: model)
    monkeypatch.setattr(evaluation_service, "new_detection_evaluation_id", lambda: "c5-evaluation")
    monkeypatch.setattr(evaluation_service, "save_detection_evaluation_adversarial_image", lambda **kw: "/results/" + kw["image_id"] + ".png")
    monkeypatch.setattr(evaluation_service, "persist_detection_evaluation", persist)
    monkeypatch.setattr(evaluation_service, "log_evaluation_profile", lambda *a: None)

    request = DetectionEvaluationRequest(
        dataset_id="dataset-1", attack_type=AttackType.SQUARE_DETECTION, parameters=_params(),
    )
    response = await evaluation_service.evaluate_detection_dataset(request)
    assert len(response.results) == 2
    assert persisted[0]["response"] is response
    assert response.summary.total_images == 2
    for item in response.results:
        trace = item.execution_trace
        assert trace is not None
        assert 1 <= trace.queries <= request.parameters.max_queries
        assert trace.seed == attack_parameters_for_image(
            attack_type=request.attack_type, parameters=request.parameters, image_id=item.image_id,
        ).seed
        assert item.perturbation_linf <= request.parameters.epsilon + 1e-4
        assert "execution_trace" in response.model_dump(mode="json")["results"][
            0 if item.image_id == "first" else 1
        ]
    assert response.results[0].execution_trace.seed != response.results[1].execution_trace.seed
    assert model.queries <= 2 * request.parameters.max_queries


def test_square_api_preserves_optional_trace(client, monkeypatch, auth_headers):
    async def handle(request, **kwargs):
        assert request.attack_type is AttackType.SQUARE_DETECTION
        assert request.parameters.max_queries == 7
        return DetectionAttackResponse(
            result_id="square-api-c5", attack_type=request.attack_type,
            parameters=request.parameters,
            original_annotated_image_url="/results/original.png",
            adversarial_image_url="/results/adversarial.png",
            adversarial_annotated_image_url="/results/adversarial-annotated.png",
            perturbation_image_url="/results/perturbation.png",
            original_detections=[], adversarial_detections=[],
            comparison=DetectionAttackComparison(
                original_detection_count=0, adversarial_detection_count=0,
                detections_difference=0, average_original_confidence=0,
                average_adversarial_confidence=0, average_confidence_difference=0,
                attack_success=False, perturbation_linf=0, perturbation_l2=0,
            ),
            execution_trace={
                "status": "skipped_no_references", "queries": 1, "max_queries": 7,
                "cache_hits": 0, "seed": 5, "attack_revision": get_attack(AttackType.SQUARE_DETECTION).revision,
                "reference_count": 0, "clean_score": None, "best_score": None,
                "objective_reached": False,
            },
        )

    monkeypatch.setattr(detection_attack_route, "apply_detection_attack", handle)
    response = client.post(
        "/detect/attack", headers=auth_headers,
        json={
            "dataset_id": "dataset-1", "image_id": "image-1",
            "attack_type": "square_detection", "parameters": {"max_queries": 7},
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["attack_type"] == "square_detection"
    assert payload["execution_trace"]["max_queries"] == 7
    assert payload["execution_trace"]["status"] == "skipped_no_references"


def test_archived_attack_remains_readable_but_cannot_be_executed():
    """La política de historial es común, incluso para ataques retirados."""
    from pydantic import ValidationError

    archived = DetectionAttackResponse(
        result_id="archived-example", attack_type="dpatch",
        parameters=AttackParameters(),
        original_annotated_image_url="/old/original.png",
        adversarial_image_url="/old/adversarial.png",
        adversarial_annotated_image_url="/old/annotated.png",
        perturbation_image_url="/old/perturbation.png",
        original_detections=[], adversarial_detections=[],
        comparison=DetectionAttackComparison(
            original_detection_count=0, adversarial_detection_count=0,
            detections_difference=0, average_original_confidence=0,
            average_adversarial_confidence=0, average_confidence_difference=0,
            attack_success=False, perturbation_linf=0, perturbation_l2=0,
        ),
    )
    assert archived.attack_type == "dpatch"
    assert "execution_trace" not in archived.model_dump(mode="json")
    with pytest.raises(ValidationError):
        DetectionAttackRequest(
            dataset_id="dataset-1", image_id="image-1", attack_type="dpatch",
        )
