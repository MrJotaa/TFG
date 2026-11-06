"""A5: PGD de localización atravesando servicios y métricas existentes.

El detector de estas pruebas es sintético: el desplazamiento de caja tras la
perturbación es controlado para aislar la evaluación, no mide eficacia real.
"""

from types import SimpleNamespace

import pytest
from PIL import Image
from torchvision.transforms.functional import to_tensor

import services.detection_attack_service as individual_service
import services.detection_evaluation_service as evaluation_service
from ml.detection_categories import DetectionClass, DetectionClassCatalog
from ml.models.detection import Detection, DetectionBox, DetectionResult
from schemas.attack import AttackParameters, AttackType
from schemas.detection_attack import DetectionAttackRequest
from schemas.detection_evaluation import DetectionEvaluationRequest
from services.annotation_importers.base import (
    CanonicalCategory,
    CanonicalObjectAnnotation,
)
from services.detection_annotation_snapshot import (
    DetectionSnapshotImage,
    MaterializedDetectionAnnotationSnapshot,
)


class _LocalizationDetector:
    id = "rf_detr_nano"
    name = "RF-DETR Nano sintético"
    task_type = "detection"
    revision = "rfdetr-test-revision"

    def __init__(self):
        self.detect_calls = 0
        self.grad_calls = 0

    def is_loaded(self):
        return True

    def prepare_detection_attack_input(self, image):
        return to_tensor(image.convert("RGB")).unsqueeze(0)

    @staticmethod
    def _result_for(image):
        # Mismo número, clase y confianza; solo cambia la localización.
        changed = image.getpixel((0, 0))[0] < 128
        box = DetectionBox(60, 60, 80, 80) if changed else DetectionBox(10, 10, 30, 30)
        return DetectionResult(
            source_width=image.width,
            source_height=image.height,
            detections=(
                Detection(class_id=0, class_name="person", confidence=0.9, box=box),
            ),
        )

    def detect(self, image):
        self.detect_calls += 1
        return self._result_for(image)

    def detect_with_visualization(self, image):
        return self._result_for(image), image.copy()

    def detect_for_evaluation(self, image, configuration):
        return self._result_for(image)

    def get_detection_class_catalog(self):
        return DetectionClassCatalog(
            model_id=self.id,
            model_name=self.name,
            classes=(DetectionClass(class_id=0, name="person"),),
        )

    def detection_localization_objective(self, candidate, references):
        assert references.source == "pseudo"
        assert len(references.objects) == 1
        self.grad_calls += 1
        return 1.0 - candidate.mean()


def _image():
    return Image.new("RGB", (100, 100), (128, 128, 128))


def _parameters():
    return AttackParameters(epsilon=0.03, alpha=0.01, steps=2)


def _snapshot(*, reviewed):
    return MaterializedDetectionAnnotationSnapshot(
        dataset_id="dataset-1",
        dataset_revision=3 if reviewed else 0,
        snapshot_id="snapshot-1" if reviewed else None,
        images=(
            DetectionSnapshotImage(
                image_id="image-1",
                filename="image-1.png",
                width=100,
                height=100,
                review_status="reviewed_with_objects" if reviewed else "unreviewed",
            ),
        ),
        categories=(
            CanonicalCategory(
                internal_id="category-person",
                external_id=91,
                name="person",
                supercategory=None,
            ),
        ) if reviewed else (),
        annotations=(
            CanonicalObjectAnnotation(
                external_id=1,
                image_id="image-1",
                category_internal_id="category-person",
                bbox_x1=10.0,
                bbox_y1=10.0,
                bbox_x2=30.0,
                bbox_y2=30.0,
                area=400.0,
                iscrowd=0,
                object_internal_id="object-1",
            ),
        ) if reviewed else (),
    )


@pytest.mark.asyncio
async def test_localization_pgd_individual_response_preserves_legacy_and_revision(
    monkeypatch, tmp_path
):
    model = _LocalizationDetector()
    persisted_metadata = []
    persisted_payloads = []

    async def get_image_from_dataset(**kwargs):
        assert kwargs["dataset_id"] == "dataset-1"
        return _image()

    monkeypatch.setattr(
        individual_service, "get_image_from_dataset", get_image_from_dataset
    )
    monkeypatch.setattr(individual_service, "get_active_model_instance", lambda: model)
    monkeypatch.setattr(
        individual_service,
        "create_result_directory",
        lambda **kw: ("result-1", tmp_path),
    )
    monkeypatch.setattr(
        individual_service,
        "save_result_image",
        lambda **kw: "/results/" + kw["filename"],
    )
    monkeypatch.setattr(
        individual_service,
        "save_result_metadata",
        lambda **kw: persisted_metadata.append(kw["metadata"]),
    )
    monkeypatch.setattr(
        individual_service,
        "save_result_payload",
        lambda **kw: persisted_payloads.append(kw["payload"]),
    )

    response = await individual_service.apply_detection_attack(
        DetectionAttackRequest(
            dataset_id="dataset-1",
            image_id="image-1",
            attack_type=AttackType.DETECTION_LOCALIZATION_PGD,
            parameters=_parameters(),
        )
    )

    assert model.detect_calls == 1  # La pseudorreferencia se fija una vez.
    assert model.grad_calls == 2
    assert response.attack_type == AttackType.DETECTION_LOCALIZATION_PGD
    assert response.parameters == _parameters()
    assert response.comparison.original_detection_count == 1
    assert response.comparison.adversarial_detection_count == 1
    assert response.comparison.average_confidence_difference == 0.0
    assert response.comparison.attack_success is False  # Semántica legacy intacta.
    assert response.comparison.perturbation_linf > 0
    assert (
        response.original_detections[0].bbox
        != response.adversarial_detections[0].bbox
    )
    assert persisted_metadata[0]["attack_revision"] == (
        "rf-detr-localization-pgd-linf-pseudo-v1"
    )
    assert persisted_metadata[0]["model_revision"] == model.revision
    assert persisted_metadata[0]["parameters"] == _parameters().model_dump(mode="json")
    assert persisted_payloads[0] == response.model_dump(mode="json")


@pytest.mark.asyncio
@pytest.mark.parametrize("reviewed", [True, False])
async def test_localization_dataset_metrics_preserve_legacy_success(
    monkeypatch, reviewed
):
    model = _LocalizationDetector()
    persisted = []

    async def get_dataset(dataset_id):
        return SimpleNamespace(
            owner_id="user-1",
            images=[SimpleNamespace(id="image-1", filename="image-1.png")],
        )

    async def get_image_from_dataset(**kwargs):
        return _image()

    async def read_detection_annotation_snapshot(dataset_id):
        return _snapshot(reviewed=reviewed)

    async def persist_detection_evaluation(**kwargs):
        persisted.append(kwargs)

    monkeypatch.setattr(evaluation_service, "get_dataset", get_dataset)
    monkeypatch.setattr(
        evaluation_service, "get_image_from_dataset", get_image_from_dataset
    )
    monkeypatch.setattr(
        evaluation_service,
        "read_detection_annotation_snapshot",
        read_detection_annotation_snapshot,
    )
    monkeypatch.setattr(evaluation_service, "get_active_model_instance", lambda: model)
    monkeypatch.setattr(
        evaluation_service, "new_detection_evaluation_id", lambda: "evaluation-1"
    )
    monkeypatch.setattr(
        evaluation_service,
        "save_detection_evaluation_adversarial_image",
        lambda **kw: "/results/adversarial.png",
    )
    monkeypatch.setattr(
        evaluation_service, "persist_detection_evaluation", persist_detection_evaluation
    )
    monkeypatch.setattr(evaluation_service, "log_evaluation_profile", lambda *a: None)

    response = await evaluation_service.evaluate_detection_dataset(
        DetectionEvaluationRequest(
            dataset_id="dataset-1",
            attack_type=AttackType.DETECTION_LOCALIZATION_PGD,
            parameters=_parameters(),
        )
    )

    assert response.attack_type == AttackType.DETECTION_LOCALIZATION_PGD
    assert response.parameters == _parameters()
    # Sin anotaciones revisadas, la evaluación legacy consulta detect()
    # dos veces más: imagen original y adversaria. La primera llamada
    # sigue correspondiendo a la pseudorreferencia del PGD.
    assert model.detect_calls == (1 if reviewed else 3)
    assert model.grad_calls == 2
    assert response.summary.total_original_detections == 1
    assert response.summary.total_adversarial_detections == 1
    assert response.summary.average_confidence_difference == 0.0
    assert response.summary.attack_success_rate == 0.0
    assert response.results[0].attack_success is False
    assert response.results[0].original_detections[0].bbox != (
        response.results[0].adversarial_detections[0].bbox
    )
    assert len(persisted) == 1
    assert persisted[0]["response"] is response
    assert persisted[0]["request"].parameters == _parameters()

    if reviewed:
        assert response.supervised.available is True
        assert response.supervised.original.aggregate.map_50_95 == pytest.approx(1.0)
        assert response.supervised.adversarial.aggregate.map_50_95 == pytest.approx(0.0)
        assert response.supervised.deltas.aggregate.map_50_95.absolute == (
            pytest.approx(1.0)
        )
        assert response.supervised.object_analysis.summary.suppressed_count == 1
    else:
        assert response.supervised.available is False
        assert response.supervised.reason == "no_annotation_snapshot"
        assert response.supervised.original is None
        assert response.supervised.adversarial is None
