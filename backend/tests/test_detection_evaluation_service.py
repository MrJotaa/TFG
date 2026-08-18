from types import SimpleNamespace

import pytest

import services.detection_evaluation_service as detection_evaluation_service
import services.detection_supervised_evaluation as supervised_evaluation
from ml.detection_categories import DetectionClass, DetectionClassCatalog
from ml.models.detection import Detection, DetectionBox, DetectionResult
from schemas.attack import AttackType
from schemas.detection_evaluation import DetectionEvaluationRequest
from schemas.detection_metrics import DetectionSupervisedConfiguration
from services.annotation_importers.base import (
    CanonicalCategory,
    CanonicalObjectAnnotation,
)
from services.detection_annotation_snapshot import (
    DetectionSnapshotImage,
    MaterializedDetectionAnnotationSnapshot,
)
from services.detection_supervised_evaluation import (
    build_legacy_detection_result,
)

TEST_DETECTION_EVALUATION_ID = (
    "detection_evaluation_00000000-0000-4000-8000-000000000001"
)


@pytest.fixture(autouse=True)
def _stub_detection_evaluation_persistence(monkeypatch):
    persisted = []
    saved_adversarial_images = []

    async def fake_persist(*, response, request, owner_id):
        assert response.result_id == TEST_DETECTION_EVALUATION_ID
        assert request.dataset_id == response.dataset_id
        assert owner_id == "user-1"
        persisted.append((response, request, owner_id))

    monkeypatch.setattr(
        detection_evaluation_service,
        "new_detection_evaluation_id",
        lambda: TEST_DETECTION_EVALUATION_ID,
    )
    monkeypatch.setattr(
        detection_evaluation_service,
        "persist_detection_evaluation",
        fake_persist,
    )
    monkeypatch.setattr(
        detection_evaluation_service,
        "save_detection_evaluation_adversarial_image",
        lambda **kwargs: (
            saved_adversarial_images.append(kwargs)
            or (
                f"/results/{kwargs['evaluation_id']}/"
                f"{kwargs['image_id']}_adversarial.png"
            )
        ),
    )
    monkeypatch.setattr(
        detection_evaluation_service,
        "remove_detection_evaluation_visual_artifacts",
        lambda result_id: None,
    )
    return persisted


def _detection(confidence: float) -> Detection:
    return Detection(
        class_id=0,
        class_name="object",
        confidence=confidence,
        box=DetectionBox(x_min=0, y_min=0, x_max=1, y_max=1),
    )


class _FakeAttack:
    id = "random_noise"

    def supports_task_type(self, task_type: str) -> bool:
        return task_type == "detection"

    def apply(self, model_wrapper, image, parameters):
        return {
            "adversarial_image": f"adversarial-{image}",
            "perturbation_linf": 0.03,
            "perturbation_l2": 0.5,
        }


class _FakeDetectionModel:
    id = "detector-1"
    name = "Detector"
    task_type = "detection"

    legacy_calls = []

    def detect(self, image):
        self.legacy_calls.append(image)
        detections_by_image = {
            "image-1": [_detection(0.9), _detection(0.8)],
            "adversarial-image-1": [_detection(0.5)],
            "image-2": [],
            "adversarial-image-2": [_detection(0.6)],
        }
        return DetectionResult(
            source_width=1,
            source_height=1,
            detections=detections_by_image[image],
        )

    def get_detection_class_catalog(self):
        raise AssertionError("catalog must not be read without supervision")


@pytest.mark.asyncio
async def test_detection_evaluation_uses_canonical_deltas_and_legacy_success(
    monkeypatch,
    _stub_detection_evaluation_persistence,
):
    _FakeDetectionModel.legacy_calls.clear()
    dataset = SimpleNamespace(
        owner_id="user-1",
        images=[
            SimpleNamespace(id="image-1", filename="image-1.png"),
            SimpleNamespace(id="image-2", filename="image-2.png"),
        ],
    )

    async def fake_get_dataset(dataset_id: str):
        assert dataset_id == "dataset-1"
        return dataset

    async def fake_get_image_from_dataset(dataset_id: str, image_id: str):
        assert dataset_id == "dataset-1"
        return image_id

    async def fake_read_snapshot(dataset_id: str):
        assert dataset_id == "dataset-1"
        return MaterializedDetectionAnnotationSnapshot(
            dataset_id=dataset_id,
            dataset_revision=0,
            snapshot_id=None,
            images=(
                DetectionSnapshotImage(
                    image_id="image-1",
                    filename="image-1.png",
                    width=1,
                    height=1,
                    review_status="unreviewed",
                ),
                DetectionSnapshotImage(
                    image_id="image-2",
                    filename="image-2.png",
                    width=1,
                    height=1,
                    review_status="unreviewed",
                ),
            ),
            categories=(),
            annotations=(),
        )

    monkeypatch.setattr(
        detection_evaluation_service,
        "get_dataset",
        fake_get_dataset,
    )
    monkeypatch.setattr(
        detection_evaluation_service,
        "get_image_from_dataset",
        fake_get_image_from_dataset,
    )
    monkeypatch.setattr(
        detection_evaluation_service,
        "read_detection_annotation_snapshot",
        fake_read_snapshot,
    )
    monkeypatch.setattr(
        detection_evaluation_service,
        "get_active_model_instance",
        _FakeDetectionModel,
    )
    monkeypatch.setattr(
        detection_evaluation_service,
        "get_attack",
        lambda attack_type: _FakeAttack(),
    )

    response = await detection_evaluation_service.evaluate_detection_dataset(
        DetectionEvaluationRequest(
            dataset_id="dataset-1",
            attack_type=AttackType.RANDOM_NOISE,
        )
    )

    assert response.result_id == TEST_DETECTION_EVALUATION_ID
    assert response.metrics_schema_version == "1.0"
    assert response.delta_convention == "original_minus_adversarial"

    assert response.results[0].detections_difference == 1
    assert response.results[0].average_confidence_difference == 0.35
    assert response.results[0].attack_success is True

    assert response.results[1].detections_difference == -1
    assert response.results[1].average_confidence_difference == -0.6
    assert response.results[1].attack_success is False

    assert response.summary.total_original_detections == 2
    assert response.summary.total_adversarial_detections == 2
    assert response.summary.detections_difference == 0
    assert response.summary.average_confidence_difference == -0.125
    assert response.summary.attack_success_rate == 0.5
    assert response.supervised.available is False
    assert response.supervised.reason == "no_annotation_snapshot"
    assert _FakeDetectionModel.legacy_calls == [
        "image-1",
        "adversarial-image-1",
        "image-2",
        "adversarial-image-2",
    ]
    assert len(_stub_detection_evaluation_persistence) == 1


class _SupervisedFakeModel:
    id = "detector-1"
    name = "Detector"
    task_type = "detection"

    def __init__(self):
        self.catalog_calls = 0
        self.detection_calls = []

    def get_detection_class_catalog(self):
        self.catalog_calls += 1
        return DetectionClassCatalog(
            model_id=self.id,
            model_name=self.name,
            classes=(
                DetectionClass(class_id=0, name="person"),
                DetectionClass(class_id=7, name="unused"),
            ),
        )

    def detect_for_evaluation(self, image, configuration):
        self.detection_calls.append((image, configuration))
        detections = {
            "reviewed": (_detection(0.9),),
            "adversarial-reviewed": (),
            "unreviewed": (_detection(0.8),),
            "adversarial-unreviewed": (_detection(0.7),),
        }[image]
        return DetectionResult(
            source_width=100,
            source_height=100,
            detections=tuple(
                Detection(
                    class_id=detection.class_id,
                    class_name=detection.class_name,
                    confidence=detection.confidence,
                    box=DetectionBox(10, 10, 30, 30),
                )
                for detection in detections
            ),
        )


def _supervised_snapshot():
    return MaterializedDetectionAnnotationSnapshot(
        dataset_id="dataset-1",
        dataset_revision=3,
        snapshot_id="snapshot-1",
        images=(
            DetectionSnapshotImage(
                image_id="reviewed",
                filename="reviewed.png",
                width=100,
                height=100,
                review_status="reviewed_with_objects",
            ),
            DetectionSnapshotImage(
                image_id="unreviewed",
                filename="unreviewed.png",
                width=100,
                height=100,
                review_status="unreviewed",
            ),
        ),
        categories=(
            CanonicalCategory(
                internal_id="category-person",
                external_id=91,
                name="person",
                supercategory=None,
            ),
            CanonicalCategory(
                internal_id="category-unused",
                external_id=3,
                name="unused",
                supercategory=None,
            ),
        ),
        annotations=(
            CanonicalObjectAnnotation(
                external_id=1001,
                image_id="reviewed",
                category_internal_id="category-person",
                bbox_x1=10.0,
                bbox_y1=10.0,
                bbox_x2=30.0,
                bbox_y2=30.0,
                area=400.0,
                iscrowd=0,
                object_internal_id="object-1001",
            ),
        ),
    )


@pytest.mark.asyncio
async def test_detection_evaluation_integrates_supervised_metrics_once(
    monkeypatch,
):
    dataset = SimpleNamespace(
        owner_id="user-1",
        images=[
            SimpleNamespace(id="reviewed", filename="reviewed.png"),
            SimpleNamespace(id="unreviewed", filename="unreviewed.png"),
        ],
    )
    model = _SupervisedFakeModel()
    resolution_calls = 0
    real_resolver = supervised_evaluation.resolve_detection_categories

    def tracked_resolver(**kwargs):
        nonlocal resolution_calls
        resolution_calls += 1
        return real_resolver(**kwargs)

    async def fake_get_dataset(dataset_id):
        return dataset

    async def fake_get_image(dataset_id, image_id):
        return image_id

    async def fake_read_snapshot(dataset_id):
        return _supervised_snapshot()

    monkeypatch.setattr(detection_evaluation_service, "get_dataset", fake_get_dataset)
    monkeypatch.setattr(
        detection_evaluation_service,
        "get_image_from_dataset",
        fake_get_image,
    )
    monkeypatch.setattr(
        detection_evaluation_service,
        "read_detection_annotation_snapshot",
        fake_read_snapshot,
    )
    monkeypatch.setattr(
        detection_evaluation_service,
        "get_active_model_instance",
        lambda: model,
    )
    monkeypatch.setattr(
        detection_evaluation_service,
        "get_attack",
        lambda attack_type: _FakeAttack(),
    )
    monkeypatch.setattr(
        supervised_evaluation,
        "resolve_detection_categories",
        tracked_resolver,
    )

    response = await detection_evaluation_service.evaluate_detection_dataset(
        DetectionEvaluationRequest(
            dataset_id="dataset-1",
            attack_type=AttackType.RANDOM_NOISE,
        )
    )

    supervised = response.supervised
    assert supervised.available is True
    assert "1 unreviewed images were excluded" in supervised.warning
    assert supervised.original.aggregate.map_50_95 == 1.0
    assert supervised.adversarial.aggregate.map_50_95 == 0.0
    assert supervised.deltas.aggregate.map_50_95.absolute == 1.0
    assert supervised.original.per_class[0].support == 1
    assert supervised.original.per_class[1].support == 0
    assert supervised.object_analysis.summary.suppressed_count == 1
    assert supervised.object_analysis.details_included is False
    assert supervised.object_analysis.by_image is None
    assert supervised.object_analysis.objects is None
    assert len(response.results) == 2
    assert response.results[0].attack_success is True
    assert response.results[1].attack_success is True
    assert model.catalog_calls == 1
    assert resolution_calls == 1
    assert len(model.detection_calls) == 4
    assert all(
        configuration == model.detection_calls[0][1]
        for _image, configuration in model.detection_calls
    )
    configuration = model.detection_calls[0][1]
    assert configuration.nms_iou_threshold == 0.7
    assert configuration.max_detections_per_image == 300


def test_legacy_detection_result_uses_strict_historical_threshold():
    scientific_result = DetectionResult(
        source_width=1,
        source_height=1,
        detections=(
            _detection(0.249999),
            _detection(0.25),
            _detection(0.250001),
        ),
    )

    legacy_result = build_legacy_detection_result(scientific_result)

    assert [item.confidence for item in legacy_result.detections] == [0.250001]


@pytest.mark.asyncio
async def test_detection_evaluation_skips_catalog_and_scientific_inference_without_reviewed_images(
    monkeypatch,
):
    dataset = SimpleNamespace(
        owner_id="user-1",
        images=[SimpleNamespace(id="image-1", filename="image-1.png")],
    )
    model = _FakeDetectionModel()
    model.legacy_calls = []

    async def fake_get_dataset(dataset_id):
        return dataset

    async def fake_get_image(dataset_id, image_id):
        return image_id

    async def fake_read_snapshot(dataset_id):
        return MaterializedDetectionAnnotationSnapshot(
            dataset_id=dataset_id,
            dataset_revision=1,
            snapshot_id="snapshot-1",
            images=(
                DetectionSnapshotImage(
                    image_id="image-1",
                    filename="image-1.png",
                    width=1,
                    height=1,
                    review_status="unreviewed",
                ),
            ),
            categories=(),
            annotations=(),
        )

    def forbidden_resolution(**kwargs):
        raise AssertionError("resolution must not run without reviewed images")

    monkeypatch.setattr(detection_evaluation_service, "get_dataset", fake_get_dataset)
    monkeypatch.setattr(
        detection_evaluation_service,
        "get_image_from_dataset",
        fake_get_image,
    )
    monkeypatch.setattr(
        detection_evaluation_service,
        "read_detection_annotation_snapshot",
        fake_read_snapshot,
    )
    monkeypatch.setattr(
        detection_evaluation_service,
        "get_active_model_instance",
        lambda: model,
    )
    monkeypatch.setattr(
        detection_evaluation_service,
        "get_attack",
        lambda attack_type: _FakeAttack(),
    )
    monkeypatch.setattr(
        supervised_evaluation,
        "resolve_detection_categories",
        forbidden_resolution,
    )

    response = await detection_evaluation_service.evaluate_detection_dataset(
        DetectionEvaluationRequest(
            dataset_id="dataset-1",
            attack_type=AttackType.RANDOM_NOISE,
        )
    )

    assert response.supervised.available is False
    assert response.supervised.reason == "no_reviewed_images"
    assert response.supervised.object_analysis is None
    assert model.legacy_calls == ["image-1", "adversarial-image-1"]


@pytest.mark.asyncio
async def test_detection_evaluation_uses_legacy_after_required_resolution_is_unavailable(
    monkeypatch,
):
    dataset = SimpleNamespace(
        owner_id="user-1",
        images=[SimpleNamespace(id="image-1", filename="image-1.png")],
    )

    class _UnavailableAfterCatalogModel(_FakeDetectionModel):
        def __init__(self):
            self.legacy_calls = []
            self.catalog_calls = 0

        def get_detection_class_catalog(self):
            self.catalog_calls += 1
            return DetectionClassCatalog(
                model_id=self.id,
                model_name=self.name,
                classes=(DetectionClass(class_id=0, name="person"),),
            )

    model = _UnavailableAfterCatalogModel()
    resolution_calls = 0
    real_resolver = supervised_evaluation.resolve_detection_categories

    def tracked_resolver(**kwargs):
        nonlocal resolution_calls
        resolution_calls += 1
        return real_resolver(**kwargs)

    async def fake_get_dataset(dataset_id):
        return dataset

    async def fake_get_image(dataset_id, image_id):
        return image_id

    async def fake_read_snapshot(dataset_id):
        return MaterializedDetectionAnnotationSnapshot(
            dataset_id=dataset_id,
            dataset_revision=1,
            snapshot_id="snapshot-1",
            images=(
                DetectionSnapshotImage(
                    image_id="image-1",
                    filename="image-1.png",
                    width=1,
                    height=1,
                    review_status="reviewed_empty",
                ),
            ),
            categories=(
                CanonicalCategory(
                    internal_id="category-spaceship",
                    external_id=99,
                    name="spaceship",
                    supercategory=None,
                ),
            ),
            annotations=(),
        )

    monkeypatch.setattr(detection_evaluation_service, "get_dataset", fake_get_dataset)
    monkeypatch.setattr(
        detection_evaluation_service,
        "get_image_from_dataset",
        fake_get_image,
    )
    monkeypatch.setattr(
        detection_evaluation_service,
        "read_detection_annotation_snapshot",
        fake_read_snapshot,
    )
    monkeypatch.setattr(
        detection_evaluation_service,
        "get_active_model_instance",
        lambda: model,
    )
    monkeypatch.setattr(
        detection_evaluation_service,
        "get_attack",
        lambda attack_type: _FakeAttack(),
    )
    monkeypatch.setattr(
        supervised_evaluation,
        "resolve_detection_categories",
        tracked_resolver,
    )

    response = await detection_evaluation_service.evaluate_detection_dataset(
        DetectionEvaluationRequest(
            dataset_id="dataset-1",
            attack_type=AttackType.RANDOM_NOISE,
        )
    )

    assert response.supervised.available is False
    assert response.supervised.reason == "no_resolved_categories"
    assert model.catalog_calls == 1
    assert resolution_calls == 1
    assert model.legacy_calls == ["image-1", "adversarial-image-1"]


@pytest.mark.parametrize(
    "configuration",
    [
        DetectionSupervisedConfiguration(),
        DetectionSupervisedConfiguration(
            score_floor=0.1,
            confidence_threshold=0.2,
            iou_threshold=0.75,
        ),
        DetectionSupervisedConfiguration(
            score_floor=0.25,
            confidence_threshold=0.9,
            iou_threshold=0.95,
            include_object_level_details=True,
        ),
    ],
)
@pytest.mark.asyncio
async def test_allowed_supervised_configuration_does_not_change_legacy_contract(
    monkeypatch,
    configuration,
):
    dataset = SimpleNamespace(
        owner_id="user-1",
        images=[SimpleNamespace(id="reviewed", filename="reviewed.png")],
    )
    model = _SupervisedFakeModel()

    async def fake_get_dataset(dataset_id):
        return dataset

    async def fake_get_image(dataset_id, image_id):
        return image_id

    async def fake_read_snapshot(dataset_id):
        snapshot = _supervised_snapshot()
        return MaterializedDetectionAnnotationSnapshot(
            dataset_id=snapshot.dataset_id,
            dataset_revision=snapshot.dataset_revision,
            snapshot_id=snapshot.snapshot_id,
            images=(snapshot.images[0],),
            categories=snapshot.categories,
            annotations=snapshot.annotations,
        )

    monkeypatch.setattr(detection_evaluation_service, "get_dataset", fake_get_dataset)
    monkeypatch.setattr(
        detection_evaluation_service,
        "get_image_from_dataset",
        fake_get_image,
    )
    monkeypatch.setattr(
        detection_evaluation_service,
        "read_detection_annotation_snapshot",
        fake_read_snapshot,
    )
    monkeypatch.setattr(
        detection_evaluation_service,
        "get_active_model_instance",
        lambda: model,
    )
    monkeypatch.setattr(
        detection_evaluation_service,
        "get_attack",
        lambda attack_type: _FakeAttack(),
    )

    response = await detection_evaluation_service.evaluate_detection_dataset(
        DetectionEvaluationRequest(
            dataset_id="dataset-1",
            attack_type=AttackType.RANDOM_NOISE,
            supervised_configuration=configuration,
        )
    )

    assert response.results[0].model_dump() == {
        "image_id": "reviewed",
        "filename": "reviewed.png",
        "original_detection_count": 1,
        "adversarial_detection_count": 0,
        "detections_difference": 1,
        "average_original_confidence": 0.9,
        "average_adversarial_confidence": 0.0,
        "average_confidence_difference": 0.9,
        "attack_success": True,
        "perturbation_linf": 0.03,
        "perturbation_l2": 0.5,
        "adversarial_image_url": (
            f"/results/{TEST_DETECTION_EVALUATION_ID}/reviewed_adversarial.png"
        ),
        "original_detections": [
            {
                "label": "object",
                "confidence": 0.9,
                "bbox": {
                    "x1": 10.0,
                    "y1": 10.0,
                    "x2": 30.0,
                    "y2": 30.0,
                },
            }
        ],
        "adversarial_detections": [],
    }
    assert response.summary.attack_success_rate == 1.0
    assert response.summary.total_original_detections == 1
    assert response.summary.total_adversarial_detections == 0
    assert response.supervised.object_analysis.details_included is (
        configuration.include_object_level_details
    )


@pytest.mark.asyncio
async def test_detection_evaluation_propagates_supervised_geometry_error(
    monkeypatch,
):
    dataset = SimpleNamespace(
        owner_id="user-1",
        images=[SimpleNamespace(id="reviewed", filename="reviewed.png")],
    )
    model = _SupervisedFakeModel()

    async def fake_get_image(dataset_id, image_id):
        return image_id

    async def fake_read_snapshot(dataset_id):
        snapshot = _supervised_snapshot()
        return MaterializedDetectionAnnotationSnapshot(
            dataset_id=snapshot.dataset_id,
            dataset_revision=snapshot.dataset_revision,
            snapshot_id=snapshot.snapshot_id,
            images=(snapshot.images[0],),
            categories=snapshot.categories,
            annotations=snapshot.annotations,
        )

    original_detect = model.detect_for_evaluation

    def wrong_geometry(image, configuration):
        result = original_detect(image, configuration)
        return DetectionResult(
            source_width=99,
            source_height=100,
            detections=result.detections,
        )

    model.detect_for_evaluation = wrong_geometry

    async def fake_get_dataset(dataset_id):
        return dataset

    monkeypatch.setattr(detection_evaluation_service, "get_dataset", fake_get_dataset)
    monkeypatch.setattr(
        detection_evaluation_service,
        "get_image_from_dataset",
        fake_get_image,
    )
    monkeypatch.setattr(
        detection_evaluation_service,
        "read_detection_annotation_snapshot",
        fake_read_snapshot,
    )
    monkeypatch.setattr(
        detection_evaluation_service,
        "get_active_model_instance",
        lambda: model,
    )
    monkeypatch.setattr(
        detection_evaluation_service,
        "get_attack",
        lambda attack_type: _FakeAttack(),
    )

    with pytest.raises(RuntimeError, match="does not match snapshot geometry"):
        await detection_evaluation_service.evaluate_detection_dataset(
            DetectionEvaluationRequest(
                dataset_id="dataset-1",
                attack_type=AttackType.RANDOM_NOISE,
            )
        )
