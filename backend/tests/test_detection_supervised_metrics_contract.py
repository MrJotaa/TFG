import json
from dataclasses import replace

import pytest
from pydantic import ValidationError

from ml.detection_categories import DetectionClass, DetectionClassCatalog
from ml.models.detection import Detection, DetectionBox, DetectionResult
from schemas.detection_evaluation import DetectionEvaluationRequest
from schemas.detection_metrics import (
    DetectionAnnotationCoverage,
    DetectionClassReference,
    DetectionCocoAggregateMetrics,
    DetectionCocoAnalysis,
    DetectionOperatingPointMetrics,
    DetectionPerClassCocoMetrics,
    DetectionPrecisionRecallCurve,
    DetectionPrecisionRecallPoint,
    DetectionSupervisedConfiguration,
    DetectionSupervisedEffectiveConfiguration,
    DetectionSupervisedUnavailableReason,
)
from services.annotation_importers.base import (
    CanonicalCategory,
    CanonicalObjectAnnotation,
)
from services.detection_annotation_snapshot import (
    DetectionSnapshotImage,
    MaterializedDetectionAnnotationSnapshot,
)
from services.detection_supervised_evaluation import (
    DetectionGroundTruthIdentityError,
    build_detection_supervised_metrics,
    prepare_detection_supervised_evaluation,
)


def _catalog(*classes):
    return DetectionClassCatalog(
        model_id="detector",
        model_name="Detector",
        classes=tuple(
            DetectionClass(class_id=class_id, name=name) for class_id, name in classes
        ),
    )


def _category(internal_id, external_id, name):
    return CanonicalCategory(
        internal_id=internal_id,
        external_id=external_id,
        name=name,
        supercategory=None,
    )


def _annotation(
    external_id,
    *,
    image_id="image-object",
    category_id="category-person",
    iscrowd=0,
):
    return CanonicalObjectAnnotation(
        external_id=external_id,
        image_id=image_id,
        category_internal_id=category_id,
        bbox_x1=10.0,
        bbox_y1=10.0,
        bbox_x2=30.0,
        bbox_y2=30.0,
        area=400.0,
        iscrowd=iscrowd,
        object_internal_id=f"object-{external_id}",
    )


def _images(*, reviewed=True):
    object_status = "reviewed_with_objects" if reviewed else "unreviewed"
    return (
        DetectionSnapshotImage(
            image_id="image-empty",
            filename="empty.png",
            width=100,
            height=100,
            review_status="reviewed_empty" if reviewed else "unreviewed",
        ),
        DetectionSnapshotImage(
            image_id="image-object",
            filename="object.png",
            width=100,
            height=100,
            review_status=object_status,
        ),
        DetectionSnapshotImage(
            image_id="image-unreviewed",
            filename="unreviewed.png",
            width=100,
            height=100,
            review_status="unreviewed",
        ),
    )


def _snapshot(
    *,
    snapshot_id="snapshot-1",
    images=None,
    categories=None,
    annotations=None,
):
    return MaterializedDetectionAnnotationSnapshot(
        dataset_id="dataset-1",
        dataset_revision=2,
        snapshot_id=snapshot_id,
        images=tuple(images if images is not None else _images()),
        categories=tuple(
            categories
            if categories is not None
            else (_category("category-person", 91, "person"),)
        ),
        annotations=tuple(
            annotations if annotations is not None else (_annotation(1001),)
        ),
    )


def _result(*detections):
    return DetectionResult(
        source_width=100,
        source_height=100,
        detections=detections,
    )


def test_preparation_rejects_snapshot_object_without_persisted_internal_id():
    snapshot = _snapshot(
        annotations=(replace(_annotation(1001), object_internal_id=None),),
    )

    _prepared_catalog = _catalog((0, "person"))

    with pytest.raises(
        DetectionGroundTruthIdentityError, match="persisted internal ID"
    ):
        prepare_detection_supervised_evaluation(
            snapshot=snapshot,
            catalog=_prepared_catalog,
        )


def _detection(box=(10.0, 10.0, 30.0, 30.0), confidence=0.9):
    return Detection(
        class_id=0,
        class_name="person",
        confidence=confidence,
        box=DetectionBox(*box),
    )


def test_old_detection_evaluation_request_gets_reproducible_defaults():
    request = DetectionEvaluationRequest(dataset_id="dataset-1")

    assert request.supervised_configuration == DetectionSupervisedConfiguration(
        score_floor=0.001,
        confidence_threshold=0.25,
        iou_threshold=0.5,
    )


@pytest.mark.parametrize(
    "configuration",
    [
        {
            "score_floor": 0.3,
            "confidence_threshold": 0.2,
        },
        {
            "iou_threshold": 0.51,
        },
        {
            "max_detections_per_image": 99,
        },
        {
            "nms_iou_threshold": 0.6,
        },
        {
            "score_floor": 0.3,
            "confidence_threshold": 0.3,
        },
    ],
)
def test_supervised_configuration_rejects_invalid_thresholds(configuration):
    with pytest.raises(ValidationError):
        DetectionSupervisedConfiguration(**configuration)


def test_effective_configuration_preserves_legacy_nms_metadata():
    configuration = DetectionSupervisedEffectiveConfiguration(
        score_floor=0.001,
        confidence_threshold=0.25,
        iou_threshold=0.5,
        nms_iou_threshold=0.7,
    )

    assert configuration.inference_mode == "nms"
    assert configuration.nms_iou_threshold == 0.7


def test_effective_configuration_rejects_nms_for_end_to_end_mode():
    with pytest.raises(ValidationError):
        DetectionSupervisedEffectiveConfiguration(
            inference_mode="end_to_end_nms_free",
            nms_iou_threshold=0.7,
        )


def test_operating_point_contract_accepts_rounded_consistent_metrics():
    metrics = DetectionOperatingPointMetrics(
        true_positives=1,
        false_positives=2,
        false_negatives=3,
        precision=0.333333,
        recall=0.25,
        f1=0.285714,
    )

    assert metrics.precision == 0.333333


@pytest.mark.parametrize(
    "values",
    [
        {
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "precision": 1.0,
            "recall": None,
            "f1": None,
        },
        {
            "true_positives": 1,
            "false_positives": 1,
            "false_negatives": 0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
        },
        {
            "true_positives": 0,
            "false_positives": 1,
            "false_negatives": 1,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        },
    ],
)
def test_operating_point_contract_rejects_incoherent_metrics(values):
    with pytest.raises(ValidationError):
        DetectionOperatingPointMetrics(**values)


def _class_reference():
    return DetectionClassReference(
        category_internal_id="category-person",
        coco_category_id=91,
        category_name="person",
        model_class_id=0,
        model_class_name="person",
    )


def _empty_operating_point():
    return DetectionOperatingPointMetrics(
        true_positives=0,
        false_positives=0,
        false_negatives=0,
        precision=None,
        recall=None,
        f1=None,
    )


def _precision_recall_curve(precision=1.0):
    return DetectionPrecisionRecallCurve(
        iou_thresholds=(0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95),
        area="all",
        max_detections=100,
        points=tuple(
            DetectionPrecisionRecallPoint(
                recall=index / 100.0,
                precision=precision,
            )
            for index in range(101)
        ),
    )


def _positive_class_metrics(
    *,
    support=1,
    true_positives=1,
    false_positives=0,
    false_negatives=0,
    average_precision=1.0,
    curve_precision=1.0,
):
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else None
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives
        else None
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall > 0
        else None
    )
    return DetectionPerClassCocoMetrics(
        reference=_class_reference(),
        support=support,
        images_with_objects=1,
        iscrowd_objects=0,
        average_precision=average_precision,
        average_recall=1.0,
        precision_recall_curve=_precision_recall_curve(curve_precision),
        operating_point=DetectionOperatingPointMetrics(
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives,
            precision=precision,
            recall=recall,
            f1=f1,
        ),
    )


def _annotation_coverage(**overrides):
    values = {
        "total_images": 3,
        "reviewed_images": 2,
        "unreviewed_images": 1,
        "reviewed_empty_images": 1,
        "reviewed_with_objects_images": 1,
        "coverage_percentage": 66.67,
        "total_categories": 2,
        "resolved_categories": 1,
        "unresolved_categories": 1,
        "ambiguous_categories": 0,
        "conflicting_categories": 0,
        "total_objects": 3,
        "evaluable_objects": 1,
        "objects_excluded_by_resolution": 1,
        "iscrowd_objects": 1,
        "object_coverage_percentage": 33.33,
    }
    values.update(overrides)
    return DetectionAnnotationCoverage(**values)


def test_annotation_coverage_contract_accepts_coherent_totals_and_percentages():
    coverage = _annotation_coverage()

    assert coverage.reviewed_images == 2
    assert coverage.object_coverage_percentage == 33.33


@pytest.mark.parametrize(
    "overrides",
    [
        {"reviewed_images": 99},
        {"reviewed_empty_images": 2},
        {"resolved_categories": 2},
        {"evaluable_objects": 3, "objects_excluded_by_resolution": 1},
        {"iscrowd_objects": 4},
        {"coverage_percentage": 50.0},
        {"object_coverage_percentage": 50.0},
    ],
)
def test_annotation_coverage_contract_rejects_incoherent_totals(overrides):
    with pytest.raises(ValidationError):
        _annotation_coverage(**overrides)


def test_per_class_contract_allows_zero_support_with_crowd_objects():
    metrics = DetectionPerClassCocoMetrics(
        reference=_class_reference(),
        support=0,
        images_with_objects=0,
        iscrowd_objects=2,
        average_precision=None,
        average_recall=None,
        precision_recall_curve=None,
        operating_point=_empty_operating_point(),
    )

    assert metrics.support == 0
    assert metrics.iscrowd_objects == 2


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "support": 0,
            "images_with_objects": 1,
        },
        {
            "support": 0,
            "average_precision": 0.0,
        },
        {
            "support": 1,
            "images_with_objects": 1,
        },
    ],
)
def test_per_class_contract_rejects_incoherent_support(overrides):
    values = {
        "reference": _class_reference(),
        "support": 0,
        "images_with_objects": 0,
        "iscrowd_objects": 0,
        "average_precision": None,
        "average_recall": None,
        "precision_recall_curve": None,
        "operating_point": _empty_operating_point(),
    }
    values.update(overrides)

    with pytest.raises(ValidationError):
        DetectionPerClassCocoMetrics(**values)


def test_per_class_contract_accepts_positive_support_with_complete_metrics():
    metrics = _positive_class_metrics()

    assert metrics.support == 1


def test_precision_recall_curve_contract_rejects_empty_points():
    with pytest.raises(ValidationError):
        DetectionPrecisionRecallCurve(
            iou_thresholds=(0.5,),
            area="all",
            max_detections=100,
            points=(),
        )


def test_precision_recall_curve_contract_rejects_non_coco_recall_grid():
    points = list(_precision_recall_curve().points)
    points[50] = DetectionPrecisionRecallPoint(recall=0.51, precision=1.0)

    with pytest.raises(ValidationError):
        DetectionPrecisionRecallCurve(
            iou_thresholds=(0.5,),
            area="all",
            max_detections=100,
            points=tuple(points),
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "support": 1,
            "true_positives": 2,
            "false_negatives": 0,
        },
        {
            "average_precision": 0.5,
            "curve_precision": 1.0,
        },
    ],
)
def test_per_class_contract_rejects_incoherent_counts_and_curve(overrides):
    with pytest.raises(ValidationError):
        _positive_class_metrics(**overrides)


def test_coco_analysis_contract_accepts_aggregate_per_class_sums():
    per_class = _positive_class_metrics()
    analysis = DetectionCocoAnalysis(
        aggregate=DetectionCocoAggregateMetrics(map_50_95=1.0),
        precision_recall_curve=_precision_recall_curve(),
        operating_point=per_class.operating_point,
        per_class=(per_class,),
    )

    assert analysis.operating_point.true_positives == 1


@pytest.mark.parametrize(
    "aggregate,curve,operating_point",
    [
        (
            DetectionCocoAggregateMetrics(map_50_95=1.0),
            _precision_recall_curve(),
            DetectionOperatingPointMetrics(
                true_positives=0,
                false_positives=0,
                false_negatives=0,
                precision=None,
                recall=None,
                f1=None,
            ),
        ),
        (
            DetectionCocoAggregateMetrics(map_50_95=0.5),
            _precision_recall_curve(1.0),
            DetectionOperatingPointMetrics(
                true_positives=1,
                false_positives=0,
                false_negatives=0,
                precision=1.0,
                recall=1.0,
                f1=1.0,
            ),
        ),
        (
            DetectionCocoAggregateMetrics(map_50_95=1.0),
            None,
            DetectionOperatingPointMetrics(
                true_positives=1,
                false_positives=0,
                false_negatives=0,
                precision=1.0,
                recall=1.0,
                f1=1.0,
            ),
        ),
    ],
)
def test_coco_analysis_contract_rejects_incoherent_aggregate(
    aggregate,
    curve,
    operating_point,
):
    _prepared_per_class = (_positive_class_metrics(),)

    with pytest.raises(ValidationError):
        DetectionCocoAnalysis(
            aggregate=aggregate,
            precision_recall_curve=curve,
            operating_point=operating_point,
            per_class=_prepared_per_class,
        )


def test_available_supervised_metrics_include_partial_coverage_and_deltas():
    snapshot = _snapshot(
        categories=(
            _category("category-person", 91, "person"),
            _category("category-unknown", 0, "spaceship"),
        ),
        annotations=(
            _annotation(1001),
            _annotation(
                1002,
                category_id="category-person",
                iscrowd=1,
            ),
            _annotation(
                1003,
                category_id="category-unknown",
            ),
        ),
    )
    context = prepare_detection_supervised_evaluation(
        snapshot=snapshot,
        catalog=_catalog((0, "person")),
    )

    metrics = build_detection_supervised_metrics(
        context=context,
        original_results_by_image_id={
            "image-empty": _result(),
            "image-object": _result(_detection()),
        },
        adversarial_results_by_image_id={
            "image-empty": _result(),
            "image-object": _result(),
        },
        configuration=DetectionSupervisedConfiguration(),
    )

    assert metrics.available is True
    assert metrics.reason is None
    assert metrics.configuration.inference_mode == "end_to_end_nms_free"
    assert metrics.configuration.nms_iou_threshold is None
    assert metrics.configuration.max_detections_per_image == 300
    assert (
        metrics.configuration.interactive_output_confidence_comparison
        == "strictly_greater_than"
    )
    assert "Imágenes sin revisar excluidas" in metrics.warning
    assert "Categorías sin resolver excluidas" in metrics.warning
    assert "Objetos excluidos por resolución de categorías" in metrics.warning
    coverage = metrics.annotation_coverage
    assert coverage.total_images == 3
    assert coverage.reviewed_images == 2
    assert coverage.reviewed_empty_images == 1
    assert coverage.unreviewed_images == 1
    assert coverage.total_categories == 2
    assert coverage.resolved_categories == 1
    assert coverage.unresolved_categories == 1
    assert coverage.total_objects == 3
    assert coverage.evaluable_objects == 1
    assert coverage.objects_excluded_by_resolution == 1
    assert coverage.iscrowd_objects == 1
    assert metrics.original.aggregate.map_50_95 == 1.0
    assert metrics.adversarial.aggregate.map_50_95 == 0.0
    assert metrics.deltas.aggregate.map_50_95.absolute == 1.0
    assert metrics.deltas.aggregate.map_50_95.relative == 1.0
    original_class = metrics.original.per_class[0]
    adversarial_class = metrics.adversarial.per_class[0]
    assert original_class.support == adversarial_class.support == 1
    assert original_class.iscrowd_objects == adversarial_class.iscrowd_objects == 1
    assert original_class.reference.coco_category_id == 91
    assert original_class.reference.model_class_id == 0


@pytest.mark.parametrize(
    ("snapshot", "catalog", "reason"),
    [
        (
            _snapshot(
                snapshot_id=None,
                categories=(),
                annotations=(),
            ),
            None,
            DetectionSupervisedUnavailableReason.NO_ANNOTATION_SNAPSHOT,
        ),
        (
            _snapshot(
                images=_images(reviewed=False),
                annotations=(),
            ),
            _catalog((0, "person")),
            DetectionSupervisedUnavailableReason.NO_REVIEWED_IMAGES,
        ),
        (
            _snapshot(
                categories=(_category("category-unknown", 1, "spaceship"),),
                annotations=(_annotation(1, category_id="category-unknown"),),
            ),
            _catalog((0, "person")),
            DetectionSupervisedUnavailableReason.NO_RESOLVED_CATEGORIES,
        ),
        (
            _snapshot(
                annotations=(_annotation(1, iscrowd=1),),
            ),
            _catalog((0, "person")),
            (DetectionSupervisedUnavailableReason.NO_RESOLVED_GROUND_TRUTH_OBJECTS),
        ),
    ],
)
def test_supervised_unavailability_is_typed(
    snapshot,
    catalog,
    reason,
):
    context = prepare_detection_supervised_evaluation(
        snapshot=snapshot,
        catalog=catalog,
    )

    metrics = build_detection_supervised_metrics(
        context=context,
        original_results_by_image_id={},
        adversarial_results_by_image_id={},
        configuration=DetectionSupervisedConfiguration(),
    )

    assert metrics.available is False
    assert metrics.reason is reason
    assert metrics.original is None
    assert metrics.adversarial is None
    assert metrics.deltas is None


def test_unknown_ambiguous_and_conflicting_categories_are_counted():
    snapshot = _snapshot(
        categories=(
            _category("unknown", 1, "spaceship"),
            _category("ambiguous", 2, "fire truck"),
            _category("person-a", 3, "person"),
            _category("person-b", 4, " PERSON "),
        ),
        annotations=(
            _annotation(1, category_id="unknown"),
            _annotation(2, category_id="ambiguous"),
            _annotation(3, category_id="person-a"),
            _annotation(4, category_id="person-b"),
        ),
    )
    context = prepare_detection_supervised_evaluation(
        snapshot=snapshot,
        catalog=_catalog(
            (0, "person"),
            (2, "fire-truck"),
            (7, "Fire__Truck"),
        ),
    )

    coverage = context.coverage
    assert coverage.resolved_categories == 0
    assert coverage.unresolved_categories == 1
    assert coverage.ambiguous_categories == 1
    assert coverage.conflicting_categories == 2
    assert coverage.objects_excluded_by_resolution == 4
    assert context.unavailable_reason is (
        DetectionSupervisedUnavailableReason.NO_RESOLVED_CATEGORIES
    )


def test_relative_delta_is_null_when_original_metric_is_zero():
    context = prepare_detection_supervised_evaluation(
        snapshot=_snapshot(),
        catalog=_catalog((0, "person")),
    )
    incorrect = _result(_detection(box=(60.0, 60.0, 80.0, 80.0)))

    metrics = build_detection_supervised_metrics(
        context=context,
        original_results_by_image_id={
            "image-empty": _result(),
            "image-object": incorrect,
        },
        adversarial_results_by_image_id={
            "image-empty": _result(),
            "image-object": _result(),
        },
        configuration=DetectionSupervisedConfiguration(),
    )

    delta = metrics.deltas.aggregate.map_50_95
    assert delta.absolute == 0.0
    assert delta.relative is None


def test_reviewed_empty_image_participates_as_a_negative_example():
    context = prepare_detection_supervised_evaluation(
        snapshot=_snapshot(),
        catalog=_catalog((0, "person")),
    )

    metrics = build_detection_supervised_metrics(
        context=context,
        original_results_by_image_id={
            "image-empty": _result(_detection()),
            "image-object": _result(_detection()),
        },
        adversarial_results_by_image_id={
            "image-empty": _result(),
            "image-object": _result(_detection()),
        },
        configuration=DetectionSupervisedConfiguration(),
    )

    assert metrics.original.operating_point.true_positives == 1
    assert metrics.original.operating_point.false_positives == 1
    assert metrics.adversarial.operating_point.true_positives == 1
    assert metrics.adversarial.operating_point.false_positives == 0


def test_supervised_contract_serializes_without_non_finite_values():
    context = prepare_detection_supervised_evaluation(
        snapshot=_snapshot(),
        catalog=_catalog((0, "person")),
    )
    metrics = build_detection_supervised_metrics(
        context=context,
        original_results_by_image_id={
            "image-empty": _result(),
            "image-object": _result(),
        },
        adversarial_results_by_image_id={
            "image-empty": _result(),
            "image-object": _result(),
        },
        configuration=DetectionSupervisedConfiguration(),
    )

    serialized = metrics.model_dump_json()
    assert "NaN" not in serialized
    assert "Infinity" not in serialized
    assert "-1.0" not in serialized
    assert json.loads(serialized)["original"]["operating_point"]["precision"] is None
