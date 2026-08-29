from dataclasses import dataclass, fields

from ml.detection_categories import (
    DetectionCategoryResolutionStatus,
    DetectionClassCatalog,
    resolve_detection_categories,
)
from ml.metrics.coco_detection import (
    CocoAggregateMetrics,
    CocoDetectionAnalysis,
    CocoEvaluationClass,
    CocoEvaluationImage,
    CocoGroundTruth,
    CocoOperatingPointConfiguration,
    CocoPerClassMetrics,
    CocoPrediction,
    OperatingPointMetrics,
    PrecisionRecallCurve,
    evaluate_coco_detections,
)
from ml.metrics.tide_detection import TideDetectionAnalysis, evaluate_tide_detections
from ml.models.detection import DetectionBox, DetectionResult
from ml.models.detection_inference import (
    LEGACY_DETECTION_CONFIDENCE_THRESHOLD,
)
from schemas.detection_metrics import (
    DetectionAnnotationCoverage,
    DetectionClassReference,
    DetectionCocoAggregateDeltas,
    DetectionCocoAggregateMetrics,
    DetectionCocoAnalysis,
    DetectionCocoDeltas,
    DetectionMetricDelta,
    DetectionOperatingPointDeltas,
    DetectionOperatingPointMetrics,
    DetectionPerClassCocoMetrics,
    DetectionPerClassDeltas,
    DetectionPrecisionRecallCurve,
    DetectionPrecisionRecallPoint,
    DetectionSupervisedConfiguration,
    DetectionSupervisedEffectiveConfiguration,
    DetectionSupervisedMetrics,
    DetectionSupervisedUnavailableReason,
    DetectionTideAnalysis,
    DetectionTideConfiguration,
    DetectionTideErrorChange,
    DetectionTideErrorMetric,
    DetectionTideRun,
)
from services.detection_annotation_snapshot import (
    MaterializedDetectionAnnotationSnapshot,
)
from services.detection_object_adversarial_analysis import (
    build_detection_object_adversarial_analysis,
)

PUBLIC_METRIC_DECIMAL_PLACES = 6
TIDE_ERROR_TYPE_BY_CODE = {
    "Cls": "classification",
    "Loc": "localization",
    "Both": "classification_localization",
    "Dupe": "duplicate",
    "Bkg": "background",
    "Miss": "missed",
    "FalsePos": "false_positive",
    "FalseNeg": "false_negative",
}


class DetectionSupervisedEvaluationError(RuntimeError):
    """Indica una incoherencia interna al preparar métricas supervisadas."""


class DetectionEvaluationGeometryError(DetectionSupervisedEvaluationError):
    """La geometría predicha no coincide con el snapshot autoritativo."""


class DetectionGroundTruthIdentityError(DetectionSupervisedEvaluationError):
    """Un objeto materializado perdió su identidad interna persistida."""


@dataclass(frozen=True, slots=True)
class PreparedDetectionSupervisedEvaluation:
    snapshot: MaterializedDetectionAnnotationSnapshot
    coverage: DetectionAnnotationCoverage
    classes: tuple[CocoEvaluationClass, ...]
    images: tuple[CocoEvaluationImage, ...]
    ground_truths: tuple[CocoGroundTruth, ...]
    unavailable_reason: DetectionSupervisedUnavailableReason | None
    warning: str | None

    @property
    def available(self) -> bool:
        return self.unavailable_reason is None


def _percentage(numerator: int, denominator: int) -> float:
    return round(numerator / denominator * 100.0, 2) if denominator else 0.0


def _effective_configuration(
    configuration: DetectionSupervisedConfiguration,
) -> DetectionSupervisedEffectiveConfiguration:
    return DetectionSupervisedEffectiveConfiguration(
        **configuration.model_dump(),
    )


def requires_detection_supervised_catalog(
    snapshot: MaterializedDetectionAnnotationSnapshot,
) -> bool:
    """Indica si la disponibilidad ya no puede decidirse sin catálogo."""
    return snapshot.has_active_snapshot and bool(snapshot.reviewed_image_ids)


def prepare_detection_supervised_evaluation(
    *,
    snapshot: MaterializedDetectionAnnotationSnapshot,
    catalog: DetectionClassCatalog | None,
) -> PreparedDetectionSupervisedEvaluation:
    _validate_object_identities(snapshot)
    status_counts = _review_status_counts(snapshot)
    reviewed_images = (
        status_counts["reviewed_empty"] + status_counts["reviewed_with_objects"]
    )
    resolution, class_id_by_category, resolved_matches = _category_resolution(
        snapshot,
        catalog,
    )
    coverage = _annotation_coverage(
        snapshot=snapshot,
        status_counts=status_counts,
        reviewed_images=reviewed_images,
        resolution=resolution,
        class_id_by_category=class_id_by_category,
        resolved_matches=resolved_matches,
    )
    unavailable_reason = _supervised_unavailable_reason(
        snapshot=snapshot,
        reviewed_images=reviewed_images,
        resolved_matches=resolved_matches,
        evaluable_objects=coverage.evaluable_objects,
    )
    warning = _supervised_warning(
        status_counts=status_counts,
        resolution=resolution,
        objects_excluded_by_resolution=coverage.objects_excluded_by_resolution,
    )
    classes, images, ground_truths = _coco_inputs(
        snapshot=snapshot,
        resolved_matches=resolved_matches,
        class_id_by_category=class_id_by_category,
    )
    return PreparedDetectionSupervisedEvaluation(
        snapshot=snapshot,
        coverage=coverage,
        classes=classes,
        images=images,
        ground_truths=ground_truths,
        unavailable_reason=unavailable_reason,
        warning=warning,
    )


def _validate_object_identities(
    snapshot: MaterializedDetectionAnnotationSnapshot,
) -> None:
    for annotation in snapshot.annotations:
        if (
            not isinstance(annotation.object_internal_id, str)
            or not annotation.object_internal_id.strip()
        ):
            raise DetectionGroundTruthIdentityError(
                "materialized annotation snapshot contains an object without "
                "a persisted internal ID"
            )


def _review_status_counts(
    snapshot: MaterializedDetectionAnnotationSnapshot,
) -> dict[str, int]:
    status_counts = {
        "unreviewed": 0,
        "reviewed_empty": 0,
        "reviewed_with_objects": 0,
    }
    for image in snapshot.images:
        status_counts[image.review_status] += 1
    return status_counts


def _category_resolution(
    snapshot: MaterializedDetectionAnnotationSnapshot,
    catalog: DetectionClassCatalog | None,
):
    if not requires_detection_supervised_catalog(snapshot):
        return None, {}, ()

    if catalog is None:
        raise DetectionSupervisedEvaluationError(
            "a reviewed active annotation snapshot requires a detection "
            "class catalog"
        )
    resolution = resolve_detection_categories(
        catalog=catalog,
        categories=snapshot.categories,
    )
    resolved_matches = tuple(
        match
        for match in resolution.matches
        if match.status is DetectionCategoryResolutionStatus.RESOLVED
    )
    return (
        resolution,
        resolution.class_id_by_category_internal_id,
        resolved_matches,
    )


def _annotation_coverage(
    *,
    snapshot: MaterializedDetectionAnnotationSnapshot,
    status_counts: dict[str, int],
    reviewed_images: int,
    resolution,
    class_id_by_category: dict[str, int],
    resolved_matches: tuple,
) -> DetectionAnnotationCoverage:
    annotations_by_category: dict[str, list] = {}
    for annotation in snapshot.annotations:
        annotations_by_category.setdefault(
            annotation.category_internal_id,
            [],
        ).append(annotation)

    total_objects = len(snapshot.annotations)
    excluded_objects = sum(
        len(annotations)
        for category_id, annotations in annotations_by_category.items()
        if category_id not in class_id_by_category
    )
    evaluable_objects = sum(
        1
        for annotation in snapshot.annotations
        if annotation.category_internal_id in class_id_by_category
        and not annotation.iscrowd
    )
    unresolved_categories = (
        sum(
            match.status is DetectionCategoryResolutionStatus.UNRESOLVED
            for match in resolution.matches
        )
        if resolution is not None
        else 0
    )
    ambiguous_categories = resolution.ambiguity_count if resolution is not None else 0
    conflicting_categories = resolution.conflict_count if resolution is not None else 0
    return DetectionAnnotationCoverage(
        total_images=len(snapshot.images),
        reviewed_images=reviewed_images,
        unreviewed_images=status_counts["unreviewed"],
        reviewed_empty_images=status_counts["reviewed_empty"],
        reviewed_with_objects_images=status_counts["reviewed_with_objects"],
        coverage_percentage=_percentage(reviewed_images, len(snapshot.images)),
        total_categories=len(snapshot.categories),
        resolved_categories=len(resolved_matches),
        unresolved_categories=unresolved_categories,
        ambiguous_categories=ambiguous_categories,
        conflicting_categories=conflicting_categories,
        total_objects=total_objects,
        evaluable_objects=evaluable_objects,
        objects_excluded_by_resolution=excluded_objects,
        iscrowd_objects=sum(bool(item.iscrowd) for item in snapshot.annotations),
        object_coverage_percentage=_percentage(evaluable_objects, total_objects),
    )


def _supervised_unavailable_reason(
    *,
    snapshot: MaterializedDetectionAnnotationSnapshot,
    reviewed_images: int,
    resolved_matches: tuple,
    evaluable_objects: int,
) -> DetectionSupervisedUnavailableReason | None:
    if not snapshot.has_active_snapshot:
        return DetectionSupervisedUnavailableReason.NO_ANNOTATION_SNAPSHOT
    if reviewed_images == 0:
        return DetectionSupervisedUnavailableReason.NO_REVIEWED_IMAGES
    if not resolved_matches:
        return DetectionSupervisedUnavailableReason.NO_RESOLVED_CATEGORIES
    if evaluable_objects == 0:
        return DetectionSupervisedUnavailableReason.NO_RESOLVED_GROUND_TRUTH_OBJECTS
    return None


def _supervised_warning(
    *,
    status_counts: dict[str, int],
    resolution,
    objects_excluded_by_resolution: int,
) -> str | None:
    warnings = []
    if status_counts["unreviewed"]:
        warnings.append(
            f"Imágenes sin revisar excluidas: {status_counts['unreviewed']}"
        )
    unresolved_total = 0
    if resolution is not None:
        unresolved_total = (
            sum(
                match.status is DetectionCategoryResolutionStatus.UNRESOLVED
                for match in resolution.matches
            )
            + resolution.ambiguity_count
            + resolution.conflict_count
        )
    if unresolved_total:
        warnings.append(f"Categorías sin resolver excluidas: {unresolved_total}")
    if objects_excluded_by_resolution:
        warnings.append(
            "Objetos excluidos por resolución de categorías: "
            f"{objects_excluded_by_resolution}"
        )
    return "; ".join(warnings) if warnings else None


def _coco_inputs(
    *,
    snapshot: MaterializedDetectionAnnotationSnapshot,
    resolved_matches: tuple,
    class_id_by_category: dict[str, int],
) -> tuple[
    tuple[CocoEvaluationClass, ...],
    tuple[CocoEvaluationImage, ...],
    tuple[CocoGroundTruth, ...],
]:
    classes = tuple(
        CocoEvaluationClass(
            category_internal_id=match.category_internal_id,
            coco_category_id=match.coco_category_id,
            category_name=match.category_name,
            model_class_id=match.resolved_class.class_id,
            model_class_name=match.resolved_class.name,
        )
        for match in sorted(
            resolved_matches,
            key=lambda item: (
                item.resolved_class.class_id,
                item.category_internal_id,
            ),
        )
    )
    reviewed_ids = snapshot.reviewed_image_ids
    images = tuple(
        CocoEvaluationImage(
            image_id=image.image_id,
            width=image.width,
            height=image.height,
        )
        for image in snapshot.images
        if image.image_id in reviewed_ids
        and image.width is not None
        and image.height is not None
    )
    ground_truths = tuple(
        CocoGroundTruth(
            annotation_id=index,
            object_internal_id=annotation.object_internal_id,
            image_id=annotation.image_id,
            model_class_id=class_id_by_category[annotation.category_internal_id],
            box=DetectionBox(
                x_min=annotation.bbox_x1,
                y_min=annotation.bbox_y1,
                x_max=annotation.bbox_x2,
                y_max=annotation.bbox_y2,
            ),
            area=annotation.area,
            iscrowd=annotation.iscrowd,
        )
        for index, annotation in enumerate(snapshot.annotations, start=1)
        if annotation.image_id in reviewed_ids
        and annotation.category_internal_id in class_id_by_category
    )
    return classes, images, ground_truths



def _predictions_for_context(
    *,
    context: PreparedDetectionSupervisedEvaluation,
    results_by_image_id: dict[str, DetectionResult],
) -> tuple[CocoPrediction, ...]:
    class_ids = {model_class.model_class_id for model_class in context.classes}
    predictions = []
    for image in context.images:
        result = results_by_image_id.get(image.image_id)
        if result is None:
            raise DetectionSupervisedEvaluationError(
                f"missing prediction for reviewed image '{image.image_id}'"
            )
        if result.source_width != image.width or result.source_height != image.height:
            raise DetectionEvaluationGeometryError(
                f"prediction geometry {result.source_width}x{result.source_height} "
                f"does not match snapshot geometry {image.width}x{image.height} "
                f"for image '{image.image_id}'"
            )
        predictions.extend(
            CocoPrediction(
                image_id=image.image_id,
                detection=detection,
            )
            for detection in result.detections
            if detection.class_id in class_ids
        )
    return tuple(predictions)


def _round_metric(value: float | None) -> float | None:
    return round(value, PUBLIC_METRIC_DECIMAL_PLACES) if value is not None else None


def _reference(
    model_class: CocoEvaluationClass,
) -> DetectionClassReference:
    return DetectionClassReference(
        category_internal_id=model_class.category_internal_id,
        coco_category_id=model_class.coco_category_id,
        category_name=model_class.category_name,
        model_class_id=model_class.model_class_id,
        model_class_name=model_class.model_class_name,
    )


def _curve(curve: PrecisionRecallCurve | None) -> DetectionPrecisionRecallCurve | None:
    if curve is None:
        return None
    return DetectionPrecisionRecallCurve(
        iou_thresholds=curve.iou_thresholds,
        area=curve.area,
        max_detections=curve.max_detections,
        points=tuple(
            DetectionPrecisionRecallPoint(
                recall=_round_metric(point.recall),
                precision=_round_metric(point.precision),
            )
            for point in curve.points
        ),
    )


def _operating(
    metrics: OperatingPointMetrics,
) -> DetectionOperatingPointMetrics:
    return DetectionOperatingPointMetrics(
        true_positives=metrics.true_positives,
        false_positives=metrics.false_positives,
        false_negatives=metrics.false_negatives,
        precision=_round_metric(metrics.precision),
        recall=_round_metric(metrics.recall),
        f1=_round_metric(metrics.f1),
    )


def _aggregate(
    metrics: CocoAggregateMetrics,
) -> DetectionCocoAggregateMetrics:
    return DetectionCocoAggregateMetrics(
        **{
            field.name: _round_metric(getattr(metrics, field.name))
            for field in fields(CocoAggregateMetrics)
        }
    )


def _class_metrics(
    metrics: CocoPerClassMetrics,
) -> DetectionPerClassCocoMetrics:
    return DetectionPerClassCocoMetrics(
        reference=_reference(metrics.reference),
        support=metrics.support,
        images_with_objects=metrics.images_with_objects,
        iscrowd_objects=metrics.iscrowd_objects,
        average_precision=_round_metric(metrics.average_precision),
        average_recall=_round_metric(metrics.average_recall),
        precision_recall_curve=_curve(metrics.precision_recall_curve),
        operating_point=_operating(metrics.operating_point),
    )


def _analysis(metrics: CocoDetectionAnalysis) -> DetectionCocoAnalysis:
    return DetectionCocoAnalysis(
        aggregate=_aggregate(metrics.aggregate),
        precision_recall_curve=_curve(metrics.precision_recall_curve),
        operating_point=_operating(metrics.operating_point),
        per_class=tuple(_class_metrics(item) for item in metrics.per_class),
    )


def _delta(
    original: float | None,
    adversarial: float | None,
) -> DetectionMetricDelta:
    if original is None or adversarial is None:
        return DetectionMetricDelta(absolute=None, relative=None)
    absolute = original - adversarial
    relative = absolute / abs(original) if original != 0 else None
    return DetectionMetricDelta(
        absolute=_round_metric(absolute),
        relative=_round_metric(relative),
    )


def _operating_deltas(
    original: OperatingPointMetrics,
    adversarial: OperatingPointMetrics,
) -> DetectionOperatingPointDeltas:
    return DetectionOperatingPointDeltas(
        precision=_delta(original.precision, adversarial.precision),
        recall=_delta(original.recall, adversarial.recall),
        f1=_delta(original.f1, adversarial.f1),
    )


def _deltas(
    original: CocoDetectionAnalysis,
    adversarial: CocoDetectionAnalysis,
) -> DetectionCocoDeltas:
    aggregate = DetectionCocoAggregateDeltas(
        **{
            field.name: _delta(
                getattr(original.aggregate, field.name),
                getattr(adversarial.aggregate, field.name),
            )
            for field in fields(CocoAggregateMetrics)
        }
    )
    per_class = []
    for original_class, adversarial_class in zip(
        original.per_class,
        adversarial.per_class,
        strict=True,
    ):
        if original_class.reference != adversarial_class.reference:
            raise DetectionSupervisedEvaluationError(
                "COCO analyses contain different class references"
            )
        per_class.append(
            DetectionPerClassDeltas(
                reference=_reference(original_class.reference),
                average_precision=_delta(
                    original_class.average_precision,
                    adversarial_class.average_precision,
                ),
                average_recall=_delta(
                    original_class.average_recall,
                    adversarial_class.average_recall,
                ),
                operating_point=_operating_deltas(
                    original_class.operating_point,
                    adversarial_class.operating_point,
                ),
            )
        )
    return DetectionCocoDeltas(
        aggregate=aggregate,
        operating_point=_operating_deltas(
            original.operating_point,
            adversarial.operating_point,
        ),
        per_class=tuple(per_class),
    )


def _tide_error(metric) -> DetectionTideErrorMetric:
    try:
        error_type = TIDE_ERROR_TYPE_BY_CODE[metric.code]
    except KeyError as error:
        raise DetectionSupervisedEvaluationError(
            f"TIDE returned unsupported error code '{metric.code}'"
        ) from error
    return DetectionTideErrorMetric(
        error_type=error_type,
        delta_ap=_round_metric(metric.delta_ap),
        count=metric.count,
    )


def _tide_run(analysis: TideDetectionAnalysis) -> DetectionTideRun:
    return DetectionTideRun(
        ap_50=_round_metric(analysis.ap_50),
        main_errors=tuple(_tide_error(item) for item in analysis.main_errors),
        special_errors=tuple(_tide_error(item) for item in analysis.special_errors),
    )


def _tide_analysis(
    *,
    original: TideDetectionAnalysis,
    adversarial: TideDetectionAnalysis,
) -> DetectionTideAnalysis:
    original_run = _tide_run(original)
    adversarial_run = _tide_run(adversarial)
    changes = tuple(
        DetectionTideErrorChange(
            error_type=original_error.error_type,
            impact_change=_round_metric(
                adversarial_error.delta_ap - original_error.delta_ap
            ),
            count_change=adversarial_error.count - original_error.count,
        )
        for original_error, adversarial_error in zip(
            original_run.main_errors,
            adversarial_run.main_errors,
            strict=True,
        )
    )
    dominant_metric = max(
        adversarial_run.main_errors,
        key=lambda item: item.delta_ap,
    )
    dominant = dominant_metric.error_type if dominant_metric.delta_ap > 0.0 else None
    return DetectionTideAnalysis(
        configuration=DetectionTideConfiguration(),
        original=original_run,
        adversarial=adversarial_run,
        changes=changes,
        dominant_adversarial_error=dominant,
    )


def build_detection_supervised_metrics(
    *,
    context: PreparedDetectionSupervisedEvaluation,
    original_results_by_image_id: dict[str, DetectionResult],
    adversarial_results_by_image_id: dict[str, DetectionResult],
    configuration: DetectionSupervisedConfiguration,
) -> DetectionSupervisedMetrics:
    effective_configuration = _effective_configuration(configuration)
    if not context.available:
        return DetectionSupervisedMetrics(
            available=False,
            reason=context.unavailable_reason,
            warning=context.warning,
            annotation_coverage=context.coverage,
            configuration=effective_configuration,
            original=None,
            adversarial=None,
            deltas=None,
            object_analysis=None,
            tide_analysis=None,
        )

    operating_point = CocoOperatingPointConfiguration(
        confidence_threshold=configuration.confidence_threshold,
        iou_threshold=configuration.iou_threshold,
    )
    original_predictions = _predictions_for_context(
        context=context,
        results_by_image_id=original_results_by_image_id,
    )
    adversarial_predictions = _predictions_for_context(
        context=context,
        results_by_image_id=adversarial_results_by_image_id,
    )
    original = evaluate_coco_detections(
        images=context.images,
        classes=context.classes,
        ground_truths=context.ground_truths,
        predictions=original_predictions,
        operating_point=operating_point,
    )
    adversarial = evaluate_coco_detections(
        images=context.images,
        classes=context.classes,
        ground_truths=context.ground_truths,
        predictions=adversarial_predictions,
        operating_point=operating_point,
    )
    object_analysis = build_detection_object_adversarial_analysis(
        ground_truths=context.ground_truths,
        classes=context.classes,
        reviewed_image_ids=context.snapshot.reviewed_image_ids,
        original_observations=original.ground_truth_observations,
        adversarial_observations=adversarial.ground_truth_observations,
        confidence_threshold=configuration.confidence_threshold,
        iou_threshold=configuration.iou_threshold,
        include_details=configuration.include_object_level_details,
    )
    original_tide = evaluate_tide_detections(
        images=context.images,
        classes=context.classes,
        ground_truths=context.ground_truths,
        predictions=original_predictions,
        name="original",
    )
    adversarial_tide = evaluate_tide_detections(
        images=context.images,
        classes=context.classes,
        ground_truths=context.ground_truths,
        predictions=adversarial_predictions,
        name="adversarial",
    )

    return DetectionSupervisedMetrics(
        available=True,
        reason=None,
        warning=context.warning,
        annotation_coverage=context.coverage,
        configuration=effective_configuration,
        original=_analysis(original),
        adversarial=_analysis(adversarial),
        deltas=_deltas(original, adversarial),
        object_analysis=object_analysis,
        tide_analysis=_tide_analysis(
            original=original_tide,
            adversarial=adversarial_tide,
        ),
    )


def build_legacy_detection_result(
    result: DetectionResult,
) -> DetectionResult:
    """Reproduce el corte histórico estricto de Ultralytics sin reinferir."""
    return DetectionResult(
        source_width=result.source_width,
        source_height=result.source_height,
        detections=tuple(
            detection
            for detection in result.detections
            if detection.confidence > LEGACY_DETECTION_CONFIDENCE_THRESHOLD
        ),
    )
