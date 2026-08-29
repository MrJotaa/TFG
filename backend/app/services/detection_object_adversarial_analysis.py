import math
from dataclasses import dataclass
from numbers import Integral, Real

from ml.metrics.coco_detection import (
    CocoEvaluationClass,
    CocoGroundTruth,
    GroundTruthMatchObservation,
)
from ml.models.detection import DetectionBox
from schemas.detection_metrics import (
    DetectionClassReference,
    DetectionMatchedObjectObservation,
    DetectionMetricBoundingBox,
    DetectionObjectAdversarialAnalysis,
    DetectionObjectAggregate,
    DetectionObjectClassAggregate,
    DetectionObjectImageAggregate,
    DetectionObjectOperatingPointConfiguration,
    DetectionObjectOutcome,
    DetectionObjectResult,
    DetectionObjectSizeAggregate,
    DetectionObjectSizeBucket,
)

PUBLIC_OBJECT_METRIC_DECIMAL_PLACES = 6


class DetectionObjectAdversarialAnalysisError(RuntimeError):
    """El universo u observaciones por objeto son estructuralmente incoherentes."""


class DetectionObjectAdversarialAnalysisInputError(
    DetectionObjectAdversarialAnalysisError
):
    """La entrada pura del análisis adversario por objeto es inválida."""


@dataclass(frozen=True, slots=True)
class _ObjectTransition:
    ground_truth: CocoGroundTruth
    category: CocoEvaluationClass
    original: GroundTruthMatchObservation
    adversarial: GroundTruthMatchObservation
    outcome: DetectionObjectOutcome
    size_bucket: DetectionObjectSizeBucket
    confidence_drop: float | None
    iou_drop: float | None

    @property
    def object_id(self) -> str:
        return self.ground_truth.object_internal_id


def _round_metric(value: float | None) -> float | None:
    return (
        round(value, PUBLIC_OBJECT_METRIC_DECIMAL_PLACES) if value is not None else None
    )


def _public_metric_difference(original: float, adversarial: float) -> float:
    """Derive a public delta from the public-precision operands."""
    public_original = round(original, PUBLIC_OBJECT_METRIC_DECIMAL_PLACES)
    public_adversarial = round(adversarial, PUBLIC_OBJECT_METRIC_DECIMAL_PLACES)
    return round(
        public_original - public_adversarial,
        PUBLIC_OBJECT_METRIC_DECIMAL_PLACES,
    )


def _require_non_empty_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DetectionObjectAdversarialAnalysisInputError(
            f"{field_name} must be a non-empty string"
        )
    return value


def _require_real(
    value: object,
    *,
    field_name: str,
    minimum: float | None = None,
    maximum: float | None = None,
    strictly_positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise DetectionObjectAdversarialAnalysisInputError(
            f"{field_name} must be a real number"
        )
    normalized = float(value)
    if not math.isfinite(normalized):
        raise DetectionObjectAdversarialAnalysisInputError(
            f"{field_name} must be finite"
        )
    if strictly_positive and normalized <= 0.0:
        raise DetectionObjectAdversarialAnalysisInputError(
            f"{field_name} must be positive"
        )
    if minimum is not None and normalized < minimum:
        raise DetectionObjectAdversarialAnalysisInputError(
            f"{field_name} must be at least {minimum}"
        )
    if maximum is not None and normalized > maximum:
        raise DetectionObjectAdversarialAnalysisInputError(
            f"{field_name} must not exceed {maximum}"
        )
    return normalized


def _require_tuple(value: object, *, field_name: str) -> tuple:
    if isinstance(value, (str, bytes)):
        raise DetectionObjectAdversarialAnalysisInputError(
            f"{field_name} must be an iterable of contract values"
        )
    try:
        return tuple(value)
    except TypeError as error:
        raise DetectionObjectAdversarialAnalysisInputError(
            f"{field_name} must be an iterable of contract values"
        ) from error


def _reference(value: CocoEvaluationClass) -> DetectionClassReference:
    return DetectionClassReference(
        category_internal_id=value.category_internal_id,
        coco_category_id=value.coco_category_id,
        category_name=value.category_name,
        model_class_id=value.model_class_id,
        model_class_name=value.model_class_name,
    )


def _public_box(value: DetectionBox) -> DetectionMetricBoundingBox:
    return DetectionMetricBoundingBox(
        x1=_round_metric(value.x_min),
        y1=_round_metric(value.y_min),
        x2=_round_metric(value.x_max),
        y2=_round_metric(value.y_max),
    )


def _validate_observation(
    observation: GroundTruthMatchObservation,
    *,
    condition: str,
    confidence_threshold: float,
    iou_threshold: float,
) -> None:
    if not isinstance(observation, GroundTruthMatchObservation):
        raise DetectionObjectAdversarialAnalysisInputError(
            f"{condition} observations contain an invalid value"
        )
    object_id = _require_non_empty_string(
        observation.object_internal_id,
        field_name=f"{condition} observation object_internal_id",
    )
    _require_non_empty_string(
        observation.image_id,
        field_name=f"{condition} observation image_id for object '{object_id}'",
    )
    if not isinstance(observation.category_reference, CocoEvaluationClass):
        raise DetectionObjectAdversarialAnalysisInputError(
            f"{condition} observation category is invalid for object '{object_id}'"
        )
    if not isinstance(observation.matched, bool):
        raise DetectionObjectAdversarialAnalysisInputError(
            f"{condition} observation matched must be boolean for object '{object_id}'"
        )
    values = (
        observation.matched_detection_confidence,
        observation.matched_detection_iou,
        observation.matched_detection_box,
    )
    if observation.matched:
        if any(value is None for value in values):
            raise DetectionObjectAdversarialAnalysisInputError(
                f"matched {condition} observation is incomplete for object "
                f"'{observation.object_internal_id}'"
            )
        confidence = _require_real(
            observation.matched_detection_confidence,
            field_name=f"matched {condition} confidence for object '{object_id}'",
            minimum=0.0,
            maximum=1.0,
        )
        iou = _require_real(
            observation.matched_detection_iou,
            field_name=f"matched {condition} IoU for object '{object_id}'",
            minimum=0.0,
            maximum=1.0,
        )
        if confidence < confidence_threshold:
            raise DetectionObjectAdversarialAnalysisInputError(
                f"matched {condition} confidence is below the operating point "
                f"for object '{object_id}'"
            )
        if iou < iou_threshold:
            raise DetectionObjectAdversarialAnalysisInputError(
                f"matched {condition} IoU is below the operating point for "
                f"object '{object_id}'"
            )
        if not isinstance(observation.matched_detection_box, DetectionBox):
            raise DetectionObjectAdversarialAnalysisInputError(
                f"matched {condition} box is invalid for object "
                f"'{observation.object_internal_id}'"
            )
    elif any(value is not None for value in values):
        raise DetectionObjectAdversarialAnalysisInputError(
            f"unmatched {condition} observation contains match data for object "
            f"'{observation.object_internal_id}'"
        )


def _observations_by_id(
    observations: tuple[GroundTruthMatchObservation, ...],
    *,
    condition: str,
    confidence_threshold: float,
    iou_threshold: float,
) -> dict[str, GroundTruthMatchObservation]:
    result = {}
    for observation in observations:
        _validate_observation(
            observation,
            condition=condition,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
        )
        if observation.object_internal_id in result:
            raise DetectionObjectAdversarialAnalysisInputError(
                f"duplicate {condition} observation for object "
                f"'{observation.object_internal_id}'"
            )
        result[observation.object_internal_id] = observation
    return result


def _size_bucket(area: float) -> DetectionObjectSizeBucket:
    if area < 32.0**2:
        return DetectionObjectSizeBucket.SMALL
    if area < 96.0**2:
        return DetectionObjectSizeBucket.MEDIUM
    return DetectionObjectSizeBucket.LARGE


def _outcome(
    original: GroundTruthMatchObservation,
    adversarial: GroundTruthMatchObservation,
) -> DetectionObjectOutcome:
    if original.matched and adversarial.matched:
        return DetectionObjectOutcome.PRESERVED
    if original.matched:
        return DetectionObjectOutcome.SUPPRESSED
    if adversarial.matched:
        return DetectionObjectOutcome.RECOVERED
    return DetectionObjectOutcome.MISSED_BOTH


def _aggregate_values(
    transitions: tuple[_ObjectTransition, ...],
) -> dict[str, object]:
    counts = {
        outcome: sum(item.outcome is outcome for item in transitions)
        for outcome in DetectionObjectOutcome
    }
    preserved = counts[DetectionObjectOutcome.PRESERVED]
    suppressed = counts[DetectionObjectOutcome.SUPPRESSED]
    recovered = counts[DetectionObjectOutcome.RECOVERED]
    originally_detected = preserved + suppressed
    adversarially_detected = preserved + recovered
    total = len(transitions)

    def fraction(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    def preserved_mean(field_name: str) -> float | None:
        if not preserved:
            return None
        return (
            sum(
                getattr(item, field_name)
                for item in transitions
                if item.outcome is DetectionObjectOutcome.PRESERVED
            )
            / preserved
        )

    return {
        "total_objects": total,
        "preserved_count": preserved,
        "suppressed_count": suppressed,
        "recovered_count": recovered,
        "missed_both_count": counts[DetectionObjectOutcome.MISSED_BOTH],
        "originally_detected_count": originally_detected,
        "adversarially_detected_count": adversarially_detected,
        "preservation_rate": _round_metric(fraction(preserved, originally_detected)),
        "suppression_rate": _round_metric(fraction(suppressed, originally_detected)),
        "original_detection_rate": _round_metric(fraction(originally_detected, total)),
        "adversarial_detection_rate": _round_metric(
            fraction(adversarially_detected, total)
        ),
        "mean_confidence_drop_preserved": _round_metric(
            preserved_mean("confidence_drop")
        ),
        "mean_iou_drop_preserved": _round_metric(preserved_mean("iou_drop")),
    }


def _public_observation(
    observation: GroundTruthMatchObservation,
) -> DetectionMatchedObjectObservation | None:
    if not observation.matched:
        return None
    return DetectionMatchedObjectObservation(
        confidence=_round_metric(observation.matched_detection_confidence),
        iou_with_ground_truth=_round_metric(observation.matched_detection_iou),
        detection_box=_public_box(observation.matched_detection_box),
    )


def build_detection_object_adversarial_analysis(
    *,
    ground_truths: tuple[CocoGroundTruth, ...],
    classes: tuple[CocoEvaluationClass, ...],
    reviewed_image_ids: frozenset[str],
    original_observations: tuple[GroundTruthMatchObservation, ...],
    adversarial_observations: tuple[GroundTruthMatchObservation, ...],
    confidence_threshold: float,
    iou_threshold: float,
    include_details: bool,
) -> DetectionObjectAdversarialAnalysis:
    """Compara dos colecciones oficiales de matches sin ejecutar matching propio."""
    confidence_threshold, iou_threshold = _validated_operating_point(
        confidence_threshold,
        iou_threshold,
    )
    if not isinstance(include_details, bool):
        raise DetectionObjectAdversarialAnalysisInputError(
            "include_details must be boolean"
        )

    classes = _require_tuple(classes, field_name="classes")
    ground_truths = _require_tuple(ground_truths, field_name="ground_truths")
    original_observations = _require_tuple(
        original_observations,
        field_name="original_observations",
    )
    adversarial_observations = _require_tuple(
        adversarial_observations,
        field_name="adversarial_observations",
    )
    reviewed_image_ids = _validated_reviewed_image_ids(reviewed_image_ids)
    class_by_id = _class_catalog_by_id(classes)
    ground_truth_by_id = _evaluable_ground_truth_by_id(
        ground_truths,
        reviewed_image_ids=reviewed_image_ids,
        class_by_id=class_by_id,
    )
    original_by_id = _observations_by_id(
        original_observations,
        condition="original",
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
    )
    adversarial_by_id = _observations_by_id(
        adversarial_observations,
        condition="adversarial",
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
    )
    _validate_observation_universe(
        ground_truth_by_id,
        original_by_id,
        adversarial_by_id,
    )

    transitions = _build_transitions(
        ground_truth_by_id=ground_truth_by_id,
        class_by_id=class_by_id,
        original_by_id=original_by_id,
        adversarial_by_id=adversarial_by_id,
    )
    summary = DetectionObjectAggregate(**_aggregate_values(transitions))
    ordered_classes, size_order, grouped = _group_transitions(
        classes=classes,
        reviewed_image_ids=reviewed_image_ids,
        transitions=transitions,
    )
    by_class, by_size = _dimension_aggregates(
        ordered_classes=ordered_classes,
        size_order=size_order,
        transitions_by_class_id=grouped["class"],
        transitions_by_size=grouped["size"],
    )
    by_image, objects = _detail_collections(
        include_details=include_details,
        transitions=transitions,
        transitions_by_image_id=grouped["image"],
    )
    return DetectionObjectAdversarialAnalysis(
        operating_point_configuration=DetectionObjectOperatingPointConfiguration(
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
        ),
        details_included=include_details,
        total_evaluable_objects=len(transitions),
        summary=summary,
        by_class=by_class,
        by_size=by_size,
        by_image=by_image,
        objects=objects,
    )


def _validated_operating_point(
    confidence_threshold: float,
    iou_threshold: float,
) -> tuple[float, float]:
    return (
        _require_real(
            confidence_threshold,
            field_name="confidence_threshold",
            minimum=0.0,
            maximum=1.0,
        ),
        _require_real(
            iou_threshold,
            field_name="iou_threshold",
            minimum=0.0,
            maximum=1.0,
        ),
    )


def _validated_reviewed_image_ids(value: object) -> frozenset[str]:
    if isinstance(value, (str, bytes)):
        raise DetectionObjectAdversarialAnalysisInputError(
            "reviewed_image_ids must be an iterable of image IDs"
        )
    try:
        reviewed_image_ids = frozenset(value)
    except TypeError as error:
        raise DetectionObjectAdversarialAnalysisInputError(
            "reviewed_image_ids must be an iterable of image IDs"
        ) from error
    for image_id in reviewed_image_ids:
        _require_non_empty_string(
            image_id,
            field_name="reviewed image ID",
        )
    return reviewed_image_ids


def _class_catalog_by_id(
    classes: tuple[CocoEvaluationClass, ...],
) -> dict[int, CocoEvaluationClass]:
    class_by_id: dict[int, CocoEvaluationClass] = {}
    for model_class in classes:
        _validate_resolved_class(model_class)
        if model_class.model_class_id in class_by_id:
            raise DetectionObjectAdversarialAnalysisInputError(
                "resolved model class IDs must be unique"
            )
        class_by_id[model_class.model_class_id] = model_class
    return class_by_id


def _validate_resolved_class(model_class: object) -> None:
    if not isinstance(model_class, CocoEvaluationClass):
        raise DetectionObjectAdversarialAnalysisInputError(
            "classes must contain CocoEvaluationClass values"
        )
    if (
        isinstance(model_class.model_class_id, bool)
        or not isinstance(model_class.model_class_id, Integral)
        or model_class.model_class_id < 0
    ):
        raise DetectionObjectAdversarialAnalysisInputError(
            "resolved model class IDs must be non-negative integers"
        )
    if isinstance(model_class.coco_category_id, bool) or not isinstance(
        model_class.coco_category_id,
        Integral,
    ):
        raise DetectionObjectAdversarialAnalysisInputError(
            "resolved COCO category IDs must be integers"
        )
    for field_name in (
        "category_internal_id",
        "category_name",
        "model_class_name",
    ):
        _require_non_empty_string(
            getattr(model_class, field_name),
            field_name=f"resolved class {field_name}",
        )


def _evaluable_ground_truth_by_id(
    ground_truths: tuple[CocoGroundTruth, ...],
    *,
    reviewed_image_ids: frozenset[str],
    class_by_id: dict[int, CocoEvaluationClass],
) -> dict[str, CocoGroundTruth]:
    ground_truth_by_id: dict[str, CocoGroundTruth] = {}
    all_ground_truth_ids: set[str] = set()
    for ground_truth in ground_truths:
        object_id = _validate_ground_truth(
            ground_truth,
            all_ground_truth_ids=all_ground_truth_ids,
        )
        if ground_truth.iscrowd == 1:
            continue
        _validate_evaluable_ground_truth(
            ground_truth,
            object_id=object_id,
            reviewed_image_ids=reviewed_image_ids,
            class_by_id=class_by_id,
            ground_truth_by_id=ground_truth_by_id,
        )
        ground_truth_by_id[object_id] = ground_truth
    return ground_truth_by_id


def _validate_ground_truth(
    ground_truth: object,
    *,
    all_ground_truth_ids: set[str],
) -> str:
    if not isinstance(ground_truth, CocoGroundTruth):
        raise DetectionObjectAdversarialAnalysisInputError(
            "ground_truths must contain CocoGroundTruth values"
        )
    object_id = _require_non_empty_string(
        ground_truth.object_internal_id,
        field_name="ground-truth object_internal_id",
    )
    if object_id in all_ground_truth_ids:
        raise DetectionObjectAdversarialAnalysisInputError(
            f"duplicate ground-truth ID '{object_id}'"
        )
    all_ground_truth_ids.add(object_id)
    _require_non_empty_string(
        ground_truth.image_id,
        field_name=f"ground-truth image_id for object '{object_id}'",
    )
    if isinstance(ground_truth.model_class_id, bool) or not isinstance(
        ground_truth.model_class_id,
        Integral,
    ):
        raise DetectionObjectAdversarialAnalysisInputError(
            f"ground truth '{object_id}' has an invalid model class ID"
        )
    if not isinstance(ground_truth.box, DetectionBox):
        raise DetectionObjectAdversarialAnalysisInputError(
            f"ground truth '{object_id}' has an invalid box"
        )
    _require_real(
        ground_truth.area,
        field_name=f"ground-truth area for object '{object_id}'",
        strictly_positive=True,
    )
    if (
        isinstance(ground_truth.iscrowd, bool)
        or not isinstance(ground_truth.iscrowd, Integral)
        or ground_truth.iscrowd not in (0, 1)
    ):
        raise DetectionObjectAdversarialAnalysisInputError(
            f"ground truth '{object_id}' has invalid iscrowd"
        )
    return object_id


def _validate_evaluable_ground_truth(
    ground_truth: CocoGroundTruth,
    *,
    object_id: str,
    reviewed_image_ids: frozenset[str],
    class_by_id: dict[int, CocoEvaluationClass],
    ground_truth_by_id: dict[str, CocoGroundTruth],
) -> None:
    if object_id in ground_truth_by_id:
        raise DetectionObjectAdversarialAnalysisInputError(
            f"duplicate evaluable ground-truth ID '{object_id}'"
        )
    if ground_truth.image_id not in reviewed_image_ids:
        raise DetectionObjectAdversarialAnalysisInputError(
            f"ground truth '{object_id}' references an unreviewed image"
        )
    if ground_truth.model_class_id not in class_by_id:
        raise DetectionObjectAdversarialAnalysisInputError(
            f"ground truth '{object_id}' references an unresolved class"
        )


def _validate_observation_universe(
    ground_truth_by_id: dict[str, CocoGroundTruth],
    original_by_id: dict[str, GroundTruthMatchObservation],
    adversarial_by_id: dict[str, GroundTruthMatchObservation],
) -> None:
    expected_ids = set(ground_truth_by_id)
    if set(original_by_id) != expected_ids or set(adversarial_by_id) != expected_ids:
        raise DetectionObjectAdversarialAnalysisInputError(
            "original and adversarial observations must cover exactly the GT universe"
        )


def _build_transitions(
    *,
    ground_truth_by_id: dict[str, CocoGroundTruth],
    class_by_id: dict[int, CocoEvaluationClass],
    original_by_id: dict[str, GroundTruthMatchObservation],
    adversarial_by_id: dict[str, GroundTruthMatchObservation],
) -> tuple[_ObjectTransition, ...]:
    return tuple(
        _build_transition(
            object_id=object_id,
            ground_truth=ground_truth,
            category=class_by_id[ground_truth.model_class_id],
            original=original_by_id[object_id],
            adversarial=adversarial_by_id[object_id],
        )
        for object_id, ground_truth in sorted(
            ground_truth_by_id.items(),
            key=lambda item: (item[1].image_id, item[0]),
        )
    )


def _build_transition(
    *,
    object_id: str,
    ground_truth: CocoGroundTruth,
    category: CocoEvaluationClass,
    original: GroundTruthMatchObservation,
    adversarial: GroundTruthMatchObservation,
) -> _ObjectTransition:
    for condition, observation in (
        ("original", original),
        ("adversarial", adversarial),
    ):
        if (
            observation.image_id != ground_truth.image_id
            or observation.category_reference != category
        ):
            raise DetectionObjectAdversarialAnalysisInputError(
                f"{condition} observation references disagree for object "
                f"'{object_id}'"
            )

    outcome = _outcome(original, adversarial)
    confidence_drop = None
    iou_drop = None
    if outcome is DetectionObjectOutcome.PRESERVED:
        confidence_drop = _public_metric_difference(
            original.matched_detection_confidence,
            adversarial.matched_detection_confidence,
        )
        iou_drop = _public_metric_difference(
            original.matched_detection_iou,
            adversarial.matched_detection_iou,
        )
    return _ObjectTransition(
        ground_truth=ground_truth,
        category=category,
        original=original,
        adversarial=adversarial,
        outcome=outcome,
        size_bucket=_size_bucket(ground_truth.area),
        confidence_drop=confidence_drop,
        iou_drop=iou_drop,
    )


def _group_transitions(
    *,
    classes: tuple[CocoEvaluationClass, ...],
    reviewed_image_ids: frozenset[str],
    transitions: tuple[_ObjectTransition, ...],
):
    ordered_classes = tuple(
        sorted(
            classes,
            key=lambda item: (item.model_class_id, item.category_internal_id),
        )
    )
    size_order = (
        DetectionObjectSizeBucket.SMALL,
        DetectionObjectSizeBucket.MEDIUM,
        DetectionObjectSizeBucket.LARGE,
    )
    grouped = {
        "class": {
            model_class.model_class_id: [] for model_class in ordered_classes
        },
        "size": {size_bucket: [] for size_bucket in size_order},
        "image": {image_id: [] for image_id in sorted(reviewed_image_ids)},
    }
    for transition in transitions:
        grouped["class"][transition.category.model_class_id].append(transition)
        grouped["size"][transition.size_bucket].append(transition)
        grouped["image"][transition.ground_truth.image_id].append(transition)
    return ordered_classes, size_order, grouped


def _dimension_aggregates(
    *,
    ordered_classes: tuple[CocoEvaluationClass, ...],
    size_order: tuple[DetectionObjectSizeBucket, ...],
    transitions_by_class_id: dict[int, list[_ObjectTransition]],
    transitions_by_size: dict[DetectionObjectSizeBucket, list[_ObjectTransition]],
) -> tuple[
    tuple[DetectionObjectClassAggregate, ...],
    tuple[DetectionObjectSizeAggregate, ...],
]:
    by_class = tuple(
        DetectionObjectClassAggregate(
            category=_reference(model_class),
            **_aggregate_values(
                tuple(transitions_by_class_id[model_class.model_class_id])
            ),
        )
        for model_class in ordered_classes
    )
    by_size = tuple(
        DetectionObjectSizeAggregate(
            size_bucket=size_bucket,
            **_aggregate_values(tuple(transitions_by_size[size_bucket])),
        )
        for size_bucket in size_order
    )
    return by_class, by_size


def _detail_collections(
    *,
    include_details: bool,
    transitions: tuple[_ObjectTransition, ...],
    transitions_by_image_id: dict[str, list[_ObjectTransition]],
) -> tuple[
    tuple[DetectionObjectImageAggregate, ...] | None,
    tuple[DetectionObjectResult, ...] | None,
]:
    if not include_details:
        return None, None

    by_image = tuple(
        _image_aggregate(image_id, tuple(image_transitions))
        for image_id, image_transitions in transitions_by_image_id.items()
    )
    objects = tuple(_public_object_result(item) for item in transitions)
    return by_image, objects


def _image_aggregate(
    image_id: str,
    transitions: tuple[_ObjectTransition, ...],
) -> DetectionObjectImageAggregate:
    aggregate_values = _aggregate_values(transitions)
    return DetectionObjectImageAggregate(
        image_id=image_id,
        object_level_attack_success=_object_level_attack_success(aggregate_values),
        **aggregate_values,
    )


def _object_level_attack_success(aggregate_values: dict[str, object]) -> bool | None:
    if aggregate_values["suppressed_count"] > 0:
        return True
    if aggregate_values["originally_detected_count"] > 0:
        return False
    return None


def _public_object_result(item: _ObjectTransition) -> DetectionObjectResult:
    return DetectionObjectResult(
        object_id=item.object_id,
        image_id=item.ground_truth.image_id,
        category=_reference(item.category),
        ground_truth_box=_public_box(item.ground_truth.box),
        area=_round_metric(item.ground_truth.area),
        size_bucket=item.size_bucket,
        outcome=item.outcome,
        original_observation=_public_observation(item.original),
        adversarial_observation=_public_observation(item.adversarial),
        confidence_drop=_round_metric(item.confidence_drop),
        iou_drop=_round_metric(item.iou_drop),
    )
