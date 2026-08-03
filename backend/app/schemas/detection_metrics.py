import math
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ml.models.detection_inference import (
    DETECTION_EVALUATION_MAX_DETECTIONS_PER_IMAGE,
    DETECTION_EVALUATION_NMS_IOU_THRESHOLD,
    LEGACY_DETECTION_CONFIDENCE_THRESHOLD,
)

PUBLIC_DETECTION_METRIC_TOLERANCE = 1e-6
PUBLIC_DETECTION_PERCENTAGE_DECIMAL_PLACES = 2
PUBLIC_DETECTION_RECALL_THRESHOLD_COUNT = 101


def _fraction_or_none(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _rounded_percentage(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(
        numerator / denominator * 100.0,
        PUBLIC_DETECTION_PERCENTAGE_DECIMAL_PLACES,
    )


def _require_metric_close(
    *,
    field_name: str,
    actual: float,
    expected: float,
) -> None:
    if not math.isclose(
        actual,
        expected,
        rel_tol=0.0,
        abs_tol=PUBLIC_DETECTION_METRIC_TOLERANCE,
    ):
        raise ValueError(f"{field_name} is mathematically inconsistent")


def _require_optional_metric(
    *,
    field_name: str,
    actual: float | None,
    expected: float | None,
) -> None:
    if expected is None:
        if actual is not None:
            raise ValueError(f"{field_name} must be null for this metric state")
        return
    if actual is None or not math.isclose(
        actual,
        expected,
        rel_tol=0.0,
        abs_tol=PUBLIC_DETECTION_METRIC_TOLERANCE,
    ):
        raise ValueError(f"{field_name} is mathematically inconsistent")


class DetectionMetricsContract(BaseModel):
    """
    Metadatos del contrato básico de métricas de detección.

    Los deltas de confianza y número de detecciones siguen la convención
    ``original - adversarial``. El campo público legacy
    ``average_confidence_difference`` corresponde al delta de confianza:
    positivo significa que baja la confianza adversaria, cero que no cambia y
    negativo que sube. Para ``detections_difference``, positivo significa que
    hay menos detecciones adversarias, cero la misma cantidad y negativo más
    detecciones adversarias. Este último signo no implica por sí mismo una
    mejora o degradación de calidad porque todavía no existe ground truth.

    Para futuras métricas, las magnitudes donde un valor mayor es mejor pueden
    expresar degradación como ``original - adversarial``. Las métricas de error,
    donde un valor mayor es peor, deben declarar su dirección o usar nombres no
    ambiguos.
    """

    metrics_schema_version: str = "1.0"
    delta_convention: Literal["original_minus_adversarial"] = (
        "original_minus_adversarial"
    )


class DetectionMetricModel(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)


class DetectionSupervisedUnavailableReason(str, Enum):
    NO_ANNOTATION_SNAPSHOT = "no_annotation_snapshot"
    NO_REVIEWED_IMAGES = "no_reviewed_images"
    NO_RESOLVED_CATEGORIES = "no_resolved_categories"
    NO_RESOLVED_GROUND_TRUTH_OBJECTS = "no_resolved_ground_truth_objects"


class DetectionSupervisedConfiguration(DetectionMetricModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")

    score_floor: float = Field(default=0.001, ge=0.0, le=1.0)
    confidence_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    include_object_level_details: bool = Field(default=False, strict=True)

    @model_validator(mode="after")
    def validate_thresholds(self):
        if self.score_floor > LEGACY_DETECTION_CONFIDENCE_THRESHOLD:
            raise ValueError(
                "score_floor must not exceed the legacy output threshold 0.25"
            )
        if self.confidence_threshold < self.score_floor:
            raise ValueError("confidence_threshold must not be lower than score_floor")
        if not any(
            abs(self.iou_threshold - threshold) <= 1e-12
            for threshold in (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)
        ):
            raise ValueError("iou_threshold must be one of the COCO IoU thresholds")
        return self


class DetectionSupervisedEffectiveConfiguration(DetectionSupervisedConfiguration):
    nms_iou_threshold: Literal[0.7] = DETECTION_EVALUATION_NMS_IOU_THRESHOLD
    max_detections_per_image: Literal[300] = (
        DETECTION_EVALUATION_MAX_DETECTIONS_PER_IMAGE
    )
    interactive_output_confidence_threshold: Literal[0.25] = (
        LEGACY_DETECTION_CONFIDENCE_THRESHOLD
    )
    interactive_output_confidence_comparison: Literal["strictly_greater_than"] = (
        "strictly_greater_than"
    )
    coco_iou_thresholds: tuple[float, ...] = (
        0.5,
        0.55,
        0.6,
        0.65,
        0.7,
        0.75,
        0.8,
        0.85,
        0.9,
        0.95,
    )
    coco_recall_threshold_count: int = 101
    coco_max_detections: tuple[int, int, int] = (1, 10, 100)
    coco_area_ranges: tuple[str, str, str, str] = (
        "all",
        "small",
        "medium",
        "large",
    )


class DetectionAnnotationCoverage(DetectionMetricModel):
    total_images: int = Field(ge=0)
    reviewed_images: int = Field(ge=0)
    unreviewed_images: int = Field(ge=0)
    reviewed_empty_images: int = Field(ge=0)
    reviewed_with_objects_images: int = Field(ge=0)
    coverage_percentage: float = Field(ge=0.0, le=100.0)
    total_categories: int = Field(ge=0)
    resolved_categories: int = Field(ge=0)
    unresolved_categories: int = Field(ge=0)
    ambiguous_categories: int = Field(ge=0)
    conflicting_categories: int = Field(ge=0)
    total_objects: int = Field(ge=0)
    evaluable_objects: int = Field(ge=0)
    objects_excluded_by_resolution: int = Field(ge=0)
    iscrowd_objects: int = Field(ge=0)
    object_coverage_percentage: float = Field(ge=0.0, le=100.0)

    @model_validator(mode="after")
    def validate_totals_and_percentages(self):
        if self.reviewed_images + self.unreviewed_images != self.total_images:
            raise ValueError(
                "reviewed_images and unreviewed_images must sum to total_images"
            )
        if (
            self.reviewed_empty_images + self.reviewed_with_objects_images
            != self.reviewed_images
        ):
            raise ValueError("reviewed image statuses must sum to reviewed_images")

        categorized = (
            self.resolved_categories
            + self.unresolved_categories
            + self.ambiguous_categories
            + self.conflicting_categories
        )
        if categorized > self.total_categories:
            raise ValueError(
                "category resolution counts must not exceed total_categories"
            )

        if (
            self.evaluable_objects + self.objects_excluded_by_resolution
            > self.total_objects
        ):
            raise ValueError(
                "evaluable and resolution-excluded objects must not exceed "
                "total_objects"
            )
        if self.iscrowd_objects > self.total_objects:
            raise ValueError("iscrowd_objects must not exceed total_objects")

        _require_metric_close(
            field_name="coverage_percentage",
            actual=self.coverage_percentage,
            expected=_rounded_percentage(
                self.reviewed_images,
                self.total_images,
            ),
        )
        _require_metric_close(
            field_name="object_coverage_percentage",
            actual=self.object_coverage_percentage,
            expected=_rounded_percentage(
                self.evaluable_objects,
                self.total_objects,
            ),
        )
        return self


class DetectionClassReference(DetectionMetricModel):
    category_internal_id: str
    coco_category_id: int = Field(strict=True)
    category_name: str
    model_class_id: int = Field(ge=0, strict=True)
    model_class_name: str

    @model_validator(mode="after")
    def validate_reference(self):
        for field_name in (
            "category_internal_id",
            "category_name",
            "model_class_name",
        ):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        return self


class DetectionPrecisionRecallPoint(DetectionMetricModel):
    recall: float = Field(ge=0.0, le=1.0)
    precision: float = Field(ge=0.0, le=1.0)


class DetectionPrecisionRecallCurve(DetectionMetricModel):
    iou_thresholds: tuple[float, ...]
    area: Literal["all", "small", "medium", "large"]
    max_detections: int = Field(ge=1)
    points: tuple[DetectionPrecisionRecallPoint, ...]

    @model_validator(mode="after")
    def validate_recall_grid(self):
        if len(self.points) != PUBLIC_DETECTION_RECALL_THRESHOLD_COUNT:
            raise ValueError(
                "precision-recall curves must contain the 101 COCO recall points"
            )
        for index, point in enumerate(self.points):
            _require_metric_close(
                field_name=f"points[{index}].recall",
                actual=point.recall,
                expected=index / 100.0,
            )
        return self


class DetectionOperatingPointMetrics(DetectionMetricModel):
    true_positives: int = Field(ge=0)
    false_positives: int = Field(ge=0)
    false_negatives: int = Field(ge=0)
    precision: float | None = Field(default=None, ge=0.0, le=1.0)
    recall: float | None = Field(default=None, ge=0.0, le=1.0)
    f1: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_joint_metrics(self):
        expected_precision = _fraction_or_none(
            self.true_positives,
            self.true_positives + self.false_positives,
        )
        expected_recall = _fraction_or_none(
            self.true_positives,
            self.true_positives + self.false_negatives,
        )
        expected_f1 = None
        if (
            expected_precision is not None
            and expected_recall is not None
            and expected_precision + expected_recall > 0
        ):
            expected_f1 = (
                2
                * expected_precision
                * expected_recall
                / (expected_precision + expected_recall)
            )

        _require_optional_metric(
            field_name="precision",
            actual=self.precision,
            expected=expected_precision,
        )
        _require_optional_metric(
            field_name="recall",
            actual=self.recall,
            expected=expected_recall,
        )
        _require_optional_metric(
            field_name="f1",
            actual=self.f1,
            expected=expected_f1,
        )
        return self


class DetectionCocoAggregateMetrics(DetectionMetricModel):
    map_50_95: float | None = Field(default=None, ge=0.0, le=1.0)
    ap_50: float | None = Field(default=None, ge=0.0, le=1.0)
    ap_75: float | None = Field(default=None, ge=0.0, le=1.0)
    ar_1: float | None = Field(default=None, ge=0.0, le=1.0)
    ar_10: float | None = Field(default=None, ge=0.0, le=1.0)
    ar_100: float | None = Field(default=None, ge=0.0, le=1.0)
    ap_small: float | None = Field(default=None, ge=0.0, le=1.0)
    ap_medium: float | None = Field(default=None, ge=0.0, le=1.0)
    ap_large: float | None = Field(default=None, ge=0.0, le=1.0)
    ar_small: float | None = Field(default=None, ge=0.0, le=1.0)
    ar_medium: float | None = Field(default=None, ge=0.0, le=1.0)
    ar_large: float | None = Field(default=None, ge=0.0, le=1.0)


class DetectionPerClassCocoMetrics(DetectionMetricModel):
    reference: DetectionClassReference
    support: int = Field(ge=0)
    images_with_objects: int = Field(ge=0)
    iscrowd_objects: int = Field(ge=0)
    average_precision: float | None = Field(default=None, ge=0.0, le=1.0)
    average_recall: float | None = Field(default=None, ge=0.0, le=1.0)
    precision_recall_curve: DetectionPrecisionRecallCurve | None = None
    operating_point: DetectionOperatingPointMetrics

    @model_validator(mode="after")
    def validate_support_contract(self):
        if self.images_with_objects > self.support:
            raise ValueError("images_with_objects must not exceed support")
        if (
            self.operating_point.true_positives + self.operating_point.false_negatives
            != self.support
        ):
            raise ValueError("true_positives plus false_negatives must equal support")
        calculated_values = (
            self.average_precision,
            self.average_recall,
            self.precision_recall_curve,
        )
        if self.support == 0 and any(value is not None for value in calculated_values):
            raise ValueError("zero-support classes require null AP, AR and PR curve")
        if self.support > 0 and any(value is None for value in calculated_values):
            raise ValueError("positive-support classes require AP, AR and PR curve")
        if self.support > 0:
            expected_average_precision = sum(
                point.precision for point in self.precision_recall_curve.points
            ) / len(self.precision_recall_curve.points)
            _require_optional_metric(
                field_name="average_precision",
                actual=self.average_precision,
                expected=expected_average_precision,
            )
        return self


class DetectionCocoAnalysis(DetectionMetricModel):
    aggregate: DetectionCocoAggregateMetrics
    precision_recall_curve: DetectionPrecisionRecallCurve | None
    operating_point: DetectionOperatingPointMetrics
    per_class: tuple[DetectionPerClassCocoMetrics, ...]

    @model_validator(mode="after")
    def validate_aggregate_contract(self):
        has_supported_class = any(item.support > 0 for item in self.per_class)
        if has_supported_class != (self.precision_recall_curve is not None):
            raise ValueError(
                "aggregate precision-recall curve presence must match class support"
            )

        aggregate_counts = {
            "true_positives": self.operating_point.true_positives,
            "false_positives": self.operating_point.false_positives,
            "false_negatives": self.operating_point.false_negatives,
        }
        for field_name, aggregate_value in aggregate_counts.items():
            per_class_total = sum(
                getattr(item.operating_point, field_name) for item in self.per_class
            )
            if aggregate_value != per_class_total:
                raise ValueError(f"aggregate {field_name} must equal the per-class sum")

        if self.precision_recall_curve is None:
            if self.aggregate.map_50_95 is not None:
                raise ValueError(
                    "map_50_95 requires an aggregate precision-recall curve"
                )
        else:
            if self.aggregate.map_50_95 is None:
                raise ValueError(
                    "an aggregate precision-recall curve requires map_50_95"
                )
            expected_map = sum(
                point.precision for point in self.precision_recall_curve.points
            ) / len(self.precision_recall_curve.points)
            _require_optional_metric(
                field_name="map_50_95",
                actual=self.aggregate.map_50_95,
                expected=expected_map,
            )
        return self


class DetectionObjectOutcome(str, Enum):
    PRESERVED = "preserved"
    SUPPRESSED = "suppressed"
    RECOVERED = "recovered"
    MISSED_BOTH = "missed_both"


class DetectionObjectSizeBucket(str, Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


class DetectionMetricBoundingBox(DetectionMetricModel):
    x1: float = Field(ge=0.0)
    y1: float = Field(ge=0.0)
    x2: float = Field(gt=0.0)
    y2: float = Field(gt=0.0)

    @model_validator(mode="after")
    def validate_geometry(self):
        if self.x1 >= self.x2 or self.y1 >= self.y2:
            raise ValueError("bounding box must have positive width and height")
        return self


class DetectionObjectOperatingPointConfiguration(DetectionMetricModel):
    confidence_threshold: float = Field(ge=0.0, le=1.0)
    iou_threshold: float = Field(ge=0.0, le=1.0)
    max_detections_per_image: Literal[100] = 100
    confidence_comparison: Literal["greater_than_or_equal"] = "greater_than_or_equal"


class DetectionMatchedObjectObservation(DetectionMetricModel):
    confidence: float = Field(ge=0.0, le=1.0)
    iou_with_ground_truth: float = Field(ge=0.0, le=1.0)
    detection_box: DetectionMetricBoundingBox


class DetectionObjectResult(DetectionMetricModel):
    object_id: str
    image_id: str
    category: DetectionClassReference
    ground_truth_box: DetectionMetricBoundingBox
    area: float = Field(gt=0.0)
    size_bucket: DetectionObjectSizeBucket
    outcome: DetectionObjectOutcome
    original_observation: DetectionMatchedObjectObservation | None = None
    adversarial_observation: DetectionMatchedObjectObservation | None = None
    confidence_drop: float | None = None
    iou_drop: float | None = None

    @model_validator(mode="after")
    def validate_outcome(self):
        if not self.object_id.strip() or not self.image_id.strip():
            raise ValueError("object and image IDs must be non-empty")
        expected_size = (
            DetectionObjectSizeBucket.SMALL
            if self.area < 32.0**2
            else DetectionObjectSizeBucket.MEDIUM
            if self.area < 96.0**2
            else DetectionObjectSizeBucket.LARGE
        )
        if self.size_bucket is not expected_size:
            raise ValueError("size_bucket is inconsistent with ground-truth area")
        expected_presence = {
            DetectionObjectOutcome.PRESERVED: (True, True, True),
            DetectionObjectOutcome.SUPPRESSED: (True, False, False),
            DetectionObjectOutcome.RECOVERED: (False, True, False),
            DetectionObjectOutcome.MISSED_BOTH: (False, False, False),
        }[self.outcome]
        actual_presence = (
            self.original_observation is not None,
            self.adversarial_observation is not None,
            self.confidence_drop is not None and self.iou_drop is not None,
        )
        if actual_presence != expected_presence:
            raise ValueError("object outcome fields are inconsistent")
        if (self.confidence_drop is None) != (self.iou_drop is None):
            raise ValueError("object deltas must be both present or both null")
        if self.outcome is DetectionObjectOutcome.PRESERVED:
            _require_metric_close(
                field_name="confidence_drop",
                actual=self.confidence_drop,
                expected=(
                    self.original_observation.confidence
                    - self.adversarial_observation.confidence
                ),
            )
            _require_metric_close(
                field_name="iou_drop",
                actual=self.iou_drop,
                expected=(
                    self.original_observation.iou_with_ground_truth
                    - self.adversarial_observation.iou_with_ground_truth
                ),
            )
        return self


class DetectionObjectAggregate(DetectionMetricModel):
    total_objects: int = Field(ge=0)
    preserved_count: int = Field(ge=0)
    suppressed_count: int = Field(ge=0)
    recovered_count: int = Field(ge=0)
    missed_both_count: int = Field(ge=0)
    originally_detected_count: int = Field(ge=0)
    adversarially_detected_count: int = Field(ge=0)
    preservation_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    suppression_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    original_detection_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    adversarial_detection_rate: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    mean_confidence_drop_preserved: float | None = Field(
        default=None,
        ge=-1.0,
        le=1.0,
    )
    mean_iou_drop_preserved: float | None = Field(
        default=None,
        ge=-1.0,
        le=1.0,
    )

    @model_validator(mode="after")
    def validate_counts_and_rates(self):
        if (
            self.preserved_count
            + self.suppressed_count
            + self.recovered_count
            + self.missed_both_count
            != self.total_objects
        ):
            raise ValueError("object outcomes must sum to total_objects")
        if self.originally_detected_count != (
            self.preserved_count + self.suppressed_count
        ):
            raise ValueError("originally_detected_count is inconsistent")
        if self.adversarially_detected_count != (
            self.preserved_count + self.recovered_count
        ):
            raise ValueError("adversarially_detected_count is inconsistent")
        expectations = (
            (
                "preservation_rate",
                _fraction_or_none(
                    self.preserved_count,
                    self.originally_detected_count,
                ),
            ),
            (
                "suppression_rate",
                _fraction_or_none(
                    self.suppressed_count,
                    self.originally_detected_count,
                ),
            ),
            (
                "original_detection_rate",
                _fraction_or_none(self.originally_detected_count, self.total_objects),
            ),
            (
                "adversarial_detection_rate",
                _fraction_or_none(
                    self.adversarially_detected_count,
                    self.total_objects,
                ),
            ),
        )
        for field_name, expected in expectations:
            _require_optional_metric(
                field_name=field_name,
                actual=getattr(self, field_name),
                expected=expected,
            )
        if self.preserved_count == 0:
            if (
                self.mean_confidence_drop_preserved is not None
                or self.mean_iou_drop_preserved is not None
            ):
                raise ValueError("preserved means require preserved objects")
        elif (
            self.mean_confidence_drop_preserved is None
            or self.mean_iou_drop_preserved is None
        ):
            raise ValueError("preserved objects require both preserved means")
        return self


class DetectionObjectClassAggregate(DetectionObjectAggregate):
    category: DetectionClassReference


class DetectionObjectSizeAggregate(DetectionObjectAggregate):
    size_bucket: DetectionObjectSizeBucket


class DetectionObjectImageAggregate(DetectionObjectAggregate):
    image_id: str
    object_level_attack_success: bool | None

    @model_validator(mode="after")
    def validate_attack_success(self):
        expected = None
        if self.suppressed_count > 0:
            expected = True
        elif self.originally_detected_count > 0:
            expected = False
        if self.object_level_attack_success is not expected:
            raise ValueError("object_level_attack_success is inconsistent")
        if not self.image_id.strip():
            raise ValueError("image_id must be non-empty")
        return self


_OBJECT_COUNT_FIELDS = (
    "total_objects",
    "preserved_count",
    "suppressed_count",
    "recovered_count",
    "missed_both_count",
    "originally_detected_count",
    "adversarially_detected_count",
)


def _validate_object_dimension(
    *,
    dimension_name: str,
    values: tuple[DetectionObjectAggregate, ...],
    summary: DetectionObjectAggregate,
) -> None:
    for field_name in _OBJECT_COUNT_FIELDS:
        if sum(getattr(value, field_name) for value in values) != getattr(
            summary,
            field_name,
        ):
            raise ValueError(f"{dimension_name} {field_name} does not sum globally")
    preserved = sum(value.preserved_count for value in values)
    for field_name in (
        "mean_confidence_drop_preserved",
        "mean_iou_drop_preserved",
    ):
        expected = None
        if preserved:
            expected = (
                sum(
                    getattr(value, field_name) * value.preserved_count
                    for value in values
                    if value.preserved_count
                )
                / preserved
            )
        _require_optional_metric(
            field_name=f"{dimension_name}.{field_name}",
            actual=getattr(summary, field_name),
            expected=expected,
        )


def _class_reference_key(reference: DetectionClassReference) -> tuple:
    return (
        reference.model_class_id,
        reference.category_internal_id,
        reference.coco_category_id,
        reference.category_name,
        reference.model_class_name,
    )


def _validate_aggregate_against_objects(
    *,
    aggregate_name: str,
    aggregate: DetectionObjectAggregate,
    objects: tuple[DetectionObjectResult, ...],
) -> None:
    outcomes = {
        outcome: sum(item.outcome is outcome for item in objects)
        for outcome in DetectionObjectOutcome
    }
    expectations = {
        "total_objects": len(objects),
        "preserved_count": outcomes[DetectionObjectOutcome.PRESERVED],
        "suppressed_count": outcomes[DetectionObjectOutcome.SUPPRESSED],
        "recovered_count": outcomes[DetectionObjectOutcome.RECOVERED],
        "missed_both_count": outcomes[DetectionObjectOutcome.MISSED_BOTH],
    }
    for field_name, expected in expectations.items():
        if getattr(aggregate, field_name) != expected:
            raise ValueError(
                f"{aggregate_name} {field_name} disagrees with object details"
            )
    preserved = tuple(
        item for item in objects if item.outcome is DetectionObjectOutcome.PRESERVED
    )
    for field_name, object_field_name in (
        ("mean_confidence_drop_preserved", "confidence_drop"),
        ("mean_iou_drop_preserved", "iou_drop"),
    ):
        expected = None
        if preserved:
            expected = sum(
                getattr(item, object_field_name) for item in preserved
            ) / len(preserved)
        _require_optional_metric(
            field_name=f"{aggregate_name}.{field_name}",
            actual=getattr(aggregate, field_name),
            expected=expected,
        )


class DetectionObjectAdversarialAnalysis(DetectionMetricModel):
    methodology_version: Literal["coco_category_aware_gt_transitions_v1"] = (
        "coco_category_aware_gt_transitions_v1"
    )
    operating_point_configuration: DetectionObjectOperatingPointConfiguration
    details_included: bool
    total_evaluable_objects: int = Field(ge=0)
    summary: DetectionObjectAggregate
    by_class: tuple[DetectionObjectClassAggregate, ...]
    by_size: tuple[DetectionObjectSizeAggregate, ...]
    by_image: tuple[DetectionObjectImageAggregate, ...] | None = None
    objects: tuple[DetectionObjectResult, ...] | None = None

    @model_validator(mode="after")
    def validate_analysis(self):
        if self.total_evaluable_objects != self.summary.total_objects:
            raise ValueError("total_evaluable_objects must match summary")
        if self.details_included != (
            self.by_image is not None and self.objects is not None
        ):
            raise ValueError("detail collections must follow details_included")
        class_keys = tuple(
            _class_reference_key(item.category) for item in self.by_class
        )
        if len(set(class_keys)) != len(self.by_class):
            raise ValueError("by_class categories must be unique")
        if tuple(
            (item.category.model_class_id, item.category.category_internal_id)
            for item in self.by_class
        ) != tuple(
            sorted(
                (
                    item.category.model_class_id,
                    item.category.category_internal_id,
                )
                for item in self.by_class
            )
        ):
            raise ValueError("by_class must use canonical order")
        if tuple(item.size_bucket for item in self.by_size) != (
            DetectionObjectSizeBucket.SMALL,
            DetectionObjectSizeBucket.MEDIUM,
            DetectionObjectSizeBucket.LARGE,
        ):
            raise ValueError("by_size must contain small, medium and large in order")
        _validate_object_dimension(
            dimension_name="by_class",
            values=self.by_class,
            summary=self.summary,
        )
        _validate_object_dimension(
            dimension_name="by_size",
            values=self.by_size,
            summary=self.summary,
        )
        if self.by_image is not None:
            image_ids = tuple(item.image_id for item in self.by_image)
            if image_ids != tuple(sorted(image_ids)) or len(set(image_ids)) != len(
                image_ids
            ):
                raise ValueError("by_image IDs must be unique and sorted")
            _validate_object_dimension(
                dimension_name="by_image",
                values=self.by_image,
                summary=self.summary,
            )
        if self.objects is not None:
            object_keys = tuple(
                (item.image_id, item.object_id) for item in self.objects
            )
            if object_keys != tuple(sorted(object_keys)) or len(
                {item.object_id for item in self.objects}
            ) != len(self.objects):
                raise ValueError("objects must have unique IDs and canonical order")
            if len(self.objects) != self.total_evaluable_objects:
                raise ValueError("objects must contain every evaluable object")
            for item in self.objects:
                for observation in (
                    item.original_observation,
                    item.adversarial_observation,
                ):
                    if observation is None:
                        continue
                    if (
                        observation.confidence
                        < self.operating_point_configuration.confidence_threshold
                        - PUBLIC_DETECTION_METRIC_TOLERANCE
                        or observation.iou_with_ground_truth
                        < self.operating_point_configuration.iou_threshold
                        - PUBLIC_DETECTION_METRIC_TOLERANCE
                    ):
                        raise ValueError(
                            "matched object observation is below the operating point"
                        )
            class_references = {
                _class_reference_key(item.category) for item in self.by_class
            }
            image_ids = {item.image_id for item in self.by_image}
            objects_by_class = {key: [] for key in class_references}
            objects_by_size = {item.size_bucket: [] for item in self.by_size}
            objects_by_image = {image_id: [] for image_id in image_ids}
            for item in self.objects:
                class_key = _class_reference_key(item.category)
                if class_key not in objects_by_class:
                    raise ValueError(
                        "object references a category absent from by_class"
                    )
                if item.image_id not in objects_by_image:
                    raise ValueError("object references an image absent from by_image")
                objects_by_class[class_key].append(item)
                objects_by_size[item.size_bucket].append(item)
                objects_by_image[item.image_id].append(item)
            _validate_aggregate_against_objects(
                aggregate_name="summary",
                aggregate=self.summary,
                objects=self.objects,
            )
            for value in self.by_class:
                key = _class_reference_key(value.category)
                _validate_aggregate_against_objects(
                    aggregate_name=f"by_class[{key}]",
                    aggregate=value,
                    objects=tuple(objects_by_class[key]),
                )
            for value in self.by_size:
                _validate_aggregate_against_objects(
                    aggregate_name=f"by_size[{value.size_bucket}]",
                    aggregate=value,
                    objects=tuple(objects_by_size[value.size_bucket]),
                )
            for value in self.by_image:
                _validate_aggregate_against_objects(
                    aggregate_name=f"by_image[{value.image_id}]",
                    aggregate=value,
                    objects=tuple(objects_by_image[value.image_id]),
                )
        return self


class DetectionMetricDelta(DetectionMetricModel):
    absolute: float | None
    relative: float | None


class DetectionCocoAggregateDeltas(DetectionMetricModel):
    map_50_95: DetectionMetricDelta
    ap_50: DetectionMetricDelta
    ap_75: DetectionMetricDelta
    ar_1: DetectionMetricDelta
    ar_10: DetectionMetricDelta
    ar_100: DetectionMetricDelta
    ap_small: DetectionMetricDelta
    ap_medium: DetectionMetricDelta
    ap_large: DetectionMetricDelta
    ar_small: DetectionMetricDelta
    ar_medium: DetectionMetricDelta
    ar_large: DetectionMetricDelta


class DetectionOperatingPointDeltas(DetectionMetricModel):
    precision: DetectionMetricDelta
    recall: DetectionMetricDelta
    f1: DetectionMetricDelta


class DetectionPerClassDeltas(DetectionMetricModel):
    reference: DetectionClassReference
    average_precision: DetectionMetricDelta
    average_recall: DetectionMetricDelta
    operating_point: DetectionOperatingPointDeltas


class DetectionCocoDeltas(DetectionMetricModel):
    aggregate: DetectionCocoAggregateDeltas
    operating_point: DetectionOperatingPointDeltas
    per_class: tuple[DetectionPerClassDeltas, ...]


class DetectionSupervisedMetrics(DetectionMetricModel):
    available: bool
    reason: DetectionSupervisedUnavailableReason | None = None
    warning: str | None = None
    annotation_coverage: DetectionAnnotationCoverage
    configuration: DetectionSupervisedEffectiveConfiguration
    original: DetectionCocoAnalysis | None = None
    adversarial: DetectionCocoAnalysis | None = None
    deltas: DetectionCocoDeltas | None = None
    object_analysis: DetectionObjectAdversarialAnalysis | None = None

    @model_validator(mode="after")
    def validate_availability(self):
        analyses = (self.original, self.adversarial, self.deltas)
        if self.available:
            if (
                self.reason is not None
                or any(value is None for value in analyses)
                or self.object_analysis is None
            ):
                raise ValueError(
                    "available supervised metrics require analyses and no reason"
                )
            original_references = tuple(
                item.reference for item in self.original.per_class
            )
            adversarial_references = tuple(
                item.reference for item in self.adversarial.per_class
            )
            delta_references = tuple(item.reference for item in self.deltas.per_class)
            if not (original_references == adversarial_references == delta_references):
                raise ValueError(
                    "original, adversarial and delta class references must match"
                )
            original_support = tuple(
                (item.support, item.images_with_objects, item.iscrowd_objects)
                for item in self.original.per_class
            )
            adversarial_support = tuple(
                (item.support, item.images_with_objects, item.iscrowd_objects)
                for item in self.adversarial.per_class
            )
            if original_support != adversarial_support:
                raise ValueError(
                    "ground-truth class support must not change between analyses"
                )
        elif (
            self.reason is None
            or any(value is not None for value in analyses)
            or self.object_analysis is not None
        ):
            raise ValueError(
                "unavailable supervised metrics require a reason and no analyses"
            )
        return self
