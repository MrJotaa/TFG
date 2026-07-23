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
            raise ValueError(
                "reviewed image statuses must sum to reviewed_images"
            )

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
            self.operating_point.true_positives
            + self.operating_point.false_negatives
            != self.support
        ):
            raise ValueError(
                "true_positives plus false_negatives must equal support"
            )
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
                raise ValueError(
                    f"aggregate {field_name} must equal the per-class sum"
                )

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

    @model_validator(mode="after")
    def validate_availability(self):
        analyses = (self.original, self.adversarial, self.deltas)
        if self.available:
            if self.reason is not None or any(value is None for value in analyses):
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
        elif self.reason is None or any(value is not None for value in analyses):
            raise ValueError(
                "unavailable supervised metrics require a reason and no analyses"
            )
        return self
