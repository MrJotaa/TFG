import contextlib
import io
import math
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from ml.models.detection import Detection, DetectionBox

COCO_IOU_THRESHOLDS = tuple(round(0.5 + index * 0.05, 2) for index in range(10))
COCO_RECALL_THRESHOLDS = tuple(round(index / 100, 2) for index in range(101))
COCO_MAX_DETECTIONS = (1, 10, 100)
COCO_AREA_LABELS = ("all", "small", "medium", "large")


class CocoDetectionMetricsError(RuntimeError):
    """Error interno del adaptador propio de COCOeval."""


class CocoDetectionMetricsInputError(ValueError):
    """La entrada agnóstica al adaptador COCO es inválida."""


@dataclass(frozen=True, slots=True)
class CocoEvaluationImage:
    image_id: str
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class CocoEvaluationClass:
    category_internal_id: str
    coco_category_id: int
    category_name: str
    model_class_id: int
    model_class_name: str


@dataclass(frozen=True, slots=True)
class CocoGroundTruth:
    annotation_id: int
    image_id: str
    model_class_id: int
    box: DetectionBox
    area: float
    iscrowd: int


@dataclass(frozen=True, slots=True)
class CocoPrediction:
    image_id: str
    detection: Detection


@dataclass(frozen=True, slots=True)
class CocoOperatingPointConfiguration:
    confidence_threshold: float
    iou_threshold: float

    def __post_init__(self) -> None:
        for field_name in ("confidence_threshold", "iou_threshold"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise CocoDetectionMetricsInputError(
                    f"{field_name} must be a real number"
                )
            normalized = float(value)
            if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
                raise CocoDetectionMetricsInputError(
                    f"{field_name} must be finite and between 0 and 1"
                )
            object.__setattr__(self, field_name, normalized)

        if not any(
            math.isclose(self.iou_threshold, threshold, abs_tol=1e-12)
            for threshold in COCO_IOU_THRESHOLDS
        ):
            raise CocoDetectionMetricsInputError(
                "iou_threshold must be one of the COCO IoU thresholds"
            )


@dataclass(frozen=True, slots=True)
class PrecisionRecallPoint:
    recall: float
    precision: float


@dataclass(frozen=True, slots=True)
class PrecisionRecallCurve:
    iou_thresholds: tuple[float, ...]
    area: str
    max_detections: int
    points: tuple[PrecisionRecallPoint, ...]


@dataclass(frozen=True, slots=True)
class OperatingPointMetrics:
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float | None
    recall: float | None
    f1: float | None


@dataclass(frozen=True, slots=True)
class CocoAggregateMetrics:
    map_50_95: float | None
    ap_50: float | None
    ap_75: float | None
    ar_1: float | None
    ar_10: float | None
    ar_100: float | None
    ap_small: float | None
    ap_medium: float | None
    ap_large: float | None
    ar_small: float | None
    ar_medium: float | None
    ar_large: float | None


@dataclass(frozen=True, slots=True)
class CocoPerClassMetrics:
    reference: CocoEvaluationClass
    support: int
    images_with_objects: int
    iscrowd_objects: int
    average_precision: float | None
    average_recall: float | None
    precision_recall_curve: PrecisionRecallCurve | None
    operating_point: OperatingPointMetrics


@dataclass(frozen=True, slots=True)
class CocoDetectionAnalysis:
    aggregate: CocoAggregateMetrics
    precision_recall_curve: PrecisionRecallCurve | None
    operating_point: OperatingPointMetrics
    per_class: tuple[CocoPerClassMetrics, ...]


def _mean_valid(values: np.ndarray) -> float | None:
    numeric = np.asarray(values, dtype=np.float64)
    if np.any(~np.isfinite(numeric)):
        raise CocoDetectionMetricsError("COCOeval returned NaN or infinite values")
    valid = numeric[numeric > -1.0]
    if valid.size == 0:
        return None
    mean = float(np.mean(valid))
    if not 0.0 <= mean <= 1.0:
        raise CocoDetectionMetricsError(
            f"COCOeval returned an out-of-range metric: {mean}"
        )
    return mean


def _metric_fraction(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _operating_metrics(tp: int, fp: int, fn: int) -> OperatingPointMetrics:
    precision = _metric_fraction(tp, tp + fp)
    recall = _metric_fraction(tp, tp + fn)
    f1 = None
    if precision is not None and recall is not None and precision + recall > 0:
        f1 = 2.0 * precision * recall / (precision + recall)
    return OperatingPointMetrics(
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def _validate_inputs(
    images: tuple[CocoEvaluationImage, ...],
    classes: tuple[CocoEvaluationClass, ...],
    ground_truths: tuple[CocoGroundTruth, ...],
    predictions: tuple[CocoPrediction, ...],
) -> None:
    if not images:
        raise CocoDetectionMetricsInputError(
            "COCO evaluation requires at least one reviewed image"
        )
    if not classes:
        raise CocoDetectionMetricsInputError(
            "COCO evaluation requires at least one resolved class"
        )

    image_by_id = {}
    for image in images:
        if not isinstance(image, CocoEvaluationImage):
            raise CocoDetectionMetricsInputError(
                "images must contain CocoEvaluationImage values"
            )
        if (
            not isinstance(image.image_id, str)
            or not image.image_id
            or isinstance(image.width, bool)
            or not isinstance(image.width, Integral)
            or image.width <= 0
            or isinstance(image.height, bool)
            or not isinstance(image.height, Integral)
            or image.height <= 0
        ):
            raise CocoDetectionMetricsInputError("evaluation image is invalid")
        if image.image_id in image_by_id:
            raise CocoDetectionMetricsInputError("evaluation image IDs must be unique")
        image_by_id[image.image_id] = image

    class_by_id = {}
    for model_class in classes:
        if not isinstance(model_class, CocoEvaluationClass):
            raise CocoDetectionMetricsInputError(
                "classes must contain CocoEvaluationClass values"
            )
        if model_class.model_class_id in class_by_id:
            raise CocoDetectionMetricsInputError(
                "evaluation model class IDs must be unique"
            )
        class_by_id[model_class.model_class_id] = model_class

    annotation_ids = set()
    for ground_truth in ground_truths:
        if not isinstance(ground_truth, CocoGroundTruth):
            raise CocoDetectionMetricsInputError(
                "ground_truths must contain CocoGroundTruth values"
            )
        if ground_truth.annotation_id in annotation_ids:
            raise CocoDetectionMetricsInputError(
                "ground-truth annotation IDs must be unique"
            )
        annotation_ids.add(ground_truth.annotation_id)
        image = image_by_id.get(ground_truth.image_id)
        if image is None:
            raise CocoDetectionMetricsInputError(
                "ground truth references an unknown image"
            )
        if ground_truth.model_class_id not in class_by_id:
            raise CocoDetectionMetricsInputError(
                "ground truth references an unknown model class"
            )
        if not isinstance(ground_truth.box, DetectionBox):
            raise CocoDetectionMetricsInputError(
                "ground-truth box must be a DetectionBox"
            )
        if (
            ground_truth.box.x_max > image.width
            or ground_truth.box.y_max > image.height
        ):
            raise CocoDetectionMetricsInputError(
                "ground-truth box exceeds source geometry"
            )
        if (
            isinstance(ground_truth.area, bool)
            or not isinstance(ground_truth.area, Real)
            or not math.isfinite(float(ground_truth.area))
            or ground_truth.area <= 0
        ):
            raise CocoDetectionMetricsInputError(
                "ground-truth area must be finite and positive"
            )
        if ground_truth.iscrowd not in (0, 1):
            raise CocoDetectionMetricsInputError("ground-truth iscrowd must be 0 or 1")

    for prediction in predictions:
        if not isinstance(prediction, CocoPrediction):
            raise CocoDetectionMetricsInputError(
                "predictions must contain CocoPrediction values"
            )
        image = image_by_id.get(prediction.image_id)
        if image is None:
            raise CocoDetectionMetricsInputError(
                "prediction references an unknown image"
            )
        if not isinstance(prediction.detection, Detection):
            raise CocoDetectionMetricsInputError(
                "prediction must contain a canonical Detection"
            )
        if prediction.detection.class_id not in class_by_id:
            raise CocoDetectionMetricsInputError(
                "prediction references an unknown evaluation class"
            )
        if (
            prediction.detection.box.x_max > image.width
            or prediction.detection.box.y_max > image.height
        ):
            raise CocoDetectionMetricsInputError(
                "prediction box exceeds source geometry"
            )


def _xywh(box: DetectionBox) -> list[float]:
    return [
        box.x_min,
        box.y_min,
        box.x_max - box.x_min,
        box.y_max - box.y_min,
    ]


def _build_coco_apis(
    *,
    images: tuple[CocoEvaluationImage, ...],
    classes: tuple[CocoEvaluationClass, ...],
    ground_truths: tuple[CocoGroundTruth, ...],
    predictions: tuple[CocoPrediction, ...],
) -> tuple[COCO, COCO, dict[str, int]]:
    image_id_map = {
        image.image_id: index
        for index, image in enumerate(
            sorted(images, key=lambda item: item.image_id),
            start=1,
        )
    }
    sorted_classes = tuple(
        sorted(
            classes,
            key=lambda item: (
                item.model_class_id,
                item.category_internal_id,
            ),
        )
    )
    coco_images = [
        {
            "id": image_id_map[image.image_id],
            "width": image.width,
            "height": image.height,
        }
        for image in sorted(images, key=lambda item: item.image_id)
    ]
    coco_categories = [
        {
            "id": model_class.model_class_id,
            "name": model_class.model_class_name,
        }
        for model_class in sorted_classes
    ]

    sorted_ground_truths = sorted(
        ground_truths,
        key=lambda item: (
            item.image_id,
            item.model_class_id,
            item.annotation_id,
        ),
    )
    coco_ground_truths = [
        {
            "id": index,
            "image_id": image_id_map[ground_truth.image_id],
            "category_id": ground_truth.model_class_id,
            "bbox": _xywh(ground_truth.box),
            "area": float(ground_truth.area),
            "iscrowd": int(ground_truth.iscrowd),
        }
        for index, ground_truth in enumerate(sorted_ground_truths, start=1)
    ]

    sorted_predictions = sorted(
        predictions,
        key=lambda item: (
            item.image_id,
            item.detection.class_id,
            -item.detection.confidence,
            item.detection.box.x_min,
            item.detection.box.y_min,
            item.detection.box.x_max,
            item.detection.box.y_max,
        ),
    )
    coco_predictions = [
        {
            "id": index,
            "image_id": image_id_map[prediction.image_id],
            "category_id": prediction.detection.class_id,
            "bbox": _xywh(prediction.detection.box),
            "area": (
                (prediction.detection.box.x_max - prediction.detection.box.x_min)
                * (prediction.detection.box.y_max - prediction.detection.box.y_min)
            ),
            "iscrowd": 0,
            "score": prediction.detection.confidence,
        }
        for index, prediction in enumerate(sorted_predictions, start=1)
    ]

    coco_ground_truth = COCO()
    coco_ground_truth.dataset = {
        "images": coco_images,
        "categories": coco_categories,
        "annotations": coco_ground_truths,
    }
    coco_detections = COCO()
    coco_detections.dataset = {
        "images": coco_images,
        "categories": coco_categories,
        "annotations": coco_predictions,
    }
    with contextlib.redirect_stdout(io.StringIO()):
        coco_ground_truth.createIndex()
        coco_detections.createIndex()
    return coco_ground_truth, coco_detections, image_id_map


def _area_index(evaluator: COCOeval, label: str) -> int:
    try:
        return evaluator.params.areaRngLbl.index(label)
    except ValueError as error:
        raise CocoDetectionMetricsError(
            f"COCOeval did not expose area range '{label}'"
        ) from error


def _max_detection_index(evaluator: COCOeval, value: int) -> int:
    try:
        return evaluator.params.maxDets.index(value)
    except ValueError as error:
        raise CocoDetectionMetricsError(
            f"COCOeval did not expose maxDet={value}"
        ) from error


def _iou_index(evaluator: COCOeval, value: float) -> int:
    indexes = np.flatnonzero(np.isclose(evaluator.params.iouThrs, value))
    if indexes.size != 1:
        raise CocoDetectionMetricsError(
            f"COCOeval did not expose IoU threshold {value}"
        )
    return int(indexes[0])


def _precision_recall_curve(
    precision: np.ndarray,
    recall_thresholds: np.ndarray,
) -> PrecisionRecallCurve | None:
    points = []
    for recall_index, recall in enumerate(recall_thresholds):
        precision_value = _mean_valid(precision[:, recall_index, ...])
        if precision_value is None:
            continue
        points.append(
            PrecisionRecallPoint(
                recall=float(recall),
                precision=precision_value,
            )
        )
    if not points:
        return None
    return PrecisionRecallCurve(
        iou_thresholds=COCO_IOU_THRESHOLDS,
        area="all",
        max_detections=100,
        points=tuple(points),
    )


def _operating_counts(
    *,
    evaluator: COCOeval,
    iou_index: int,
    confidence_threshold: float,
    category_id: int | None,
) -> OperatingPointMetrics:
    all_range = evaluator.params.areaRng[_area_index(evaluator, "all")]
    tp = 0
    fp = 0
    fn = 0
    for evaluation_image in evaluator.evalImgs:
        if evaluation_image is None:
            continue
        if evaluation_image["aRng"] != all_range:
            continue
        if category_id is not None and evaluation_image["category_id"] != category_id:
            continue

        scores = np.asarray(evaluation_image["dtScores"], dtype=np.float64)
        selected = scores >= confidence_threshold
        matches = np.asarray(evaluation_image["dtMatches"])[iou_index]
        ignored_detections = np.asarray(
            evaluation_image["dtIgnore"],
            dtype=bool,
        )[iou_index]
        true_positive_mask = selected & ~ignored_detections & (matches > 0)
        false_positive_mask = selected & ~ignored_detections & (matches == 0)
        tp += int(np.count_nonzero(true_positive_mask))
        fp += int(np.count_nonzero(false_positive_mask))

        ground_truth_matches = np.asarray(evaluation_image["gtMatches"])[iou_index]
        ignored_ground_truths = np.asarray(
            evaluation_image["gtIgnore"],
            dtype=bool,
        )
        selected_detection_ids = np.asarray(
            evaluation_image["dtIds"],
            dtype=np.int64,
        )[selected]
        matched_by_selected_detection = np.isin(
            ground_truth_matches,
            selected_detection_ids,
        )
        fn += int(
            np.count_nonzero(~ignored_ground_truths & ~matched_by_selected_detection)
        )
    return _operating_metrics(tp, fp, fn)


def evaluate_coco_detections(
    *,
    images: tuple[CocoEvaluationImage, ...],
    classes: tuple[CocoEvaluationClass, ...],
    ground_truths: tuple[CocoGroundTruth, ...],
    predictions: tuple[CocoPrediction, ...],
    operating_point: CocoOperatingPointConfiguration,
) -> CocoDetectionAnalysis:
    """Ejecuta COCOeval sin filtrar tipos, sentinels ni stdout del proveedor."""
    images = tuple(images)
    classes = tuple(classes)
    ground_truths = tuple(ground_truths)
    predictions = tuple(predictions)
    if not isinstance(operating_point, CocoOperatingPointConfiguration):
        raise CocoDetectionMetricsInputError(
            "operating_point must be a CocoOperatingPointConfiguration"
        )
    _validate_inputs(images, classes, ground_truths, predictions)

    try:
        coco_ground_truth, coco_detections, _image_id_map = _build_coco_apis(
            images=images,
            classes=classes,
            ground_truths=ground_truths,
            predictions=predictions,
        )
        evaluator = COCOeval(coco_ground_truth, coco_detections, "bbox")
        evaluator.params.imgIds = sorted(coco_ground_truth.getImgIds())
        evaluator.params.catIds = sorted(
            model_class.model_class_id for model_class in classes
        )
        evaluator.params.iouThrs = np.asarray(
            COCO_IOU_THRESHOLDS,
            dtype=np.float64,
        )
        evaluator.params.recThrs = np.asarray(
            COCO_RECALL_THRESHOLDS,
            dtype=np.float64,
        )
        evaluator.params.maxDets = list(COCO_MAX_DETECTIONS)
        with contextlib.redirect_stdout(io.StringIO()):
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()
    except CocoDetectionMetricsError:
        raise
    except Exception as error:
        raise CocoDetectionMetricsError(
            "COCOeval failed while evaluating canonical detections"
        ) from error

    precision = np.asarray(evaluator.eval.get("precision"), dtype=np.float64)
    recall = np.asarray(evaluator.eval.get("recall"), dtype=np.float64)
    expected_precision_shape = (
        len(COCO_IOU_THRESHOLDS),
        len(COCO_RECALL_THRESHOLDS),
        len(classes),
        len(COCO_AREA_LABELS),
        len(COCO_MAX_DETECTIONS),
    )
    expected_recall_shape = (
        len(COCO_IOU_THRESHOLDS),
        len(classes),
        len(COCO_AREA_LABELS),
        len(COCO_MAX_DETECTIONS),
    )
    if precision.shape != expected_precision_shape:
        raise CocoDetectionMetricsError(
            f"COCOeval precision shape {precision.shape} is invalid"
        )
    if recall.shape != expected_recall_shape:
        raise CocoDetectionMetricsError(
            f"COCOeval recall shape {recall.shape} is invalid"
        )

    all_area = _area_index(evaluator, "all")
    small_area = _area_index(evaluator, "small")
    medium_area = _area_index(evaluator, "medium")
    large_area = _area_index(evaluator, "large")
    max_1 = _max_detection_index(evaluator, 1)
    max_10 = _max_detection_index(evaluator, 10)
    max_100 = _max_detection_index(evaluator, 100)
    iou_50 = _iou_index(evaluator, 0.5)
    iou_75 = _iou_index(evaluator, 0.75)
    operating_iou = _iou_index(evaluator, operating_point.iou_threshold)

    aggregate = CocoAggregateMetrics(
        map_50_95=_mean_valid(precision[:, :, :, all_area, max_100]),
        ap_50=_mean_valid(precision[iou_50, :, :, all_area, max_100]),
        ap_75=_mean_valid(precision[iou_75, :, :, all_area, max_100]),
        ar_1=_mean_valid(recall[:, :, all_area, max_1]),
        ar_10=_mean_valid(recall[:, :, all_area, max_10]),
        ar_100=_mean_valid(recall[:, :, all_area, max_100]),
        ap_small=_mean_valid(precision[:, :, :, small_area, max_100]),
        ap_medium=_mean_valid(precision[:, :, :, medium_area, max_100]),
        ap_large=_mean_valid(precision[:, :, :, large_area, max_100]),
        ar_small=_mean_valid(recall[:, :, small_area, max_100]),
        ar_medium=_mean_valid(recall[:, :, medium_area, max_100]),
        ar_large=_mean_valid(recall[:, :, large_area, max_100]),
    )
    aggregate_curve = _precision_recall_curve(
        precision[:, :, :, all_area, max_100],
        evaluator.params.recThrs,
    )
    aggregate_operating = _operating_counts(
        evaluator=evaluator,
        iou_index=operating_iou,
        confidence_threshold=operating_point.confidence_threshold,
        category_id=None,
    )

    support_by_class: dict[int, int] = {}
    crowd_by_class: dict[int, int] = {}
    images_by_class: dict[int, set[str]] = {}
    for ground_truth in ground_truths:
        if ground_truth.iscrowd:
            crowd_by_class[ground_truth.model_class_id] = (
                crowd_by_class.get(ground_truth.model_class_id, 0) + 1
            )
            continue
        support_by_class[ground_truth.model_class_id] = (
            support_by_class.get(ground_truth.model_class_id, 0) + 1
        )
        images_by_class.setdefault(ground_truth.model_class_id, set()).add(
            ground_truth.image_id
        )

    category_indexes = {
        category_id: index for index, category_id in enumerate(evaluator.params.catIds)
    }
    per_class = []
    for model_class in sorted(
        classes,
        key=lambda item: (
            item.model_class_id,
            item.category_internal_id,
        ),
    ):
        class_index = category_indexes[model_class.model_class_id]
        support = support_by_class.get(model_class.model_class_id, 0)
        class_precision = precision[
            :,
            :,
            class_index : class_index + 1,
            all_area,
            max_100,
        ]
        class_recall = recall[
            :,
            class_index : class_index + 1,
            all_area,
            max_100,
        ]
        per_class.append(
            CocoPerClassMetrics(
                reference=model_class,
                support=support,
                images_with_objects=len(
                    images_by_class.get(model_class.model_class_id, set())
                ),
                iscrowd_objects=crowd_by_class.get(
                    model_class.model_class_id,
                    0,
                ),
                average_precision=(_mean_valid(class_precision) if support else None),
                average_recall=(_mean_valid(class_recall) if support else None),
                precision_recall_curve=(
                    _precision_recall_curve(
                        class_precision,
                        evaluator.params.recThrs,
                    )
                    if support
                    else None
                ),
                operating_point=_operating_counts(
                    evaluator=evaluator,
                    iou_index=operating_iou,
                    confidence_threshold=operating_point.confidence_threshold,
                    category_id=model_class.model_class_id,
                ),
            )
        )

    return CocoDetectionAnalysis(
        aggregate=aggregate,
        precision_recall_curve=aggregate_curve,
        operating_point=aggregate_operating,
        per_class=tuple(per_class),
    )
