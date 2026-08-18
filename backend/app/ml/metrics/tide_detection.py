import math
from dataclasses import dataclass

from tidecv import TIDE
from tidecv.data import Data

from ml.metrics.coco_detection import (
    CocoEvaluationClass,
    CocoEvaluationImage,
    CocoGroundTruth,
    CocoPrediction,
)

TIDE_POSITION_THRESHOLD = 0.5
TIDE_BACKGROUND_THRESHOLD = 0.1
TIDE_MAX_DETECTIONS_PER_IMAGE = 100
TIDE_MAIN_ERROR_ORDER = ("Cls", "Loc", "Both", "Dupe", "Bkg", "Miss")
TIDE_SPECIAL_ERROR_ORDER = ("FalsePos", "FalseNeg")


class TideDetectionMetricsError(RuntimeError):
    """TIDE no pudo evaluar las detecciones canónicas."""


@dataclass(frozen=True, slots=True)
class TideErrorMetric:
    code: str
    delta_ap: float
    count: int


@dataclass(frozen=True, slots=True)
class TideDetectionAnalysis:
    ap_50: float
    main_errors: tuple[TideErrorMetric, ...]
    special_errors: tuple[TideErrorMetric, ...]


def _xywh(box) -> list[float]:
    return [
        float(box.x_min),
        float(box.y_min),
        float(box.x_max - box.x_min),
        float(box.y_max - box.y_min),
    ]


def _intersection(left, right) -> float:
    width = max(0.0, min(left.x_max, right.x_max) - max(left.x_min, right.x_min))
    height = max(0.0, min(left.y_max, right.y_max) - max(left.y_min, right.y_min))
    return width * height


def _area(box) -> float:
    return (box.x_max - box.x_min) * (box.y_max - box.y_min)


def _iou(left, right) -> float:
    intersection = _intersection(left, right)
    union = _area(left) + _area(right) - intersection
    return intersection / union if union else 0.0


def _exclude_crowd_ignored_predictions(
    *,
    ground_truths: tuple[CocoGroundTruth, ...],
    predictions: tuple[CocoPrediction, ...],
) -> tuple[CocoPrediction, ...]:
    """Replica el ignore COCO de crowds antes del conteo de errores TIDE."""
    ground_truths_by_key: dict[tuple[str, int], list[CocoGroundTruth]] = {}
    for ground_truth in ground_truths:
        ground_truths_by_key.setdefault(
            (ground_truth.image_id, ground_truth.model_class_id),
            [],
        ).append(ground_truth)

    included = []
    for prediction in predictions:
        candidates = ground_truths_by_key.get(
            (prediction.image_id, prediction.detection.class_id),
            (),
        )
        matches_regular = any(
            not ground_truth.iscrowd
            and _iou(prediction.detection.box, ground_truth.box)
            >= TIDE_POSITION_THRESHOLD
            for ground_truth in candidates
        )
        ignored_by_crowd = any(
            ground_truth.iscrowd
            and _intersection(prediction.detection.box, ground_truth.box)
            / _area(prediction.detection.box)
            >= TIDE_POSITION_THRESHOLD
            for ground_truth in candidates
        )
        if matches_regular or not ignored_by_crowd:
            included.append(prediction)
    return tuple(included)


def _build_ground_truth_data(
    *,
    images: tuple[CocoEvaluationImage, ...],
    classes: tuple[CocoEvaluationClass, ...],
    ground_truths: tuple[CocoGroundTruth, ...],
) -> tuple[Data, dict[str, int]]:
    data = Data("ground_truth", max_dets=TIDE_MAX_DETECTIONS_PER_IMAGE)
    image_id_map = {
        image.image_id: index
        for index, image in enumerate(
            sorted(images, key=lambda item: item.image_id),
            start=1,
        )
    }
    for image in images:
        data.add_image(image_id_map[image.image_id], image.image_id)
    for model_class in classes:
        data.add_class(model_class.model_class_id, model_class.model_class_name)
    for ground_truth in ground_truths:
        arguments = (
            image_id_map[ground_truth.image_id],
            ground_truth.model_class_id,
            _xywh(ground_truth.box),
        )
        if ground_truth.iscrowd:
            data.add_ignore_region(*arguments)
        else:
            data.add_ground_truth(*arguments)
    return data, image_id_map


def _build_prediction_data(
    *,
    name: str,
    images: tuple[CocoEvaluationImage, ...],
    classes: tuple[CocoEvaluationClass, ...],
    predictions: tuple[CocoPrediction, ...],
    image_id_map: dict[str, int],
) -> Data:
    data = Data(name, max_dets=TIDE_MAX_DETECTIONS_PER_IMAGE)
    for image in images:
        data.add_image(image_id_map[image.image_id], image.image_id)
    for model_class in classes:
        data.add_class(model_class.model_class_id, model_class.model_class_name)
    for prediction in predictions:
        data.add_detection(
            image_id_map[prediction.image_id],
            prediction.detection.class_id,
            float(prediction.detection.confidence),
            _xywh(prediction.detection.box),
        )
    return data


def _finite_metric(value, *, name: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise TideDetectionMetricsError(f"TIDE returned an invalid {name}")
    # Los oráculos especiales de tidecv no aplican el clamp que sí utilizan
    # los seis errores principales y pueden devolver dAP negativo cuando el
    # AP ya es perfecto. Un oráculo expresa ganancia potencial, nunca pérdida.
    return max(numeric, 0.0)


def _empty_prediction_analysis(
    ground_truths: tuple[CocoGroundTruth, ...],
) -> TideDetectionAnalysis:
    missed_count = sum(not item.iscrowd for item in ground_truths)
    main_errors = tuple(
        TideErrorMetric(
            code=code,
            delta_ap=0.0,
            count=missed_count if code == "Miss" else 0,
        )
        for code in TIDE_MAIN_ERROR_ORDER
    )
    return TideDetectionAnalysis(
        ap_50=0.0,
        main_errors=main_errors,
        special_errors=(
            TideErrorMetric(code="FalsePos", delta_ap=0.0, count=0),
            TideErrorMetric(
                code="FalseNeg",
                delta_ap=0.0,
                count=missed_count,
            ),
        ),
    )


def evaluate_tide_detections(
    *,
    images: tuple[CocoEvaluationImage, ...],
    classes: tuple[CocoEvaluationClass, ...],
    ground_truths: tuple[CocoGroundTruth, ...],
    predictions: tuple[CocoPrediction, ...],
    name: str,
) -> TideDetectionAnalysis:
    """Calcula AP50, oráculos dAP y conteos TIDE sin parsear stdout."""
    images = tuple(images)
    classes = tuple(classes)
    ground_truths = tuple(ground_truths)
    predictions = tuple(predictions)
    if not images or not classes or not any(not item.iscrowd for item in ground_truths):
        raise TideDetectionMetricsError(
            "TIDE requires images, classes and evaluable ground truth objects"
        )
    if not predictions:
        # tidecv 1.0.1 divide por cero al construir sus oráculos sin ninguna
        # clase presente en el AP de predicciones. Conservamos su escala y
        # taxonomía, y materializamos explícitamente los objetos omitidos.
        return _empty_prediction_analysis(ground_truths)
    predictions = _exclude_crowd_ignored_predictions(
        ground_truths=ground_truths,
        predictions=predictions,
    )
    if not predictions:
        return _empty_prediction_analysis(ground_truths)

    try:
        ground_truth_data, image_id_map = _build_ground_truth_data(
            images=images,
            classes=classes,
            ground_truths=ground_truths,
        )
        prediction_data = _build_prediction_data(
            name=name,
            images=images,
            classes=classes,
            predictions=predictions,
            image_id_map=image_id_map,
        )
        tide = TIDE(
            pos_threshold=TIDE_POSITION_THRESHOLD,
            background_threshold=TIDE_BACKGROUND_THRESHOLD,
            mode=TIDE.BOX,
        )
        run = tide.evaluate(
            ground_truth_data,
            prediction_data,
            name=name,
            mode=TIDE.BOX,
        )
        main_values = tide.get_main_errors()[name]
        special_values = tide.get_special_errors()[name]
        counts = {
            error_type.short_name: int(count)
            for error_type, count in run.count_errors().items()
        }
        false_negative_count = sum(len(items) for items in run.false_negatives.values())
    except TideDetectionMetricsError:
        raise
    except Exception as error:
        raise TideDetectionMetricsError(
            "TIDE failed while evaluating canonical detections"
        ) from error

    if tuple(main_values) != TIDE_MAIN_ERROR_ORDER:
        raise TideDetectionMetricsError("TIDE returned unexpected main error types")
    if tuple(special_values) != TIDE_SPECIAL_ERROR_ORDER:
        raise TideDetectionMetricsError("TIDE returned unexpected special error types")
    if set(counts) != set(TIDE_MAIN_ERROR_ORDER):
        raise TideDetectionMetricsError("TIDE returned incomplete error counts")

    main_errors = tuple(
        TideErrorMetric(
            code=code,
            delta_ap=_finite_metric(main_values[code], name=f"{code} dAP"),
            count=counts[code],
        )
        for code in TIDE_MAIN_ERROR_ORDER
    )
    false_positive_count = sum(
        item.count for item in main_errors if item.code != "Miss"
    )
    special_counts = {
        "FalsePos": false_positive_count,
        "FalseNeg": false_negative_count,
    }
    return TideDetectionAnalysis(
        ap_50=_finite_metric(run.ap, name="AP50"),
        main_errors=main_errors,
        special_errors=tuple(
            TideErrorMetric(
                code=code,
                delta_ap=_finite_metric(special_values[code], name=f"{code} dAP"),
                count=special_counts[code],
            )
            for code in TIDE_SPECIAL_ERROR_ORDER
        ),
    )
