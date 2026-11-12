from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
from scipy.optimize import linear_sum_assignment

from ml.models.detection import Detection, DetectionBox


# Versioned semantics. The old formula is retained only for interpreting old
# experiments and for explicit regression tests, never for new evaluations.
LEGACY_DETECTION_SUCCESS = "legacy_count_or_confidence_v1"
MATCHED_DETECTION_SUCCESS = "matched_detections_iou50_v1"
CURRENT_DETECTION_METRICS_SCHEMA_VERSION = "1.1"
DETECTION_SUCCESS_IOU_THRESHOLD = 0.5
DetectionSuccessCriterion = Literal[
    "legacy_count_or_confidence_v1", "matched_detections_iou50_v1"
]


class DetectionWithConfidence(Protocol):
    confidence: float


@dataclass(frozen=True)
class DetectionComparisonMetrics:
    original_count: int
    adversarial_count: int
    detections_difference: int
    average_original_confidence: float
    average_adversarial_confidence: float
    average_confidence_difference: float
    attack_success: bool


def average(values: Sequence[float]) -> float:
    if not values:
        return 0.0

    return round(sum(values) / len(values), 4)


def average_detection_confidence(
    detections: Sequence[DetectionWithConfidence],
) -> float:
    if not detections:
        return 0.0

    return sum(detection.confidence for detection in detections) / len(detections)


def _legacy_average_detection_confidence(
    detections: Sequence[DetectionWithConfidence],
) -> float:
    """
    Reproduce la media histórica sobre confianzas ya presentadas por YOLO.

    Antes del contrato canónico, cada confianza se redondeaba a cuatro
    decimales al crear ``DetectionPrediction``. Esta normalización se limita a
    la clasificación legacy de éxito y no afecta a las métricas canónicas.
    """
    if not detections:
        return 0.0

    return sum(round(detection.confidence, 4) for detection in detections) / len(
        detections
    )


def _box_iou(left: DetectionBox, right: DetectionBox) -> float:
    overlap_width = max(
        0.0, min(left.x_max, right.x_max) - max(left.x_min, right.x_min)
    )
    overlap_height = max(
        0.0, min(left.y_max, right.y_max) - max(left.y_min, right.y_min)
    )
    intersection = overlap_width * overlap_height
    left_area = (left.x_max - left.x_min) * (left.y_max - left.y_min)
    right_area = (right.x_max - right.x_min) * (right.y_max - right.y_min)
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _detection_consistency_success(
    original: Sequence[Detection], adversarial: Sequence[Detection]
) -> bool:
    """True iff at least one clean detection loses a 1:1 same-class match.

    A detection survives if another detection of the same class has IoU >= 0.5.
    Maximum-cardinality bipartite matching prevents one adversarial box from
    "saving" several clean boxes. Confidence changes by themselves never count;
    the model's existing output threshold determines which boxes exist.

    No clean detections => no possible loss (new false positives are not counted).
    This is prediction stability, NOT ground-truth correctness or COCO AP.
    """
    if not original:
        return False
    if not adversarial:
        return True
    if not all(isinstance(item, Detection) for item in (*original, *adversarial)):
        raise TypeError("matching success requires canonical Detection instances")

    valid = np.zeros((len(original), len(adversarial)), dtype=np.int8)
    for i, reference in enumerate(original):
        for j, candidate in enumerate(adversarial):
            if (
                reference.class_id == candidate.class_id
                and _box_iou(reference.box, candidate.box)
                >= DETECTION_SUCCESS_IOU_THRESHOLD
            ):
                valid[i, j] = 1
    if not valid.any():
        return True
    rows, cols = linear_sum_assignment(valid, maximize=True)
    return int(valid[rows, cols].sum()) < len(original)


def calculate_detection_comparison(
    original_detections: Sequence[DetectionWithConfidence],
    adversarial_detections: Sequence[DetectionWithConfidence],
    *,
    success_criterion: DetectionSuccessCriterion = MATCHED_DETECTION_SUCCESS,
) -> DetectionComparisonMetrics:
    """Deltas constantes; éxito de detección por correspondencia de cajas.

    El criterio actual exige que una detección original deje de tener pareja
    de la misma clase con IoU >= 0.5. La confianza solo influye si la detección
    deja de aparecer tras el umbral de salida del modelo.

    El criterio legacy puede solicitarse *explícitamente* para reproducir
    resultados históricos; nunca se usa en nuevas ejecuciones.
    """

    original_count = len(original_detections)
    adversarial_count = len(adversarial_detections)

    average_original_confidence = average_detection_confidence(original_detections)
    average_adversarial_confidence = average_detection_confidence(
        adversarial_detections
    )

    average_confidence_difference = (
        average_original_confidence - average_adversarial_confidence
    )

    if success_criterion == MATCHED_DETECTION_SUCCESS:
        attack_success = _detection_consistency_success(
            original_detections, adversarial_detections
        )
    elif success_criterion == LEGACY_DETECTION_SUCCESS:
        legacy_original_average = _legacy_average_detection_confidence(
            original_detections
        )
        legacy_adversarial_average = _legacy_average_detection_confidence(
            adversarial_detections
        )
        legacy_confidence_difference = round(
            round(legacy_adversarial_average, 4) - round(legacy_original_average, 4),
            4,
        )
        attack_success = (
            adversarial_count < original_count or legacy_confidence_difference < 0
        )
    else:
        raise ValueError("unsupported detection success criterion")

    return DetectionComparisonMetrics(
        original_count=original_count,
        adversarial_count=adversarial_count,
        detections_difference=(original_count - adversarial_count),
        average_original_confidence=(average_original_confidence),
        average_adversarial_confidence=(average_adversarial_confidence),
        average_confidence_difference=(average_confidence_difference),
        attack_success=attack_success,
    )
