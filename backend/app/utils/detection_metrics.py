from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


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


def calculate_detection_comparison(
    original_detections: Sequence[DetectionWithConfidence],
    adversarial_detections: Sequence[DetectionWithConfidence],
) -> DetectionComparisonMetrics:
    """
    Calcula los deltas canónicos sin redondear valores intermedios.

    Ambos deltas siguen ``original - adversarial``:

    - confianza positiva: la confianza adversaria ha bajado;
    - detecciones positivas: hay menos detecciones adversarias, sin afirmar
      calidad porque todavía no existe ground truth.

    ``attack_success`` mantiene deliberadamente la definición legacy previa.
    Esta redondeaba individualmente cada confianza a cuatro decimales antes de
    calcular y comparar las medias. Se conserva separada del signo canónico
    para que corregir los deltas no reclasifique muestras.
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

    legacy_original_average = _legacy_average_detection_confidence(original_detections)
    legacy_adversarial_average = _legacy_average_detection_confidence(
        adversarial_detections
    )
    legacy_confidence_difference = round(
        round(legacy_adversarial_average, 4) - round(legacy_original_average, 4),
        4,
    )

    return DetectionComparisonMetrics(
        original_count=original_count,
        adversarial_count=adversarial_count,
        detections_difference=(original_count - adversarial_count),
        average_original_confidence=(average_original_confidence),
        average_adversarial_confidence=(average_adversarial_confidence),
        average_confidence_difference=(average_confidence_difference),
        attack_success=(
            adversarial_count < original_count or legacy_confidence_difference < 0
        ),
    )
