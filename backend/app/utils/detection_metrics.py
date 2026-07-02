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

    return average([detection.confidence for detection in detections])


def calculate_detection_comparison(
    original_detections: Sequence[DetectionWithConfidence],
    adversarial_detections: Sequence[DetectionWithConfidence],
) -> DetectionComparisonMetrics:
    original_count = len(original_detections)
    adversarial_count = len(adversarial_detections)

    average_original_confidence = average_detection_confidence(original_detections)
    average_adversarial_confidence = average_detection_confidence(
        adversarial_detections
    )

    average_confidence_difference = round(
        average_adversarial_confidence - average_original_confidence,
        4,
    )

    return DetectionComparisonMetrics(
        original_count=original_count,
        adversarial_count=adversarial_count,
        detections_difference=(adversarial_count - original_count),
        average_original_confidence=(average_original_confidence),
        average_adversarial_confidence=(average_adversarial_confidence),
        average_confidence_difference=(average_confidence_difference),
        attack_success=(
            adversarial_count < original_count or average_confidence_difference < 0
        ),
    )
