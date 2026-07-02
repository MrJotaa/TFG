from types import SimpleNamespace

import pytest

from utils.detection_metrics import (
    average,
    average_detection_confidence,
    calculate_detection_comparison,
)


def _detection(confidence: float):
    return SimpleNamespace(confidence=confidence)


def test_average_returns_zero_for_empty_values():
    assert average([]) == 0.0


def test_average_rounds_to_four_decimals():
    assert average([0.1, 0.2, 0.4]) == 0.2333


def test_average_detection_confidence():
    detections = [
        _detection(0.9),
        _detection(0.7),
    ]

    assert average_detection_confidence(detections) == 0.8


def test_detection_comparison_detects_reduced_count():
    metrics = calculate_detection_comparison(
        original_detections=[
            _detection(0.9),
            _detection(0.8),
        ],
        adversarial_detections=[
            _detection(0.5),
        ],
    )

    assert metrics.original_count == 2
    assert metrics.adversarial_count == 1
    assert metrics.detections_difference == -1
    assert metrics.average_original_confidence == 0.85
    assert metrics.average_adversarial_confidence == 0.5
    assert metrics.average_confidence_difference == -0.35
    assert metrics.attack_success is True


def test_detection_comparison_detects_confidence_reduction():
    metrics = calculate_detection_comparison(
        original_detections=[
            _detection(0.9),
        ],
        adversarial_detections=[
            _detection(0.6),
        ],
    )

    assert metrics.detections_difference == 0
    assert metrics.attack_success is True


def test_detection_comparison_reports_unsuccessful_attack():
    metrics = calculate_detection_comparison(
        original_detections=[
            _detection(0.7),
        ],
        adversarial_detections=[
            _detection(0.8),
        ],
    )

    assert metrics.detections_difference == 0
    assert metrics.average_confidence_difference == pytest.approx(0.1)
    assert metrics.attack_success is False
