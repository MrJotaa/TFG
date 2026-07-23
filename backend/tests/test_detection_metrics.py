import math
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
    assert metrics.detections_difference == 1
    assert metrics.average_original_confidence == pytest.approx(0.85)
    assert metrics.average_adversarial_confidence == 0.5
    assert metrics.average_confidence_difference == pytest.approx(0.35)
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
    assert metrics.average_confidence_difference == pytest.approx(0.3)
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
    assert metrics.average_confidence_difference == pytest.approx(-0.1)
    assert metrics.attack_success is False


def test_detection_comparison_reports_unchanged_confidence_and_count():
    metrics = calculate_detection_comparison(
        original_detections=[_detection(0.7)],
        adversarial_detections=[_detection(0.7)],
    )

    assert metrics.detections_difference == 0
    assert metrics.average_confidence_difference == 0.0
    assert metrics.attack_success is False


def test_detection_comparison_reports_new_detections_with_negative_delta():
    metrics = calculate_detection_comparison(
        original_detections=[_detection(0.7)],
        adversarial_detections=[
            _detection(0.7),
            _detection(0.8),
        ],
    )

    assert metrics.detections_difference == -1
    assert metrics.attack_success is False


def test_detection_comparison_handles_no_detections():
    metrics = calculate_detection_comparison(
        original_detections=[],
        adversarial_detections=[],
    )

    assert metrics.original_count == 0
    assert metrics.adversarial_count == 0
    assert metrics.detections_difference == 0
    assert metrics.average_original_confidence == 0.0
    assert metrics.average_adversarial_confidence == 0.0
    assert metrics.average_confidence_difference == 0.0
    assert math.isfinite(metrics.average_confidence_difference)
    assert metrics.attack_success is False


@pytest.mark.parametrize(
    ("original", "adversarial", "expected_count_delta", "expected_confidence_delta"),
    [
        ([0.8], [], 1, 0.8),
        ([], [0.8], -1, -0.8),
    ],
)
def test_detection_comparison_handles_detections_on_only_one_variant(
    original,
    adversarial,
    expected_count_delta,
    expected_confidence_delta,
):
    metrics = calculate_detection_comparison(
        original_detections=[_detection(value) for value in original],
        adversarial_detections=[_detection(value) for value in adversarial],
    )

    assert metrics.detections_difference == expected_count_delta
    assert metrics.average_confidence_difference == expected_confidence_delta
    assert math.isfinite(metrics.average_confidence_difference)


def test_detection_comparison_does_not_round_before_calculating_delta():
    metrics = calculate_detection_comparison(
        original_detections=[
            _detection(0.12344),
            _detection(0.12344),
        ],
        adversarial_detections=[
            _detection(0.12336),
            _detection(0.12336),
        ],
    )

    assert metrics.average_original_confidence == 0.12344
    assert metrics.average_adversarial_confidence == 0.12336
    assert metrics.average_confidence_difference == pytest.approx(0.00008)


def test_legacy_attack_success_rounds_each_confidence_before_averaging():
    original_detections = [
        _detection(0.500049),
        _detection(0.500149),
    ]
    adversarial_detections = [
        _detection(0.499951),
        _detection(0.500051),
    ]

    assert [round(detection.confidence, 4) for detection in original_detections] == [
        0.5,
        0.5001,
    ]
    assert [round(detection.confidence, 4) for detection in adversarial_detections] == [
        0.5,
        0.5001,
    ]

    metrics = calculate_detection_comparison(
        original_detections=original_detections,
        adversarial_detections=adversarial_detections,
    )

    assert metrics.average_original_confidence == pytest.approx(0.500099)
    assert metrics.average_adversarial_confidence == pytest.approx(0.500001)
    assert metrics.average_adversarial_confidence < metrics.average_original_confidence
    assert metrics.average_confidence_difference == pytest.approx(0.000098)
    assert metrics.attack_success is False


@pytest.mark.parametrize(
    ("original", "adversarial"),
    [
        ([0.9, 0.8], [0.5]),
        ([0.9], [0.6]),
        ([0.7], [0.8]),
        ([0.7], [0.7]),
        ([], []),
        ([], [0.8]),
        ([0.8], []),
        ([0.12344], [0.12336]),
    ],
)
def test_detection_comparison_preserves_legacy_attack_success(
    original,
    adversarial,
):
    original_detections = [_detection(value) for value in original]
    adversarial_detections = [_detection(value) for value in adversarial]

    legacy_original_average = average_detection_confidence(
        [
            _detection(round(detection.confidence, 4))
            for detection in original_detections
        ]
    )
    legacy_adversarial_average = average_detection_confidence(
        [
            _detection(round(detection.confidence, 4))
            for detection in adversarial_detections
        ]
    )
    legacy_confidence_difference = round(
        round(legacy_adversarial_average, 4) - round(legacy_original_average, 4),
        4,
    )
    legacy_attack_success = (
        len(adversarial_detections) < len(original_detections)
        or legacy_confidence_difference < 0
    )

    metrics = calculate_detection_comparison(
        original_detections=original_detections,
        adversarial_detections=adversarial_detections,
    )

    assert metrics.attack_success is legacy_attack_success
