import math

import pytest

from ml.models.detection import Detection, DetectionBox
from utils.detection_metrics import (
    average,
    average_detection_confidence,
    calculate_detection_comparison,
    LEGACY_DETECTION_SUCCESS,
    MATCHED_DETECTION_SUCCESS,
)


def _detection(
    confidence: float,
    *,
    class_id: int = 0,
    bbox: tuple[float, float, float, float] = (10, 10, 30, 30),
):
    return Detection(class_id, confidence, DetectionBox(*bbox))


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


def test_confidence_decrease_alone_does_not_count_as_success():
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
    assert metrics.attack_success is False


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


def test_localization_shift_counts_as_matching_failure():
    """Count and confidence remain unchanged despite a large box shift."""
    original = Detection(0, 0.9, DetectionBox(10, 10, 30, 30))
    shifted = Detection(0, 0.9, DetectionBox(60, 60, 80, 80))

    metrics = calculate_detection_comparison([original], [shifted])

    assert metrics.original_count == metrics.adversarial_count == 1
    assert metrics.average_confidence_difference == 0.0
    assert metrics.attack_success is True


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
        success_criterion=LEGACY_DETECTION_SUCCESS,
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
        success_criterion=LEGACY_DETECTION_SUCCESS,
    )

    assert metrics.attack_success is legacy_attack_success


@pytest.mark.parametrize("confidence", [0.899, 0.8, 0.25])
def test_confidence_reduction_without_box_or_class_change_is_not_success(confidence):
    metrics = calculate_detection_comparison(
        [_detection(0.9)], [_detection(confidence)]
    )
    assert metrics.attack_success is False
    assert metrics.average_confidence_difference > 0


def test_class_change_same_box_counts_as_success():
    metrics = calculate_detection_comparison(
        [_detection(0.9, class_id=0)], [_detection(0.9, class_id=1)]
    )
    assert metrics.attack_success is True
    assert metrics.detections_difference == 0


def test_iou_threshold_inclusive_and_significant_shift():
    original = [_detection(0.9, bbox=(0, 0, 30, 20))]
    # Exact 0.5 IoU: horizontal overlap 20, union 40.
    boundary = [_detection(0.9, bbox=(10, 0, 40, 20))]
    shifted = [_detection(0.9, bbox=(11, 0, 41, 20))]
    assert calculate_detection_comparison(original, boundary).attack_success is False
    assert calculate_detection_comparison(original, shifted).attack_success is True


def test_one_to_one_matching_prevents_double_use_of_one_candidate():
    originals = [
        _detection(0.8, bbox=(0, 0, 20, 20)),
        _detection(0.9, bbox=(1, 0, 21, 20)),
    ]
    adversarial = [_detection(0.9, bbox=(0, 0, 20, 20))]
    assert calculate_detection_comparison(originals, adversarial).attack_success is True


def test_matching_uses_maximum_cardinality_not_confidence_or_greedy_order():
    originals = [
        _detection(0.8, bbox=(0, 0, 20, 20)),
        _detection(0.9, bbox=(10, 0, 30, 20)),
    ]
    adversarial = [
        _detection(0.6, bbox=(5, 0, 25, 20)),  # overlaps both at IoU 0.6
        _detection(0.3, bbox=(0, 0, 20, 20)),  # overlaps only first at 1.0
    ]
    assert calculate_detection_comparison(originals, adversarial).attack_success is False


def test_extra_detections_do_not_mask_lost_original_detection():
    originals = [
        _detection(0.9, bbox=(0, 0, 20, 20)),
        _detection(0.9, bbox=(50, 0, 70, 20)),
    ]
    adversarial = [
        _detection(0.9, bbox=(0, 0, 20, 20)),
        _detection(0.9, bbox=(90, 0, 110, 20)),
    ]
    assert calculate_detection_comparison(originals, adversarial).attack_success is True
    assert calculate_detection_comparison([], adversarial).attack_success is False


def test_legacy_formula_can_still_be_requested_explicitly():
    original = [_detection(0.9)]
    adversarial = [_detection(0.899)]
    assert calculate_detection_comparison(original, adversarial).attack_success is False
    assert calculate_detection_comparison(
        original, adversarial, success_criterion=LEGACY_DETECTION_SUCCESS
    ).attack_success is True
    assert MATCHED_DETECTION_SUCCESS != LEGACY_DETECTION_SUCCESS
