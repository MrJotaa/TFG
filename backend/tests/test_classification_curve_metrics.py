from math import isfinite

import pytest

from ml.metrics.classification_curves import (
    calculate_classification_curves,
)
from ml.models.classification import (
    ClassificationClassScore,
    DetailedClassificationPrediction,
)

CLASS_LABELS = (
    "cat",
    "dog",
    "car",
)


def _prediction(
    probabilities: tuple[float, ...],
) -> DetailedClassificationPrediction:
    ranked_indices = sorted(
        range(len(probabilities)),
        key=lambda index: probabilities[index],
        reverse=True,
    )

    top_k = tuple(
        ClassificationClassScore(
            index=index,
            label=CLASS_LABELS[index],
            probability=probabilities[index],
        )
        for index in ranked_indices
    )

    class_index = ranked_indices[0]

    margin = None

    if len(ranked_indices) >= 2:
        margin = probabilities[ranked_indices[0]] - probabilities[ranked_indices[1]]

    return DetailedClassificationPrediction(
        class_index=class_index,
        class_label=CLASS_LABELS[class_index],
        confidence=probabilities[class_index],
        probabilities=probabilities,
        top_k=top_k,
        prediction_margin=margin,
    )


def test_perfect_original_curves_have_maximum_scores():
    y_true = [0, 0, 1, 1]

    original_predictions = [
        _prediction((0.90, 0.05, 0.05)),
        _prediction((0.80, 0.10, 0.10)),
        _prediction((0.10, 0.80, 0.10)),
        _prediction((0.05, 0.90, 0.05)),
    ]

    adversarial_predictions = [
        _prediction((0.10, 0.80, 0.10)),
        _prediction((0.20, 0.70, 0.10)),
        _prediction((0.80, 0.10, 0.10)),
        _prediction((0.70, 0.20, 0.10)),
    ]

    metrics = calculate_classification_curves(
        y_true=y_true,
        original_predictions=original_predictions,
        adversarial_predictions=(adversarial_predictions),
        metric_labels=[0, 1],
        class_labels=CLASS_LABELS,
    )

    assert metrics.roc.micro.original.available
    assert metrics.roc.micro.original.score == 1.0

    assert metrics.roc.micro.adversarial.available
    assert metrics.roc.micro.adversarial.score == 0.0

    assert metrics.roc.macro_original.value == 1.0
    assert metrics.roc.macro_adversarial.value == 0.0

    assert metrics.precision_recall.micro.original.score == 1.0

    assert metrics.precision_recall.micro.adversarial.score < 1.0

    for class_index in (0, 1):
        class_metrics = metrics.per_class_metrics[class_index]

        assert class_metrics.original_roc_auc.value == 1.0

        assert class_metrics.adversarial_roc_auc.value == 0.0

        assert class_metrics.original_average_precision.value == 1.0


def test_roc_infinite_threshold_is_serialized_as_none():
    metrics = calculate_classification_curves(
        y_true=[0, 1],
        original_predictions=[
            _prediction((0.9, 0.1, 0.0)),
            _prediction((0.1, 0.9, 0.0)),
        ],
        adversarial_predictions=[
            _prediction((0.8, 0.2, 0.0)),
            _prediction((0.2, 0.8, 0.0)),
        ],
        metric_labels=[0, 1],
        class_labels=CLASS_LABELS,
    )

    points = metrics.roc.micro.original.points

    assert points[0].threshold is None

    assert all(point.threshold is None or isfinite(point.threshold) for point in points)


def test_precision_recall_last_point_has_no_threshold():
    metrics = calculate_classification_curves(
        y_true=[0, 1],
        original_predictions=[
            _prediction((0.9, 0.1, 0.0)),
            _prediction((0.1, 0.9, 0.0)),
        ],
        adversarial_predictions=[
            _prediction((0.8, 0.2, 0.0)),
            _prediction((0.2, 0.8, 0.0)),
        ],
        metric_labels=[0, 1],
        class_labels=CLASS_LABELS,
    )

    points = metrics.precision_recall.micro.original.points

    assert points[-1].threshold is None
    assert points[-1].x == 0.0
    assert points[-1].y == 1.0


def test_single_ground_truth_class_disables_roc():
    metrics = calculate_classification_curves(
        y_true=[0, 0],
        original_predictions=[
            _prediction((0.9, 0.05, 0.05)),
            _prediction((0.8, 0.10, 0.10)),
        ],
        adversarial_predictions=[
            _prediction((0.7, 0.20, 0.10)),
            _prediction((0.6, 0.30, 0.10)),
        ],
        metric_labels=[0],
        class_labels=CLASS_LABELS,
    )

    assert metrics.roc.micro.original.available is False

    assert "muestra negativa" in metrics.roc.micro.original.reason

    assert metrics.roc.macro_original.available is False

    class_roc = metrics.roc.per_class[0]

    assert class_roc.curves.original.available is False


def test_single_ground_truth_class_keeps_pr_available():
    metrics = calculate_classification_curves(
        y_true=[0, 0],
        original_predictions=[
            _prediction((0.9, 0.05, 0.05)),
            _prediction((0.8, 0.10, 0.10)),
        ],
        adversarial_predictions=[
            _prediction((0.7, 0.20, 0.10)),
            _prediction((0.6, 0.30, 0.10)),
        ],
        metric_labels=[0],
        class_labels=CLASS_LABELS,
    )

    precision_recall = metrics.precision_recall

    assert precision_recall.micro.original.available is True

    assert precision_recall.micro.original.score == 1.0

    assert "no contiene muestras negativas" in precision_recall.micro.original.warning


def test_low_class_support_adds_warning():
    metrics = calculate_classification_curves(
        y_true=[0, 1],
        original_predictions=[
            _prediction((0.9, 0.1, 0.0)),
            _prediction((0.1, 0.9, 0.0)),
        ],
        adversarial_predictions=[
            _prediction((0.8, 0.2, 0.0)),
            _prediction((0.2, 0.8, 0.0)),
        ],
        metric_labels=[0, 1],
        class_labels=CLASS_LABELS,
    )

    class_curve = metrics.roc.per_class[0]

    assert class_curve.support == 1

    assert "menos de 2 muestras positivas" in class_curve.curves.original.warning


def test_curve_analysis_rejects_mismatched_lengths():
    with pytest.raises(
        ValueError,
        match=("Original prediction count must match"),
    ):
        calculate_classification_curves(
            y_true=[0, 1],
            original_predictions=[
                _prediction((0.9, 0.1, 0.0)),
            ],
            adversarial_predictions=[
                _prediction((0.8, 0.2, 0.0)),
                _prediction((0.2, 0.8, 0.0)),
            ],
            metric_labels=[0, 1],
            class_labels=CLASS_LABELS,
        )
