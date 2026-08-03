import pytest

from ml.metrics.classification import (
    ClassificationEvaluationSample,
    build_evaluation_image_diagnostics,
    calculate_advanced_classification_metrics,
)
from ml.models.classification import (
    ClassificationClassScore,
    DetailedClassificationPrediction,
)
from schemas.classification_metrics import (
    ClassReference,
    LabelResolutionSummary,
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
        for index in ranked_indices[:5]
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


def _sample(
    ground_truth_index: int | None,
    original: tuple[float, ...],
    adversarial: tuple[float, ...],
) -> ClassificationEvaluationSample:
    ground_truth = None

    if ground_truth_index is not None:
        ground_truth = ClassReference(
            index=ground_truth_index,
            label=CLASS_LABELS[ground_truth_index],
        )

    return ClassificationEvaluationSample(
        ground_truth=ground_truth,
        original=_prediction(original),
        adversarial=_prediction(adversarial),
        perturbation_linf=0.03,
        perturbation_l2=1.0,
    )


def _resolution(
    resolved_labels: int,
    labeled_images: int | None = None,
) -> LabelResolutionSummary:
    if labeled_images is None:
        labeled_images = resolved_labels

    return LabelResolutionSummary(
        labeled_images=labeled_images,
        resolved_labels=resolved_labels,
        unresolved_labels=(labeled_images - resolved_labels),
        coverage=(resolved_labels / labeled_images if labeled_images else 0.0),
        unresolved=[
            {
                "image_id": f"unresolved-{index}",
                "filename": f"image-{index}.png",
                "provided_label": "unknown",
                "reason": ("Label does not match any model class"),
            }
            for index in range(labeled_images - resolved_labels)
        ],
    )


def test_calculates_robustness_transitions():
    samples = [
        # correcta -> incorrecta
        _sample(
            0,
            original=(0.8, 0.1, 0.1),
            adversarial=(0.2, 0.7, 0.1),
        ),
        # correcta -> correcta
        _sample(
            1,
            original=(0.1, 0.8, 0.1),
            adversarial=(0.1, 0.7, 0.2),
        ),
        # incorrecta -> correcta
        _sample(
            2,
            original=(0.1, 0.8, 0.1),
            adversarial=(0.1, 0.2, 0.7),
        ),
        # incorrecta -> incorrecta
        _sample(
            0,
            original=(0.1, 0.2, 0.7),
            adversarial=(0.1, 0.3, 0.6),
        ),
    ]

    metrics = calculate_advanced_classification_metrics(
        samples=samples,
        label_resolution=_resolution(4),
        class_labels=CLASS_LABELS,
    )

    robustness = metrics.robustness

    assert robustness.changed_predictions == 2
    assert robustness.unchanged_predictions == 2
    assert robustness.prediction_flip_rate == 0.5

    assert robustness.correct_to_incorrect == 1
    assert robustness.incorrect_to_correct == 1

    assert robustness.clean_accuracy.value == 0.5
    assert robustness.robust_accuracy.value == 0.5
    assert robustness.accuracy_drop.value == 0.0

    assert robustness.conditional_attack_success_rate.value == 0.5

    assert robustness.corrected_by_attack_rate.value == 0.5


def test_calculates_aggregate_metrics_and_deltas():
    samples = [
        _sample(
            0,
            original=(0.8, 0.1, 0.1),
            adversarial=(0.2, 0.7, 0.1),
        ),
        _sample(
            1,
            original=(0.1, 0.8, 0.1),
            adversarial=(0.1, 0.7, 0.2),
        ),
    ]

    metrics = calculate_advanced_classification_metrics(
        samples=samples,
        label_resolution=_resolution(2),
        class_labels=CLASS_LABELS,
    )

    aggregate = metrics.supervised.aggregate

    assert aggregate.original.accuracy == 1.0
    assert aggregate.adversarial.accuracy == 0.5
    assert aggregate.delta.accuracy == 0.5

    assert aggregate.original.f1_macro == 1.0
    assert aggregate.adversarial.f1_macro < 1.0

    assert aggregate.original.top_5_accuracy == 1.0
    assert aggregate.adversarial.top_5_accuracy == 1.0
    assert metrics.supervised.roc.micro.original.available
    assert metrics.supervised.precision_recall.micro.original.available

    assert metrics.supervised.roc.macro_original.value == 1.0

    for class_metrics in metrics.supervised.per_class:
        assert class_metrics.original.roc_auc.available
        assert class_metrics.original.average_precision.available


def test_confusion_matrix_includes_predicted_only_class():
    samples = [
        _sample(
            0,
            original=(0.8, 0.1, 0.1),
            adversarial=(0.1, 0.2, 0.7),
        ),
        _sample(
            1,
            original=(0.1, 0.8, 0.1),
            adversarial=(0.1, 0.8, 0.1),
        ),
    ]

    metrics = calculate_advanced_classification_metrics(
        samples=samples,
        label_resolution=_resolution(2),
        class_labels=CLASS_LABELS,
    )

    matrices = metrics.supervised.confusion_matrices

    assert [reference.index for reference in matrices.labels] == [0, 1, 2]

    assert matrices.original == [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
    ]

    assert matrices.adversarial == [
        [0, 0, 1],
        [0, 1, 0],
        [0, 0, 0],
    ]


def test_reports_undefined_class_precision():
    samples = [
        _sample(
            0,
            original=(0.8, 0.1, 0.1),
            adversarial=(0.1, 0.8, 0.1),
        ),
    ]

    metrics = calculate_advanced_classification_metrics(
        samples=samples,
        label_resolution=_resolution(1),
        class_labels=CLASS_LABELS,
    )

    class_metrics = metrics.supervised.per_class[0]

    assert class_metrics.original.precision.available
    assert class_metrics.adversarial.precision.available is False

    assert "no samples were predicted" in class_metrics.adversarial.precision.reason


def test_no_resolved_labels_disables_supervised_metrics():
    samples = [
        _sample(
            None,
            original=(0.8, 0.1, 0.1),
            adversarial=(0.2, 0.7, 0.1),
        ),
    ]

    metrics = calculate_advanced_classification_metrics(
        samples=samples,
        label_resolution=_resolution(
            resolved_labels=0,
            labeled_images=0,
        ),
        class_labels=CLASS_LABELS,
    )

    assert metrics.supervised is None
    assert metrics.supervised_status.available is False

    assert metrics.robustness.clean_accuracy.available is False


def test_diagnostics_calculates_true_class_probability():
    ground_truth = ClassReference(
        index=0,
        label="cat",
    )

    diagnostics = build_evaluation_image_diagnostics(
        ground_truth=ground_truth,
        original=_prediction((0.8, 0.1, 0.1)),
        adversarial=_prediction((0.2, 0.7, 0.1)),
    )

    assert diagnostics.transition == ("correct_to_incorrect")

    assert diagnostics.original_true_class_probability == pytest.approx(0.8)

    assert diagnostics.adversarial_true_class_probability == pytest.approx(0.2)

    assert diagnostics.true_class_probability_drop == pytest.approx(0.6)

    assert diagnostics.prediction_margin_drop == (pytest.approx(0.2))
