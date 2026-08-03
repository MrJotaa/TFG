from collections.abc import Sequence
from dataclasses import dataclass

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from ml.metrics.classification_curves import (
    PerClassCurveMetricValues,
    calculate_classification_curves,
)
from ml.models.classification import (
    DetailedClassificationPrediction,
)
from schemas.classification_metrics import (
    AdvancedClassificationMetrics,
    AggregateMetricDelta,
    AggregateMetricsComparison,
    AggregateMetricSnapshot,
    AvailabilityStatus,
    ClassReference,
    ConfusionMatrices,
    EvaluationImageDiagnostics,
    LabelResolutionSummary,
    MetricValue,
    PerClassMetricsComparison,
    PerClassMetricSnapshot,
    RobustnessMetrics,
    SupervisedClassificationAnalysis,
)


@dataclass(frozen=True, slots=True)
class ClassificationEvaluationSample:
    """
    Resultado interno de evaluar una imagen original y adversaria.
    """

    ground_truth: ClassReference | None

    original: DetailedClassificationPrediction
    adversarial: DetailedClassificationPrediction

    perturbation_linf: float
    perturbation_l2: float


def _round_metric(value: float) -> float:
    return round(float(value), 4)


def _average(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("Cannot calculate the average of empty values")

    return sum(values) / len(values)


def _available_metric(
    value: float,
    warning: str | None = None,
) -> MetricValue:
    return MetricValue(
        available=True,
        value=_round_metric(value),
        warning=warning,
    )


def _unavailable_metric(reason: str) -> MetricValue:
    return MetricValue(
        available=False,
        reason=reason,
    )


def _metric_difference(
    original: MetricValue,
    adversarial: MetricValue,
) -> MetricValue:
    if (
        not original.available
        or not adversarial.available
        or original.value is None
        or adversarial.value is None
    ):
        return _unavailable_metric(
            "Delta is unavailable because the metric is "
            "undefined for original or adversarial predictions"
        )

    return _available_metric(original.value - adversarial.value)


def _validate_samples(
    samples: Sequence[ClassificationEvaluationSample],
    class_labels: tuple[str, ...],
) -> None:
    if not samples:
        raise ValueError("At least one evaluation sample is required")

    if not class_labels:
        raise ValueError("Classification model has no class labels")

    class_count = len(class_labels)

    for sample in samples:
        predictions = (
            sample.original,
            sample.adversarial,
        )

        for prediction in predictions:
            if len(prediction.probabilities) != class_count:
                raise ValueError(
                    "Prediction probability count does not match the model class count"
                )

        if sample.ground_truth is None:
            continue

        if not (0 <= sample.ground_truth.index < class_count):
            raise ValueError(
                "Ground-truth class index is outside the model class catalog"
            )


def build_evaluation_image_diagnostics(
    ground_truth: ClassReference | None,
    original: DetailedClassificationPrediction,
    adversarial: DetailedClassificationPrediction,
    label_resolution_reason: str | None = None,
) -> EvaluationImageDiagnostics:
    original_true_class_probability = None
    adversarial_true_class_probability = None
    true_class_probability_drop = None

    original_top_5_correct = None
    adversarial_top_5_correct = None

    transition = None

    if ground_truth is not None:
        class_index = ground_truth.index

        original_true_class_probability = original.probabilities[class_index]
        adversarial_true_class_probability = adversarial.probabilities[class_index]

        true_class_probability_drop = (
            original_true_class_probability - adversarial_true_class_probability
        )

        original_top_5_correct = any(
            score.index == class_index for score in original.top_k
        )
        adversarial_top_5_correct = any(
            score.index == class_index for score in adversarial.top_k
        )

        original_correct = original.class_index == class_index
        adversarial_correct = adversarial.class_index == class_index

        if original_correct and adversarial_correct:
            transition = "correct_to_correct"
        elif original_correct:
            transition = "correct_to_incorrect"
        elif adversarial_correct:
            transition = "incorrect_to_correct"
        else:
            transition = "incorrect_to_incorrect"

    prediction_margin_drop = None

    if (
        original.prediction_margin is not None
        and adversarial.prediction_margin is not None
    ):
        prediction_margin_drop = (
            original.prediction_margin - adversarial.prediction_margin
        )

    return EvaluationImageDiagnostics(
        ground_truth_class=ground_truth,
        label_resolution_reason=(label_resolution_reason),
        original_class_index=original.class_index,
        adversarial_class_index=(adversarial.class_index),
        original_true_class_probability=(original_true_class_probability),
        adversarial_true_class_probability=(adversarial_true_class_probability),
        true_class_probability_drop=(true_class_probability_drop),
        original_prediction_margin=(original.prediction_margin),
        adversarial_prediction_margin=(adversarial.prediction_margin),
        prediction_margin_drop=(prediction_margin_drop),
        original_top_5_correct=(original_top_5_correct),
        adversarial_top_5_correct=(adversarial_top_5_correct),
        transition=transition,
    )


def _calculate_aggregate_snapshot(
    y_true: list[int],
    y_pred: list[int],
    top_5_correct: list[bool],
    metric_labels: list[int],
) -> AggregateMetricSnapshot:
    (
        precision_macro,
        recall_macro,
        f1_macro,
        _,
    ) = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=metric_labels,
        average="macro",
        zero_division=0,
    )

    (
        precision_weighted,
        recall_weighted,
        f1_weighted,
        _,
    ) = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=metric_labels,
        average="weighted",
        zero_division=0,
    )

    top_5_accuracy = sum(top_5_correct) / len(top_5_correct)

    return AggregateMetricSnapshot(
        accuracy=_round_metric(accuracy_score(y_true, y_pred)),
        precision_macro=_round_metric(precision_macro),
        recall_macro=_round_metric(recall_macro),
        f1_macro=_round_metric(f1_macro),
        precision_weighted=_round_metric(precision_weighted),
        recall_weighted=_round_metric(recall_weighted),
        f1_weighted=_round_metric(f1_weighted),
        top_5_accuracy=_round_metric(top_5_accuracy),
    )


def _calculate_aggregate_delta(
    original: AggregateMetricSnapshot,
    adversarial: AggregateMetricSnapshot,
) -> AggregateMetricDelta:
    return AggregateMetricDelta(
        accuracy=_round_metric(original.accuracy - adversarial.accuracy),
        precision_macro=_round_metric(
            original.precision_macro - adversarial.precision_macro
        ),
        recall_macro=_round_metric(original.recall_macro - adversarial.recall_macro),
        f1_macro=_round_metric(original.f1_macro - adversarial.f1_macro),
        precision_weighted=_round_metric(
            original.precision_weighted - adversarial.precision_weighted
        ),
        recall_weighted=_round_metric(
            original.recall_weighted - adversarial.recall_weighted
        ),
        f1_weighted=_round_metric(original.f1_weighted - adversarial.f1_weighted),
        top_5_accuracy=_round_metric(
            original.top_5_accuracy - adversarial.top_5_accuracy
        ),
    )


def _build_per_class_snapshot(
    precision: float,
    recall: float,
    f1: float,
    predicted_count: int,
    support: int,
    roc_auc: MetricValue,
    average_precision: MetricValue,
) -> PerClassMetricSnapshot:
    warning = None

    if support < 2:
        warning = "Class support is lower than 2 samples"

    if predicted_count == 0:
        precision_metric = MetricValue(
            available=False,
            reason=(
                "Precision is undefined because no samples were predicted as this class"
            ),
            warning=warning,
        )
    else:
        precision_metric = _available_metric(
            precision,
            warning=warning,
        )

    return PerClassMetricSnapshot(
        precision=precision_metric,
        recall=_available_metric(
            recall,
            warning=warning,
        ),
        f1=_available_metric(
            f1,
            warning=warning,
        ),
        roc_auc=roc_auc,
        average_precision=average_precision,
    )


def _calculate_per_class_metrics(
    y_true: list[int],
    original_predictions: list[int],
    adversarial_predictions: list[int],
    metric_labels: list[int],
    class_labels: tuple[str, ...],
    curve_metrics_by_class: dict[
        int,
        PerClassCurveMetricValues,
    ],
) -> list[PerClassMetricsComparison]:
    (
        original_precision,
        original_recall,
        original_f1,
        supports,
    ) = precision_recall_fscore_support(
        y_true,
        original_predictions,
        labels=metric_labels,
        average=None,
        zero_division=0,
    )

    (
        adversarial_precision,
        adversarial_recall,
        adversarial_f1,
        _,
    ) = precision_recall_fscore_support(
        y_true,
        adversarial_predictions,
        labels=metric_labels,
        average=None,
        zero_division=0,
    )

    comparisons = []

    for position, class_index in enumerate(metric_labels):
        support = int(supports[position])
        curve_metrics = curve_metrics_by_class[class_index]

        original_snapshot = _build_per_class_snapshot(
            precision=float(original_precision[position]),
            recall=float(original_recall[position]),
            f1=float(original_f1[position]),
            predicted_count=sum(
                prediction == class_index for prediction in original_predictions
            ),
            support=support,
            roc_auc=curve_metrics.original_roc_auc,
            average_precision=(curve_metrics.original_average_precision),
        )

        adversarial_snapshot = _build_per_class_snapshot(
            precision=float(adversarial_precision[position]),
            recall=float(adversarial_recall[position]),
            f1=float(adversarial_f1[position]),
            predicted_count=sum(
                prediction == class_index for prediction in adversarial_predictions
            ),
            support=support,
            roc_auc=(curve_metrics.adversarial_roc_auc),
            average_precision=(curve_metrics.adversarial_average_precision),
        )

        delta_snapshot = PerClassMetricSnapshot(
            precision=_metric_difference(
                original_snapshot.precision,
                adversarial_snapshot.precision,
            ),
            recall=_metric_difference(
                original_snapshot.recall,
                adversarial_snapshot.recall,
            ),
            f1=_metric_difference(
                original_snapshot.f1,
                adversarial_snapshot.f1,
            ),
            roc_auc=_metric_difference(
                original_snapshot.roc_auc,
                adversarial_snapshot.roc_auc,
            ),
            average_precision=_metric_difference(
                original_snapshot.average_precision,
                adversarial_snapshot.average_precision,
            ),
        )

        comparisons.append(
            PerClassMetricsComparison(
                class_reference=ClassReference(
                    index=class_index,
                    label=class_labels[class_index],
                ),
                support=support,
                original=original_snapshot,
                adversarial=(adversarial_snapshot),
                delta=delta_snapshot,
            )
        )

    return comparisons


def _round_normalized_matrix(
    matrix,
) -> list[list[float]]:
    return [[_round_metric(value) for value in row] for row in matrix.tolist()]


def _calculate_confusion_matrices(
    y_true: list[int],
    original_predictions: list[int],
    adversarial_predictions: list[int],
    class_labels: tuple[str, ...],
) -> ConfusionMatrices:
    matrix_indices = sorted(
        set(y_true) | set(original_predictions) | set(adversarial_predictions)
    )

    original_matrix = confusion_matrix(
        y_true,
        original_predictions,
        labels=matrix_indices,
    )

    adversarial_matrix = confusion_matrix(
        y_true,
        adversarial_predictions,
        labels=matrix_indices,
    )

    original_normalized = confusion_matrix(
        y_true,
        original_predictions,
        labels=matrix_indices,
        normalize="true",
    )

    adversarial_normalized = confusion_matrix(
        y_true,
        adversarial_predictions,
        labels=matrix_indices,
        normalize="true",
    )

    return ConfusionMatrices(
        labels=[
            ClassReference(
                index=class_index,
                label=class_labels[class_index],
            )
            for class_index in matrix_indices
        ],
        original=original_matrix.tolist(),
        adversarial=adversarial_matrix.tolist(),
        original_normalized=(_round_normalized_matrix(original_normalized)),
        adversarial_normalized=(_round_normalized_matrix(adversarial_normalized)),
    )


def calculate_advanced_classification_metrics(
    samples: Sequence[ClassificationEvaluationSample],
    label_resolution: LabelResolutionSummary,
    class_labels: tuple[str, ...],
) -> AdvancedClassificationMetrics:
    _validate_samples(
        samples=samples,
        class_labels=class_labels,
    )

    total_images = len(samples)

    changed_predictions = sum(
        sample.original.class_index != sample.adversarial.class_index
        for sample in samples
    )

    unchanged_predictions = total_images - changed_predictions

    resolved_samples = [sample for sample in samples if sample.ground_truth is not None]

    if len(resolved_samples) != label_resolution.resolved_labels:
        raise ValueError(
            "Resolved sample count does not match label resolution summary"
        )

    correct_to_incorrect = sum(
        sample.original.class_index == sample.ground_truth.index
        and sample.adversarial.class_index != sample.ground_truth.index
        for sample in resolved_samples
    )

    incorrect_to_correct = sum(
        sample.original.class_index != sample.ground_truth.index
        and sample.adversarial.class_index == sample.ground_truth.index
        for sample in resolved_samples
    )

    average_linf = _average([sample.perturbation_linf for sample in samples])

    average_l2 = _average([sample.perturbation_l2 for sample in samples])

    margin_drops = [
        (sample.original.prediction_margin - sample.adversarial.prediction_margin)
        for sample in samples
        if (
            sample.original.prediction_margin is not None
            and sample.adversarial.prediction_margin is not None
        )
    ]

    if margin_drops:
        average_margin_drop = _available_metric(_average(margin_drops))
    else:
        average_margin_drop = _unavailable_metric(
            "Prediction margin requires at least two model classes"
        )

    supervised = None

    if resolved_samples:
        correct_original = sum(
            sample.original.class_index == sample.ground_truth.index
            for sample in resolved_samples
        )

        correct_adversarial = sum(
            sample.adversarial.class_index == sample.ground_truth.index
            for sample in resolved_samples
        )

        incorrect_original = len(resolved_samples) - correct_original

        clean_accuracy_value = correct_original / len(resolved_samples)
        robust_accuracy_value = correct_adversarial / len(resolved_samples)

        clean_accuracy = _available_metric(clean_accuracy_value)
        robust_accuracy = _available_metric(robust_accuracy_value)
        accuracy_drop = _available_metric(clean_accuracy_value - robust_accuracy_value)

        if correct_original:
            conditional_asr = _available_metric(correct_to_incorrect / correct_original)
        else:
            conditional_asr = _unavailable_metric(
                "Conditional attack success rate requires "
                "at least one correct original prediction"
            )

        if incorrect_original:
            corrected_by_attack_rate = _available_metric(
                incorrect_to_correct / incorrect_original
            )
        else:
            corrected_by_attack_rate = _unavailable_metric(
                "Corrected-by-attack rate requires at "
                "least one incorrect original prediction"
            )

        true_class_probability_drops = [
            (
                sample.original.probabilities[sample.ground_truth.index]
                - sample.adversarial.probabilities[sample.ground_truth.index]
            )
            for sample in resolved_samples
        ]

        average_true_class_probability_drop = _available_metric(
            _average(true_class_probability_drops)
        )

        y_true = [sample.ground_truth.index for sample in resolved_samples]

        original_detailed_predictions = [sample.original for sample in resolved_samples]

        adversarial_detailed_predictions = [
            sample.adversarial for sample in resolved_samples
        ]

        original_predictions = [
            sample.original.class_index for sample in resolved_samples
        ]

        adversarial_predictions = [
            sample.adversarial.class_index for sample in resolved_samples
        ]

        original_top_5 = [
            any(
                score.index == sample.ground_truth.index
                for score in sample.original.top_k
            )
            for sample in resolved_samples
        ]

        adversarial_top_5 = [
            any(
                score.index == sample.ground_truth.index
                for score in sample.adversarial.top_k
            )
            for sample in resolved_samples
        ]

        metric_labels = sorted(set(y_true))

        curve_metrics = calculate_classification_curves(
            y_true=y_true,
            original_predictions=(original_detailed_predictions),
            adversarial_predictions=(adversarial_detailed_predictions),
            metric_labels=metric_labels,
            class_labels=class_labels,
        )

        original_aggregate = _calculate_aggregate_snapshot(
            y_true=y_true,
            y_pred=original_predictions,
            top_5_correct=original_top_5,
            metric_labels=metric_labels,
        )

        adversarial_aggregate = _calculate_aggregate_snapshot(
            y_true=y_true,
            y_pred=adversarial_predictions,
            top_5_correct=adversarial_top_5,
            metric_labels=metric_labels,
        )

        aggregate_comparison = AggregateMetricsComparison(
            original=original_aggregate,
            adversarial=(adversarial_aggregate),
            delta=_calculate_aggregate_delta(
                original_aggregate,
                adversarial_aggregate,
            ),
        )

        supervised = SupervisedClassificationAnalysis(
            aggregate=aggregate_comparison,
            per_class=_calculate_per_class_metrics(
                y_true=y_true,
                original_predictions=(original_predictions),
                adversarial_predictions=(adversarial_predictions),
                metric_labels=metric_labels,
                class_labels=class_labels,
                curve_metrics_by_class=(curve_metrics.per_class_metrics),
            ),
            confusion_matrices=(
                _calculate_confusion_matrices(
                    y_true=y_true,
                    original_predictions=(original_predictions),
                    adversarial_predictions=(adversarial_predictions),
                    class_labels=class_labels,
                )
            ),
            roc=curve_metrics.roc,
            precision_recall=(curve_metrics.precision_recall),
        )

        warning = None

        if label_resolution.unresolved_labels:
            warning = (
                f"{label_resolution.unresolved_labels} "
                "labeled images were excluded because "
                "their labels could not be resolved"
            )

        supervised_status = AvailabilityStatus(
            available=True,
            warning=warning,
        )
    else:
        no_labels_reason = (
            "Dataset has no labeled images"
            if not label_resolution.labeled_images
            else ("No dataset labels match the active model classes")
        )

        clean_accuracy = _unavailable_metric(no_labels_reason)
        robust_accuracy = _unavailable_metric(no_labels_reason)
        accuracy_drop = _unavailable_metric(no_labels_reason)
        conditional_asr = _unavailable_metric(no_labels_reason)
        corrected_by_attack_rate = _unavailable_metric(no_labels_reason)
        average_true_class_probability_drop = _unavailable_metric(no_labels_reason)

        supervised_status = AvailabilityStatus(
            available=False,
            reason=no_labels_reason,
        )

    robustness = RobustnessMetrics(
        total_images=total_images,
        changed_predictions=changed_predictions,
        unchanged_predictions=(unchanged_predictions),
        prediction_flip_rate=_round_metric(changed_predictions / total_images),
        correct_to_incorrect=(correct_to_incorrect),
        incorrect_to_correct=(incorrect_to_correct),
        clean_accuracy=clean_accuracy,
        robust_accuracy=robust_accuracy,
        accuracy_drop=accuracy_drop,
        conditional_attack_success_rate=(conditional_asr),
        corrected_by_attack_rate=(corrected_by_attack_rate),
        average_true_class_probability_drop=(average_true_class_probability_drop),
        average_prediction_margin_drop=(average_margin_drop),
        average_perturbation_linf=_round_metric(average_linf),
        average_perturbation_l2=_round_metric(average_l2),
    )

    return AdvancedClassificationMetrics(
        label_resolution=label_resolution,
        robustness=robustness,
        supervised_status=supervised_status,
        supervised=supervised,
    )
