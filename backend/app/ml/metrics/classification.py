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


@dataclass(frozen=True, slots=True)
class _SupervisedMetricResult:
    clean_accuracy: MetricValue
    robust_accuracy: MetricValue
    accuracy_drop: MetricValue
    conditional_asr: MetricValue
    corrected_by_attack_rate: MetricValue
    average_true_class_probability_drop: MetricValue
    status: AvailabilityStatus
    analysis: SupervisedClassificationAnalysis | None


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
            "La diferencia no está disponible porque la métrica "
            "no está definida para las predicciones originales o tras el ataque"
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
        warning = "El soporte de la clase es inferior a 2 muestras"

    if predicted_count == 0:
        precision_metric = MetricValue(
            available=False,
            reason=(
                "La precisión no está definida porque ninguna muestra fue "
                "predicha como esta clase"
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


def _ratio_metric_or_unavailable(
    numerator: int,
    denominator: int,
    *,
    unavailable_reason: str,
) -> MetricValue:
    if denominator:
        return _available_metric(numerator / denominator)
    return _unavailable_metric(unavailable_reason)


def _unavailable_supervised_metrics(
    label_resolution: LabelResolutionSummary,
) -> _SupervisedMetricResult:
    reason = (
        "El conjunto de datos no contiene imágenes etiquetadas"
        if not label_resolution.labeled_images
        else (
            "Ninguna etiqueta del conjunto de datos coincide con las "
            "clases del modelo activo"
        )
    )
    unavailable = _unavailable_metric(reason)
    return _SupervisedMetricResult(
        clean_accuracy=unavailable,
        robust_accuracy=unavailable,
        accuracy_drop=unavailable,
        conditional_asr=unavailable,
        corrected_by_attack_rate=unavailable,
        average_true_class_probability_drop=unavailable,
        status=AvailabilityStatus(available=False, reason=reason),
        analysis=None,
    )


def _build_supervised_analysis(
    resolved_samples: Sequence[ClassificationEvaluationSample],
    class_labels: tuple[str, ...],
) -> SupervisedClassificationAnalysis:
    y_true = [sample.ground_truth.index for sample in resolved_samples]
    original_detailed_predictions = [sample.original for sample in resolved_samples]
    adversarial_detailed_predictions = [sample.adversarial for sample in resolved_samples]
    original_predictions = [sample.original.class_index for sample in resolved_samples]
    adversarial_predictions = [
        sample.adversarial.class_index for sample in resolved_samples
    ]
    original_top_5 = [
        any(score.index == sample.ground_truth.index for score in sample.original.top_k)
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
        original_predictions=original_detailed_predictions,
        adversarial_predictions=adversarial_detailed_predictions,
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
        adversarial=adversarial_aggregate,
        delta=_calculate_aggregate_delta(
            original_aggregate,
            adversarial_aggregate,
        ),
    )
    return SupervisedClassificationAnalysis(
        aggregate=aggregate_comparison,
        per_class=_calculate_per_class_metrics(
            y_true=y_true,
            original_predictions=original_predictions,
            adversarial_predictions=adversarial_predictions,
            metric_labels=metric_labels,
            class_labels=class_labels,
            curve_metrics_by_class=curve_metrics.per_class_metrics,
        ),
        confusion_matrices=_calculate_confusion_matrices(
            y_true=y_true,
            original_predictions=original_predictions,
            adversarial_predictions=adversarial_predictions,
            class_labels=class_labels,
        ),
        roc=curve_metrics.roc,
        precision_recall=curve_metrics.precision_recall,
    )


def _available_supervised_metrics(
    resolved_samples: Sequence[ClassificationEvaluationSample],
    label_resolution: LabelResolutionSummary,
    class_labels: tuple[str, ...],
    *,
    correct_to_incorrect: int,
    incorrect_to_correct: int,
) -> _SupervisedMetricResult:
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
    true_class_probability_drops = [
        (
            sample.original.probabilities[sample.ground_truth.index]
            - sample.adversarial.probabilities[sample.ground_truth.index]
        )
        for sample in resolved_samples
    ]
    warning = (
        "Imágenes etiquetadas excluidas por etiquetas no resueltas: "
        f"{label_resolution.unresolved_labels}"
        if label_resolution.unresolved_labels
        else None
    )
    return _SupervisedMetricResult(
        clean_accuracy=_available_metric(clean_accuracy_value),
        robust_accuracy=_available_metric(robust_accuracy_value),
        accuracy_drop=_available_metric(
            clean_accuracy_value - robust_accuracy_value
        ),
        conditional_asr=_ratio_metric_or_unavailable(
            correct_to_incorrect,
            correct_original,
            unavailable_reason=(
                "La tasa de éxito condicional del ataque requiere "
                "al menos una predicción original correcta"
            ),
        ),
        corrected_by_attack_rate=_ratio_metric_or_unavailable(
            incorrect_to_correct,
            incorrect_original,
            unavailable_reason=(
                "La tasa de corrección tras el ataque requiere "
                "al menos una predicción original incorrecta"
            ),
        ),
        average_true_class_probability_drop=_available_metric(
            _average(true_class_probability_drops)
        ),
        status=AvailabilityStatus(available=True, warning=warning),
        analysis=_build_supervised_analysis(resolved_samples, class_labels),
    )


def _supervised_metrics(
    resolved_samples: Sequence[ClassificationEvaluationSample],
    label_resolution: LabelResolutionSummary,
    class_labels: tuple[str, ...],
    *,
    correct_to_incorrect: int,
    incorrect_to_correct: int,
) -> _SupervisedMetricResult:
    if not resolved_samples:
        return _unavailable_supervised_metrics(label_resolution)
    return _available_supervised_metrics(
        resolved_samples,
        label_resolution,
        class_labels,
        correct_to_incorrect=correct_to_incorrect,
        incorrect_to_correct=incorrect_to_correct,
    )


def _transition_counts(
    resolved_samples: Sequence[ClassificationEvaluationSample],
) -> tuple[int, int]:
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
    return correct_to_incorrect, incorrect_to_correct


def _average_margin_drop(
    samples: Sequence[ClassificationEvaluationSample],
) -> MetricValue:
    margin_drops = [
        sample.original.prediction_margin - sample.adversarial.prediction_margin
        for sample in samples
        if sample.original.prediction_margin is not None
        and sample.adversarial.prediction_margin is not None
    ]
    if margin_drops:
        return _available_metric(_average(margin_drops))
    return _unavailable_metric(
        "El margen de predicción requiere al menos dos clases del modelo"
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

    correct_to_incorrect, incorrect_to_correct = _transition_counts(
        resolved_samples
    )
    average_linf = _average([sample.perturbation_linf for sample in samples])
    average_l2 = _average([sample.perturbation_l2 for sample in samples])
    average_margin_drop = _average_margin_drop(samples)

    supervised_metrics = _supervised_metrics(
        resolved_samples,
        label_resolution,
        class_labels,
        correct_to_incorrect=correct_to_incorrect,
        incorrect_to_correct=incorrect_to_correct,
    )

    robustness = RobustnessMetrics(
        total_images=total_images,
        changed_predictions=changed_predictions,
        unchanged_predictions=(unchanged_predictions),
        prediction_flip_rate=_round_metric(changed_predictions / total_images),
        correct_to_incorrect=(correct_to_incorrect),
        incorrect_to_correct=(incorrect_to_correct),
        clean_accuracy=supervised_metrics.clean_accuracy,
        robust_accuracy=supervised_metrics.robust_accuracy,
        accuracy_drop=supervised_metrics.accuracy_drop,
        conditional_attack_success_rate=supervised_metrics.conditional_asr,
        corrected_by_attack_rate=supervised_metrics.corrected_by_attack_rate,
        average_true_class_probability_drop=(
            supervised_metrics.average_true_class_probability_drop
        ),
        average_prediction_margin_drop=(average_margin_drop),
        average_perturbation_linf=_round_metric(average_linf),
        average_perturbation_l2=_round_metric(average_l2),
    )

    return AdvancedClassificationMetrics(
        label_resolution=label_resolution,
        robustness=robustness,
        supervised_status=supervised_metrics.status,
        supervised=supervised_metrics.analysis,
    )
