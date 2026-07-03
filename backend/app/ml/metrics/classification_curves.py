from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite

from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)

from ml.models.classification import (
    DetailedClassificationPrediction,
)
from schemas.classification_metrics import (
    ClassReference,
    CurveAnalysis,
    CurveComparison,
    CurvePoint,
    CurveSeries,
    MetricValue,
    PerClassCurveComparison,
)


@dataclass(frozen=True, slots=True)
class PerClassCurveMetricValues:
    """
    Valores escalares de las curvas para una clase concreta.
    """

    original_roc_auc: MetricValue
    adversarial_roc_auc: MetricValue

    original_average_precision: MetricValue
    adversarial_average_precision: MetricValue


@dataclass(frozen=True, slots=True)
class ClassificationCurveMetrics:
    """
    Resultado completo del análisis ROC y Precision-Recall.
    """

    roc: CurveAnalysis
    precision_recall: CurveAnalysis

    per_class_metrics: dict[
        int,
        PerClassCurveMetricValues,
    ]


@dataclass(frozen=True, slots=True)
class _BinaryCurveResult:
    """
    Resultado interno de una curva binaria.

    raw_score conserva la precisión completa para calcular
    agregaciones macro antes de redondear.
    """

    series: CurveSeries
    metric: MetricValue
    raw_score: float | None


def _round_score(value: float) -> float:
    return round(float(value), 4)


def _round_point(value: float) -> float:
    return round(float(value), 6)


def _available_metric(
    value: float,
    warning: str | None = None,
) -> MetricValue:
    return MetricValue(
        available=True,
        value=_round_score(value),
        warning=warning,
    )


def _unavailable_metric(
    reason: str,
    warning: str | None = None,
) -> MetricValue:
    return MetricValue(
        available=False,
        reason=reason,
        warning=warning,
    )


def _safe_threshold(
    threshold: float,
) -> float | None:
    threshold_value = float(threshold)

    if not isfinite(threshold_value):
        return None

    return _round_point(threshold_value)


def _support_warning(
    positive_count: int,
    negative_count: int,
) -> str | None:
    warnings = []

    if positive_count < 2:
        warnings.append("Curve contains fewer than 2 positive samples")

    if negative_count == 0:
        warnings.append("Curve contains no negative samples and is degenerate")
    elif negative_count < 2:
        warnings.append("Curve contains fewer than 2 negative samples")

    if not warnings:
        return None

    return ". ".join(warnings)


def _unavailable_curve(
    reason: str,
    warning: str | None = None,
) -> _BinaryCurveResult:
    return _BinaryCurveResult(
        series=CurveSeries(
            available=False,
            reason=reason,
            warning=warning,
        ),
        metric=_unavailable_metric(
            reason=reason,
            warning=warning,
        ),
        raw_score=None,
    )


def _available_curve(
    score: float,
    points: list[CurvePoint],
    warning: str | None,
) -> _BinaryCurveResult:
    rounded_score = _round_score(score)

    return _BinaryCurveResult(
        series=CurveSeries(
            available=True,
            score=rounded_score,
            warning=warning,
            points=points,
        ),
        metric=MetricValue(
            available=True,
            value=rounded_score,
            warning=warning,
        ),
        raw_score=float(score),
    )


def _calculate_roc_curve(
    binary_targets: Sequence[int],
    scores: Sequence[float],
) -> _BinaryCurveResult:
    positive_count = sum(binary_targets)
    negative_count = len(binary_targets) - positive_count

    warning = _support_warning(
        positive_count=positive_count,
        negative_count=negative_count,
    )

    if positive_count == 0:
        return _unavailable_curve(
            reason=("ROC requires at least one positive sample"),
            warning=warning,
        )

    if negative_count == 0:
        return _unavailable_curve(
            reason=("ROC requires at least one negative sample"),
            warning=warning,
        )

    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        binary_targets,
        scores,
        drop_intermediate=True,
    )

    score = auc(
        false_positive_rate,
        true_positive_rate,
    )

    points = [
        CurvePoint(
            x=_round_point(false_positive_rate[index]),
            y=_round_point(true_positive_rate[index]),
            threshold=_safe_threshold(thresholds[index]),
        )
        for index in range(len(false_positive_rate))
    ]

    return _available_curve(
        score=float(score),
        points=points,
        warning=warning,
    )


def _calculate_precision_recall_curve(
    binary_targets: Sequence[int],
    scores: Sequence[float],
) -> _BinaryCurveResult:
    positive_count = sum(binary_targets)
    negative_count = len(binary_targets) - positive_count

    warning = _support_warning(
        positive_count=positive_count,
        negative_count=negative_count,
    )

    if positive_count == 0:
        return _unavailable_curve(
            reason=("Precision-recall analysis requires at least one positive sample"),
            warning=warning,
        )

    precision, recall, thresholds = precision_recall_curve(
        binary_targets,
        scores,
        drop_intermediate=True,
    )

    score = average_precision_score(
        binary_targets,
        scores,
    )

    points = []

    for index in range(len(precision)):
        threshold = None

        if index < len(thresholds):
            threshold = _safe_threshold(thresholds[index])

        points.append(
            CurvePoint(
                x=_round_point(recall[index]),
                y=_round_point(precision[index]),
                threshold=threshold,
            )
        )

    return _available_curve(
        score=float(score),
        points=points,
        warning=warning,
    )


def _calculate_macro_metric(
    curve_results: Sequence[_BinaryCurveResult],
    metric_name: str,
) -> MetricValue:
    if not curve_results:
        return _unavailable_metric(f"Macro {metric_name} requires at least one class")

    raw_scores = [
        result.raw_score for result in curve_results if result.raw_score is not None
    ]

    if len(raw_scores) != len(curve_results):
        return _unavailable_metric(
            f"Macro {metric_name} is unavailable because "
            "at least one class metric is undefined"
        )

    warning = None

    if any(result.series.warning is not None for result in curve_results):
        warning = "At least one class has limited positive or negative support"

    return _available_metric(
        value=sum(raw_scores) / len(raw_scores),
        warning=warning,
    )


def _validate_curve_inputs(
    y_true: Sequence[int],
    original_predictions: Sequence[DetailedClassificationPrediction],
    adversarial_predictions: Sequence[DetailedClassificationPrediction],
    metric_labels: Sequence[int],
    class_labels: tuple[str, ...],
) -> None:
    sample_count = len(y_true)

    if sample_count == 0:
        raise ValueError("Curve analysis requires at least one sample")

    if len(original_predictions) != sample_count:
        raise ValueError("Original prediction count must match ground-truth count")

    if len(adversarial_predictions) != sample_count:
        raise ValueError("Adversarial prediction count must match ground-truth count")

    if not metric_labels:
        raise ValueError("Curve analysis requires at least one class")

    if len(metric_labels) != len(set(metric_labels)):
        raise ValueError("Curve metric labels must be unique")

    class_count = len(class_labels)

    for class_index in metric_labels:
        if not 0 <= class_index < class_count:
            raise ValueError("Curve class index is outside the model class catalog")

    if not set(y_true).issubset(set(metric_labels)):
        raise ValueError(
            "All ground-truth classes must be included in curve metric labels"
        )

    for prediction in (
        *original_predictions,
        *adversarial_predictions,
    ):
        if len(prediction.probabilities) != class_count:
            raise ValueError(
                "Prediction probability count does not match the model class count"
            )


def calculate_classification_curves(
    y_true: Sequence[int],
    original_predictions: Sequence[DetailedClassificationPrediction],
    adversarial_predictions: Sequence[DetailedClassificationPrediction],
    metric_labels: Sequence[int],
    class_labels: tuple[str, ...],
) -> ClassificationCurveMetrics:
    """
    Calcula curvas One-vs-Rest para las clases presentes en
    el ground truth resuelto.

    No se incluyen las clases del modelo sin muestras positivas,
    porque su ROC-AUC no sería calculable y distorsionaría las
    agregaciones macro.
    """
    _validate_curve_inputs(
        y_true=y_true,
        original_predictions=original_predictions,
        adversarial_predictions=(adversarial_predictions),
        metric_labels=metric_labels,
        class_labels=class_labels,
    )

    original_roc_results = []
    adversarial_roc_results = []

    original_pr_results = []
    adversarial_pr_results = []

    roc_per_class = []
    precision_recall_per_class = []

    per_class_metrics = {}

    for class_index in metric_labels:
        binary_targets = [int(target == class_index) for target in y_true]

        original_scores = [
            prediction.probabilities[class_index] for prediction in original_predictions
        ]

        adversarial_scores = [
            prediction.probabilities[class_index]
            for prediction in adversarial_predictions
        ]

        original_roc = _calculate_roc_curve(
            binary_targets=binary_targets,
            scores=original_scores,
        )

        adversarial_roc = _calculate_roc_curve(
            binary_targets=binary_targets,
            scores=adversarial_scores,
        )

        original_pr = _calculate_precision_recall_curve(
            binary_targets=binary_targets,
            scores=original_scores,
        )

        adversarial_pr = _calculate_precision_recall_curve(
            binary_targets=binary_targets,
            scores=adversarial_scores,
        )

        original_roc_results.append(original_roc)
        adversarial_roc_results.append(adversarial_roc)

        original_pr_results.append(original_pr)
        adversarial_pr_results.append(adversarial_pr)

        class_reference = ClassReference(
            index=class_index,
            label=class_labels[class_index],
        )

        support = sum(binary_targets)

        roc_per_class.append(
            PerClassCurveComparison(
                class_reference=class_reference,
                support=support,
                curves=CurveComparison(
                    original=original_roc.series,
                    adversarial=(adversarial_roc.series),
                ),
            )
        )

        precision_recall_per_class.append(
            PerClassCurveComparison(
                class_reference=class_reference,
                support=support,
                curves=CurveComparison(
                    original=original_pr.series,
                    adversarial=(adversarial_pr.series),
                ),
            )
        )

        per_class_metrics[class_index] = PerClassCurveMetricValues(
            original_roc_auc=(original_roc.metric),
            adversarial_roc_auc=(adversarial_roc.metric),
            original_average_precision=(original_pr.metric),
            adversarial_average_precision=(adversarial_pr.metric),
        )

    micro_targets = []
    original_micro_scores = []
    adversarial_micro_scores = []

    for sample_index, target in enumerate(y_true):
        for class_index in metric_labels:
            micro_targets.append(int(target == class_index))

            original_micro_scores.append(
                original_predictions[sample_index].probabilities[class_index]
            )

            adversarial_micro_scores.append(
                adversarial_predictions[sample_index].probabilities[class_index]
            )

    original_micro_roc = _calculate_roc_curve(
        binary_targets=micro_targets,
        scores=original_micro_scores,
    )

    adversarial_micro_roc = _calculate_roc_curve(
        binary_targets=micro_targets,
        scores=adversarial_micro_scores,
    )

    original_micro_pr = _calculate_precision_recall_curve(
        binary_targets=micro_targets,
        scores=original_micro_scores,
    )

    adversarial_micro_pr = _calculate_precision_recall_curve(
        binary_targets=micro_targets,
        scores=adversarial_micro_scores,
    )

    roc_analysis = CurveAnalysis(
        micro=CurveComparison(
            original=original_micro_roc.series,
            adversarial=(adversarial_micro_roc.series),
        ),
        macro_original=_calculate_macro_metric(
            curve_results=original_roc_results,
            metric_name="ROC-AUC",
        ),
        macro_adversarial=_calculate_macro_metric(
            curve_results=adversarial_roc_results,
            metric_name="ROC-AUC",
        ),
        per_class=roc_per_class,
    )

    precision_recall_analysis = CurveAnalysis(
        micro=CurveComparison(
            original=original_micro_pr.series,
            adversarial=(adversarial_micro_pr.series),
        ),
        macro_original=_calculate_macro_metric(
            curve_results=original_pr_results,
            metric_name="Average Precision",
        ),
        macro_adversarial=_calculate_macro_metric(
            curve_results=adversarial_pr_results,
            metric_name="Average Precision",
        ),
        per_class=precision_recall_per_class,
    )

    return ClassificationCurveMetrics(
        roc=roc_analysis,
        precision_recall=(precision_recall_analysis),
        per_class_metrics=per_class_metrics,
    )
