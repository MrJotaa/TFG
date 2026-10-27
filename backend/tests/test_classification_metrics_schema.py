import pytest
from pydantic import ValidationError

from schemas.classification_metrics import (
    AdvancedClassificationMetrics,
    AggregateMetricDelta,
    AggregateMetricsComparison,
    AggregateMetricSnapshot,
    AvailabilityStatus,
    ClassReference,
    ConfusionMatrices,
    CurveAnalysis,
    CurveComparison,
    CurvePoint,
    CurveSeries,
    LabelResolutionSummary,
    MetricValue,
    RobustnessMetrics,
    SupervisedClassificationAnalysis,
)
from schemas.evaluation import EvaluationResponse


def _available_metric(value: float) -> MetricValue:
    return MetricValue(
        available=True,
        value=value,
    )


def _available_curve(score: float) -> CurveSeries:
    return CurveSeries(
        available=True,
        score=score,
        points=[
            CurvePoint(
                x=0.0,
                y=0.0,
                threshold=None,
            ),
            CurvePoint(
                x=1.0,
                y=1.0,
                threshold=0.0,
            ),
        ],
    )


def _aggregate_snapshot(
    value: float,
) -> AggregateMetricSnapshot:
    return AggregateMetricSnapshot(
        accuracy=value,
        precision_macro=value,
        recall_macro=value,
        f1_macro=value,
        precision_weighted=value,
        recall_weighted=value,
        f1_weighted=value,
        top_5_accuracy=value,
    )


def _aggregate_comparison():
    return AggregateMetricsComparison(
        original=_aggregate_snapshot(0.8),
        adversarial=_aggregate_snapshot(0.5),
        delta=AggregateMetricDelta(
            accuracy=0.3,
            precision_macro=0.3,
            recall_macro=0.3,
            f1_macro=0.3,
            precision_weighted=0.3,
            recall_weighted=0.3,
            f1_weighted=0.3,
            top_5_accuracy=0.3,
        ),
    )


def _curve_analysis() -> CurveAnalysis:
    comparison = CurveComparison(
        original=_available_curve(0.9),
        adversarial=_available_curve(0.7),
    )

    return CurveAnalysis(
        micro=comparison,
        macro_original=_available_metric(0.9),
        macro_adversarial=_available_metric(0.7),
    )


def _supervised_analysis():
    class_reference = ClassReference(
        index=0,
        label="class-a",
    )

    return SupervisedClassificationAnalysis(
        aggregate=_aggregate_comparison(),
        per_class=[],
        confusion_matrices=ConfusionMatrices(
            labels=[class_reference],
            original=[[2]],
            adversarial=[[1]],
            original_normalized=[[1.0]],
            adversarial_normalized=[[0.5]],
        ),
        roc=_curve_analysis(),
        precision_recall=_curve_analysis(),
    )


def _robustness_metrics():
    return RobustnessMetrics(
        total_images=2,
        changed_predictions=1,
        unchanged_predictions=1,
        prediction_flip_rate=0.5,
        correct_to_incorrect=1,
        incorrect_to_correct=0,
        clean_accuracy=_available_metric(0.8),
        robust_accuracy=_available_metric(0.5),
        accuracy_drop=_available_metric(0.3),
        conditional_attack_success_rate=(_available_metric(0.5)),
        corrected_by_attack_rate=(_available_metric(0.0)),
        average_true_class_probability_drop=(_available_metric(0.2)),
        average_prediction_margin_drop=(_available_metric(0.15)),
        average_perturbation_linf=0.03,
        average_perturbation_l2=1.2,
    )


def test_legacy_evaluation_response_remains_valid():
    response = EvaluationResponse(
        result_id="evaluation-result",
        dataset_id="dataset-1",
        model_id="resnet18",
        attack_type="fgsm",
        parameters={
            "epsilon": 0.03,
            "alpha": 0.01,
            "steps": 10,
        },
        summary={
            "total_images": 1,
            "changed_predictions": 0,
            "unchanged_predictions": 1,
            "attack_success_rate": 0.0,
            "average_original_confidence": 0.9,
            "average_adversarial_confidence": 0.8,
            "average_confidence_difference": -0.1,
            "average_perturbation_linf": 0.03,
            "average_perturbation_l2": 1.0,
        },
        results=[
            {
                "image_id": "image-1",
                "filename": "image.png",
                "original_label": "tabby cat",
                "adversarial_label": "tabby cat",
                "original_confidence": 0.9,
                "adversarial_confidence": 0.8,
                "confidence_difference": -0.1,
                "prediction_changed": False,
                "attack_success": False,
                "perturbation_linf": 0.03,
                "perturbation_l2": 1.0,
            }
        ],
    )

    assert response.model_name is None
    assert response.advanced_metrics is None
    assert response.results[0].diagnostics is None


def test_advanced_metrics_contains_expected_sections():
    advanced_metrics = AdvancedClassificationMetrics(
        label_resolution=LabelResolutionSummary(
            labeled_images=2,
            resolved_labels=2,
            unresolved_labels=0,
            coverage=1.0,
        ),
        robustness=_robustness_metrics(),
        supervised_status=AvailabilityStatus(available=True),
        supervised=_supervised_analysis(),
    )

    data = advanced_metrics.model_dump()

    assert data["metrics_schema_version"] == "1.0"
    assert data["delta_convention"] == "original_minus_adversarial"

    assert data["robustness"]["prediction_flip_rate"] == 0.5

    assert data["supervised"]["aggregate"]["delta"]["accuracy"] == 0.3

    assert data["supervised"]["confusion_matrices"]["labels"] == [
        {
            "index": 0,
            "label": "class-a",
        }
    ]

    assert "roc" in data["supervised"]
    assert "precision_recall" in data["supervised"]


def test_unavailable_metric_requires_reason():
    with pytest.raises(
        ValidationError,
        match="must include a reason",
    ):
        MetricValue(available=False)


def test_available_metric_requires_value():
    with pytest.raises(
        ValidationError,
        match="must include a value",
    ):
        MetricValue(available=True)


def test_label_resolution_counts_must_be_consistent():
    with pytest.raises(
        ValidationError,
        match="must match labeled images",
    ):
        LabelResolutionSummary(
            labeled_images=3,
            resolved_labels=1,
            unresolved_labels=1,
            coverage=0.5,
        )


def test_confusion_matrices_must_match_labels():
    _prepared_labels = [
        ClassReference(
            index=0,
            label="class-a",
        ),
        ClassReference(
            index=1,
            label="class-b",
        ),
    ]

    with pytest.raises(
        ValidationError,
        match="must be square",
    ):
        ConfusionMatrices(
            labels=_prepared_labels,
            original=[[1]],
            adversarial=[
                [1, 0],
                [0, 1],
            ],
            original_normalized=[
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            adversarial_normalized=[
                [1.0, 0.0],
                [0.0, 1.0],
            ],
        )


def test_prediction_flip_rate_must_match_counts():
    _prepared_clean_accuracy = _available_metric(1.0)
    _prepared_robust_accuracy = _available_metric(0.5)
    _prepared_accuracy_drop = _available_metric(0.5)
    _prepared_conditional_attack_success_rate = _available_metric(0.5)
    _prepared_corrected_by_attack_rate = _available_metric(0.0)
    _prepared_average_true_class_probability_drop = _available_metric(0.2)
    _prepared_average_prediction_margin_drop = _available_metric(0.1)

    with pytest.raises(
        ValidationError,
        match="flip rate does not match",
    ):
        RobustnessMetrics(
            total_images=2,
            changed_predictions=1,
            unchanged_predictions=1,
            prediction_flip_rate=0.9,
            correct_to_incorrect=1,
            incorrect_to_correct=0,
            clean_accuracy=_prepared_clean_accuracy,
            robust_accuracy=_prepared_robust_accuracy,
            accuracy_drop=_prepared_accuracy_drop,
            conditional_attack_success_rate=(_prepared_conditional_attack_success_rate),
            corrected_by_attack_rate=(_prepared_corrected_by_attack_rate),
            average_true_class_probability_drop=(
                _prepared_average_true_class_probability_drop
            ),
            average_prediction_margin_drop=(_prepared_average_prediction_margin_drop),
            average_perturbation_linf=0.03,
            average_perturbation_l2=1.0,
        )


def test_unavailable_supervised_metrics_require_reason():
    with pytest.raises(
        ValidationError,
        match="must include a reason",
    ):
        AvailabilityStatus(available=False)
