from typing import Literal

from pydantic import BaseModel, Field, model_validator


class AvailabilityStatus(BaseModel):
    """
    Indica si una sección completa de métricas puede calcularse.
    """

    available: bool
    reason: str | None = None
    warning: str | None = None

    @model_validator(mode="after")
    def validate_status(self):
        if not self.available and not self.reason:
            raise ValueError("Unavailable data must include a reason")

        return self


class MetricValue(BaseModel):
    """
    Representa una métrica que puede no estar matemáticamente
    disponible.

    Ejemplo:
    - ROC-AUC no disponible porque no existen muestras negativas.
    - Conditional ASR no disponible porque ninguna predicción
      original era correcta.
    """

    available: bool
    value: float | None = None
    reason: str | None = None
    warning: str | None = None

    @model_validator(mode="after")
    def validate_metric(self):
        if self.available and self.value is None:
            raise ValueError("Available metrics must include a value")

        if not self.available and not self.reason:
            raise ValueError("Unavailable metrics must include a reason")

        return self


class ClassReference(BaseModel):
    """
    Referencia canónica a una clase del modelo.
    """

    index: int = Field(ge=0)
    label: str = Field(min_length=1)


class UnresolvedLabel(BaseModel):
    """
    Etiqueta introducida por el usuario que no pudo asociarse
    a una clase canónica del modelo.
    """

    image_id: str
    filename: str
    provided_label: str
    reason: str

    candidate_classes: list[ClassReference] = Field(default_factory=list)


class LabelResolutionSummary(BaseModel):
    """
    Resultado de resolver las etiquetas del dataset contra el
    catálogo de clases del modelo.
    """

    labeled_images: int = Field(ge=0)
    resolved_labels: int = Field(ge=0)
    unresolved_labels: int = Field(ge=0)
    coverage: float = Field(ge=0.0, le=1.0)

    unresolved: list[UnresolvedLabel] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_counts(self):
        resolved_total = self.resolved_labels + self.unresolved_labels

        if resolved_total != self.labeled_images:
            raise ValueError("Resolved and unresolved labels must match labeled images")

        if len(self.unresolved) != self.unresolved_labels:
            raise ValueError(
                "Unresolved label details must match unresolved label count"
            )

        expected_coverage = (
            self.resolved_labels / self.labeled_images if self.labeled_images else 0.0
        )

        if abs(self.coverage - expected_coverage) > 1e-4:
            raise ValueError("Label coverage does not match resolved labels")

        return self


class RobustnessMetrics(BaseModel):
    """
    Métricas principales del impacto adversario.

    Las métricas que necesitan ground truth utilizan MetricValue
    para poder indicar explícitamente cuándo no están disponibles.
    """

    total_images: int = Field(ge=0)

    changed_predictions: int = Field(ge=0)
    unchanged_predictions: int = Field(ge=0)
    prediction_flip_rate: float = Field(
        ge=0.0,
        le=1.0,
    )

    correct_to_incorrect: int = Field(ge=0)
    incorrect_to_correct: int = Field(ge=0)

    clean_accuracy: MetricValue
    robust_accuracy: MetricValue
    accuracy_drop: MetricValue

    conditional_attack_success_rate: MetricValue
    corrected_by_attack_rate: MetricValue

    average_true_class_probability_drop: MetricValue
    average_prediction_margin_drop: MetricValue

    average_perturbation_linf: float = Field(ge=0.0)
    average_perturbation_l2: float = Field(ge=0.0)

    @model_validator(mode="after")
    def validate_prediction_counts(self):
        predictions_total = self.changed_predictions + self.unchanged_predictions

        if predictions_total != self.total_images:
            raise ValueError(
                "Changed and unchanged predictions must match total images"
            )

        expected_flip_rate = (
            self.changed_predictions / self.total_images if self.total_images else 0.0
        )

        if abs(self.prediction_flip_rate - expected_flip_rate) > 1e-4:
            raise ValueError("Prediction flip rate does not match prediction counts")

        return self


class AggregateMetricSnapshot(BaseModel):
    """
    Métricas agregadas para imágenes originales o adversarias.
    """

    accuracy: float = Field(ge=0.0, le=1.0)

    precision_macro: float = Field(ge=0.0, le=1.0)
    recall_macro: float = Field(ge=0.0, le=1.0)
    f1_macro: float = Field(ge=0.0, le=1.0)

    precision_weighted: float = Field(
        ge=0.0,
        le=1.0,
    )
    recall_weighted: float = Field(
        ge=0.0,
        le=1.0,
    )
    f1_weighted: float = Field(
        ge=0.0,
        le=1.0,
    )

    top_5_accuracy: float = Field(
        ge=0.0,
        le=1.0,
    )


class AggregateMetricDelta(BaseModel):
    """
    Variación original menos adversarial.

    Un valor positivo representa una degradación producida por
    el ataque. Un valor negativo representa una mejora.
    """

    accuracy: float

    precision_macro: float
    recall_macro: float
    f1_macro: float

    precision_weighted: float
    recall_weighted: float
    f1_weighted: float

    top_5_accuracy: float


class AggregateMetricsComparison(BaseModel):
    original: AggregateMetricSnapshot
    adversarial: AggregateMetricSnapshot
    delta: AggregateMetricDelta


class PerClassMetricSnapshot(BaseModel):
    """
    Métricas para una clase concreta.

    Cada valor puede ser no calculable, por ejemplo si no existen
    predicciones positivas o si ROC no dispone de positivos y
    negativos.
    """

    precision: MetricValue
    recall: MetricValue
    f1: MetricValue
    roc_auc: MetricValue
    average_precision: MetricValue


class PerClassMetricsComparison(BaseModel):
    class_reference: ClassReference
    support: int = Field(ge=0)

    original: PerClassMetricSnapshot
    adversarial: PerClassMetricSnapshot
    delta: PerClassMetricSnapshot


class ConfusionMatrices(BaseModel):
    """
    Matrices de confusión.

    Las filas representan las clases reales.
    Las columnas representan las clases predichas.
    """

    labels: list[ClassReference] = Field(min_length=1)

    original: list[list[int]]
    adversarial: list[list[int]]

    original_normalized: list[list[float]]
    adversarial_normalized: list[list[float]]

    @model_validator(mode="after")
    def validate_dimensions(self):
        size = len(self.labels)

        matrix_names = (
            "original",
            "adversarial",
            "original_normalized",
            "adversarial_normalized",
        )

        for matrix_name in matrix_names:
            matrix = getattr(self, matrix_name)

            is_square = len(matrix) == size and all(len(row) == size for row in matrix)

            if not is_square:
                raise ValueError(
                    f"Confusion matrix '{matrix_name}' "
                    "must be square and match the label count"
                )

        return self


class CurvePoint(BaseModel):
    """
    Punto genérico de una curva.

    Para ROC:
    - x = false positive rate
    - y = true positive rate

    Para Precision-Recall:
    - x = recall
    - y = precision
    """

    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    threshold: float | None = None


class CurveSeries(BaseModel):
    """
    Una curva original o adversaria.

    score representa:
    - ROC-AUC para curvas ROC.
    - Average Precision para curvas Precision-Recall.
    """

    available: bool
    score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    reason: str | None = None
    warning: str | None = None

    points: list[CurvePoint] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_curve(self):
        if self.available:
            if self.score is None:
                raise ValueError("Available curves must include a score")

            if not self.points:
                raise ValueError("Available curves must include points")

        elif not self.reason:
            raise ValueError("Unavailable curves must include a reason")

        return self


class CurveComparison(BaseModel):
    original: CurveSeries
    adversarial: CurveSeries


class PerClassCurveComparison(BaseModel):
    class_reference: ClassReference
    support: int = Field(ge=0)
    curves: CurveComparison


class CurveAnalysis(BaseModel):
    """
    Análisis de curvas multiclase One-vs-Rest.
    """

    micro: CurveComparison

    macro_original: MetricValue
    macro_adversarial: MetricValue

    per_class: list[PerClassCurveComparison] = Field(default_factory=list)


class SupervisedClassificationAnalysis(BaseModel):
    """
    Métricas que requieren etiquetas reales resueltas.
    """

    aggregate: AggregateMetricsComparison

    per_class: list[PerClassMetricsComparison]

    confusion_matrices: ConfusionMatrices

    roc: CurveAnalysis
    precision_recall: CurveAnalysis


class AdvancedClassificationMetrics(BaseModel):
    """
    Contrato principal de las métricas avanzadas.

    Los deltas se calculan como:

        original - adversarial

    Por tanto, un delta positivo representa degradación.
    """

    metrics_schema_version: str = "1.0"

    delta_convention: Literal["original_minus_adversarial"] = (
        "original_minus_adversarial"
    )

    label_resolution: LabelResolutionSummary
    robustness: RobustnessMetrics

    supervised_status: AvailabilityStatus

    supervised: SupervisedClassificationAnalysis | None = None

    @model_validator(mode="after")
    def validate_supervised_analysis(self):
        if self.supervised_status.available and self.supervised is None:
            raise ValueError(
                "Available supervised metrics must include supervised analysis"
            )

        if not self.supervised_status.available and self.supervised is not None:
            raise ValueError(
                "Unavailable supervised metrics cannot include supervised analysis"
            )

        return self


EvaluationTransition = Literal[
    "correct_to_correct",
    "correct_to_incorrect",
    "incorrect_to_correct",
    "incorrect_to_incorrect",
]


class EvaluationImageDiagnostics(BaseModel):
    """
    Datos avanzados por imagen utilizados por la pestaña
    de resultados y los filtros visuales.
    """

    ground_truth_class: ClassReference | None = None
    label_resolution_reason: str | None = None

    original_class_index: int | None = Field(
        default=None,
        ge=0,
    )
    adversarial_class_index: int | None = Field(
        default=None,
        ge=0,
    )

    original_true_class_probability: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    adversarial_true_class_probability: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    true_class_probability_drop: float | None = None

    original_prediction_margin: float | None = None
    adversarial_prediction_margin: float | None = None
    prediction_margin_drop: float | None = None

    original_top_5_correct: bool | None = None
    adversarial_top_5_correct: bool | None = None

    transition: EvaluationTransition | None = None
