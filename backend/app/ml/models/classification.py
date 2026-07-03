from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ClassificationClassScore:
    """
    Puntuación de una clase concreta del modelo.
    """

    index: int
    label: str
    probability: float


@dataclass(frozen=True, slots=True)
class DetailedClassificationPrediction:
    """
    Resultado interno completo de una inferencia de clasificación.

    probabilities conserva el vector completo sin redondear.
    Este objeto no forma parte directamente del contrato HTTP.
    """

    class_index: int
    class_label: str
    confidence: float

    probabilities: tuple[float, ...]
    top_k: tuple[ClassificationClassScore, ...]

    prediction_margin: float | None
