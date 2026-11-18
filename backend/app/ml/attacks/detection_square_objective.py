"""Objetivo agnóstico, no diferenciable, para Square-style Detection.

Las referencias son detecciones *fijas* de la imagen limpia, nunca etiquetas COCO.
Se optimiza la supervivencia de esas referencias mediante consultas a la
interfaz canónica de detección. No equivale a mAP, recall ni a `attack_success`.

C2 sólo define la función objetivo: no registra ni ejecuta un nuevo ataque.
"""

from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real

import numpy as np
from scipy.optimize import linear_sum_assignment

from ml.models.detection import Detection, DetectionBox, DetectionResult
from ml.models.detection_inference import LEGACY_DETECTION_CONFIDENCE_THRESHOLD

DEFAULT_REFERENCE_CONFIDENCE = LEGACY_DETECTION_CONFIDENCE_THRESHOLD
DEFAULT_MAX_REFERENCES = 100


def _intersection_over_union(left: DetectionBox, right: DetectionBox) -> float:
    """IoU en coordenadas originales xyxy; sin redondeo de píxeles."""
    overlap_width = max(0.0, min(left.x_max, right.x_max) - max(left.x_min, right.x_min))
    overlap_height = max(0.0, min(left.y_max, right.y_max) - max(left.y_min, right.y_min))
    intersection = overlap_width * overlap_height
    left_area = (left.x_max - left.x_min) * (left.y_max - left.y_min)
    right_area = (right.x_max - right.x_min) * (right.y_max - right.y_min)
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


@dataclass(frozen=True, slots=True)
class SquareDetectionObjective:
    """Supervivencia media de referencias con correspondencia máxima 1:1.

    Cada pareja de la misma clase aporta ``IoU * min(1, conf_adv/conf_ref)``.
    Un candidato adversario sólo puede asignarse a UNA referencia; se elige
    el emparejamiento de peso total máximo. Los objetos no emparejados aportan
    cero, igual que los cambios de clase o las desapariciones.

    El promedio se calcula por referencia, no por detección adversaria. Las
    apariciones de falsos positivos no son el objetivo de esta variante.
    """

    source_width: int
    source_height: int
    references: tuple[Detection, ...]

    @classmethod
    def from_clean(
        cls,
        clean: DetectionResult,
        *,
        min_confidence: float = DEFAULT_REFERENCE_CONFIDENCE,
        max_references: int = DEFAULT_MAX_REFERENCES,
    ) -> "SquareDetectionObjective | None":
        """Selecciona anclas fiables; None indica que el ataque debe omitirse.

        El ranking es estable frente al orden de salida, salvo duplicados
        verdaderamente idénticos; la selección se realiza una única vez.
        """
        if not isinstance(clean, DetectionResult):
            raise TypeError("clean must be a canonical DetectionResult")
        if (
            isinstance(min_confidence, bool)
            or not isinstance(min_confidence, Real)
            or not isfinite(float(min_confidence))
            or not 0.0 < float(min_confidence) <= 1.0
        ):
            raise ValueError("min_confidence must be finite and in (0, 1]")
        if (
            isinstance(max_references, bool)
            or not isinstance(max_references, Integral)
            or not 1 <= int(max_references) <= 300
        ):
            raise ValueError("max_references must be an integer from 1 to 300")

        ranked = sorted(
            (item for item in clean.detections if item.confidence >= min_confidence),
            key=lambda item: (
                -item.confidence,
                item.class_id,
                item.box.x_min,
                item.box.y_min,
                item.box.x_max,
                item.box.y_max,
            ),
        )
        selected = tuple(ranked[:max_references])
        if not selected:
            return None
        return cls(clean.source_width, clean.source_height, selected)

    def __post_init__(self) -> None:
        if not self.references:
            raise ValueError("Square detection requires non-empty clean references")
        if any(not isinstance(item, Detection) or item.confidence <= 0 for item in self.references):
            raise ValueError("Square detection references must have positive confidence")
        # Reutilizar la validación canónica de dimensiones y límites de cajas.
        DetectionResult(self.source_width, self.source_height, self.references)

    def score(self, prediction: DetectionResult) -> float:
        """En [0,1]. Menor puntuación = menor supervivencia de referencias.

        No realiza consultas al detector ni lee parámetros internos. La
        función es determinista y no altera las referencias ni las predicciones.
        """
        if not isinstance(prediction, DetectionResult):
            raise TypeError("prediction must be a canonical DetectionResult")
        if (prediction.source_width, prediction.source_height) != (
            self.source_width,
            self.source_height,
        ):
            raise ValueError("Prediction/reference source geometries differ")

        reference_classes = {item.class_id for item in self.references}
        candidates = tuple(
            item
            for item in prediction.detections
            if item.class_id in reference_classes and item.confidence > 0.0
        )
        if not candidates:
            return 0.0

        # Matriz rectangular: cuando faltan candidatos algunas referencias
        # quedan sin pareja. No se permite reutilizar una caja adversaria.
        weights = np.zeros((len(self.references), len(candidates)), dtype=np.float64)
        for row, reference in enumerate(self.references):
            for column, candidate in enumerate(candidates):
                if reference.class_id != candidate.class_id:
                    continue
                iou = _intersection_over_union(reference.box, candidate.box)
                if iou:
                    weights[row, column] = iou * min(
                        1.0, candidate.confidence / reference.confidence
                    )
        rows, columns = linear_sum_assignment(weights, maximize=True)
        return float(weights[rows, columns].sum() / len(self.references))

    def is_success(self, prediction: DetectionResult, *, max_retention: float = 0.8) -> bool:
        """Criterio operacional del ataque, distinto del éxito legacy o COCO."""
        if (
            isinstance(max_retention, bool)
            or not isinstance(max_retention, Real)
            or not isfinite(float(max_retention))
            or not 0.0 <= float(max_retention) < 1.0
        ):
            raise ValueError("max_retention must be finite and in [0, 1)")
        return self.score(prediction) <= float(max_retention)
