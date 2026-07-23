from typing import Literal

from pydantic import BaseModel


class DetectionMetricsContract(BaseModel):
    """
    Metadatos del contrato básico de métricas de detección.

    Los deltas de confianza y número de detecciones siguen la convención
    ``original - adversarial``. El campo público legacy
    ``average_confidence_difference`` corresponde al delta de confianza:
    positivo significa que baja la confianza adversaria, cero que no cambia y
    negativo que sube. Para ``detections_difference``, positivo significa que
    hay menos detecciones adversarias, cero la misma cantidad y negativo más
    detecciones adversarias. Este último signo no implica por sí mismo una
    mejora o degradación de calidad porque todavía no existe ground truth.

    Para futuras métricas, las magnitudes donde un valor mayor es mejor pueden
    expresar degradación como ``original - adversarial``. Las métricas de error,
    donde un valor mayor es peor, deben declarar su dirección o usar nombres no
    ambiguos.
    """

    metrics_schema_version: str = "1.0"
    delta_convention: Literal["original_minus_adversarial"] = (
        "original_minus_adversarial"
    )
