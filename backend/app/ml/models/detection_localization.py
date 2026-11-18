"""Referencias inmutables en geometría fuente para ataques de localización.

La construcción de referencias está separada de la optimización: PGD debe
mantenerlas fijas durante todas sus iteraciones. Los IDs supervisados han de
estar previamente resueltos contra el catálogo de clases del modelo.
"""

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Literal

from ml.models.detection import DetectionBox, DetectionResult


@dataclass(frozen=True, slots=True)
class DetectionLocalizationReference:
    class_id: int
    box: DetectionBox

    def __post_init__(self) -> None:
        if isinstance(self.class_id, bool) or not isinstance(self.class_id, Integral):
            raise TypeError("localization class_id must be an integer")
        if self.class_id < 0:
            raise ValueError("localization class_id must not be negative")
        if not isinstance(self.box, DetectionBox):
            raise TypeError("localization box must be a DetectionBox")
        object.__setattr__(self, "class_id", int(self.class_id))


@dataclass(frozen=True, slots=True)
class DetectionLocalizationReferences:
    """Cajas xyxy y clases sobre la imagen original, nunca sobre el resize.

    ``source`` distingue anotaciones reales de pseudorreferencias del modelo
    limpio. Un conjunto vacío es válido para describir una imagen sin objetos,
    pero no tiene pérdida de localización: el objetivo debe rechazarlo.
    """

    source_width: int
    source_height: int
    objects: tuple[DetectionLocalizationReference, ...]
    source: Literal["supervised", "pseudo"]

    def __post_init__(self) -> None:
        for name in ("source_width", "source_height"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, int(value))
        if self.source not in ("supervised", "pseudo"):
            raise ValueError(
                "localization reference source must be supervised or pseudo"
            )

        objects = tuple(self.objects)
        for item in objects:
            if not isinstance(item, DetectionLocalizationReference):
                raise TypeError(
                    "localization objects must be DetectionLocalizationReference"
                )
            if (
                item.box.x_max > self.source_width
                or item.box.y_max > self.source_height
            ):
                raise ValueError("localization box exceeds source image geometry")
        object.__setattr__(self, "objects", objects)

    @classmethod
    def from_detections(
        cls,
        result: DetectionResult,
        *,
        min_confidence: float,
    ) -> "DetectionLocalizationReferences":
        """Fija pseudorreferencias a partir de detecciones de la imagen limpia."""
        if not isinstance(result, DetectionResult):
            raise TypeError("result must be a DetectionResult")
        if (
            isinstance(min_confidence, bool)
            or not isinstance(min_confidence, Real)
            or not math.isfinite(float(min_confidence))
            or not 0.0 <= min_confidence <= 1.0
        ):
            raise ValueError("min_confidence must be finite and between 0 and 1")
        return cls(
            source_width=result.source_width,
            source_height=result.source_height,
            objects=tuple(
                DetectionLocalizationReference(item.class_id, item.box)
                for item in result.detections
                if item.confidence >= min_confidence
            ),
            source="pseudo",
        )

    @classmethod
    def from_supervised_objects(
        cls,
        *,
        source_width: int,
        source_height: int,
        objects: tuple[tuple[int, DetectionBox], ...],
    ) -> "DetectionLocalizationReferences":
        """Recibe pares (ID de clase DEL MODELO, caja) ya resueltos.

        En evaluación COCO el llamador debe tomar ``model_class_id`` de los
        ground truths resueltos, nunca el ID externo de la categoría COCO.
        Deben excluirse previamente los objetos iscrowd.
        """
        return cls(
            source_width=source_width,
            source_height=source_height,
            objects=tuple(
                DetectionLocalizationReference(class_id, box)
                for class_id, box in objects
            ),
            source="supervised",
        )
