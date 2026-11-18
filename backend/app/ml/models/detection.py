import math
from dataclasses import dataclass
from numbers import Integral, Real


class DetectionContractError(ValueError):
    """Indica que una predicción no cumple el contrato canónico."""


def _finite_float(value: Real, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise DetectionContractError(f"{field_name} must be a real number")

    normalized = float(value)
    if not math.isfinite(normalized):
        raise DetectionContractError(f"{field_name} must be finite")

    return normalized


def _positive_dimension(value: int, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise DetectionContractError(f"{field_name} must be an integer")

    normalized = int(value)
    if normalized <= 0:
        raise DetectionContractError(f"{field_name} must be positive")

    return normalized


@dataclass(frozen=True, slots=True)
class DetectionBox:
    """
    Caja ``xyxy`` en píxeles de la imagen fuente.

    Las coordenadas son continuas y conservan toda su precisión. El extremo
    máximo puede coincidir con la dimensión correspondiente de la imagen.
    """

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        x_min = _finite_float(self.x_min, field_name="x_min")
        y_min = _finite_float(self.y_min, field_name="y_min")
        x_max = _finite_float(self.x_max, field_name="x_max")
        y_max = _finite_float(self.y_max, field_name="y_max")

        if min(x_min, y_min, x_max, y_max) < 0:
            raise DetectionContractError("box coordinates must not be negative")
        if x_min >= x_max or y_min >= y_max:
            raise DetectionContractError(
                "box must have positive width and height in xyxy order"
            )

        object.__setattr__(self, "x_min", x_min)
        object.__setattr__(self, "y_min", y_min)
        object.__setattr__(self, "x_max", x_max)
        object.__setattr__(self, "y_max", y_max)


@dataclass(frozen=True, slots=True)
class Detection:
    """Detección individual independiente del modelo o framework."""

    class_id: int
    confidence: float
    box: DetectionBox
    class_name: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.class_id, bool) or not isinstance(self.class_id, Integral):
            raise DetectionContractError("class_id must be an integer")

        class_id = int(self.class_id)
        if class_id < 0:
            raise DetectionContractError("class_id must not be negative")

        confidence = _finite_float(self.confidence, field_name="confidence")
        if not 0.0 <= confidence <= 1.0:
            raise DetectionContractError("confidence must be between 0 and 1")

        if not isinstance(self.box, DetectionBox):
            raise DetectionContractError("box must be a DetectionBox")

        if self.class_name is not None:
            if not isinstance(self.class_name, str) or not self.class_name.strip():
                raise DetectionContractError(
                    "class_name must be a non-empty string when provided"
                )

        object.__setattr__(self, "class_id", class_id)
        object.__setattr__(self, "confidence", confidence)


@dataclass(frozen=True, slots=True)
class DetectionResult:
    """
    Predicción canónica para una imagen.

    ``detections`` se copia a una tupla para impedir mutaciones compartidas.
    Todas las cajas pertenecen al espacio de la imagen fuente.
    """

    source_width: int
    source_height: int
    detections: tuple[Detection, ...] = ()

    def __post_init__(self) -> None:
        source_width = _positive_dimension(
            self.source_width,
            field_name="source_width",
        )
        source_height = _positive_dimension(
            self.source_height,
            field_name="source_height",
        )

        try:
            detections = tuple(self.detections)
        except TypeError as error:
            raise DetectionContractError("detections must be an iterable") from error

        for detection in detections:
            if not isinstance(detection, Detection):
                raise DetectionContractError(
                    "detections must contain only Detection values"
                )
            if detection.box.x_max > source_width:
                raise DetectionContractError("box exceeds source_width")
            if detection.box.y_max > source_height:
                raise DetectionContractError("box exceeds source_height")

        object.__setattr__(self, "source_width", source_width)
        object.__setattr__(self, "source_height", source_height)
        object.__setattr__(self, "detections", detections)
