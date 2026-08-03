import math
from dataclasses import dataclass, field
from numbers import Real

LEGACY_DETECTION_CONFIDENCE_THRESHOLD = 0.25
DETECTION_EVALUATION_NMS_IOU_THRESHOLD = 0.7
DETECTION_EVALUATION_MAX_DETECTIONS_PER_IMAGE = 300


class DetectionInferenceConfigurationError(ValueError):
    """Indica que la configuración científica de inferencia es inválida."""


def _probability(value: Real, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise DetectionInferenceConfigurationError(
            f"{field_name} must be a real number"
        )
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise DetectionInferenceConfigurationError(
            f"{field_name} must be finite and between 0 and 1"
        )
    return normalized


@dataclass(frozen=True, slots=True)
class DetectionEvaluationInferenceConfiguration:
    """Configuración explícita usada para obtener el ranking de detecciones."""

    score_floor: float = 0.001
    nms_iou_threshold: float = field(
        default=DETECTION_EVALUATION_NMS_IOU_THRESHOLD,
        init=False,
    )
    max_detections_per_image: int = field(
        default=DETECTION_EVALUATION_MAX_DETECTIONS_PER_IMAGE,
        init=False,
    )

    def __post_init__(self) -> None:
        score_floor = _probability(self.score_floor, field_name="score_floor")
        if score_floor > LEGACY_DETECTION_CONFIDENCE_THRESHOLD:
            raise DetectionInferenceConfigurationError(
                "score_floor must not exceed the legacy detection threshold"
            )

        object.__setattr__(self, "score_floor", score_floor)
