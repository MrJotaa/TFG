from dataclasses import dataclass
from numbers import Real

from ml.models.detection_inference import LEGACY_DETECTION_CONFIDENCE_THRESHOLD


def _normalized_required_text(value: str, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")

    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")

    return normalized


@dataclass(frozen=True, slots=True)
class RfDetrDetectionModelConfig:
    """Declarative configuration for an RF-DETR checkpoint from Hugging Face."""

    id: str
    name: str
    revision: str
    checkpoint: str
    checkpoint_revision: str
    confidence_threshold: float = LEGACY_DETECTION_CONFIDENCE_THRESHOLD

    def __post_init__(self) -> None:
        for field_name in (
            "id",
            "name",
            "revision",
            "checkpoint",
            "checkpoint_revision",
        ):
            object.__setattr__(
                self,
                field_name,
                _normalized_required_text(
                    getattr(self, field_name),
                    field_name=field_name,
                ),
            )

        if isinstance(self.confidence_threshold, bool) or not isinstance(
            self.confidence_threshold,
            Real,
        ):
            raise TypeError("confidence_threshold must be a real number")

        confidence_threshold = float(self.confidence_threshold)
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")

        object.__setattr__(
            self,
            "confidence_threshold",
            confidence_threshold,
        )
