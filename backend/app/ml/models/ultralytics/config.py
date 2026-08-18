from dataclasses import dataclass


def _normalized_required_text(value: str, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")

    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")

    return normalized


@dataclass(frozen=True, slots=True)
class UltralyticsDetectionModelConfig:
    """Declarative configuration for an Ultralytics detection model."""

    id: str
    name: str
    revision: str
    weights: str
    image_size: int = 640
    end_to_end: bool = False
    explainability_target_layer_index: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "id",
            _normalized_required_text(self.id, field_name="id"),
        )
        object.__setattr__(
            self,
            "name",
            _normalized_required_text(self.name, field_name="name"),
        )
        object.__setattr__(
            self,
            "revision",
            _normalized_required_text(self.revision, field_name="revision"),
        )
        object.__setattr__(
            self,
            "weights",
            _normalized_required_text(self.weights, field_name="weights"),
        )

        if isinstance(self.image_size, bool) or not isinstance(self.image_size, int):
            raise TypeError("image_size must be an integer")
        if self.image_size <= 0:
            raise ValueError("image_size must be greater than zero")

        if not isinstance(self.end_to_end, bool):
            raise TypeError("end_to_end must be a boolean")

        layer_index = self.explainability_target_layer_index
        if layer_index is not None and (
            isinstance(layer_index, bool) or not isinstance(layer_index, int)
        ):
            raise TypeError("explainability_target_layer_index must be an integer")
