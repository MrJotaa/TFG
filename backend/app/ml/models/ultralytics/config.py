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
    weights: str
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
            "weights",
            _normalized_required_text(self.weights, field_name="weights"),
        )

        layer_index = self.explainability_target_layer_index
        if layer_index is not None and (
            isinstance(layer_index, bool) or not isinstance(layer_index, int)
        ):
            raise TypeError("explainability_target_layer_index must be an integer")
