from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

import torch
from PIL import Image

from schemas.attack import AttackParameters

AttackParameterValueType = Literal["float", "integer", "boolean", "string"]
AttackParameterControl = Literal["range", "number", "select", "checkbox"]
AttackKnowledgeType = Literal["white_box", "black_box", "baseline"]


@dataclass(frozen=True, slots=True)
class AttackParameterChoice:
    value: str | int | float | bool
    label: str


@dataclass(frozen=True, slots=True)
class AttackParameterSpec:
    key: str
    label: str
    summary_label: str
    description: str
    value_type: AttackParameterValueType
    default: str | int | float | bool
    control: AttackParameterControl = "number"
    minimum: float | int | None = None
    maximum: float | int | None = None
    step: float | int | None = None
    choices: tuple[AttackParameterChoice, ...] = ()

    def resolve(self, raw_value: Any) -> str | int | float | bool:
        value = self._coerce(raw_value)

        if self.minimum is not None and value < self.minimum:
            raise ValueError(
                f"Parameter '{self.key}' must be greater than or equal to "
                f"{self.minimum}"
            )
        if self.maximum is not None and value > self.maximum:
            raise ValueError(
                f"Parameter '{self.key}' must be less than or equal to {self.maximum}"
            )

        if self.choices:
            allowed_values = {choice.value for choice in self.choices}
            if value not in allowed_values:
                raise ValueError(
                    f"Parameter '{self.key}' must be one of "
                    f"{sorted(allowed_values, key=str)}"
                )

        return value

    def _coerce(self, raw_value: Any) -> str | int | float | bool:
        if self.value_type == "float":
            if isinstance(raw_value, bool):
                raise ValueError(f"Parameter '{self.key}' must be numeric")
            try:
                return float(raw_value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"Parameter '{self.key}' must be numeric") from error

        if self.value_type == "integer":
            if isinstance(raw_value, bool):
                raise ValueError(f"Parameter '{self.key}' must be an integer")
            try:
                numeric_value = float(raw_value)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Parameter '{self.key}' must be an integer"
                ) from error
            if not numeric_value.is_integer():
                raise ValueError(f"Parameter '{self.key}' must be an integer")
            return int(numeric_value)

        if self.value_type == "boolean":
            if not isinstance(raw_value, bool):
                raise ValueError(f"Parameter '{self.key}' must be a boolean")
            return raw_value

        if self.value_type == "string":
            if not isinstance(raw_value, str):
                raise ValueError(f"Parameter '{self.key}' must be a string")
            return raw_value

        raise ValueError(f"Unsupported parameter type '{self.value_type}'")

    def as_catalog_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "summary_label": self.summary_label,
            "description": self.description,
            "value_type": self.value_type,
            "control": self.control,
            "default": self.default,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "step": self.step,
            "choices": [
                {"value": choice.value, "label": choice.label}
                for choice in self.choices
            ],
        }


@dataclass(frozen=True, slots=True)
class AdversarialAttackResult:
    """Resultado tensorial de un ataque en la entrada de clasificación."""

    clean_model_input: torch.Tensor
    adversarial_model_input: torch.Tensor
    model_space_delta: torch.Tensor

    adversarial_image: Image.Image
    perturbation_image: Image.Image

    perturbation_linf: float
    perturbation_l2: float


class BaseAdversarialAttack(ABC):
    """
    Clase base para ataques adversarios sobre imágenes.

    Además de la interfaz de ejecución, cada implementación declara su catálogo
    de capacidades y parámetros. Esa declaración es la fuente de verdad común
    para validación backend y representación frontend.
    """

    id: str
    name: str
    description: str
    knowledge_type: AttackKnowledgeType
    revision: str | None = None
    supported_task_types: tuple[str, ...] = ()
    parameter_specs: tuple[AttackParameterSpec, ...] = ()

    def supports_task_type(self, task_type: str) -> bool:
        return task_type in self.supported_task_types

    def resolve_parameters(self, parameters: AttackParameters) -> AttackParameters:
        supplied = parameters.model_dump(mode="python")
        spec_by_key = {spec.key: spec for spec in self.parameter_specs}
        supplied_keys = set(supplied)
        supported_keys = set(spec_by_key)

        # Compatibilidad de transición con resultados/clientes antiguos, que
        # enviaban epsilon/alpha/steps/seed a todos los ataques aunque no se usasen.
        legacy_keys = {"epsilon", "alpha", "steps", "seed"}
        unsupported_keys = supplied_keys - supported_keys - legacy_keys
        if unsupported_keys:
            joined = ", ".join(sorted(unsupported_keys))
            raise ValueError(f"Unsupported parameters for attack '{self.id}': {joined}")

        resolved = {}
        for spec in self.parameter_specs:
            raw_value = supplied.get(spec.key, spec.default)
            resolved[spec.key] = spec.resolve(raw_value)

        return AttackParameters(**resolved)

    def create_runner(self, model_wrapper, parameters, cancellation_token=None):
        """Return an execution-scoped runner. Stateless attacks reuse themselves."""
        return self

    def catalog_entry(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "knowledge_type": self.knowledge_type,
            "supported_task_types": list(self.supported_task_types),
            "parameters": [spec.as_catalog_dict() for spec in self.parameter_specs],
        }

    @abstractmethod
    def apply(
        self,
        model_wrapper,
        image: Image.Image,
        parameters,
        original_class_index: int | None = None,
    ):
        """
        Aplica el ataque adversario sobre una imagen.

        Args:
            model_wrapper: modelo activo del sistema.
            image: imagen original como objeto PIL.
            parameters: parámetros específicos del ataque ya resueltos.
            original_class_index: clase original ya calculada, cuando
                el servicio puede reutilizarla y evitar una inferencia.

        Returns:
            Resultado interno definido por el ataque concreto.
        """
        pass
