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
                f"El parámetro '{self.key}' debe ser mayor o igual que "
                f"{self.minimum}"
            )
        if self.maximum is not None and value > self.maximum:
            raise ValueError(
                f"El parámetro '{self.key}' debe ser menor o igual que {self.maximum}"
            )

        if self.choices:
            allowed_values = {choice.value for choice in self.choices}
            if value not in allowed_values:
                raise ValueError(
                    f"El parámetro '{self.key}' debe ser uno de "
                    f"{sorted(allowed_values, key=str)}"
                )

        return value

    def _coerce(self, raw_value: Any) -> str | int | float | bool:
        coercers = {
            "float": self._coerce_float,
            "integer": self._coerce_integer,
            "boolean": self._coerce_boolean,
            "string": self._coerce_string,
        }
        coercer = coercers.get(self.value_type)
        if coercer is None:
            raise ValueError(f"Tipo de parámetro no compatible: '{self.value_type}'")
        return coercer(raw_value)

    def _coerce_float(self, raw_value: Any) -> float:
        if isinstance(raw_value, bool):
            raise ValueError(f"El parámetro '{self.key}' debe ser numérico")
        try:
            return float(raw_value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"El parámetro '{self.key}' debe ser numérico"
            ) from error

    def _coerce_integer(self, raw_value: Any) -> int:
        if isinstance(raw_value, bool):
            raise ValueError(f"El parámetro '{self.key}' debe ser un entero")
        try:
            numeric_value = float(raw_value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"El parámetro '{self.key}' debe ser un entero"
            ) from error
        if not numeric_value.is_integer():
            raise ValueError(f"El parámetro '{self.key}' debe ser un entero")
        return int(numeric_value)

    def _coerce_boolean(self, raw_value: Any) -> bool:
        if not isinstance(raw_value, bool):
            raise ValueError(f"El parámetro '{self.key}' debe ser booleano")
        return raw_value

    def _coerce_string(self, raw_value: Any) -> str:
        if not isinstance(raw_value, str):
            raise ValueError(
                f"El parámetro '{self.key}' debe ser una cadena de texto"
            )
        return raw_value

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
class AdversarialEvaluationResult:
    """Resultado tensorial mínimo reutilizado por evaluaciones de datasets."""

    clean_model_input: torch.Tensor
    adversarial_model_input: torch.Tensor
    model_space_delta: torch.Tensor
    perturbation_linf: float
    perturbation_l2: float


@dataclass(frozen=True, slots=True)
class AdversarialAttackResult:
    """Resultado completo de un ataque para flujos interactivos."""

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
    # None: cualquier modelo de los tipos de tarea admitidos. Los ataques
    # específicos pueden restringir esta capacidad a IDs de modelo concretos.
    supported_model_ids: tuple[str, ...] | None = None
    parameter_specs: tuple[AttackParameterSpec, ...] = ()
    requires_exclusive_model_execution = False
    evaluation_micro_batch_size = 1

    @staticmethod
    def _perturbation_norms(perturbation: torch.Tensor) -> tuple[float, float]:
        perturbation_linf = torch.max(torch.abs(perturbation)).item()
        perturbation_l2 = torch.linalg.vector_norm(
            perturbation.reshape(-1),
            ord=2,
            dim=0,
        ).item()

        return perturbation_linf, perturbation_l2

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
            raise ValueError(
                f"Parámetros no compatibles con el ataque '{self.id}': {joined}"
            )

        resolved = {}
        for spec in self.parameter_specs:
            raw_value = supplied.get(spec.key, spec.default)
            resolved[spec.key] = spec.resolve(raw_value)

        return AttackParameters(**resolved)

    def create_runner(self, model_wrapper, parameters, cancellation_token=None):
        """Return an execution-scoped runner. Stateless attacks reuse themselves."""
        return self

    def apply_for_evaluation(
        self,
        model_wrapper,
        image: Image.Image,
        prepared_input: torch.Tensor,
        parameters,
        original_class_index: int | None = None,
    ):
        """
        Optional optimized execution path for dataset evaluations.

        The default keeps third-party and legacy attacks compatible by delegating
        to ``apply``. Attacks that operate in model space may override this method
        to reuse ``prepared_input`` and skip presentation-only work.
        """
        return self.apply(
            model_wrapper=model_wrapper,
            image=image,
            parameters=parameters,
            original_class_index=original_class_index,
        )

    def apply_many_for_evaluation(
        self,
        model_wrapper,
        images,
        prepared_inputs,
        parameters,
        original_class_indices,
    ):
        """Optional micro-batch path for dataset evaluations.

        The default preserves compatibility and semantics by delegating each
        sample to ``apply_for_evaluation`` independently. Concrete attacks may
        override this when their mathematics are batch-separable.
        """
        if not (
            len(images)
            == len(prepared_inputs)
            == len(parameters)
            == len(original_class_indices)
        ):
            raise ValueError(
                "Evaluation attack batch inputs must have matching lengths"
            )

        return tuple(
            self.apply_for_evaluation(
                model_wrapper=model_wrapper,
                image=image,
                prepared_input=prepared_input,
                parameters=sample_parameters,
                original_class_index=original_class_index,
            )
            for image, prepared_input, sample_parameters, original_class_index in zip(
                images,
                prepared_inputs,
                parameters,
                original_class_indices,
            )
        )

    def catalog_entry(self) -> dict[str, Any]:
        entry = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "knowledge_type": self.knowledge_type,
            "supported_task_types": list(self.supported_task_types),
            "parameters": [spec.as_catalog_dict() for spec in self.parameter_specs],
        }
        if self.supported_model_ids is not None:
            entry["supported_model_ids"] = list(self.supported_model_ids)
        return entry

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
