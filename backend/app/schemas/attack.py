from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from schemas.predict import PredictResponse


class AttackType(str, Enum):
    FGSM = "fgsm"
    RANDOM_NOISE = "random_noise"
    PGD = "pgd"
    CARLINI_WAGNER = "carlini_wagner"
    DEEPFOOL = "deepfool"
    HOP_SKIP_JUMP = "hop_skip_jump"
    ZOO = "zoo"
    # ID legacy serializado: se conserva para clientes y resultados existentes.
    YOLO_CONFIDENCE = "yolo_confidence"
    DETECTION_LOCALIZATION_PGD = "detection_localization_pgd"


AttackTaskType = Literal["classification", "detection"]
AttackKnowledgeType = Literal["white_box", "black_box", "baseline"]
AttackParameterValueType = Literal["float", "integer", "boolean", "string"]
AttackParameterControl = Literal["range", "number", "select", "checkbox"]


class AttackParameterChoice(BaseModel):
    value: str | int | float | bool
    label: str


class AttackParameterDefinition(BaseModel):
    key: str = Field(min_length=1)
    label: str = Field(min_length=1)
    summary_label: str = Field(min_length=1)
    description: str = Field(min_length=1)
    value_type: AttackParameterValueType
    control: AttackParameterControl
    default: str | int | float | bool
    minimum: float | int | None = None
    maximum: float | int | None = None
    step: float | int | None = None
    choices: list[AttackParameterChoice] = Field(default_factory=list)


class AttackCatalogItem(BaseModel):
    id: AttackType
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    knowledge_type: AttackKnowledgeType
    supported_task_types: list[AttackTaskType] = Field(min_length=1)
    # Ausencia/None: compatible con cualquier modelo de las tareas declaradas.
    supported_model_ids: list[str] | None = Field(default=None, min_length=1)
    parameters: list[AttackParameterDefinition]

    @field_validator("supported_model_ids")
    @classmethod
    def validate_supported_model_ids(cls, model_ids: list[str] | None):
        if model_ids is None:
            return None
        if any(
            not model_id.strip() or model_id != model_id.strip()
            for model_id in model_ids
        ):
            raise ValueError("Los identificadores de modelo no pueden estar vacíos")
        if len(model_ids) != len(set(model_ids)):
            raise ValueError("Los identificadores de modelo deben ser únicos")
        return model_ids


class AttackParameters(BaseModel):
    """
    Contenedor extensible de parámetros de ataque.

    La validación específica pertenece a cada ataque y se ejecuta al resolverlo
    desde el registro. Se mantienen aquí las restricciones y defaults de acceso
    legacy para conservar el contrato existente durante la migración. Los
    defaults legacy no se serializan si no pertenecen al ataque resuelto.
    """

    model_config = ConfigDict(extra="allow")

    _LEGACY_DEFAULTS = {
        "epsilon": 0.03,
        "alpha": 0.01,
        "steps": 10,
        "seed": 0,
    }

    def __getattr__(self, item: str):
        try:
            return super().__getattr__(item)
        except AttributeError:
            if item in self._LEGACY_DEFAULTS:
                return self._LEGACY_DEFAULTS[item]
            raise

    @model_validator(mode="after")
    def validate_legacy_parameters(self):
        values = self.__pydantic_extra__ or {}

        self._validate_numeric(values, "epsilon", minimum=0.0, maximum=0.3)
        self._validate_numeric(values, "alpha", minimum=0.0, maximum=0.1)
        self._validate_integer(values, "steps", minimum=1, maximum=100)
        self._validate_integer(
            values,
            "seed",
            minimum=0,
            maximum=2_147_483_647,
        )

        return self

    @staticmethod
    def _validate_numeric(
        values: dict[str, Any],
        key: str,
        *,
        minimum: float,
        maximum: float,
    ) -> None:
        if key not in values:
            return

        value = values[key]
        if isinstance(value, bool):
            raise ValueError(f"{key} must be numeric")

        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{key} must be numeric") from error

        if not minimum <= numeric_value <= maximum:
            raise ValueError(f"{key} must be between {minimum} and {maximum}")

        values[key] = numeric_value

    @staticmethod
    def _validate_integer(
        values: dict[str, Any],
        key: str,
        *,
        minimum: int,
        maximum: int,
    ) -> None:
        if key not in values:
            return

        value = values[key]
        if isinstance(value, bool):
            raise ValueError(f"{key} must be an integer")

        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{key} must be an integer") from error

        if not numeric_value.is_integer():
            raise ValueError(f"{key} must be an integer")

        integer_value = int(numeric_value)
        if not minimum <= integer_value <= maximum:
            raise ValueError(f"{key} must be between {minimum} and {maximum}")

        values[key] = integer_value


class AttackRequest(BaseModel):
    dataset_id: str
    image_id: str
    attack_type: AttackType = AttackType.FGSM
    parameters: AttackParameters = Field(default_factory=AttackParameters)


class AttackComparison(BaseModel):
    original_label: str
    adversarial_label: str
    original_confidence: float
    adversarial_confidence: float
    confidence_difference: float
    prediction_changed: bool
    attack_success: bool
    perturbation_linf: float
    perturbation_l2: float


class AttackResponse(BaseModel):
    result_id: str
    attack_type: AttackType
    parameters: AttackParameters

    adversarial_image_url: str
    perturbation_image_url: str
    original_gradcam_image_url: str
    adversarial_gradcam_image_url: str

    original_prediction: PredictResponse
    adversarial_prediction: PredictResponse
    comparison: AttackComparison
