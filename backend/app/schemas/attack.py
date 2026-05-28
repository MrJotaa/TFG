from enum import Enum

from pydantic import BaseModel, Field

from schemas.predict import PredictResponse


class AttackType(str, Enum):
    FGSM = "fgsm"


class AttackParameters(BaseModel):
    epsilon: float = Field(
        default=0.03,
        ge=0.0,
        le=0.3,
        description="Perturbation intensity for FGSM attack",
    )


class AttackRequest(BaseModel):
    dataset_id: str
    image_id: str
    attack_type: AttackType = AttackType.FGSM
    parameters: AttackParameters = AttackParameters()


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
    attack_type: AttackType
    parameters: AttackParameters
    adversarial_image: str
    original_prediction: PredictResponse
    adversarial_prediction: PredictResponse
    comparison: AttackComparison