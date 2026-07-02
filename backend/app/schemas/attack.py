from enum import Enum

from pydantic import BaseModel, Field

from schemas.predict import PredictResponse


class AttackType(str, Enum):
    FGSM = "fgsm"
    RANDOM_NOISE = "random_noise"
    PGD = "pgd"
    YOLO_CONFIDENCE = "yolo_confidence"


class AttackParameters(BaseModel):
    epsilon: float = Field(
        default=0.03,
        ge=0.0,
        le=0.3,
        description="Maximum perturbation intensity",
    )
    alpha: float = Field(
        default=0.01,
        ge=0.0,
        le=0.1,
        description="Step size for iterative attacks such as PGD",
    )
    steps: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of iterations for iterative attacks such as PGD",
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
