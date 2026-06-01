from pydantic import BaseModel

from schemas.attack import AttackParameters, AttackType


class EvaluationRequest(BaseModel):
    dataset_id: str
    attack_type: AttackType = AttackType.FGSM
    parameters: AttackParameters = AttackParameters()


class EvaluationImageResult(BaseModel):
    image_id: str
    filename: str

    original_image_url: str | None = None
    adversarial_image_url: str | None = None
    perturbation_image_url: str | None = None

    original_label: str
    adversarial_label: str

    original_confidence: float
    adversarial_confidence: float
    confidence_difference: float

    prediction_changed: bool
    attack_success: bool

    perturbation_linf: float
    perturbation_l2: float


class EvaluationSummary(BaseModel):
    total_images: int
    changed_predictions: int
    unchanged_predictions: int
    attack_success_rate: float

    average_original_confidence: float
    average_adversarial_confidence: float
    average_confidence_difference: float

    average_perturbation_linf: float
    average_perturbation_l2: float


class EvaluationResponse(BaseModel):
    result_id: str
    dataset_id: str
    model_id: str
    attack_type: AttackType
    parameters: AttackParameters

    summary: EvaluationSummary
    results: list[EvaluationImageResult]