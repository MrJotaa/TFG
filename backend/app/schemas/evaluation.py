from pydantic import BaseModel, Field

from schemas.attack import AttackParameters, AttackType
from schemas.classification_metrics import (
    AdvancedClassificationMetrics,
    EvaluationImageDiagnostics,
)


class EvaluationRequest(BaseModel):
    dataset_id: str
    attack_type: AttackType = AttackType.FGSM
    parameters: AttackParameters = Field(default_factory=AttackParameters)
    force_recompute: bool = Field(default=False, exclude=True)


class EvaluationImageResult(BaseModel):
    image_id: str
    filename: str

    original_image_url: str | None = None
    adversarial_image_url: str | None = None
    perturbation_image_url: str | None = None

    ground_truth_label: str | None = None

    original_label: str
    adversarial_label: str

    original_confidence: float
    adversarial_confidence: float
    confidence_difference: float

    prediction_changed: bool
    attack_success: bool

    original_correct: bool | None = None
    adversarial_correct: bool | None = None

    perturbation_linf: float
    perturbation_l2: float

    diagnostics: EvaluationImageDiagnostics | None = None


class EvaluationSummary(BaseModel):
    """
    Resumen legado.

    Se mantiene temporalmente para que el frontend actual continúe
    funcionando durante la migración a advanced_metrics.
    """

    total_images: int
    changed_predictions: int
    unchanged_predictions: int
    attack_success_rate: float

    average_original_confidence: float
    average_adversarial_confidence: float
    average_confidence_difference: float

    average_perturbation_linf: float
    average_perturbation_l2: float

    labeled_images: int = 0

    original_accuracy: float | None = None
    adversarial_accuracy: float | None = None
    accuracy_drop: float | None = None

    correct_before_attack: int | None = None
    correct_after_attack: int | None = None
    correct_before_wrong_after: int | None = None


class EvaluationResponse(BaseModel):
    result_id: str
    dataset_id: str
    dataset_revision: int = Field(default=0, ge=0)

    model_id: str
    model_name: str | None = None

    attack_type: AttackType
    parameters: AttackParameters

    summary: EvaluationSummary

    advanced_metrics: AdvancedClassificationMetrics | None = None

    results: list[EvaluationImageResult]


class EvaluationGradCamResponse(BaseModel):
    evaluation_id: str
    image_id: str
    original_gradcam_image_url: str
    adversarial_gradcam_image_url: str
