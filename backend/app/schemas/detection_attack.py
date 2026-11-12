from pydantic import BaseModel, Field

from schemas.attack import ArchivedAttackId, AttackParameters, AttackType
from schemas.detection import DetectionPrediction
from schemas.detection_metrics import DetectionMetricsContract
from schemas.detection_square_trace import SquareDetectionExecutionTrace


class DetectionAttackRequest(BaseModel):
    dataset_id: str
    image_id: str
    attack_type: AttackType = AttackType.RANDOM_NOISE
    parameters: AttackParameters = Field(default_factory=AttackParameters)


class DetectionAttackComparison(BaseModel):
    """
    Comparación básica con deltas ``original - adversarial``.

    Desde metrics_schema_version 1.1, ``attack_success`` es verdadero si
    desaparece al menos una correspondencia uno a uno de la misma clase con
    IoU >= 0.5, no por una variación aislada de confianza. Los artefactos
    antiguos conservan el significado indicado en attack_success_criterion.
    """

    original_detection_count: int
    adversarial_detection_count: int
    detections_difference: int
    average_original_confidence: float
    average_adversarial_confidence: float
    average_confidence_difference: float
    attack_success: bool
    perturbation_linf: float
    perturbation_l2: float


class DetectionAttackResponse(DetectionMetricsContract):
    result_id: str
    attack_type: ArchivedAttackId
    parameters: AttackParameters

    original_annotated_image_url: str
    adversarial_image_url: str
    adversarial_annotated_image_url: str
    perturbation_image_url: str

    original_explanation_image_url: str | None = None
    adversarial_explanation_image_url: str | None = None

    original_detections: list[DetectionPrediction]
    adversarial_detections: list[DetectionPrediction]
    comparison: DetectionAttackComparison
    execution_trace: SquareDetectionExecutionTrace | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
