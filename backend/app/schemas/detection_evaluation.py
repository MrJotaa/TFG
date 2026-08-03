from datetime import datetime

from pydantic import BaseModel, Field

from schemas.attack import AttackParameters, AttackType
from schemas.detection import DetectionPrediction
from schemas.detection_metrics import (
    DetectionMetricsContract,
    DetectionSupervisedConfiguration,
    DetectionSupervisedMetrics,
    DetectionSupervisedUnavailableReason,
)


class DetectionEvaluationRequest(BaseModel):
    dataset_id: str
    attack_type: AttackType = AttackType.YOLO_CONFIDENCE
    parameters: AttackParameters = AttackParameters()
    supervised_configuration: DetectionSupervisedConfiguration = Field(
        default_factory=DetectionSupervisedConfiguration
    )
    force_recompute: bool = Field(default=False, exclude=True)


class DetectionEvaluationImageResult(BaseModel):
    """
    Métricas por imagen con deltas ``original - adversarial``.

    ``attack_success`` es una clasificación legacy basada en reducción de
    conteo o confianza, no una evaluación por objetos contra ground truth.
    """

    image_id: str
    filename: str

    original_detection_count: int
    adversarial_detection_count: int
    detections_difference: int

    average_original_confidence: float
    average_adversarial_confidence: float
    average_confidence_difference: float

    attack_success: bool
    perturbation_linf: float
    perturbation_l2: float

    adversarial_image_url: str | None = None

    original_detections: list[DetectionPrediction]
    adversarial_detections: list[DetectionPrediction]


class DetectionEvaluationSummary(BaseModel):
    """
    Resumen básico con deltas ``original - adversarial``.

    ``attack_success_rate`` agrega la clasificación legacy de cada imagen.
    """

    total_images: int

    total_original_detections: int
    total_adversarial_detections: int
    detections_difference: int

    average_original_detections_per_image: float
    average_adversarial_detections_per_image: float

    average_original_confidence: float
    average_adversarial_confidence: float
    average_confidence_difference: float

    images_with_reduced_detections: int
    images_with_no_adversarial_detections: int

    attack_success_rate: float

    average_perturbation_linf: float
    average_perturbation_l2: float


class DetectionEvaluationResponse(DetectionMetricsContract):
    result_id: str
    model_id: str
    model_name: str
    dataset_id: str
    dataset_revision: int = Field(ge=0)
    annotation_snapshot_id: str | None = None
    attack_type: AttackType
    parameters: AttackParameters

    summary: DetectionEvaluationSummary
    results: list[DetectionEvaluationImageResult]
    supervised: DetectionSupervisedMetrics | None = None


class DetectionEvaluationHistoryItem(DetectionMetricsContract):
    result_id: str
    dataset_id: str
    dataset_revision: int = Field(ge=0)
    annotation_snapshot_id: str | None = None
    model_id: str
    model_name: str
    attack_type: AttackType
    parameters: AttackParameters
    supervised_configuration: DetectionSupervisedConfiguration
    summary: DetectionEvaluationSummary
    supervised_available: bool
    supervised_reason: DetectionSupervisedUnavailableReason | None = None
    created_at: datetime
    expires_at: datetime


class DetectionEvaluationEigenCamResponse(BaseModel):
    evaluation_id: str
    image_id: str
    explanation_type: str = "eigencam"
    original_eigencam_image_url: str
    adversarial_eigencam_image_url: str
