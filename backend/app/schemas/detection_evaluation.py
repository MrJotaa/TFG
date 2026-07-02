from pydantic import BaseModel

from schemas.attack import AttackParameters, AttackType
from schemas.detection import DetectionPrediction


class DetectionEvaluationRequest(BaseModel):
    dataset_id: str
    attack_type: AttackType = AttackType.YOLO_CONFIDENCE
    parameters: AttackParameters = AttackParameters()


class DetectionEvaluationImageResult(BaseModel):
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

    original_detections: list[DetectionPrediction]
    adversarial_detections: list[DetectionPrediction]


class DetectionEvaluationSummary(BaseModel):
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


class DetectionEvaluationResponse(BaseModel):
    model_id: str
    model_name: str
    dataset_id: str
    attack_type: AttackType
    parameters: AttackParameters

    summary: DetectionEvaluationSummary
    results: list[DetectionEvaluationImageResult]
