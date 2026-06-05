from pydantic import BaseModel

from schemas.attack import AttackParameters, AttackType
from schemas.detection import DetectionPrediction


class DetectionAttackRequest(BaseModel):
    dataset_id: str
    image_id: str
    attack_type: AttackType = AttackType.RANDOM_NOISE
    parameters: AttackParameters = AttackParameters()


class DetectionAttackComparison(BaseModel):
    original_detection_count: int
    adversarial_detection_count: int
    detections_difference: int
    average_original_confidence: float
    average_adversarial_confidence: float
    average_confidence_difference: float
    attack_success: bool
    perturbation_linf: float
    perturbation_l2: float


class DetectionAttackResponse(BaseModel):
    result_id: str
    attack_type: AttackType
    parameters: AttackParameters

    original_annotated_image_url: str
    adversarial_image_url: str
    adversarial_annotated_image_url: str
    perturbation_image_url: str

    original_detections: list[DetectionPrediction]
    adversarial_detections: list[DetectionPrediction]
    comparison: DetectionAttackComparison