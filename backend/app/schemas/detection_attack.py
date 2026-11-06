from pydantic import BaseModel, Field

from schemas.attack import AttackParameters, AttackType
from schemas.detection import DetectionPrediction
from schemas.detection_metrics import DetectionMetricsContract


class DetectionAttackRequest(BaseModel):
    dataset_id: str
    image_id: str
    attack_type: AttackType = AttackType.RANDOM_NOISE
    parameters: AttackParameters = Field(default_factory=AttackParameters)


class DetectionAttackComparison(BaseModel):
    """
    Comparación básica con deltas ``original - adversarial``.

    ``attack_success`` conserva la definición legacy: es verdadero cuando
    disminuye el número de detecciones o la confianza media adversaria.
    No mide desplazamientos de cajas: debe interpretarse con precaución para
    PGD de localización.
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
    attack_type: AttackType
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
