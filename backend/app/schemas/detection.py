from collections.abc import Sequence

from pydantic import BaseModel

from ml.models.detection import Detection


class DetectionRequest(BaseModel):
    dataset_id: str
    image_id: str


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class DetectionPrediction(BaseModel):
    label: str
    confidence: float
    bbox: BoundingBox


class DetectionResponse(BaseModel):
    model_id: str
    model_name: str
    detections: list[DetectionPrediction]
    annotated_image_url: str | None = None


def to_detection_prediction(detection: Detection) -> DetectionPrediction:
    """Presenta una detección canónica mediante el contrato HTTP existente."""
    label = (
        detection.class_name
        if detection.class_name is not None
        else str(detection.class_id)
    )

    return DetectionPrediction(
        label=label,
        confidence=round(detection.confidence, 4),
        bbox=BoundingBox(
            x1=round(detection.box.x_min, 2),
            y1=round(detection.box.y_min, 2),
            x2=round(detection.box.x_max, 2),
            y2=round(detection.box.y_max, 2),
        ),
    )


def to_detection_predictions(
    detections: Sequence[Detection],
) -> list[DetectionPrediction]:
    return [to_detection_prediction(detection) for detection in detections]
