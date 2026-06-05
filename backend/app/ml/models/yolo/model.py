from PIL import Image
from ultralytics import YOLO

from ml.models.base import BaseObjectDetectionModel
from schemas.detection import BoundingBox, DetectionPrediction, DetectionResponse


class YOLODetectionModel(BaseObjectDetectionModel):
    id = "yolov8n"
    name = "YOLOv8n pretrained"
    task_type = "detection"

    def __init__(self):
        self.model = None
        self.model_path = "yolov8n.pt"

    def load(self) -> None:
        if self.model is not None:
            return

        self.model = YOLO(self.model_path)

    def unload(self) -> None:
        self.model = None

    def is_loaded(self) -> bool:
        return self.model is not None

    def detect(self, image: Image.Image) -> DetectionResponse:
        if self.model is None:
            self.load()

        results = self.model(image)
        result = results[0]

        detections = []

        if result.boxes is not None:
            for box in result.boxes:
                xyxy = box.xyxy[0].tolist()
                confidence = float(box.conf[0].item())
                class_id = int(box.cls[0].item())
                label = result.names[class_id]

                detections.append(
                    DetectionPrediction(
                        label=label,
                        confidence=round(confidence, 4),
                        bbox=BoundingBox(
                            x1=round(float(xyxy[0]), 2),
                            y1=round(float(xyxy[1]), 2),
                            x2=round(float(xyxy[2]), 2),
                            y2=round(float(xyxy[3]), 2),
                        ),
                    )
                )

        return DetectionResponse(
            model_id=self.id,
            model_name=self.name,
            detections=detections,
            annotated_image_url=None,
        )

    def get_annotated_image(self, image: Image.Image) -> Image.Image:
        if self.model is None:
            self.load()

        results = self.model(image)
        result = results[0]

        annotated_array = result.plot()
        return Image.fromarray(annotated_array)