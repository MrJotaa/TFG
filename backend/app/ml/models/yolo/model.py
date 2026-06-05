from PIL import Image
from ultralytics import YOLO
import torch

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

    def get_torch_model(self):
        if self.model is None:
            self.load()

        return self.model.model

    def get_explainability_target_layer(self):
        if self.model is None:
            self.load()

        # Capa profunda antes del head de detección.
        # Para YOLOv8n suele ser una buena primera opción para EigenCAM.
        return self.model.model.model[-2]

    def _run_inference(self, image: Image.Image):
        if self.model is None:
            self.load()

        with torch.no_grad():
            results = self.model(image)

        return results[0]

    def _result_to_response(self, result) -> DetectionResponse:
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

    def _result_to_annotated_image(self, result) -> Image.Image:
        annotated_array = result.plot()

        # Ultralytics/OpenCV puede devolver la imagen anotada en BGR.
        # PIL espera RGB, por eso invertimos los canales.
        annotated_array = annotated_array[:, :, ::-1].copy()

        return Image.fromarray(annotated_array)

    def detect(self, image: Image.Image) -> DetectionResponse:
        result = self._run_inference(image)
        return self._result_to_response(result)

    def get_annotated_image(self, image: Image.Image) -> Image.Image:
        result = self._run_inference(image)
        return self._result_to_annotated_image(result)

    def detect_with_visualization(
        self,
        image: Image.Image,
    ) -> tuple[DetectionResponse, Image.Image]:
        result = self._run_inference(image)

        detection_response = self._result_to_response(result)
        annotated_image = self._result_to_annotated_image(result)

        return detection_response, annotated_image