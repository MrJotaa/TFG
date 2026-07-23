from PIL import Image
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results

from ml.models.base import BaseObjectDetectionModel
from ml.models.detection import DetectionResult
from ml.models.yolo.adapter import UltralyticsDetectionAdapter


class YOLODetectionModel(BaseObjectDetectionModel):
    id = "yolov8n"
    name = "YOLOv8n pretrained"
    task_type = "detection"

    def __init__(self):
        self.model = None
        self.model_path = "yolov8n.pt"
        self._prediction_adapter = UltralyticsDetectionAdapter()

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

    def _run_inference(self, image: Image.Image) -> Results:
        if self.model is None:
            self.load()

        with torch.no_grad():
            results = self.model(image)

        return results[0]

    def _result_to_prediction(
        self,
        result: Results,
        image: Image.Image,
    ) -> DetectionResult:
        return self._prediction_adapter.adapt(
            result,
            source_width=image.width,
            source_height=image.height,
        )

    def _result_to_annotated_image(self, result: Results) -> Image.Image:
        annotated_array = result.plot()

        # Ultralytics/OpenCV puede devolver la imagen anotada en BGR.
        # PIL espera RGB, por eso invertimos los canales.
        annotated_array = annotated_array[:, :, ::-1].copy()

        return Image.fromarray(annotated_array)

    def detect(self, image: Image.Image) -> DetectionResult:
        result = self._run_inference(image)
        return self._result_to_prediction(result, image)

    def get_annotated_image(self, image: Image.Image) -> Image.Image:
        result = self._run_inference(image)
        return self._result_to_annotated_image(result)

    def detect_with_visualization(
        self,
        image: Image.Image,
    ) -> tuple[DetectionResult, Image.Image]:
        result = self._run_inference(image)

        detection_result = self._result_to_prediction(result, image)
        annotated_image = self._result_to_annotated_image(result)

        return detection_result, annotated_image
