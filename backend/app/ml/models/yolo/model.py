import torch
from PIL import Image
from ultralytics import YOLO
from ultralytics.engine.results import Results

from ml.detection_categories import DetectionClassCatalog
from ml.models.base import BaseObjectDetectionModel
from ml.models.detection import DetectionResult
from ml.models.detection_inference import (
    DetectionEvaluationInferenceConfiguration,
)
from ml.models.yolo.adapter import UltralyticsDetectionAdapter
from ml.models.yolo.catalog import (
    UltralyticsClassCatalogError,
    build_ultralytics_class_catalog,
)


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

    def get_detection_class_catalog(self) -> DetectionClassCatalog:
        if self.model is None:
            try:
                self.load()
            except Exception as error:
                raise UltralyticsClassCatalogError(
                    f"could not load Ultralytics model {self.id!r} "
                    f"({self.name!r}) to obtain its detection class catalog"
                ) from error

        try:
            names = self.model.names
        except Exception as error:
            raise UltralyticsClassCatalogError(
                f"could not read the detection class catalog for loaded "
                f"Ultralytics model {self.id!r} ({self.name!r})"
            ) from error

        return build_ultralytics_class_catalog(
            model_id=self.id,
            model_name=self.name,
            names=names,
        )

    def _run_inference(
        self,
        image: Image.Image,
        configuration: DetectionEvaluationInferenceConfiguration | None = None,
    ) -> Results:
        if self.model is None:
            self.load()

        with torch.no_grad():
            if configuration is None:
                results = self.model(image)
            else:
                results = self.model(
                    image,
                    conf=configuration.score_floor,
                    iou=configuration.nms_iou_threshold,
                    max_det=configuration.max_detections_per_image,
                    verbose=False,
                )

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

    def detect_for_evaluation(
        self,
        image: Image.Image,
        configuration: DetectionEvaluationInferenceConfiguration,
    ) -> DetectionResult:
        if not isinstance(
            configuration,
            DetectionEvaluationInferenceConfiguration,
        ):
            raise TypeError(
                "configuration must be a DetectionEvaluationInferenceConfiguration"
            )
        result = self._run_inference(image, configuration)
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
