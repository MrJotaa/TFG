import torch
from PIL import Image

from ml.detection_categories import DetectionClassCatalog
from ml.models.base import BaseObjectDetectionModel
from ml.models.detection import DetectionResult
from ml.models.detection_inference import (
    DetectionEvaluationInferenceConfiguration,
)
from ml.models.ultralytics.adapter import UltralyticsDetectionAdapter
from ml.models.ultralytics.catalog import (
    UltralyticsClassCatalogError,
    build_ultralytics_class_catalog,
)
from ml.models.ultralytics.config import UltralyticsDetectionModelConfig
from ultralytics import YOLO
from ultralytics.engine.results import Results


class UltralyticsDetectionModel(BaseObjectDetectionModel):
    """Object detector backed by an Ultralytics model checkpoint."""

    task_type = "detection"

    def __init__(self, config: UltralyticsDetectionModelConfig):
        if not isinstance(config, UltralyticsDetectionModelConfig):
            raise TypeError("config must be an UltralyticsDetectionModelConfig")

        self.config = config
        self.id = config.id
        self.name = config.name
        self.revision = config.revision
        self.model = None
        self._prediction_adapter = UltralyticsDetectionAdapter()

    def load(self) -> None:
        if self.model is not None:
            return

        self.model = YOLO(self.config.weights)

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

        layer_index = self.config.explainability_target_layer_index
        if layer_index is None:
            raise NotImplementedError(
                f"Model '{self.id}' does not define an explainability target layer"
            )

        return self.model.model.model[layer_index]

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
                results = self.model(
                    image,
                    end2end=self.config.end_to_end,
                )
            else:
                results = self.model(
                    image,
                    conf=configuration.score_floor,
                    max_det=configuration.max_detections_per_image,
                    end2end=self.config.end_to_end,
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
