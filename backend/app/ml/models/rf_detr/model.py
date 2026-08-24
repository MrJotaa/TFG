import gc
from collections.abc import Mapping

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection

from ml.detection_categories import DetectionClassCatalog
from ml.models.base import BaseObjectDetectionModel
from ml.models.detection import DetectionResult
from ml.models.detection_inference import (
    DETECTION_EVALUATION_MAX_DETECTIONS_PER_IMAGE,
    DetectionEvaluationInferenceConfiguration,
)
from ml.models.detection_visualization import draw_detections_on_image
from ml.models.rf_detr.adapter import RfDetrDetectionAdapter
from ml.models.rf_detr.catalog import (
    RfDetrClassCatalogError,
    build_rf_detr_class_catalog,
    normalize_rf_detr_id2label,
)
from ml.models.rf_detr.config import RfDetrDetectionModelConfig


class RfDetrDetectionModel(BaseObjectDetectionModel):
    """RF-DETR object detector backed by the native Transformers integration."""

    task_type = "detection"

    def __init__(self, config: RfDetrDetectionModelConfig):
        if not isinstance(config, RfDetrDetectionModelConfig):
            raise TypeError("config must be an RfDetrDetectionModelConfig")

        self.config = config
        self.id = config.id
        self.name = config.name
        self.revision = config.revision
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: torch.nn.Module | None = None
        self.image_processor = None
        self._class_catalog: DetectionClassCatalog | None = None
        self._id2label: dict[int, str] | None = None
        self._prediction_adapter = RfDetrDetectionAdapter()

    def load(self) -> None:
        """Load one pinned checkpoint revision and keep it lazy until selected."""
        if self.is_loaded():
            return

        image_processor = AutoImageProcessor.from_pretrained(
            self.config.checkpoint,
            revision=self.config.checkpoint_revision,
        )
        model = AutoModelForObjectDetection.from_pretrained(
            self.config.checkpoint,
            revision=self.config.checkpoint_revision,
        )
        model.to(self.device)
        model.eval()

        try:
            raw_id2label = getattr(model.config, "id2label", None)
            if raw_id2label is None:
                raise RfDetrClassCatalogError(
                    "RF-DETR checkpoint configuration does not expose id2label"
                )

            id2label = normalize_rf_detr_id2label(raw_id2label)
            class_catalog = build_rf_detr_class_catalog(
                model_id=self.id,
                model_name=self.name,
                id2label=id2label,
            )
        except Exception:
            del model
            del image_processor
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise

        self.model = model
        self.image_processor = image_processor
        self._id2label = id2label
        self._class_catalog = class_catalog

    def unload(self) -> None:
        self.model = None
        self.image_processor = None
        self._class_catalog = None
        self._id2label = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def is_loaded(self) -> bool:
        return (
            self.model is not None
            and self.image_processor is not None
            and self._class_catalog is not None
            and self._id2label is not None
        )

    def get_torch_model(self) -> torch.nn.Module:
        if not self.is_loaded():
            self.load()

        if self.model is None:
            raise RuntimeError("RF-DETR model did not finish loading")
        return self.model

    def get_detection_class_catalog(self) -> DetectionClassCatalog:
        if not self.is_loaded():
            try:
                self.load()
            except Exception as error:
                if isinstance(error, RfDetrClassCatalogError):
                    raise
                raise RfDetrClassCatalogError(
                    f"could not load RF-DETR model {self.id!r} ({self.name!r}) "
                    "to obtain its detection class catalog"
                ) from error

        if self._class_catalog is None:
            raise RuntimeError("RF-DETR class catalog did not finish loading")
        return self._class_catalog

    def _prepare_inputs(self, image: Image.Image) -> dict[str, object]:
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL Image")
        if not self.is_loaded():
            self.load()

        if self.image_processor is None:
            raise RuntimeError("RF-DETR image processor did not finish loading")

        batch = self.image_processor(
            images=image.convert("RGB"),
            return_tensors="pt",
        )
        return {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }

    def _run_inference(
        self,
        image: Image.Image,
        *,
        threshold: float,
        top_k: int,
    ) -> Mapping[str, torch.Tensor]:
        inputs = self._prepare_inputs(image)
        if self.model is None or self.image_processor is None:
            raise RuntimeError("RF-DETR resources did not finish loading")

        with torch.inference_mode():
            outputs = self.model(**inputs)
            target_sizes = torch.tensor(
                [[image.height, image.width]],
                device=self.device,
            )
            results = self.image_processor.post_process_object_detection(
                outputs,
                threshold=threshold,
                target_sizes=target_sizes,
                top_k=top_k,
            )

        if not isinstance(results, list) or len(results) != 1:
            raise RuntimeError("RF-DETR postprocessor returned an unexpected batch")

        return results[0]

    def _result_to_prediction(
        self,
        native_result: Mapping[str, torch.Tensor],
        image: Image.Image,
    ) -> DetectionResult:
        if self._id2label is None:
            raise RuntimeError("RF-DETR class mapping is not loaded")

        return self._prediction_adapter.adapt(
            native_result,
            source_width=image.width,
            source_height=image.height,
            id2label=self._id2label,
        )

    def detect(self, image: Image.Image) -> DetectionResult:
        native_result = self._run_inference(
            image,
            threshold=self.config.confidence_threshold,
            top_k=DETECTION_EVALUATION_MAX_DETECTIONS_PER_IMAGE,
        )
        return self._result_to_prediction(native_result, image)

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

        native_result = self._run_inference(
            image,
            threshold=configuration.score_floor,
            top_k=configuration.max_detections_per_image,
        )
        return self._result_to_prediction(native_result, image)

    def get_annotated_image(self, image: Image.Image) -> Image.Image:
        result = self.detect(image)
        return draw_detections_on_image(image, result.detections)

    def detect_with_visualization(
        self,
        image: Image.Image,
    ) -> tuple[DetectionResult, Image.Image]:
        result = self.detect(image)
        return result, draw_detections_on_image(image, result.detections)
