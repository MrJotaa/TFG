import gc
from collections.abc import Mapping

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_tensor
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
    is_rf_detr_placeholder_label,
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

    def supports_detection_explainability(self) -> bool:
        return True

    def get_explainability_target_layers(self) -> list[torch.nn.Module]:
        """Use the spatial projector consumed directly by the DETR decoder."""
        torch_model = self.get_torch_model()
        # Transformers convierte aquí las features multietapa de DINOv2 en el
        # mapa espacial [B, C, H, W] que RF-DETR aplana y entrega al decoder.
        try:
            target_layer = torch_model.model.backbone.projector.layer_norm
        except AttributeError as error:
            raise RuntimeError(
                "RF-DETR explainability target layer is incompatible with the "
                "pinned Transformers architecture"
            ) from error

        if not isinstance(target_layer, torch.nn.Module):
            raise RuntimeError("RF-DETR explainability target is not a torch module")
        return [target_layer]

    def prepare_detection_explainability_input(
        self,
        image: Image.Image,
    ) -> torch.Tensor:
        """Reproduce the checkpoint preprocessing while retaining CAM geometry."""
        source_tensor = self.prepare_detection_attack_input(image)
        pixel_values, _pixel_mask = self._differentiable_attack_model_input(
            source_tensor
        )
        return pixel_values

    def project_detection_explainability_map(
        self,
        normalized_cam: np.ndarray,
        source_image: Image.Image,
    ) -> np.ndarray:
        """Project the fixed-size RF-DETR CAM back to source-image geometry."""
        if not isinstance(normalized_cam, np.ndarray) or normalized_cam.ndim != 2:
            raise TypeError("normalized_cam must be a two-dimensional numpy array")
        if not isinstance(source_image, Image.Image):
            raise TypeError("source_image must be a PIL Image")

        cam_image = Image.fromarray(
            np.uint8(np.clip(normalized_cam, 0.0, 1.0) * 255),
            mode="L",
        )
        projected = cam_image.resize(source_image.size, Image.Resampling.BILINEAR)
        return np.asarray(projected, dtype=np.float32) / 255.0

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

    def _torch_model_device_and_dtype(self) -> tuple[torch.device, torch.dtype]:
        torch_model = self.get_torch_model()
        try:
            parameter = next(torch_model.parameters())
        except StopIteration:
            return self.device, torch.float32
        return parameter.device, parameter.dtype

    def prepare_detection_attack_input(
        self,
        image: Image.Image,
    ) -> torch.Tensor:
        """Return source-geometry pixels in [0, 1] for adversarial updates."""
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL Image")

        device, dtype = self._torch_model_device_and_dtype()
        return (
            to_tensor(image.convert("RGB")).unsqueeze(0).to(device=device, dtype=dtype)
        )

    def _attack_processor_configuration(
        self,
    ) -> tuple[tuple[int, int], torch.Tensor, torch.Tensor]:
        if not self.is_loaded():
            self.load()
        if self.image_processor is None:
            raise RuntimeError("RF-DETR image processor did not finish loading")

        processor = self.image_processor
        required_flags = {
            "do_resize": True,
            "do_rescale": True,
            "do_normalize": True,
        }
        for attribute, expected in required_flags.items():
            if getattr(processor, attribute, None) is not expected:
                raise NotImplementedError(
                    f"RF-DETR attack preprocessing requires {attribute}={expected}"
                )

        size = getattr(processor, "size", None)
        if isinstance(size, Mapping):
            height = size.get("height")
            width = size.get("width")
        else:
            height = getattr(size, "height", None)
            width = getattr(size, "width", None)
        if not isinstance(height, int) or not isinstance(width, int):
            raise NotImplementedError(
                "RF-DETR differentiable attacks require an exact "
                "height/width processor size"
            )
        if height <= 0 or width <= 0:
            raise RuntimeError("RF-DETR image processor exposes an invalid size")

        mean = getattr(processor, "image_mean", None)
        std = getattr(processor, "image_std", None)
        if not isinstance(mean, (list, tuple)) or len(mean) != 3:
            raise NotImplementedError(
                "RF-DETR differentiable attacks require a three-channel image mean"
            )
        if not isinstance(std, (list, tuple)) or len(std) != 3:
            raise NotImplementedError(
                "RF-DETR differentiable attacks require a three-channel image std"
            )
        if any(float(value) <= 0 for value in std):
            raise RuntimeError("RF-DETR image processor exposes an invalid image std")

        device, dtype = self._torch_model_device_and_dtype()
        mean_tensor = torch.tensor(mean, device=device, dtype=dtype).view(1, 3, 1, 1)
        std_tensor = torch.tensor(std, device=device, dtype=dtype).view(1, 3, 1, 1)
        return (height, width), mean_tensor, std_tensor

    def _differentiable_attack_model_input(
        self,
        prepared_input: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            not isinstance(prepared_input, torch.Tensor)
            or prepared_input.ndim != 4
            or prepared_input.shape[1] != 3
        ):
            raise ValueError(
                "prepared detection attack input must have shape [B, 3, H, W]"
            )
        if not prepared_input.is_floating_point():
            raise TypeError("prepared detection attack input must be floating point")

        (height, width), mean, std = self._attack_processor_configuration()
        # ``prepared_input`` ya representa el rescale 1/255 del procesador
        # oficial: mantenemos [0, 1] en geometría fuente durante el ataque.
        resized = F.interpolate(
            prepared_input,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        pixel_values = (resized - mean) / std
        pixel_mask = torch.ones(
            (prepared_input.shape[0], height, width),
            device=prepared_input.device,
            dtype=torch.long,
        )
        return pixel_values, pixel_mask

    def _valid_attack_class_ids(self, *, num_classes: int) -> tuple[int, ...]:
        if self._id2label is None:
            raise RuntimeError("RF-DETR class mapping is not loaded")

        class_ids = tuple(
            class_id
            for class_id, class_name in sorted(self._id2label.items())
            if not is_rf_detr_placeholder_label(class_name)
        )
        if not class_ids:
            raise RuntimeError(
                "RF-DETR does not expose any valid classes for confidence suppression"
            )
        if min(class_ids) < 0 or max(class_ids) >= num_classes:
            raise ValueError(
                "RF-DETR class catalog does not match the logits class dimension"
            )
        return class_ids

    @staticmethod
    def _query_confidence_objective(
        logits: torch.Tensor,
        *,
        valid_class_ids: tuple[int, ...],
    ) -> torch.Tensor:
        if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
            raise ValueError("Unexpected RF-DETR object-detection logits structure")
        if logits.shape[0] <= 0 or logits.shape[1] <= 0 or logits.shape[2] <= 0:
            raise ValueError("RF-DETR logits do not contain detection queries")
        if not valid_class_ids:
            raise ValueError("RF-DETR confidence objective requires valid class ids")
        if min(valid_class_ids) < 0 or max(valid_class_ids) >= logits.shape[2]:
            raise ValueError("RF-DETR confidence class ids exceed the logits dimension")

        class_index = torch.tensor(
            valid_class_ids,
            device=logits.device,
            dtype=torch.long,
        )
        valid_probabilities = logits.index_select(dim=2, index=class_index).sigmoid()
        query_confidences = valid_probabilities.amax(dim=2)
        return query_confidences.mean()

    def detection_confidence_objective(
        self,
        prepared_input: torch.Tensor,
    ) -> torch.Tensor:
        torch_model = self.get_torch_model()
        torch_model.eval()

        pixel_values, pixel_mask = self._differentiable_attack_model_input(
            prepared_input
        )
        outputs = torch_model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
        )
        logits = getattr(outputs, "logits", None)
        if not isinstance(logits, torch.Tensor):
            raise ValueError("RF-DETR output does not expose object-detection logits")

        valid_class_ids = self._valid_attack_class_ids(num_classes=logits.shape[-1])
        return self._query_confidence_objective(
            logits,
            valid_class_ids=valid_class_ids,
        )

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
