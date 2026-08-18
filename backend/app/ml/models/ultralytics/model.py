import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_tensor

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

    def get_explainability_target_layers(self):
        if self.model is None:
            self.load()

        layer_indices = self.config.explainability_target_layer_indices
        if not layer_indices:
            raise NotImplementedError(
                f"Model '{self.id}' does not define explainability target layers"
            )

        layers = self.model.model.model
        try:
            return [layers[layer_index] for layer_index in layer_indices]
        except IndexError as error:
            raise RuntimeError(
                f"Model '{self.id}' explainability layer configuration is invalid"
            ) from error

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
                    imgsz=self.config.image_size,
                    rect=True,
                    end2end=self.config.end_to_end,
                )
            else:
                results = self.model(
                    image,
                    imgsz=self.config.image_size,
                    rect=True,
                    conf=configuration.score_floor,
                    max_det=configuration.max_detections_per_image,
                    end2end=self.config.end_to_end,
                    verbose=False,
                )

        return results[0]

    def _torch_model_device_and_dtype(self) -> tuple[torch.device, torch.dtype]:
        torch_model = self.get_torch_model()

        try:
            parameter = next(torch_model.parameters())
        except StopIteration:
            return torch.device("cpu"), torch.float32

        return parameter.device, parameter.dtype

    def prepare_detection_attack_input(
        self,
        image: Image.Image,
    ) -> torch.Tensor:
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL Image")

        device, dtype = self._torch_model_device_and_dtype()
        return (
            to_tensor(image.convert("RGB")).unsqueeze(0).to(device=device, dtype=dtype)
        )

    def _model_stride(self) -> int:
        torch_model = self.get_torch_model()
        stride = getattr(torch_model, "stride", None)
        if stride is None:
            raise RuntimeError(
                f"Ultralytics model '{self.id}' does not expose its detection stride"
            )

        stride_value = int(torch.as_tensor(stride).max().item())
        if stride_value <= 0:
            raise RuntimeError(
                f"Ultralytics model '{self.id}' exposes an invalid detection stride"
            )

        return stride_value

    def _letterbox_geometry(
        self,
        *,
        source_height: int,
        source_width: int,
    ) -> tuple[int, int, int, int, int, int]:
        if source_height <= 0 or source_width <= 0:
            raise ValueError("source geometry must be positive")

        target_size = self.config.image_size
        scale = min(
            target_size / source_height,
            target_size / source_width,
        )
        resized_height = max(1, round(source_height * scale))
        resized_width = max(1, round(source_width * scale))

        stride = self._model_stride()
        padding_height = (target_size - resized_height) % stride
        padding_width = (target_size - resized_width) % stride
        half_height = padding_height / 2
        half_width = padding_width / 2

        top = round(half_height - 0.1)
        bottom = round(half_height + 0.1)
        left = round(half_width - 0.1)
        right = round(half_width + 0.1)
        return resized_height, resized_width, top, bottom, left, right

    def _differentiable_letterbox(
        self,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        if tensor.ndim != 4 or tensor.shape[1] != 3:
            raise ValueError(
                "prepared detection attack input must have shape [B, 3, H, W]"
            )

        source_height, source_width = tensor.shape[-2:]
        (
            resized_height,
            resized_width,
            top,
            bottom,
            left,
            right,
        ) = self._letterbox_geometry(
            source_height=source_height,
            source_width=source_width,
        )

        if (resized_height, resized_width) == (source_height, source_width):
            resized = tensor
        else:
            resized = F.interpolate(
                tensor,
                size=(resized_height, resized_width),
                mode="bilinear",
                align_corners=False,
            )

        if top == bottom == left == right == 0:
            return resized

        return F.pad(
            resized,
            (left, right, top, bottom),
            value=114 / 255,
        )

    def prepare_detection_explainability_input(
        self,
        image: Image.Image,
    ) -> torch.Tensor:
        source_tensor = self.prepare_detection_attack_input(image)
        return self._differentiable_letterbox(source_tensor)

    def project_detection_explainability_map(
        self,
        normalized_cam: np.ndarray,
        source_image: Image.Image,
    ) -> np.ndarray:
        if not isinstance(normalized_cam, np.ndarray) or normalized_cam.ndim != 2:
            raise TypeError("normalized_cam must be a two-dimensional numpy array")

        (
            resized_height,
            resized_width,
            top,
            _bottom,
            left,
            _right,
        ) = self._letterbox_geometry(
            source_height=source_image.height,
            source_width=source_image.width,
        )

        cam_height, cam_width = normalized_cam.shape
        expected_height = top + resized_height
        expected_width = left + resized_width
        if expected_height > cam_height or expected_width > cam_width:
            raise ValueError(
                "CAM geometry is smaller than the Ultralytics image region"
            )

        cropped = normalized_cam[
            top : top + resized_height,
            left : left + resized_width,
        ]
        cam_image = Image.fromarray(
            np.uint8(np.clip(cropped, 0.0, 1.0) * 255),
            mode="L",
        )
        projected = cam_image.resize(source_image.size, Image.Resampling.BILINEAR)
        return np.asarray(projected, dtype=np.float32) / 255.0

    @staticmethod
    def _end_to_end_confidence_objective(output) -> torch.Tensor:
        if (
            not isinstance(output, (list, tuple))
            or len(output) < 2
            or not isinstance(output[1], dict)
            or not isinstance(output[1].get("one2one"), dict)
        ):
            raise ValueError(
                "Unexpected Ultralytics end-to-end detection output structure"
            )

        scores = output[1]["one2one"].get("scores")
        if not isinstance(scores, torch.Tensor) or scores.ndim != 3:
            raise ValueError("Unexpected Ultralytics one-to-one score tensor structure")
        if scores.shape[1] <= 0 or scores.shape[2] <= 0:
            raise ValueError(
                "Ultralytics one-to-one scores do not contain detection candidates"
            )

        class_probabilities = scores.sigmoid()
        candidate_confidences = class_probabilities.amax(dim=1)
        return candidate_confidences.mean()

    @staticmethod
    def _normalize_detection_geometry_cache_for_autograd(
        torch_model: torch.nn.Module,
    ) -> None:
        """
        Convierte a tensores normales los caches geométricos creados en inference mode.

        Ultralytics cachea ``anchors`` y ``strides`` en el head de detección. Si
        una inferencia pública los creó dentro de ``torch.inference_mode()``, no
        pueden reutilizarse después en un forward seguido por autograd.
        """
        for module in torch_model.modules():
            anchors = getattr(module, "anchors", None)
            strides = getattr(module, "strides", None)

            if not isinstance(anchors, torch.Tensor) or not isinstance(
                strides, torch.Tensor
            ):
                continue

            if torch.is_inference(anchors):
                module.anchors = anchors.clone()
            if torch.is_inference(strides):
                module.strides = strides.clone()

    def detection_confidence_objective(
        self,
        prepared_input: torch.Tensor,
    ) -> torch.Tensor:
        if not self.config.end_to_end:
            raise NotImplementedError(
                f"Model '{self.id}' does not use an end-to-end detection head"
            )

        torch_model = self.get_torch_model()
        torch_model.eval()
        self._normalize_detection_geometry_cache_for_autograd(torch_model)

        model_input = self._differentiable_letterbox(prepared_input)
        output = torch_model.forward(model_input)

        return self._end_to_end_confidence_objective(output)

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
