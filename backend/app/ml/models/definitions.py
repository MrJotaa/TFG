from ml.models.registry import ModelRegistry
from ml.models.resnet18.model import ResNet18Model
from ml.models.rf_detr.config import RfDetrDetectionModelConfig
from ml.models.rf_detr.model import RfDetrDetectionModel
from ml.models.swin_v2_t.model import SwinV2TinyModel
from ml.models.ultralytics.config import UltralyticsDetectionModelConfig
from ml.models.ultralytics.model import UltralyticsDetectionModel

_YOLO26N_CONFIG = UltralyticsDetectionModelConfig(
    id="yolo26n",
    name="YOLO26n pretrained",
    revision="ultralytics-yolo26n-coco-8.4.121",
    weights="yolo26n.pt",
    image_size=640,
    end_to_end=True,
    # YOLO26 Detect consumes the P3/8, P4/16 and P5/32 feature maps from
    # layers 16, 19 and 22 respectively. EigenCAM aggregates all three scales.
    explainability_target_layer_indices=(16, 19, 22),
)

_RF_DETR_NANO_CONFIG = RfDetrDetectionModelConfig(
    id="rf_detr_nano",
    name="RF-DETR Nano pretrained",
    revision="transformers-5.15.1-rf-detr-nano-coco-9f201577",
    checkpoint="Roboflow/rf-detr-nano",
    checkpoint_revision="9f201577d9415f0378084748d27e68586e6b3600",
)


def build_model_registry() -> ModelRegistry:
    """Build the application model registry at the composition boundary."""
    return ModelRegistry(
        models=(
            ResNet18Model(),
            SwinV2TinyModel(),
            UltralyticsDetectionModel(_YOLO26N_CONFIG),
            RfDetrDetectionModel(_RF_DETR_NANO_CONFIG),
        ),
        default_model_id="resnet18",
    )
