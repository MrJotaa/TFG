from ml.models.registry import ModelRegistry
from ml.models.resnet18.model import ResNet18Model
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


def build_model_registry() -> ModelRegistry:
    """Build the application model registry at the composition boundary."""
    return ModelRegistry(
        models=(
            ResNet18Model(),
            UltralyticsDetectionModel(_YOLO26N_CONFIG),
        ),
        default_model_id="resnet18",
    )
