from ml.models.registry import ModelRegistry
from ml.models.resnet18.model import ResNet18Model
from ml.models.ultralytics.config import UltralyticsDetectionModelConfig
from ml.models.ultralytics.model import UltralyticsDetectionModel

_YOLO26N_CONFIG = UltralyticsDetectionModelConfig(
    id="yolo26n",
    name="YOLO26n pretrained",
    revision="ultralytics-yolo26n-coco-8.4.121",
    weights="yolo26n.pt",
    # Kept temporarily until the dedicated YOLO26 explainability sub-issue
    # validates the most appropriate feature layer scientifically.
    explainability_target_layer_index=-2,
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
