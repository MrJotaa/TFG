from ml.models.registry import ModelRegistry
from ml.models.resnet18.model import ResNet18Model
from ml.models.ultralytics.config import UltralyticsDetectionModelConfig
from ml.models.ultralytics.model import UltralyticsDetectionModel

# Temporary definition kept intentionally unchanged during the architectural
# refactor. The next migration sub-issue replaces it with YOLO26n.
_YOLOV8N_CONFIG = UltralyticsDetectionModelConfig(
    id="yolov8n",
    name="YOLOv8n pretrained",
    weights="yolov8n.pt",
    explainability_target_layer_index=-2,
)


def build_model_registry() -> ModelRegistry:
    """Build the application model registry at the composition boundary."""
    return ModelRegistry(
        models=(
            ResNet18Model(),
            UltralyticsDetectionModel(_YOLOV8N_CONFIG),
        ),
        default_model_id="resnet18",
    )
