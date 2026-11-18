import torch
from torchvision import models

from ml.models.torchvision_classification import TorchvisionImageClassificationModel


class ResNet18Model(TorchvisionImageClassificationModel):
    id = "resnet18"
    name = "ResNet-18 · preentrenado en ImageNet-1K"
    revision = "torchvision-default-v1"
    task_type = "classification"
    evaluation_attack_micro_batch_size = 4

    input_size = (224, 224)
    normalization_mean = (0.485, 0.456, 0.406)
    normalization_std = (0.229, 0.224, 0.225)

    def _build_model(self) -> tuple[torch.nn.Module, tuple[str, ...]]:
        """Construye ResNet18 y el catálogo ImageNet de sus mismos pesos."""
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        categories = tuple(weights.meta["categories"])

        return model, categories

    def get_gradcam_target_layer(self):
        if not self.is_loaded():
            raise ValueError("Model is not loaded")

        return self.get_torch_model().layer4[-1]
