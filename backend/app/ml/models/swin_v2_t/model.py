import torch
from torchvision import models
from torchvision.transforms import InterpolationMode

from ml.models.torchvision_classification import TorchvisionImageClassificationModel


class SwinV2TinyModel(TorchvisionImageClassificationModel):
    """Swin Transformer V2 Tiny pretrained on ImageNet-1K."""

    id = "swin_v2_t"
    name = "Swin Transformer V2 Tiny · preentrenado en ImageNet-1K"
    revision = "torchvision-swin-v2-t-imagenet1k-v1"
    task_type = "classification"
    evaluation_attack_micro_batch_size = 2

    # The official weights use a 256x256 crop after resizing to 260. The
    # application keeps a direct resize to model geometry so adversarial
    # tensors preserve a complete, reversible correspondence with the source
    # image instead of silently discarding pixels through a center crop.
    input_size = (256, 256)
    input_interpolation = InterpolationMode.BICUBIC
    normalization_mean = (0.485, 0.456, 0.406)
    normalization_std = (0.229, 0.224, 0.225)

    def _build_model(self) -> tuple[torch.nn.Module, tuple[str, ...]]:
        """Build Swin V2 Tiny with the matching ImageNet-1K class catalog."""
        weights = models.Swin_V2_T_Weights.IMAGENET1K_V1
        model = models.swin_v2_t(weights=weights)
        categories = tuple(weights.meta["categories"])

        return model, categories

    def get_gradcam_target_layer(self) -> torch.nn.Module:
        """Use Swin's final spatial normalization before global pooling."""
        if not self.is_loaded():
            raise ValueError("Model is not loaded")

        return self.get_torch_model().norm

    def reshape_gradcam_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Convert Swin's [B, H, W, C] activations to [B, C, H, W]."""
        if tensor.ndim != 4:
            raise ValueError("Swin V2 Grad-CAM tensor must have shape [B, H, W, C]")

        return tensor.permute(0, 3, 1, 2)
