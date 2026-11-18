from dataclasses import dataclass

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, to_pil_image


@dataclass(frozen=True, slots=True)
class SourceGeometryAttackReconstruction:
    """Artefactos visuales reconstruidos en la geometría de la imagen fuente."""

    source_space_delta: torch.Tensor
    adversarial_image: Image.Image
    perturbation_visualization: Image.Image


def perturbation_to_visualization(perturbation: torch.Tensor) -> Image.Image:
    """Reproduce la visualización min-max usada por FGSM y PGD."""
    visible_perturbation = perturbation.detach().cpu().float().squeeze(0).abs()

    min_value = visible_perturbation.min()
    max_value = visible_perturbation.max()
    visible_perturbation = (visible_perturbation - min_value) / (
        max_value - min_value + 1e-8
    )

    return to_pil_image(visible_perturbation)


def reconstruct_attack_in_source_geometry(
    source_image: Image.Image,
    model_space_delta: torch.Tensor,
) -> SourceGeometryAttackReconstruction:
    """Proyecta un delta de modelo y construye artefactos visuales de fuente."""
    _validate_model_space_delta(model_space_delta)

    source_rgb = source_image.convert("RGB")
    source_width, source_height = source_rgb.size

    detached_delta = model_space_delta.detach().to(device="cpu", copy=True)
    if detached_delta.shape[-2:] == (source_height, source_width):
        source_space_delta = detached_delta
    else:
        source_space_delta = F.interpolate(
            detached_delta,
            size=(source_height, source_width),
            mode="bilinear",
            align_corners=False,
        )

    source_tensor = pil_to_tensor(source_rgb).to(dtype=source_space_delta.dtype)
    source_tensor = source_tensor.div(255).unsqueeze(0)
    source_geometry_adversarial = torch.clamp(
        source_tensor + source_space_delta,
        min=0.0,
        max=1.0,
    )

    adversarial_image = to_pil_image(source_geometry_adversarial.squeeze(0).float())
    perturbation_visualization = perturbation_to_visualization(source_space_delta)

    return SourceGeometryAttackReconstruction(
        source_space_delta=source_space_delta,
        adversarial_image=adversarial_image,
        perturbation_visualization=perturbation_visualization,
    )


def _validate_model_space_delta(model_space_delta: torch.Tensor) -> None:
    if not isinstance(model_space_delta, torch.Tensor):
        raise TypeError("model_space_delta must be a torch.Tensor")
    if model_space_delta.ndim != 4:
        raise ValueError("model_space_delta must have shape [1, 3, H, W]")
    if model_space_delta.shape[0] != 1:
        raise ValueError("model_space_delta must contain exactly one image")
    if model_space_delta.shape[1] != 3:
        raise ValueError("model_space_delta must contain exactly three RGB channels")
    if not model_space_delta.is_floating_point():
        raise TypeError("model_space_delta must have a floating-point dtype")
