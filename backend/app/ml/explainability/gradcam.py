import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


def _reshape_gradcam_tensor(
    model_wrapper,
    tensor: torch.Tensor,
    *,
    tensor_name: str,
) -> torch.Tensor:
    """Normaliza activaciones y gradientes al contrato espacial de Grad-CAM."""
    reshaped = model_wrapper.reshape_gradcam_tensor(tensor)

    if not isinstance(reshaped, torch.Tensor):
        raise TypeError(f"Grad-CAM {tensor_name} transform must return a tensor")

    if reshaped.ndim != 4:
        raise ValueError(
            f"Grad-CAM {tensor_name} must have shape [B, C, H, W] after "
            f"model transformation, got {tuple(reshaped.shape)}"
        )

    return reshaped


def _calculate_gradcam(model_wrapper, model_input: torch.Tensor):
    """Calcula el mapa normalizado sobre la entrada científica del modelo."""
    model = model_wrapper.get_torch_model()
    target_layer = model_wrapper.get_gradcam_target_layer()

    activations = []

    def forward_hook(module, input, output):
        activations.append(output)

    forward_handle = target_layer.register_forward_hook(forward_hook)

    try:
        with torch.enable_grad():
            output = model(model_input)
            probabilities = F.softmax(output[0], dim=0)

            confidence, predicted_index = torch.max(probabilities, dim=0)
            class_labels = model_wrapper.get_class_labels()
            predicted_label = class_labels[predicted_index.item()]

            raw_activation = activations[0]
            (raw_gradient,) = torch.autograd.grad(
                outputs=output[0, predicted_index],
                inputs=raw_activation,
            )

        activation = _reshape_gradcam_tensor(
            model_wrapper,
            raw_activation.detach(),
            tensor_name="activation",
        )
        gradient = _reshape_gradcam_tensor(
            model_wrapper,
            raw_gradient.detach(),
            tensor_name="gradient",
        )

        if activation.shape != gradient.shape:
            raise ValueError(
                "Grad-CAM activation and gradient shapes must match after model "
                "transformation"
            )

        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activation).sum(dim=1)[0]

        cam = torch.relu(cam)
        cam = cam.cpu().numpy()

        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)

        return {
            "prediction": predicted_label,
            "confidence": round(confidence.item(), 4),
            "normalized_cam": cam,
        }
    finally:
        forward_handle.remove()
        activations.clear()


def _render_gradcam_on_source(
    normalized_cam: np.ndarray,
    source_image: Image.Image,
) -> Image.Image:
    """Proyecta el mapa y lo mezcla sobre una copia RGB de la imagen fuente."""
    source_rgb = source_image.copy().convert("RGB")
    cam_image = Image.fromarray(np.uint8(normalized_cam * 255)).resize(source_rgb.size)
    heatmap = cm.jet(np.array(cam_image) / 255.0)[:, :, :3]
    heatmap = np.uint8(heatmap * 255)

    original = np.array(source_rgb)
    overlay = np.uint8(0.45 * heatmap + 0.55 * original)

    return Image.fromarray(overlay, mode="RGB")


def generate_clean_gradcam(
    model_wrapper,
    model_input: torch.Tensor,
    source_image: Image.Image,
):
    """
    Genera el Grad-CAM limpio usando representaciones científica y visual separadas.
    """
    result = _calculate_gradcam(model_wrapper, model_input)
    overlay_image = _render_gradcam_on_source(
        normalized_cam=result["normalized_cam"],
        source_image=source_image,
    )

    return {
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "gradcam_pil_image": overlay_image,
    }


def generate_gradcam_from_prepared_input(
    model_wrapper,
    model_input: torch.Tensor,
    visualization_image: Image.Image,
):
    """
    Genera Grad-CAM desde un tensor de ataque preparado en el dominio [0, 1].

    Solo aplica la normalización del modelo; la imagen visual no participa en
    el forward.
    """
    normalized_model_input = model_wrapper.normalize_attack_input(model_input)
    return generate_clean_gradcam(
        model_wrapper=model_wrapper,
        model_input=normalized_model_input,
        source_image=visualization_image,
    )


def generate_gradcam(model_wrapper, image: Image.Image):
    """
    Genera Grad-CAM con el contrato legacy utilizado por los ataques.
    """
    model_input = model_wrapper.preprocess(image)
    result = generate_clean_gradcam(
        model_wrapper=model_wrapper,
        model_input=model_input,
        source_image=image,
    )

    return {
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "gradcam_pil_image": result["gradcam_pil_image"],
    }


class _DetectionEigenCamForward(torch.nn.Module):
    """Ejecuta cualquier detector sin construir un grafo que EigenCAM no necesita."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.model(input_tensor)

        # BaseCAM espera una salida tipo clasificación cuando ``targets=None``.
        # EigenCAM no utiliza ese target, así que exponemos un tensor neutro y
        # mantenemos completamente opacas las salidas reales del detector.
        return input_tensor.new_zeros((input_tensor.shape[0], 1))


def generate_detection_eigencam(model_wrapper, image: Image.Image) -> Image.Image:
    """Genera EigenCAM sin conocer la arquitectura concreta del detector."""
    torch_model = model_wrapper.get_detection_explainability_model()
    target_layers = model_wrapper.get_explainability_target_layers()
    input_tensor = model_wrapper.prepare_detection_explainability_input(image)
    reshape_transform = model_wrapper.reshape_detection_explainability_tensor

    torch_model.eval()
    forward_model = _DetectionEigenCamForward(torch_model)

    with EigenCAM(
        model=forward_model,
        target_layers=target_layers,
        reshape_transform=reshape_transform,
    ) as cam:
        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=None,
        )[0]

    projected_cam = model_wrapper.project_detection_explainability_map(
        grayscale_cam,
        image,
    )
    source_rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    cam_image = show_cam_on_image(
        source_rgb,
        projected_cam,
        use_rgb=True,
    )

    return Image.fromarray(cam_image, mode="RGB")
