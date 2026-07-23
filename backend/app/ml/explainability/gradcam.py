import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


def _calculate_gradcam(model_wrapper, model_input: torch.Tensor):
    """Calcula el mapa normalizado sobre la entrada científica del modelo."""
    model = model_wrapper.get_torch_model()
    target_layer = model_wrapper.get_gradcam_target_layer()

    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        output = model(model_input)
        probabilities = F.softmax(output[0], dim=0)

        confidence, predicted_index = torch.max(probabilities, dim=0)
        predicted_label = model_wrapper.categories[predicted_index.item()]

        model.zero_grad()
        output[0, predicted_index].backward()

        activation = activations[0].detach()
        gradient = gradients[0].detach()

        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activation).sum(dim=1).squeeze()

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
        backward_handle.remove()
        activations.clear()
        gradients.clear()
        model.zero_grad()


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


class _YOLOOutputWrapper(torch.nn.Module):
    """
    Wrapper pequeño para que pytorch-grad-cam trabaje con la salida principal
    del modelo interno de YOLO.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_tensor):
        output = self.model(input_tensor)

        if isinstance(output, torch.Tensor):
            return output

        if isinstance(output, (list, tuple)):
            for item in output:
                if isinstance(item, torch.Tensor):
                    return item

                if isinstance(item, (list, tuple)):
                    for nested_item in item:
                        if isinstance(nested_item, torch.Tensor):
                            return nested_item

        raise ValueError("Could not extract tensor output from YOLO model")


def generate_detection_eigencam(model_wrapper, image: Image.Image) -> Image.Image:
    """
    Genera un EigenCAM global para un modelo de detección YOLO.

    Devuelve una imagen PIL con el heatmap superpuesto sobre la imagen original.
    """
    torch_model = model_wrapper.get_torch_model()
    target_layer = model_wrapper.get_explainability_target_layer()

    torch_model.eval()

    input_size = 640

    resized_image = image.convert("RGB").resize((input_size, input_size))

    rgb_np = np.array(resized_image).astype(np.float32) / 255.0

    input_tensor = torch.from_numpy(rgb_np).permute(2, 0, 1).unsqueeze(0)

    try:
        device = next(torch_model.parameters()).device
        input_tensor = input_tensor.to(device)
    except StopIteration:
        pass

    wrapped_model = _YOLOOutputWrapper(torch_model)

    cam = EigenCAM(
        model=wrapped_model,
        target_layers=[target_layer],
    )

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=None,
    )[0]

    cam_image = show_cam_on_image(
        rgb_np,
        grayscale_cam,
        use_rgb=True,
    )

    return Image.fromarray(cam_image).resize(image.size)


def draw_detections_on_image(image: Image.Image, detections: list) -> Image.Image:
    """
    Dibuja bounding boxes y etiquetas sobre una imagen PIL.
    """
    output_image = image.convert("RGB").copy()
    draw = ImageDraw.Draw(output_image)

    for detection in detections:
        bbox = detection.bbox

        x1 = float(bbox.x1)
        y1 = float(bbox.y1)
        x2 = float(bbox.x2)
        y2 = float(bbox.y2)

        label = detection.label
        confidence = detection.confidence

        draw.rectangle(
            [(x1, y1), (x2, y2)],
            outline=(0, 255, 255),
            width=3,
        )

        text = f"{label} {confidence:.2f}"

        text_x = x1
        text_y = max(0, y1 - 18)

        text_bbox = draw.textbbox((text_x, text_y), text)
        draw.rectangle(text_bbox, fill=(0, 255, 255))
        draw.text((text_x, text_y), text, fill=(0, 0, 0))

    return output_image
