import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from utils.image_encoding import image_to_base64


def generate_gradcam(model_wrapper, image: Image.Image):
    """
    Genera Grad-CAM para la clase predicha por el modelo activo.
    Devuelve:
    - etiqueta predicha
    - confianza
    - imagen Grad-CAM en base64
    """
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

    input_tensor = model_wrapper.preprocess(image)

    output = model(input_tensor)
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

    cam_image = Image.fromarray(np.uint8(cam * 255)).resize(image.size)

    heatmap = cm.jet(np.array(cam_image) / 255.0)[:, :, :3]
    heatmap = np.uint8(heatmap * 255)

    original = np.array(image.convert("RGB"))
    overlay = np.uint8(0.45 * heatmap + 0.55 * original)

    overlay_image = Image.fromarray(overlay)

    forward_handle.remove()
    backward_handle.remove()

    return {
        "prediction": predicted_label,
        "confidence": round(confidence.item(), 4),
        "gradcam_image": image_to_base64(overlay_image),
    }