import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from ml.attacks.base import BaseAdversarialAttack


class FGSMAttack(BaseAdversarialAttack):
    id = "fgsm"
    name = "Fast Gradient Sign Method"
    supported_task_types = ["classification"]

    def __init__(self): # Inicializa las transformaciones necesarias para el ataque.
        self.to_tensor = transforms.ToTensor()

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        self.to_pil = transforms.ToPILImage()

    def apply(self, model_wrapper, image: Image.Image, parameters):
        """
        Aplica FGSM sobre una imagen usando el modelo activo.

        El ataque se aplica en espacio [0, 1], antes de la normalización.
        Esto hace que epsilon sea más interpretable.
        """
        if not model_wrapper.is_loaded():
            raise ValueError("Model is not loaded")

        # Extraemos el valor de epsilon de los parámetros del ataque.
        epsilon = parameters.epsilon
        model = model_wrapper.get_torch_model()

        # Convertimos la imagen a tensor y habilitamos el cálculo de gradientes.
        original_tensor = self.to_tensor(image).unsqueeze(0)
        original_tensor.requires_grad = True

        # Normalizamos el tensor para que el modelo lo procese correctamente.
        normalized_input = self._normalize_batch(original_tensor)

        # Ejecutamos la inferencia para obtener la predicción original y calcular el gradiente.
        output = model(normalized_input)
        probabilities = F.softmax(output[0], dim=0)

        # Obtenemos la clase original para calcular la pérdida y el gradiente.
        _, original_class = torch.max(probabilities, dim=0)

        # Calculamos la pérdida de entropía cruzada para la clase original.
        loss = F.cross_entropy(
            output,
            original_class.unsqueeze(0),
        )

        # Calculamos el gradiente de la pérdida con respecto a la imagen de entrada.
        model.zero_grad()
        loss.backward()

        # Obtenemos el signo del gradiente para cada píxel.
        gradient_sign = original_tensor.grad.data.sign()

        # Creamos la imagen adversaria sumando la perturbación al tensor original.
        adversarial_tensor = original_tensor + epsilon * gradient_sign
        adversarial_tensor = torch.clamp(adversarial_tensor, 0, 1)

        # Calculamos la perturbación aplicada para métricas de comparación.
        perturbation = adversarial_tensor - original_tensor


        # Calculamos las métricas de la perturbación: norma L-infinito y L2.
        perturbation_linf = torch.max(torch.abs(perturbation)).item()
        perturbation_l2 = torch.norm(perturbation).item()

        # Convertimos el tensor adversario y la perturbación a imágenes para la respuesta.
        adversarial_image = self._tensor_to_image(adversarial_tensor)
        perturbation_image = self._perturbation_to_image(perturbation)

        return { # Devolvemos la imagen adversaria y las métricas de la perturbación.
            "adversarial_image": adversarial_image,
            "perturbation_image": perturbation_image,
            "perturbation_linf": round(perturbation_linf, 4),
            "perturbation_l2": round(perturbation_l2, 4),
        }

    def _normalize_batch(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Normaliza un batch de imágenes en formato [B, C, H, W].
        """
        normalized_images = []

        for image_tensor in tensor:
            normalized_images.append(self.normalize(image_tensor))

        return torch.stack(normalized_images)

    def _tensor_to_image(self, tensor: torch.Tensor) -> Image.Image:
        """
        Convierte un tensor [1, C, H, W] en imagen PIL.
        """
        tensor = tensor.detach().cpu().squeeze(0)
        tensor = torch.clamp(tensor, 0, 1)
        return self.to_pil(tensor)

    def _perturbation_to_image(self, perturbation: torch.Tensor) -> Image.Image:
        """
        Convierte la perturbación en una imagen visible.

        La perturbación real puede ser casi imperceptible, así que se normaliza
        visualmente para que pueda interpretarse en la interfaz.
        """
        perturbation = perturbation.detach().cpu().squeeze(0)

        perturbation = perturbation.abs()

        min_value = perturbation.min()
        max_value = perturbation.max()

        perturbation = (perturbation - min_value) / (max_value - min_value + 1e-8)

        return self.to_pil(perturbation)