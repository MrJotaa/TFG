import torch
from PIL import Image
from torchvision import transforms

from ml.attacks.base import BaseAdversarialAttack


class RandomNoiseAttack(BaseAdversarialAttack):
    id = "random_noise"
    name = "Random Noise Baseline"

    def __init__(self):
        self.to_tensor = transforms.ToTensor()
        self.to_pil = transforms.ToPILImage()

    def apply(self, model_wrapper, image: Image.Image, parameters):
        """
        Aplica ruido aleatorio uniforme sobre la imagen.

        Este ataque no usa gradientes. Sirve como baseline para comparar
        frente a ataques adversarios como FGSM o PGD.
        """
        epsilon = parameters.epsilon

        original_tensor = self.to_tensor(image).unsqueeze(0)

        noise = torch.empty_like(original_tensor).uniform_(
            -epsilon,
            epsilon,
        )

        adversarial_tensor = original_tensor + noise
        adversarial_tensor = torch.clamp(adversarial_tensor, 0, 1)

        perturbation = adversarial_tensor - original_tensor

        perturbation_linf = torch.max(torch.abs(perturbation)).item()
        perturbation_l2 = torch.norm(perturbation).item()

        adversarial_image = self._tensor_to_image(adversarial_tensor)
        perturbation_image = self._perturbation_to_image(perturbation)

        return {
            "adversarial_image": adversarial_image,
            "perturbation_image": perturbation_image,
            "perturbation_linf": round(perturbation_linf, 4),
            "perturbation_l2": round(perturbation_l2, 4),
        }

    def _tensor_to_image(self, tensor: torch.Tensor) -> Image.Image:
        tensor = tensor.detach().cpu().squeeze(0)
        tensor = torch.clamp(tensor, 0, 1)
        return self.to_pil(tensor)

    def _perturbation_to_image(self, perturbation: torch.Tensor) -> Image.Image:
        """
        Convierte la perturbación en una imagen visible.
        Se normaliza visualmente para facilitar su interpretación.
        """
        perturbation = perturbation.detach().cpu().squeeze(0)
        perturbation = perturbation.abs()

        min_value = perturbation.min()
        max_value = perturbation.max()

        perturbation = (perturbation - min_value) / (max_value - min_value + 1e-8)

        return self.to_pil(perturbation)