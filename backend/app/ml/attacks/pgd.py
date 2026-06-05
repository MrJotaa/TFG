import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from ml.attacks.base import BaseAdversarialAttack


class PGDAttack(BaseAdversarialAttack):
    id = "pgd"
    name = "Projected Gradient Descent"
    supported_task_types = ["classification"]

    def __init__(self):
        self.to_tensor = transforms.ToTensor()
        self.to_pil = transforms.ToPILImage()

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def apply(self, model_wrapper, image: Image.Image, parameters):
        """
        Aplica PGD sobre una imagen usando el modelo activo.

        PGD es una versión iterativa de FGSM:
        - aplica varios pasos de tamaño alpha
        - proyecta la perturbación para que no supere epsilon
        - mantiene la imagen en el rango [0, 1]
        """
        if not model_wrapper.is_loaded():
            raise ValueError("Model is not loaded")

        epsilon = parameters.epsilon
        alpha = parameters.alpha
        steps = parameters.steps

        model = model_wrapper.get_torch_model()

        original_tensor = self.to_tensor(image).unsqueeze(0)
        adversarial_tensor = original_tensor.clone().detach()

        normalized_original = self._normalize_batch(original_tensor)
        original_output = model(normalized_original)
        probabilities = F.softmax(original_output[0], dim=0)

        _, original_class = torch.max(probabilities, dim=0)
        target = original_class.unsqueeze(0)

        for _ in range(steps):
            adversarial_tensor.requires_grad = True

            normalized_adversarial = self._normalize_batch(adversarial_tensor)
            output = model(normalized_adversarial)

            loss = F.cross_entropy(output, target)

            model.zero_grad()
            loss.backward()

            gradient_sign = adversarial_tensor.grad.data.sign()

            adversarial_tensor = adversarial_tensor + alpha * gradient_sign

            perturbation = torch.clamp(
                adversarial_tensor - original_tensor,
                min=-epsilon,
                max=epsilon,
            )

            adversarial_tensor = torch.clamp(
                original_tensor + perturbation,
                min=0,
                max=1,
            ).detach()

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

    def _normalize_batch(self, tensor: torch.Tensor) -> torch.Tensor:
        normalized_images = []

        for image_tensor in tensor:
            normalized_images.append(self.normalize(image_tensor))

        return torch.stack(normalized_images)

    def _tensor_to_image(self, tensor: torch.Tensor) -> Image.Image:
        tensor = tensor.detach().cpu().squeeze(0)
        tensor = torch.clamp(tensor, 0, 1)

        return self.to_pil(tensor)

    def _perturbation_to_image(self, perturbation: torch.Tensor) -> Image.Image:
        perturbation = perturbation.detach().cpu().squeeze(0)
        perturbation = perturbation.abs()

        min_value = perturbation.min()
        max_value = perturbation.max()

        perturbation = (perturbation - min_value) / (max_value - min_value + 1e-8)

        return self.to_pil(perturbation)