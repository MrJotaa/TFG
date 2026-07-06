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

    def apply(
        self,
        model_wrapper,
        image: Image.Image,
        parameters,
        original_class_index: int | None = None,
    ):
        """
        Aplica PGD en el espacio de entrada real del modelo.

        El propio modelo define el tamaño y la normalización. PGD
        mantiene la perturbación dentro de la bola L-infinito de
        radio epsilon y los píxeles dentro del rango [0, 1].
        """
        if not model_wrapper.is_loaded():
            raise ValueError("Model is not loaded")

        epsilon = parameters.epsilon
        alpha = parameters.alpha
        steps = parameters.steps

        model = model_wrapper.get_torch_model()
        model.eval()

        original_tensor = model_wrapper.prepare_attack_input(image).detach()
        adversarial_tensor = original_tensor.clone()

        target = None

        if original_class_index is not None:
            target = torch.tensor(
                [original_class_index],
                device=original_tensor.device,
                dtype=torch.long,
            )
        else:
            with torch.no_grad():
                original_output = model(
                    model_wrapper.normalize_attack_input(original_tensor)
                )
                target = original_output.argmax(dim=1)

        for step_index in range(steps):
            adversarial_tensor = adversarial_tensor.detach().requires_grad_(True)

            normalized_adversarial = model_wrapper.normalize_attack_input(
                adversarial_tensor
            )

            output = model(normalized_adversarial)

            if (
                step_index == 0
                and original_class_index is not None
                and not 0 <= original_class_index < output.shape[1]
            ):
                raise ValueError(
                    "Original class index is outside the model output range"
                )

            loss = F.cross_entropy(output, target)

            gradient = torch.autograd.grad(
                loss,
                adversarial_tensor,
                only_inputs=True,
            )[0]

            with torch.no_grad():
                stepped_tensor = adversarial_tensor + alpha * gradient.sign()

                perturbation = torch.clamp(
                    stepped_tensor - original_tensor,
                    min=-epsilon,
                    max=epsilon,
                )

                adversarial_tensor = torch.clamp(
                    original_tensor + perturbation,
                    min=0.0,
                    max=1.0,
                )

        perturbation = adversarial_tensor - original_tensor

        perturbation_linf = perturbation.abs().max().item()
        perturbation_l2 = torch.linalg.vector_norm(perturbation).item()

        adversarial_image = self._tensor_to_image(adversarial_tensor)
        perturbation_image = self._perturbation_to_image(perturbation)

        return {
            "adversarial_image": adversarial_image,
            "perturbation_image": perturbation_image,
            "perturbation_linf": round(
                perturbation_linf,
                4,
            ),
            "perturbation_l2": round(
                perturbation_l2,
                4,
            ),
        }

    def _tensor_to_image(
        self,
        tensor: torch.Tensor,
    ) -> Image.Image:
        tensor = tensor.detach().cpu().squeeze(0)
        tensor = torch.clamp(tensor, 0, 1)
        return self.to_pil(tensor)

    def _perturbation_to_image(
        self,
        perturbation: torch.Tensor,
    ) -> Image.Image:
        perturbation = perturbation.detach().cpu().squeeze(0)
        perturbation = perturbation.abs()

        min_value = perturbation.min()
        max_value = perturbation.max()

        perturbation = (perturbation - min_value) / (max_value - min_value + 1e-8)

        return self.to_pil(perturbation)
