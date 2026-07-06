import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from ml.attacks.base import BaseAdversarialAttack


class FGSMAttack(BaseAdversarialAttack):
    id = "fgsm"
    name = "Fast Gradient Sign Method"
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
        Aplica FGSM en el espacio de entrada real del modelo.

        El ataque trabaja sobre píxeles en [0, 1]. El propio modelo
        decide el tamaño de entrada y la normalización necesaria.
        """
        if not model_wrapper.is_loaded():
            raise ValueError("Model is not loaded")

        epsilon = parameters.epsilon
        model = model_wrapper.get_torch_model()
        model.eval()

        original_tensor = (
            model_wrapper.prepare_attack_input(image).detach().requires_grad_(True)
        )

        normalized_input = model_wrapper.normalize_attack_input(original_tensor)

        output = model(normalized_input)

        if original_class_index is None:
            target = output.detach().argmax(dim=1)
        else:
            if not 0 <= original_class_index < output.shape[1]:
                raise ValueError(
                    "Original class index is outside the model output range"
                )

            target = torch.tensor(
                [original_class_index],
                device=output.device,
                dtype=torch.long,
            )

        loss = F.cross_entropy(output, target)

        gradient = torch.autograd.grad(
            loss,
            original_tensor,
            only_inputs=True,
        )[0]

        with torch.no_grad():
            adversarial_tensor = torch.clamp(
                original_tensor + epsilon * gradient.sign(),
                min=0.0,
                max=1.0,
            )

        perturbation = adversarial_tensor - original_tensor.detach()

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
