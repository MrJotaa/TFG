import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from ml.attacks.base import (
    AdversarialAttackResult,
    AttackParameterSpec,
    BaseAdversarialAttack,
)
from ml.attacks.presentation import perturbation_to_visualization


class PGDAttack(BaseAdversarialAttack):
    id = "pgd"
    name = "PGD"
    description = "Ataque iterativo basado en gradiente"
    knowledge_type = "white_box"
    supported_task_types = ("classification",)
    parameter_specs = (
        AttackParameterSpec(
            key="epsilon",
            label="Epsilon",
            summary_label="ε",
            description="Presupuesto máximo de perturbación L-infinito",
            value_type="float",
            default=0.03,
            control="range",
            minimum=0.0,
            maximum=0.3,
            step=0.01,
        ),
        AttackParameterSpec(
            key="alpha",
            label="Alpha",
            summary_label="α",
            description="Tamaño de paso de cada iteración",
            value_type="float",
            default=0.01,
            control="range",
            minimum=0.0,
            maximum=0.1,
            step=0.001,
        ),
        AttackParameterSpec(
            key="steps",
            label="Steps",
            summary_label="steps",
            description="Número de iteraciones del ataque",
            value_type="integer",
            default=10,
            control="number",
            minimum=1,
            maximum=100,
            step=1,
        ),
    )

    def __init__(self):
        self.to_tensor = transforms.ToTensor()
        self.to_pil = transforms.ToPILImage()

    def apply(
        self,
        model_wrapper,
        image: Image.Image,
        parameters,
        original_class_index: int | None = None,
    ) -> AdversarialAttackResult:
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

        clean_model_input = original_tensor.detach().clone()
        adversarial_model_input = adversarial_tensor.detach().clone()
        model_space_delta = adversarial_model_input - clean_model_input

        perturbation_linf = model_space_delta.abs().max().item()
        perturbation_l2 = torch.linalg.vector_norm(model_space_delta).item()

        adversarial_image = self._tensor_to_image(adversarial_model_input)
        perturbation_image = self._perturbation_to_image(model_space_delta)

        return AdversarialAttackResult(
            clean_model_input=clean_model_input,
            adversarial_model_input=adversarial_model_input,
            model_space_delta=model_space_delta,
            adversarial_image=adversarial_image,
            perturbation_image=perturbation_image,
            perturbation_linf=round(
                perturbation_linf,
                4,
            ),
            perturbation_l2=round(
                perturbation_l2,
                4,
            ),
        )

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
        return perturbation_to_visualization(perturbation)
