import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from ml.attacks.base import (
    AdversarialAttackResult,
    AdversarialEvaluationResult,
    AttackParameterSpec,
    BaseAdversarialAttack,
)
from ml.attacks.presentation import perturbation_to_visualization


class FGSMAttack(BaseAdversarialAttack):
    id = "fgsm"
    name = "FGSM"
    description = "Ataque rápido basado en gradiente"
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
        """Aplica FGSM y construye también las visualizaciones interactivas."""
        if not model_wrapper.is_loaded():
            raise ValueError("Model is not loaded")
        prepared_input = model_wrapper.prepare_attack_input(image).detach()
        tensor_result = self._apply_prepared_input(
            model_wrapper=model_wrapper,
            prepared_input=prepared_input,
            parameters=parameters,
            original_class_index=original_class_index,
        )
        return AdversarialAttackResult(
            clean_model_input=tensor_result.clean_model_input,
            adversarial_model_input=tensor_result.adversarial_model_input,
            model_space_delta=tensor_result.model_space_delta,
            adversarial_image=self._tensor_to_image(
                tensor_result.adversarial_model_input
            ),
            perturbation_image=self._perturbation_to_image(
                tensor_result.model_space_delta
            ),
            perturbation_linf=tensor_result.perturbation_linf,
            perturbation_l2=tensor_result.perturbation_l2,
        )

    def apply_for_evaluation(
        self,
        model_wrapper,
        image: Image.Image,
        prepared_input: torch.Tensor,
        parameters,
        original_class_index: int | None = None,
    ) -> AdversarialEvaluationResult:
        del image
        return self._apply_prepared_input(
            model_wrapper=model_wrapper,
            prepared_input=prepared_input,
            parameters=parameters,
            original_class_index=original_class_index,
        )

    def _apply_prepared_input(
        self,
        *,
        model_wrapper,
        prepared_input: torch.Tensor,
        parameters,
        original_class_index: int | None,
    ) -> AdversarialEvaluationResult:
        """Ejecuta FGSM reutilizando una entrada ya preparada en [0, 1]."""
        if not model_wrapper.is_loaded():
            raise ValueError("Model is not loaded")

        epsilon = parameters.epsilon
        model = model_wrapper.get_torch_model()
        model.eval()

        original_tensor = prepared_input.detach().requires_grad_(True)
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

        clean_model_input = original_tensor.detach().clone()
        adversarial_model_input = adversarial_tensor.detach().clone()
        model_space_delta = adversarial_model_input - clean_model_input

        perturbation_linf = model_space_delta.abs().max().item()
        perturbation_l2 = torch.linalg.vector_norm(model_space_delta).item()

        return AdversarialEvaluationResult(
            clean_model_input=clean_model_input,
            adversarial_model_input=adversarial_model_input,
            model_space_delta=model_space_delta,
            perturbation_linf=round(perturbation_linf, 4),
            perturbation_l2=round(perturbation_l2, 4),
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
