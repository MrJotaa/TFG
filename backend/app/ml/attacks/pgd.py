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


class PGDAttack(BaseAdversarialAttack):
    id = "pgd"
    name = "PGD"
    description = "Ataque iterativo basado en gradiente"
    knowledge_type = "white_box"
    supported_task_types = ("classification",)
    evaluation_micro_batch_size = 4
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
        """Aplica PGD y construye también las visualizaciones interactivas."""
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

    def apply_many_for_evaluation(
        self,
        model_wrapper,
        images,
        prepared_inputs,
        parameters,
        original_class_indices,
    ) -> tuple[AdversarialEvaluationResult, ...]:
        if not (
            len(images)
            == len(prepared_inputs)
            == len(parameters)
            == len(original_class_indices)
        ):
            raise ValueError(
                "Evaluation attack batch inputs must have matching lengths"
            )
        if not prepared_inputs:
            return ()

        sample_shapes = {tuple(tensor.shape) for tensor in prepared_inputs}
        if len(sample_shapes) != 1:
            return super().apply_many_for_evaluation(
                model_wrapper=model_wrapper,
                images=images,
                prepared_inputs=prepared_inputs,
                parameters=parameters,
                original_class_indices=original_class_indices,
            )

        return self._apply_prepared_batch(
            model_wrapper=model_wrapper,
            prepared_inputs=prepared_inputs,
            parameters=parameters,
            original_class_indices=original_class_indices,
        )

    def _apply_prepared_batch(
        self,
        *,
        model_wrapper,
        prepared_inputs,
        parameters,
        original_class_indices,
    ) -> tuple[AdversarialEvaluationResult, ...]:
        if not model_wrapper.is_loaded():
            raise ValueError("Model is not loaded")

        model = model_wrapper.get_torch_model()
        model.eval()

        original_tensor = torch.cat(
            [prepared_input.detach() for prepared_input in prepared_inputs],
            dim=0,
        )
        adversarial_tensor = original_tensor.clone()

        if all(
            original_class_index is not None
            for original_class_index in original_class_indices
        ):
            target = torch.tensor(
                original_class_indices,
                device=original_tensor.device,
                dtype=torch.long,
            )
        else:
            with torch.no_grad():
                original_output = model(
                    model_wrapper.normalize_attack_input(original_tensor)
                )
            target = original_output.argmax(dim=1)
            for sample_index, original_class_index in enumerate(
                original_class_indices
            ):
                if original_class_index is not None:
                    target[sample_index] = original_class_index

        epsilon = torch.tensor(
            [sample_parameters.epsilon for sample_parameters in parameters],
            device=original_tensor.device,
            dtype=original_tensor.dtype,
        ).view(-1, 1, 1, 1)
        alpha = torch.tensor(
            [sample_parameters.alpha for sample_parameters in parameters],
            device=original_tensor.device,
            dtype=original_tensor.dtype,
        ).view(-1, 1, 1, 1)
        step_counts = {sample_parameters.steps for sample_parameters in parameters}
        if len(step_counts) != 1:
            return tuple(
                self._apply_prepared_input(
                    model_wrapper=model_wrapper,
                    prepared_input=prepared_input,
                    parameters=sample_parameters,
                    original_class_index=original_class_index,
                )
                for prepared_input, sample_parameters, original_class_index in zip(
                    prepared_inputs,
                    parameters,
                    original_class_indices,
                )
            )
        steps = next(iter(step_counts))

        for step_index in range(steps):
            adversarial_tensor = adversarial_tensor.detach().requires_grad_(True)
            normalized_adversarial = model_wrapper.normalize_attack_input(
                adversarial_tensor
            )
            output = model(normalized_adversarial)
            if step_index == 0 and any(
                original_class_index is not None
                and not 0 <= original_class_index < output.shape[1]
                for original_class_index in original_class_indices
            ):
                raise ValueError(
                    "Original class index is outside the model output range"
                )
            loss = F.cross_entropy(output, target, reduction="sum")
            gradient = torch.autograd.grad(
                loss,
                adversarial_tensor,
                only_inputs=True,
            )[0]

            with torch.no_grad():
                stepped_tensor = adversarial_tensor + alpha * gradient.sign()
                perturbation = torch.maximum(
                    torch.minimum(
                        stepped_tensor - original_tensor,
                        epsilon,
                    ),
                    -epsilon,
                )
                adversarial_tensor = torch.clamp(
                    original_tensor + perturbation,
                    min=0.0,
                    max=1.0,
                )

        return tuple(
            self._build_evaluation_result(
                original_tensor[sample_index : sample_index + 1],
                adversarial_tensor[sample_index : sample_index + 1],
            )
            for sample_index in range(len(prepared_inputs))
        )

    @staticmethod
    def _build_evaluation_result(
        original_tensor: torch.Tensor,
        adversarial_tensor: torch.Tensor,
    ) -> AdversarialEvaluationResult:
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

    def _apply_prepared_input(
        self,
        *,
        model_wrapper,
        prepared_input: torch.Tensor,
        parameters,
        original_class_index: int | None,
    ) -> AdversarialEvaluationResult:
        """Ejecuta PGD reutilizando una entrada ya preparada en [0, 1]."""
        if not model_wrapper.is_loaded():
            raise ValueError("Model is not loaded")

        epsilon = parameters.epsilon
        alpha = parameters.alpha
        steps = parameters.steps

        model = model_wrapper.get_torch_model()
        model.eval()

        original_tensor = prepared_input.detach()
        adversarial_tensor = original_tensor.clone()

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

        return self._build_evaluation_result(
            original_tensor,
            adversarial_tensor,
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
