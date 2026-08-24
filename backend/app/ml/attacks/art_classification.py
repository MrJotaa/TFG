from __future__ import annotations

from abc import abstractmethod
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image

from core.cancellation import CancellationToken
from ml.adapters.art_classification import ARTClassificationAdapter
from ml.attacks.base import (
    AdversarialAttackResult,
    AdversarialEvaluationResult,
    BaseAdversarialAttack,
)
from ml.attacks.presentation import perturbation_to_visualization


class ARTClassificationAttackRunner:
    """Reusable execution context for ART classification attacks.

    The adapter and ART attack instance are intentionally scoped to one runner so
    dataset evaluations can reuse them without retaining model references beyond
    the evaluation lifetime.
    """

    def __init__(
        self,
        attack: BaseARTClassificationAttack,
        model_wrapper,
        parameters,
        cancellation_token: CancellationToken | None = None,
    ) -> None:
        self._attack = attack
        self._model_wrapper = model_wrapper
        self._parameters = parameters

        torch_model = model_wrapper.get_torch_model()
        torch_model.eval()

        self._adapter = ARTClassificationAdapter(
            model_wrapper,
            cancellation_token=cancellation_token,
        )
        self._art_attack = attack.build_art_attack(
            estimator=self._adapter.estimator,
            parameters=parameters,
        )

    def apply(
        self,
        model_wrapper,
        image: Image.Image,
        parameters,
        original_class_index: int | None = None,
    ) -> AdversarialAttackResult:
        self._validate_execution(model_wrapper, parameters)
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
            perturbation_image=perturbation_to_visualization(
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
        self._validate_execution(model_wrapper, parameters)

        clean_model_input = prepared_input.detach().clone()
        art_input = self._adapter.to_art_input(clean_model_input)
        labels = self._build_labels(original_class_index)

        adversarial_art_input = self._art_attack.generate(
            x=art_input,
            y=labels,
        )
        adversarial_model_input = self._adapter.to_model_input(
            adversarial_art_input
        ).detach()
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

    def _validate_execution(self, model_wrapper, parameters) -> None:
        if model_wrapper is not self._model_wrapper:
            raise ValueError("ART attack runner cannot be reused with another model")
        if parameters.model_dump(mode="python") != self._parameters.model_dump(
            mode="python"
        ):
            raise ValueError("ART attack runner parameters changed during execution")

    def _build_labels(self, original_class_index: int | None) -> np.ndarray | None:
        if original_class_index is None:
            return None
        if not 0 <= original_class_index < self._adapter.nb_classes:
            raise ValueError("Original class index is outside the model output range")

        return np.asarray([original_class_index], dtype=np.int64)

    @staticmethod
    def _tensor_to_image(tensor: torch.Tensor) -> Image.Image:
        image_tensor = tensor.detach().to(device="cpu", dtype=torch.float32).squeeze(0)
        return to_pil_image(torch.clamp(image_tensor, min=0.0, max=1.0))


class BaseARTClassificationAttack(BaseAdversarialAttack):
    """Base class for ART attacks over classification models."""

    def create_runner(
        self,
        model_wrapper,
        parameters,
        cancellation_token: CancellationToken | None = None,
    ) -> ARTClassificationAttackRunner:
        return ARTClassificationAttackRunner(
            attack=self,
            model_wrapper=model_wrapper,
            parameters=parameters,
            cancellation_token=cancellation_token,
        )

    def apply(
        self,
        model_wrapper,
        image: Image.Image,
        parameters,
        original_class_index: int | None = None,
    ) -> AdversarialAttackResult:
        runner = self.create_runner(
            model_wrapper=model_wrapper,
            parameters=parameters,
        )
        return runner.apply(
            model_wrapper=model_wrapper,
            image=image,
            parameters=parameters,
            original_class_index=original_class_index,
        )

    @abstractmethod
    def build_art_attack(self, estimator, parameters) -> Any:
        """Build the concrete ART evasion attack for one execution."""
        raise NotImplementedError
