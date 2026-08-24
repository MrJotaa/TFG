import math
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from torchvision import transforms

from ml.attacks.base import (
    AdversarialAttackResult,
    AdversarialEvaluationResult,
)
from ml.attacks.fgsm import FGSMAttack


class TinyGradientModel(torch.nn.Module):
    """
    Modelo binario controlado.

    Para una imagen gris de valor 128, la primera clase es la predicha
    y el gradiente de la pérdida respecto a todos los píxeles es negativo.
    Por tanto, FGSM debe restar epsilon a todos los píxeles.
    """

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        score = inputs.mean(dim=(1, 2, 3))

        return torch.stack(
            (
                score,
                -score,
            ),
            dim=1,
        )


class CountingGradientModel(TinyGradientModel):
    def __init__(self):
        super().__init__()
        self.forward_calls = 0

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        return super().forward(inputs)


class UpperBoundaryGradientModel(torch.nn.Module):
    """
    Modelo controlado que produce gradiente positivo para imágenes claras.

    Permite comprobar que FGSM limita el tensor adversario al rango [0, 1].
    """

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        score = inputs.mean(dim=(1, 2, 3))

        class_zero_logit = 3.0 - score
        class_one_logit = torch.zeros_like(score)

        return torch.stack(
            (
                class_zero_logit,
                class_one_logit,
            ),
            dim=1,
        )


class FakeClassificationModelWrapper:
    def __init__(
        self,
        model: torch.nn.Module,
        loaded: bool = True,
    ):
        self.model = model
        self.loaded = loaded

    def is_loaded(self) -> bool:
        return self.loaded

    def get_torch_model(self) -> torch.nn.Module:
        return self.model

    def prepare_attack_input(
        self,
        image: Image.Image,
    ) -> torch.Tensor:
        return transforms.ToTensor()(image).unsqueeze(0)

    def normalize_attack_input(
        self,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        return tensor


def _create_test_image(
    size: tuple[int, int] = (8, 8),
    pixel_value: int = 128,
) -> Image.Image:
    return Image.new(
        "RGB",
        size,
        color=(pixel_value, pixel_value, pixel_value),
    )


def _create_parameters(epsilon: float):
    return SimpleNamespace(epsilon=epsilon)


def _capture_adversarial_tensor(
    attack: FGSMAttack,
    monkeypatch,
) -> dict[str, torch.Tensor]:
    """
    Captura el tensor adversario antes de convertirlo a imagen PIL.

    Esto permite validar las propiedades matemáticas sin introducir
    el redondeo de una imagen RGB de 8 bits.
    """
    captured = {}
    original_tensor_to_image = attack._tensor_to_image

    def capture_tensor(tensor: torch.Tensor) -> Image.Image:
        captured["tensor"] = tensor.detach().clone()
        return original_tensor_to_image(tensor)

    monkeypatch.setattr(
        attack,
        "_tensor_to_image",
        capture_tensor,
    )

    return captured


def test_fgsm_returns_expected_result_structure():
    attack = FGSMAttack()
    wrapper = FakeClassificationModelWrapper(TinyGradientModel())
    original_image = _create_test_image()

    result = attack.apply(
        model_wrapper=wrapper,
        image=original_image,
        parameters=_create_parameters(epsilon=0.05),
    )

    assert isinstance(result, AdversarialAttackResult)

    assert isinstance(result.adversarial_image, Image.Image)
    assert isinstance(result.perturbation_image, Image.Image)

    assert result.adversarial_image.size == original_image.size
    assert result.perturbation_image.size == original_image.size

    assert result.adversarial_image.mode == "RGB"
    assert result.perturbation_image.mode == "RGB"

    tensors = (
        result.clean_model_input,
        result.adversarial_model_input,
        result.model_space_delta,
    )

    assert all(tensor.shape == tensors[0].shape for tensor in tensors)
    assert all(tensor.is_floating_point() for tensor in tensors)
    assert all(not tensor.requires_grad for tensor in tensors)
    assert result.adversarial_model_input.min().item() >= 0.0
    assert result.adversarial_model_input.max().item() <= 1.0
    assert torch.equal(
        result.model_space_delta,
        result.adversarial_model_input - result.clean_model_input,
    )
    assert result.model_space_delta.abs().max().item() <= 0.05 + 1e-7
    assert isinstance(result.perturbation_linf, float)
    assert isinstance(result.perturbation_l2, float)


def test_fgsm_requires_loaded_model():
    attack = FGSMAttack()

    wrapper = FakeClassificationModelWrapper(
        model=TinyGradientModel(),
        loaded=False,
    )

    with pytest.raises(ValueError, match="Model is not loaded"):
        attack.apply(
            model_wrapper=wrapper,
            image=_create_test_image(),
            parameters=_create_parameters(epsilon=0.05),
        )


def test_fgsm_follows_gradient_sign(monkeypatch):
    epsilon = 0.05

    attack = FGSMAttack()
    wrapper = FakeClassificationModelWrapper(TinyGradientModel())
    original_image = _create_test_image(pixel_value=128)

    captured = _capture_adversarial_tensor(
        attack,
        monkeypatch,
    )

    result = attack.apply(
        model_wrapper=wrapper,
        image=original_image,
        parameters=_create_parameters(epsilon=epsilon),
    )

    original_tensor = attack.to_tensor(original_image).unsqueeze(0)
    adversarial_tensor = captured["tensor"]

    expected_adversarial_tensor = original_tensor - epsilon

    assert torch.allclose(
        adversarial_tensor,
        expected_adversarial_tensor,
        atol=1e-6,
    )

    assert torch.all(adversarial_tensor < original_tensor)

    expected_l2 = epsilon * math.sqrt(original_tensor.numel())

    assert result.perturbation_linf == pytest.approx(
        epsilon,
        abs=1e-4,
    )

    assert result.perturbation_l2 == pytest.approx(
        round(expected_l2, 4),
        abs=1e-4,
    )


def test_fgsm_respects_epsilon_bound_and_pixel_range(monkeypatch):
    epsilon = 0.1

    attack = FGSMAttack()

    wrapper = FakeClassificationModelWrapper(
        UpperBoundaryGradientModel(),
    )

    original_image = _create_test_image(pixel_value=250)

    captured = _capture_adversarial_tensor(
        attack,
        monkeypatch,
    )

    result = attack.apply(
        model_wrapper=wrapper,
        image=original_image,
        parameters=_create_parameters(epsilon=epsilon),
    )

    original_tensor = attack.to_tensor(original_image).unsqueeze(0)
    adversarial_tensor = captured["tensor"]

    perturbation = adversarial_tensor - original_tensor
    observed_linf = torch.max(torch.abs(perturbation)).item()

    assert torch.min(adversarial_tensor).item() >= 0.0
    assert torch.max(adversarial_tensor).item() <= 1.0

    assert observed_linf <= epsilon + 1e-7
    assert result.perturbation_linf <= epsilon

    assert torch.any(adversarial_tensor == 1.0)


def test_fgsm_with_zero_epsilon_does_not_modify_image(monkeypatch):
    attack = FGSMAttack()
    wrapper = FakeClassificationModelWrapper(TinyGradientModel())
    original_image = _create_test_image()

    captured = _capture_adversarial_tensor(
        attack,
        monkeypatch,
    )

    result = attack.apply(
        model_wrapper=wrapper,
        image=original_image,
        parameters=_create_parameters(epsilon=0.0),
    )

    original_tensor = attack.to_tensor(original_image).unsqueeze(0)
    adversarial_tensor = captured["tensor"]

    assert torch.equal(
        adversarial_tensor,
        original_tensor,
    )

    assert result.adversarial_image.tobytes() == original_image.tobytes()

    assert result.perturbation_linf == 0.0
    assert result.perturbation_l2 == 0.0


def test_fgsm_uses_model_prepared_input_size():
    class ResizingWrapper(FakeClassificationModelWrapper):
        def prepare_attack_input(
            self,
            image: Image.Image,
        ) -> torch.Tensor:
            resized = image.resize((16, 16))
            return transforms.ToTensor()(resized).unsqueeze(0)

    attack = FGSMAttack()
    wrapper = ResizingWrapper(TinyGradientModel())

    result = attack.apply(
        model_wrapper=wrapper,
        image=_create_test_image(size=(1920, 1279)),
        parameters=_create_parameters(epsilon=0.05),
        original_class_index=0,
    )

    assert result.adversarial_image.size == (16, 16)
    assert result.perturbation_image.size == (16, 16)


def test_fgsm_evaluation_path_reuses_prepared_input(monkeypatch):
    attack = FGSMAttack()
    wrapper = FakeClassificationModelWrapper(TinyGradientModel())
    image = _create_test_image()
    prepared_input = wrapper.prepare_attack_input(image).detach()

    def fail_if_prepared_again(_image):
        raise AssertionError("Evaluation must reuse the already prepared input")

    monkeypatch.setattr(wrapper, "prepare_attack_input", fail_if_prepared_again)
    monkeypatch.setattr(
        attack,
        "_tensor_to_image",
        lambda tensor: (_ for _ in ()).throw(
            AssertionError("Evaluation must not build model-space images")
        ),
    )

    result = attack.apply_for_evaluation(
        model_wrapper=wrapper,
        image=image,
        prepared_input=prepared_input,
        parameters=_create_parameters(epsilon=0.05),
        original_class_index=0,
    )

    assert isinstance(result, AdversarialEvaluationResult)
    assert torch.equal(result.clean_model_input, prepared_input)
    assert result.adversarial_model_input.shape == prepared_input.shape


def test_fgsm_evaluation_micro_batch_matches_individual_results():
    attack = FGSMAttack()
    wrapper = FakeClassificationModelWrapper(TinyGradientModel())
    images = [
        _create_test_image(pixel_value=pixel_value)
        for pixel_value in (64, 96, 128, 192)
    ]
    prepared_inputs = [wrapper.prepare_attack_input(image) for image in images]
    parameters = [_create_parameters(epsilon=0.05) for _ in images]
    original_class_indices = [0 for _ in images]

    individual_results = tuple(
        attack.apply_for_evaluation(
            model_wrapper=wrapper,
            image=image,
            prepared_input=prepared_input,
            parameters=sample_parameters,
            original_class_index=original_class_index,
        )
        for image, prepared_input, sample_parameters, original_class_index in zip(
            images,
            prepared_inputs,
            parameters,
            original_class_indices,
        )
    )

    batch_results = attack.apply_many_for_evaluation(
        model_wrapper=wrapper,
        images=images,
        prepared_inputs=prepared_inputs,
        parameters=parameters,
        original_class_indices=original_class_indices,
    )

    assert attack.evaluation_micro_batch_size == 4
    assert len(batch_results) == len(individual_results)
    for individual, batched in zip(individual_results, batch_results):
        assert torch.allclose(
            batched.adversarial_model_input,
            individual.adversarial_model_input,
            atol=1e-7,
        )
        assert torch.allclose(
            batched.model_space_delta,
            individual.model_space_delta,
            atol=1e-7,
        )
        assert batched.perturbation_linf == individual.perturbation_linf
        assert batched.perturbation_l2 == individual.perturbation_l2


def test_fgsm_micro_batch_uses_single_model_forward():
    model = CountingGradientModel()
    attack = FGSMAttack()
    wrapper = FakeClassificationModelWrapper(model)
    images = [_create_test_image(pixel_value=value) for value in (64, 96, 128, 192)]
    prepared_inputs = [wrapper.prepare_attack_input(image) for image in images]

    attack.apply_many_for_evaluation(
        model_wrapper=wrapper,
        images=images,
        prepared_inputs=prepared_inputs,
        parameters=[_create_parameters(epsilon=0.03) for _ in images],
        original_class_indices=[0 for _ in images],
    )

    assert model.forward_calls == 1
