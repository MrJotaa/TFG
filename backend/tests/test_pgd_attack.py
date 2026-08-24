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
from ml.attacks.pgd import PGDAttack


class TinyGradientModel(torch.nn.Module):
    """
    Modelo binario controlado.

    Para la imagen gris utilizada en los tests, predice la clase 0.
    El gradiente de la pérdida respecto a los píxeles es negativo,
    por lo que PGD debe restar alpha en cada iteración.
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
    """
    Registra cuántas veces se ejecuta el modelo.
    """

    def __init__(self):
        super().__init__()
        self.forward_calls = 0

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        return super().forward(inputs)


class UpperBoundaryGradientModel(torch.nn.Module):
    """
    Genera un gradiente positivo para una imagen clara.

    Permite comprobar que PGD limita la salida al máximo 1.0.
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


def _create_parameters(
    epsilon: float,
    alpha: float,
    steps: int,
):
    return SimpleNamespace(
        epsilon=epsilon,
        alpha=alpha,
        steps=steps,
    )


def _capture_adversarial_tensor(
    attack: PGDAttack,
    monkeypatch,
) -> dict[str, torch.Tensor]:
    """
    Captura el tensor adversario antes de convertirlo a PIL.

    Así se pueden comprobar las propiedades matemáticas sin el
    redondeo propio de las imágenes RGB de 8 bits.
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


def test_pgd_returns_expected_result_structure():
    attack = PGDAttack()
    wrapper = FakeClassificationModelWrapper(TinyGradientModel())
    original_image = _create_test_image()

    result = attack.apply(
        model_wrapper=wrapper,
        image=original_image,
        parameters=_create_parameters(
            epsilon=0.1,
            alpha=0.02,
            steps=3,
        ),
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
    assert result.model_space_delta.abs().max().item() <= 0.1 + 1e-7
    assert isinstance(result.perturbation_linf, float)
    assert isinstance(result.perturbation_l2, float)


def test_pgd_requires_loaded_model():
    attack = PGDAttack()

    wrapper = FakeClassificationModelWrapper(
        model=TinyGradientModel(),
        loaded=False,
    )

    with pytest.raises(ValueError, match="Model is not loaded"):
        attack.apply(
            model_wrapper=wrapper,
            image=_create_test_image(),
            parameters=_create_parameters(
                epsilon=0.1,
                alpha=0.02,
                steps=3,
            ),
        )


def test_pgd_uses_alpha_in_each_iteration(monkeypatch):
    epsilon = 0.2
    alpha = 0.02
    steps = 3

    attack = PGDAttack()
    wrapper = FakeClassificationModelWrapper(TinyGradientModel())
    original_image = _create_test_image(pixel_value=128)

    captured = _capture_adversarial_tensor(
        attack,
        monkeypatch,
    )

    result = attack.apply(
        model_wrapper=wrapper,
        image=original_image,
        parameters=_create_parameters(
            epsilon=epsilon,
            alpha=alpha,
            steps=steps,
        ),
    )

    original_tensor = attack.to_tensor(original_image).unsqueeze(0)
    adversarial_tensor = captured["tensor"]

    expected_perturbation = alpha * steps
    expected_adversarial_tensor = original_tensor - expected_perturbation

    assert torch.allclose(
        adversarial_tensor,
        expected_adversarial_tensor,
        atol=1e-6,
    )

    assert torch.all(adversarial_tensor < original_tensor)

    assert result.perturbation_linf == pytest.approx(
        expected_perturbation,
        abs=1e-4,
    )

    expected_l2 = expected_perturbation * math.sqrt(original_tensor.numel())

    assert result.perturbation_l2 == pytest.approx(
        round(expected_l2, 4),
        abs=1e-4,
    )


def test_pgd_projects_perturbation_to_epsilon(monkeypatch):
    epsilon = 0.08
    alpha = 0.03
    steps = 10

    attack = PGDAttack()
    wrapper = FakeClassificationModelWrapper(TinyGradientModel())
    original_image = _create_test_image(pixel_value=128)

    captured = _capture_adversarial_tensor(
        attack,
        monkeypatch,
    )

    result = attack.apply(
        model_wrapper=wrapper,
        image=original_image,
        parameters=_create_parameters(
            epsilon=epsilon,
            alpha=alpha,
            steps=steps,
        ),
    )

    original_tensor = attack.to_tensor(original_image).unsqueeze(0)
    adversarial_tensor = captured["tensor"]

    perturbation = adversarial_tensor - original_tensor
    observed_linf = torch.max(torch.abs(perturbation)).item()

    expected_adversarial_tensor = original_tensor - epsilon

    assert torch.allclose(
        adversarial_tensor,
        expected_adversarial_tensor,
        atol=1e-6,
    )

    assert observed_linf <= epsilon + 1e-7
    assert observed_linf == pytest.approx(epsilon, abs=1e-6)

    assert result.perturbation_linf == pytest.approx(
        epsilon,
        abs=1e-4,
    )

    expected_l2 = epsilon * math.sqrt(original_tensor.numel())

    assert result.perturbation_l2 == pytest.approx(
        round(expected_l2, 4),
        abs=1e-4,
    )


def test_pgd_runs_configured_number_of_steps():
    steps = 4

    model = CountingGradientModel()
    attack = PGDAttack()

    wrapper = FakeClassificationModelWrapper(model)

    attack.apply(
        model_wrapper=wrapper,
        image=_create_test_image(),
        parameters=_create_parameters(
            epsilon=0.1,
            alpha=0.02,
            steps=steps,
        ),
    )

    # Una inferencia obtiene la clase original.
    # Después se ejecuta una inferencia por cada paso de PGD.
    assert model.forward_calls == steps + 1


def test_pgd_output_remains_in_valid_pixel_range(monkeypatch):
    attack = PGDAttack()

    wrapper = FakeClassificationModelWrapper(UpperBoundaryGradientModel())

    original_image = _create_test_image(pixel_value=250)

    captured = _capture_adversarial_tensor(
        attack,
        monkeypatch,
    )

    result = attack.apply(
        model_wrapper=wrapper,
        image=original_image,
        parameters=_create_parameters(
            epsilon=0.2,
            alpha=0.1,
            steps=3,
        ),
    )

    original_tensor = attack.to_tensor(original_image).unsqueeze(0)
    adversarial_tensor = captured["tensor"]

    perturbation = adversarial_tensor - original_tensor
    observed_linf = torch.max(torch.abs(perturbation)).item()

    assert torch.min(adversarial_tensor).item() >= 0.0
    assert torch.max(adversarial_tensor).item() <= 1.0

    assert torch.any(adversarial_tensor == 1.0)

    assert observed_linf <= 0.2 + 1e-7
    assert result.perturbation_linf <= 0.2


def test_pgd_with_zero_epsilon_does_not_modify_image(monkeypatch):
    attack = PGDAttack()
    wrapper = FakeClassificationModelWrapper(TinyGradientModel())
    original_image = _create_test_image()

    captured = _capture_adversarial_tensor(
        attack,
        monkeypatch,
    )

    result = attack.apply(
        model_wrapper=wrapper,
        image=original_image,
        parameters=_create_parameters(
            epsilon=0.0,
            alpha=0.02,
            steps=5,
        ),
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


def test_pgd_reuses_provided_original_class():
    steps = 4

    model = CountingGradientModel()
    attack = PGDAttack()
    wrapper = FakeClassificationModelWrapper(model)

    attack.apply(
        model_wrapper=wrapper,
        image=_create_test_image(),
        parameters=_create_parameters(
            epsilon=0.1,
            alpha=0.02,
            steps=steps,
        ),
        original_class_index=0,
    )

    assert model.forward_calls == steps


def test_pgd_uses_model_prepared_input_size():
    class ResizingWrapper(FakeClassificationModelWrapper):
        def prepare_attack_input(
            self,
            image: Image.Image,
        ) -> torch.Tensor:
            resized = image.resize((16, 16))
            return transforms.ToTensor()(resized).unsqueeze(0)

    attack = PGDAttack()
    wrapper = ResizingWrapper(TinyGradientModel())

    result = attack.apply(
        model_wrapper=wrapper,
        image=_create_test_image(size=(1920, 1279)),
        parameters=_create_parameters(
            epsilon=0.1,
            alpha=0.02,
            steps=2,
        ),
        original_class_index=0,
    )

    assert result.adversarial_image.size == (16, 16)
    assert result.perturbation_image.size == (16, 16)


def test_pgd_evaluation_path_reuses_prepared_input(monkeypatch):
    attack = PGDAttack()
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
        parameters=_create_parameters(epsilon=0.1, alpha=0.02, steps=2),
        original_class_index=0,
    )

    assert isinstance(result, AdversarialEvaluationResult)
    assert torch.equal(result.clean_model_input, prepared_input)
    assert result.adversarial_model_input.shape == prepared_input.shape


def test_pgd_evaluation_micro_batch_matches_individual_results():
    attack = PGDAttack()
    wrapper = FakeClassificationModelWrapper(TinyGradientModel())
    images = [
        _create_test_image(pixel_value=pixel_value)
        for pixel_value in (64, 96, 128, 192)
    ]
    prepared_inputs = [wrapper.prepare_attack_input(image) for image in images]
    parameters = [_create_parameters(epsilon=0.08, alpha=0.02, steps=3) for _ in images]
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


def test_pgd_micro_batch_uses_one_forward_per_step():
    steps = 4
    model = CountingGradientModel()
    attack = PGDAttack()
    wrapper = FakeClassificationModelWrapper(model)
    images = [_create_test_image(pixel_value=value) for value in (64, 96, 128, 192)]
    prepared_inputs = [wrapper.prepare_attack_input(image) for image in images]

    attack.apply_many_for_evaluation(
        model_wrapper=wrapper,
        images=images,
        prepared_inputs=prepared_inputs,
        parameters=[
            _create_parameters(epsilon=0.08, alpha=0.02, steps=steps) for _ in images
        ],
        original_class_indices=[0 for _ in images],
    )

    assert model.forward_calls == steps
