import math
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

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

    assert set(result) == {
        "adversarial_image",
        "perturbation_image",
        "perturbation_linf",
        "perturbation_l2",
    }

    assert isinstance(result["adversarial_image"], Image.Image)
    assert isinstance(result["perturbation_image"], Image.Image)

    assert result["adversarial_image"].size == original_image.size
    assert result["perturbation_image"].size == original_image.size

    assert result["adversarial_image"].mode == "RGB"
    assert result["perturbation_image"].mode == "RGB"


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

    assert result["perturbation_linf"] == pytest.approx(
        epsilon,
        abs=1e-4,
    )

    assert result["perturbation_l2"] == pytest.approx(
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
    assert result["perturbation_linf"] <= epsilon

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

    assert (
        result["adversarial_image"].tobytes()
        == original_image.tobytes()
    )

    assert result["perturbation_linf"] == 0.0
    assert result["perturbation_l2"] == 0.0