from types import SimpleNamespace

import torch
from PIL import Image, ImageChops
from torchvision import transforms

from ml.attacks.random_noise import RandomNoiseAttack


def _create_test_image(
    size: tuple[int, int] = (16, 16),
    pixel_value: int = 128,
) -> Image.Image:
    return Image.new(
        "RGB",
        size,
        color=(pixel_value, pixel_value, pixel_value),
    )


def _create_parameters(epsilon: float):
    return SimpleNamespace(epsilon=epsilon)


def test_random_noise_returns_expected_result_structure():
    torch.manual_seed(1234)

    attack = RandomNoiseAttack()
    original_image = _create_test_image()
    parameters = _create_parameters(epsilon=0.1)

    result = attack.apply(
        model_wrapper=None,
        image=original_image,
        parameters=parameters,
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


def test_random_noise_respects_epsilon_bound():
    torch.manual_seed(1234)

    epsilon = 0.1
    attack = RandomNoiseAttack()
    original_image = _create_test_image()
    parameters = _create_parameters(epsilon=epsilon)

    result = attack.apply(
        model_wrapper=None,
        image=original_image,
        parameters=parameters,
    )

    assert result["perturbation_linf"] <= epsilon
    assert result["perturbation_linf"] >= 0
    assert result["perturbation_l2"] >= 0


def test_random_noise_output_pixels_remain_in_valid_range():
    torch.manual_seed(1234)

    epsilon = 0.1
    attack = RandomNoiseAttack()
    original_image = _create_test_image()
    parameters = _create_parameters(epsilon=epsilon)

    result = attack.apply(
        model_wrapper=None,
        image=original_image,
        parameters=parameters,
    )

    to_tensor = transforms.ToTensor()

    original_tensor = to_tensor(original_image)
    adversarial_tensor = to_tensor(result["adversarial_image"])

    assert torch.min(adversarial_tensor).item() >= 0.0
    assert torch.max(adversarial_tensor).item() <= 1.0

    observed_perturbation = adversarial_tensor - original_tensor
    observed_linf = torch.max(torch.abs(observed_perturbation)).item()

    quantization_tolerance = 1.0 / 255.0

    assert observed_linf <= epsilon + quantization_tolerance


def test_random_noise_with_zero_epsilon_does_not_modify_image():
    torch.manual_seed(1234)

    attack = RandomNoiseAttack()
    original_image = _create_test_image()
    parameters = _create_parameters(epsilon=0.0)

    result = attack.apply(
        model_wrapper=None,
        image=original_image,
        parameters=parameters,
    )

    difference = ImageChops.difference(
        original_image,
        result["adversarial_image"],
    )

    assert difference.getbbox() is None
    assert result["perturbation_linf"] == 0.0
    assert result["perturbation_l2"] == 0.0


def test_random_noise_is_reproducible_with_fixed_seed():
    attack = RandomNoiseAttack()
    original_image = _create_test_image()
    parameters = _create_parameters(epsilon=0.1)

    torch.manual_seed(2026)

    first_result = attack.apply(
        model_wrapper=None,
        image=original_image,
        parameters=parameters,
    )

    torch.manual_seed(2026)

    second_result = attack.apply(
        model_wrapper=None,
        image=original_image,
        parameters=parameters,
    )

    first_pixels = first_result["adversarial_image"].tobytes()
    second_pixels = second_result["adversarial_image"].tobytes()

    assert first_pixels == second_pixels
    assert first_result["perturbation_linf"] == second_result["perturbation_linf"]
    assert first_result["perturbation_l2"] == second_result["perturbation_l2"]


def test_random_noise_with_positive_epsilon_modifies_image():
    torch.manual_seed(1234)

    attack = RandomNoiseAttack()
    original_image = _create_test_image()
    parameters = _create_parameters(epsilon=0.1)

    result = attack.apply(
        model_wrapper=None,
        image=original_image,
        parameters=parameters,
    )

    difference = ImageChops.difference(
        original_image,
        result["adversarial_image"],
    )

    assert difference.getbbox() is not None
    assert result["perturbation_linf"] > 0
    assert result["perturbation_l2"] > 0
