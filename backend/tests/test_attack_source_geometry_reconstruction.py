import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

from ml.attacks.presentation import reconstruct_attack_in_source_geometry


@pytest.mark.parametrize(
    ("source_size", "model_size"),
    [
        ((31, 9), (7, 11)),
        ((8, 29), (13, 5)),
        ((17, 17), (6, 6)),
        ((227, 113), (224, 224)),
        ((1, 2), (3, 4)),
    ],
)
def test_reconstruction_preserves_source_geometry(source_size, model_size):
    source = Image.new("RGB", source_size, color=(64, 128, 192))
    model_width, model_height = model_size
    delta = torch.zeros((1, 3, model_height, model_width), dtype=torch.float32)

    result = reconstruct_attack_in_source_geometry(source, delta)

    assert result.source_space_delta.shape == (
        1,
        3,
        source_size[1],
        source_size[0],
    )
    assert result.adversarial_image.size == source_size
    assert result.perturbation_visualization.size == source_size
    assert result.adversarial_image.mode == "RGB"
    assert result.perturbation_visualization.mode == "RGB"


def test_same_geometry_preserves_delta_exactly():
    source = Image.new("RGB", (5, 4), color=(128, 128, 128))
    delta = torch.linspace(-0.2, 0.2, steps=60).reshape(1, 3, 4, 5)

    result = reconstruct_attack_in_source_geometry(source, delta)

    assert torch.equal(result.source_space_delta, delta)
    assert result.source_space_delta.data_ptr() != delta.data_ptr()


def test_zero_delta_preserves_source_and_uses_current_zero_visualization():
    source = Image.new("RGB", (4, 3), color=(17, 101, 233))
    delta = torch.zeros((1, 3, 2, 2), dtype=torch.float32)

    result = reconstruct_attack_in_source_geometry(source, delta)

    assert torch.count_nonzero(result.source_space_delta) == 0
    assert result.adversarial_image.tobytes() == source.tobytes()
    assert result.perturbation_visualization.getextrema() == ((0, 0),) * 3


def test_signed_delta_and_rgb_channels_are_preserved():
    source = Image.new("RGB", (2, 2), color=(128, 128, 128))
    delta = torch.tensor([0.1, -0.2, 0.05], dtype=torch.float32).view(1, 3, 1, 1)

    result = reconstruct_attack_in_source_geometry(source, delta)

    expected_delta = delta.expand(1, 3, 2, 2)
    assert torch.equal(result.source_space_delta, expected_delta)

    adversarial = pil_to_tensor(result.adversarial_image).float().div(255)
    expected_pixel = torch.tensor([128 / 255 + 0.1, 128 / 255 - 0.2, 128 / 255 + 0.05])
    assert torch.allclose(adversarial[:, 0, 0], expected_pixel, atol=1 / 255)

    visible_delta = expected_delta.squeeze(0).abs()
    expected_visualization = to_pil_image(
        (visible_delta - visible_delta.min())
        / (visible_delta.max() - visible_delta.min() + 1e-8)
    )
    assert (
        result.perturbation_visualization.tobytes() == expected_visualization.tobytes()
    )


def test_reconstruction_clips_lower_and_upper_pixel_bounds():
    source = Image.new("RGB", (1, 1), color=(10, 128, 250))
    delta = torch.tensor([-0.5, 0.0, 0.5], dtype=torch.float32).view(1, 3, 1, 1)

    result = reconstruct_attack_in_source_geometry(source, delta)

    assert result.adversarial_image.getpixel((0, 0)) == (0, 128, 255)


def test_projection_matches_bilinear_interpolation_without_aligning_corners():
    source = Image.new("RGB", (7, 5), color=(128, 128, 128))
    delta = torch.tensor(
        [[[[0.0, 0.2], [-0.1, 0.3]]] * 3],
        dtype=torch.float32,
    )
    expected = F.interpolate(
        delta,
        size=(5, 7),
        mode="bilinear",
        align_corners=False,
    )

    result = reconstruct_attack_in_source_geometry(source, delta)

    assert torch.equal(result.source_space_delta, expected)


def test_reconstruction_does_not_mutate_inputs():
    source = Image.new("RGB", (5, 3), color=(12, 34, 56))
    source_before = source.copy()
    delta = torch.randn((1, 3, 2, 4), dtype=torch.float32)
    delta_before = delta.clone()

    reconstruct_attack_in_source_geometry(source, delta)

    assert source.mode == source_before.mode
    assert source.size == source_before.size
    assert source.tobytes() == source_before.tobytes()
    assert torch.equal(delta, delta_before)


def test_reconstruction_detaches_autograd_history_and_builds_on_cpu():
    source = Image.new("RGB", (4, 3), color=(128, 128, 128))
    base = torch.ones((1, 3, 2, 2), dtype=torch.float32, requires_grad=True)
    delta_with_history = base * 0.1

    result = reconstruct_attack_in_source_geometry(source, delta_with_history)

    assert result.source_space_delta.device.type == "cpu"
    assert result.source_space_delta.requires_grad is False
    assert result.source_space_delta.grad_fn is None


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float64])
def test_reconstruction_accepts_floating_point_dtypes(dtype):
    source = Image.new("RGB", (4, 3), color=(128, 128, 128))
    delta = torch.zeros((1, 3, 2, 2), dtype=dtype)

    result = reconstruct_attack_in_source_geometry(source, delta)

    assert result.source_space_delta.dtype == dtype
    assert result.adversarial_image.mode == "RGB"
    assert result.perturbation_visualization.getextrema() == ((0, 0),) * 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_reconstruction_moves_cuda_delta_to_cpu_for_visual_construction():
    source = Image.new("RGB", (4, 3), color=(128, 128, 128))
    delta = torch.zeros((1, 3, 2, 2), device="cuda")

    result = reconstruct_attack_in_source_geometry(source, delta)

    assert result.source_space_delta.device.type == "cpu"


@pytest.mark.parametrize(
    ("delta", "exception", "message"),
    [
        (
            torch.zeros((2, 3, 4, 4)),
            ValueError,
            "exactly one image",
        ),
        (
            torch.zeros((1, 1, 4, 4)),
            ValueError,
            "exactly three RGB channels",
        ),
        (
            torch.zeros((3, 4, 4)),
            ValueError,
            r"shape \[1, 3, H, W\]",
        ),
        (
            torch.zeros((1, 3, 4, 4), dtype=torch.int64),
            TypeError,
            "floating-point dtype",
        ),
    ],
)
def test_reconstruction_rejects_invalid_delta(delta, exception, message):
    source = Image.new("RGB", (4, 4))

    with pytest.raises(exception, match=message):
        reconstruct_attack_in_source_geometry(source, delta)
