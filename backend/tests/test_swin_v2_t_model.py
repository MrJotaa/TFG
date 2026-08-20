import torch
from torchvision import models
from torchvision.transforms import InterpolationMode

from ml.models.swin_v2_t.model import SwinV2TinyModel


class _FakeSwinBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(8)


class _FakeSwin(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Identity(),
            torch.nn.Sequential(_FakeSwinBlock()),
        )
        self.norm = torch.nn.LayerNorm(8)

    def forward(self, inputs):
        return torch.zeros((inputs.shape[0], 1000), device=inputs.device)


def test_swin_v2_t_declares_scientific_input_contract():
    model = SwinV2TinyModel()

    assert model.id == "swin_v2_t"
    assert model.task_type == "classification"
    assert model.revision == "torchvision-swin-v2-t-imagenet1k-v1"
    assert model.input_size == (256, 256)
    assert model.input_interpolation is InterpolationMode.BICUBIC
    assert model.get_attack_input_shape() == (3, 256, 256)


def test_swin_v2_t_builds_official_imagenet_weights_without_extra_dependencies(
    monkeypatch,
):
    captured = {}
    fake_model = _FakeSwin()

    def fake_swin_v2_t(*, weights):
        captured["weights"] = weights
        return fake_model

    monkeypatch.setattr(models, "swin_v2_t", fake_swin_v2_t)

    model = SwinV2TinyModel()
    model.load()

    assert captured["weights"] is models.Swin_V2_T_Weights.IMAGENET1K_V1
    assert model.get_torch_model() is fake_model
    assert len(model.get_class_labels()) == 1000
    assert model.get_class_labels() == tuple(
        models.Swin_V2_T_Weights.IMAGENET1K_V1.meta["categories"]
    )


def test_swin_v2_t_exposes_gradcam_layer_and_converts_channels_last():
    model = SwinV2TinyModel()
    fake_model = _FakeSwin()
    model.model = fake_model
    model.categories = ("class-0",)

    target_layer = model.get_gradcam_target_layer()
    channels_last = torch.zeros((1, 8, 8, 768))
    channels_first = model.reshape_gradcam_tensor(channels_last)

    assert target_layer is fake_model.norm
    assert channels_first.shape == (1, 768, 8, 8)


def test_swin_v2_t_rejects_non_spatial_gradcam_tensor():
    model = SwinV2TinyModel()

    try:
        model.reshape_gradcam_tensor(torch.zeros((1, 64, 768)))
    except ValueError as error:
        assert "[B, H, W, C]" in str(error)
    else:
        raise AssertionError("Expected invalid Swin Grad-CAM shape to be rejected")
