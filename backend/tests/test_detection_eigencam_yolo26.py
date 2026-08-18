import numpy as np
import torch
from PIL import Image

import ml.explainability.gradcam as gradcam


class _FakeCAM:
    captured = {}

    def __init__(self, *, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.captured.update(target_layers=target_layers)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def __call__(self, *, input_tensor, targets):
        self.captured.update(input_shape=tuple(input_tensor.shape), targets=targets)
        return np.ones(
            (1, input_tensor.shape[-2], input_tensor.shape[-1]),
            dtype=np.float32,
        )


class _TorchModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        return torch.zeros(batch_size, 300, 6, device=inputs.device)


class _Wrapper:
    def __init__(self):
        self.torch_model = _TorchModel()
        self.layers = [torch.nn.Identity(), torch.nn.Identity(), torch.nn.Identity()]
        self.projected = None

    def get_torch_model(self):
        return self.torch_model

    def get_explainability_target_layers(self):
        return self.layers

    def prepare_detection_explainability_input(self, image):
        assert image.size == (23, 17)
        return torch.zeros(1, 3, 480, 640)

    def project_detection_explainability_map(self, normalized_cam, source_image):
        self.projected = (normalized_cam.shape, source_image.size)
        return np.ones((source_image.height, source_image.width), dtype=np.float32)


def test_detection_eigencam_uses_multiscale_layers_and_source_geometry(monkeypatch):
    wrapper = _Wrapper()
    source_image = Image.new("RGB", (23, 17), color=(30, 40, 50))
    _FakeCAM.captured = {}

    monkeypatch.setattr(gradcam, "EigenCAM", _FakeCAM)
    monkeypatch.setattr(
        gradcam,
        "show_cam_on_image",
        lambda rgb, cam, use_rgb: np.zeros((17, 23, 3), dtype=np.uint8),
    )

    result = gradcam.generate_detection_eigencam(wrapper, source_image)

    assert _FakeCAM.captured["target_layers"] == wrapper.layers
    assert _FakeCAM.captured["input_shape"] == (1, 3, 480, 640)
    assert _FakeCAM.captured["targets"] is None
    assert wrapper.projected == ((480, 640), (23, 17))
    assert result.size == source_image.size
    assert result.mode == "RGB"
