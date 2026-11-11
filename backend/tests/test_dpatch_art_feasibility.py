"""B1: prueba de viabilidad de DPatch/ART, sin adaptadores de producción.

El detector sintético prueba la integración real con ART 1.20.1. Las dos
pruebas opcionales con checkpoints solo verifican gradientes existentes:
NO demuestran compatibilidad directa de esos modelos con DPatch.
"""

import os

import numpy as np
import pytest
import torch
from PIL import Image

from art.attacks.evasion import DPatch
from art.estimators.estimator import BaseEstimator, LossGradientsMixin
from art.estimators.object_detection import PyTorchObjectDetector
from art.estimators.object_detection.object_detector import ObjectDetectorMixin


class _TinyDetector(torch.nn.Module):
    """Detector torchvision-like mínimo con pérdida diferenciable."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, images, targets=None):
        signal = images.mean(dim=(1, 2, 3)) * self.scale
        if self.training:
            if targets is None or len(targets) != images.shape[0]:
                raise ValueError("Training forward requires detection targets")
            return {"loss_box_reg": ((signal - 0.2) ** 2).mean()}

        return [
            {
                "boxes": torch.tensor(
                    [[3.0, 4.0, 12.0, 13.0]],
                    device=images.device,
                    dtype=images.dtype,
                ),
                "labels": torch.tensor([1], device=images.device),
                "scores": torch.sigmoid(value).reshape(1),
            }
            for value in signal
        ]


def _estimator_and_input():
    model = _TinyDetector()
    estimator = PyTorchObjectDetector(
        model=model,
        input_shape=(3, 16, 16),
        channels_first=True,
        clip_values=(0.0, 1.0),
        attack_losses=("loss_box_reg",),
        device_type="cpu",
    )
    x = np.full((1, 3, 16, 16), 0.5, dtype=np.float32)
    return model, estimator, x


def test_dpatch_estimator_requires_detection_gradients():
    assert DPatch._estimator_requirements == (
        BaseEstimator,
        LossGradientsMixin,
        ObjectDetectorMixin,
    )


def test_dpatch_art_1201_predict_gradient_generate_and_apply_patch():
    model, estimator, x = _estimator_and_input()
    weights_before = model.scale.detach().clone()

    predictions = estimator.predict(x)
    assert len(predictions) == 1
    assert set(predictions[0]) >= {"boxes", "labels", "scores"}
    assert predictions[0]["boxes"].shape == (1, 4)
    assert predictions[0]["labels"].shape == (1,)
    assert predictions[0]["scores"].shape == (1,)

    gradients = estimator.loss_gradient(x, predictions)
    assert gradients.shape == x.shape
    assert np.isfinite(gradients).all()
    assert np.any(gradients != 0)

    # El estimador genérico de ART activa train() para loss_gradient;
    # B2 deberá aislar/restaurar ese estado en los detectores reales.
    assert model.training is True

    attack = DPatch(
        estimator=estimator,
        patch_shape=(3, 4, 4),  # ART espera C,H,W con channels_first=True
        learning_rate=0.02,
        max_iter=2,
        batch_size=1,
        verbose=False,
    )
    patch = attack.generate(x=x)
    adversarial = attack.apply_patch(x=x, random_location=False)

    assert patch.shape == (3, 4, 4)
    assert np.isfinite(patch).all()
    assert np.all((0.0 <= patch) & (patch <= 1.0))
    assert adversarial.shape == x.shape
    assert np.isfinite(adversarial).all()
    assert np.all((0.0 <= adversarial) & (adversarial <= 1.0))
    np.testing.assert_allclose(adversarial[0, :, :4, :4], patch)
    np.testing.assert_array_equal(adversarial[0, :, 4:, :], x[0, :, 4:, :])
    np.testing.assert_array_equal(adversarial[0, :, :4, 4:], x[0, :, :4, 4:])
    torch.testing.assert_close(model.scale.detach(), weights_before)


@pytest.mark.real_model
@pytest.mark.parametrize("model_id", ["rf_detr_nano", "yolo26n"])
@pytest.mark.skipif(
    os.getenv("RUN_DPATCH_B1_REAL_SMOKE") != "1",
    reason="Set RUN_DPATCH_B1_REAL_SMOKE=1 for checkpoint gradient feasibility",
)
def test_real_detector_has_input_gradients_but_no_art_adapter_yet(model_id):
    """Comprueba solo la capacidad diferenciable existente, NO DPatch real."""
    from ml.models.definitions import build_model_registry

    model = build_model_registry().get_model(model_id)
    image = Image.new("RGB", (96, 64), color=(145, 90, 45))

    try:
        model.load()
        tensor = model.prepare_detection_attack_input(image).detach().requires_grad_(True)
        parameters = tuple(model.get_torch_model().parameters())
        first_weight_before = parameters[0].detach().clone()
        grads_before = [parameter.grad for parameter in parameters]
        objective = model.detection_confidence_objective(tensor)
        (gradient,) = torch.autograd.grad(objective, tensor)

        assert objective.ndim == 0
        assert torch.isfinite(objective)
        assert gradient.shape == tensor.shape
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) > 0
        torch.testing.assert_close(parameters[0].detach(), first_weight_before)
        assert all(parameter.grad is before for parameter, before in zip(parameters, grads_before))
    finally:
        model.unload()
