import pytest
import torch
from PIL import Image

from ml.models.ultralytics.config import UltralyticsDetectionModelConfig
from ml.models.ultralytics.model import UltralyticsDetectionModel


class TinyEndToEndYOLO26(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.register_buffer("stride", torch.tensor([8.0, 16.0, 32.0]))
        self.last_input_shape = None

    def forward(self, inputs: torch.Tensor):
        self.last_input_shape = tuple(inputs.shape)

        batch_size = inputs.shape[0]
        candidate_count = 3
        image_signal = inputs.mean(dim=(1, 2, 3)) * self.scale
        primary_logits = (image_signal - 0.5) * 4.0
        primary_logits = primary_logits[:, None].expand(batch_size, candidate_count)
        secondary_logits = primary_logits - 2.0
        confidence = torch.sigmoid(primary_logits)

        boxes = torch.zeros(
            batch_size,
            candidate_count,
            4,
            device=inputs.device,
            dtype=inputs.dtype,
        )
        class_ids = torch.full(
            (batch_size, candidate_count, 1),
            79.0,
            device=inputs.device,
            dtype=inputs.dtype,
        )
        predictions = torch.cat(
            (
                boxes,
                confidence.unsqueeze(-1),
                class_ids,
            ),
            dim=-1,
        )

        raw_predictions = {
            "one2many": {"scores": torch.zeros(1, device=inputs.device)},
            "one2one": {
                "scores": torch.stack(
                    (primary_logits, secondary_logits),
                    dim=1,
                )
            },
        }
        return predictions, raw_predictions


class ProviderModel:
    def __init__(self, torch_model: torch.nn.Module):
        self.model = torch_model


def _build_model(
    *,
    end_to_end: bool = True,
    image_size: int = 640,
) -> tuple[UltralyticsDetectionModel, TinyEndToEndYOLO26]:
    torch_model = TinyEndToEndYOLO26()
    model = UltralyticsDetectionModel(
        UltralyticsDetectionModelConfig(
            id="yolo26n",
            name="YOLO26n · preentrenado en COCO",
            revision="ultralytics-yolo26n-coco-8.4.121",
            weights="yolo26n.pt",
            image_size=image_size,
            end_to_end=end_to_end,
        )
    )
    model.model = ProviderModel(torch_model)
    return model, torch_model


def test_ultralytics_attack_input_preserves_source_geometry_and_model_device():
    model, torch_model = _build_model()
    image = Image.new("RGB", (640, 300), color=(128, 64, 32))

    prepared = model.prepare_detection_attack_input(image)

    assert prepared.shape == (1, 3, 300, 640)
    assert prepared.device == torch_model.scale.device
    assert prepared.dtype == torch_model.scale.dtype
    assert prepared.min().item() >= 0.0
    assert prepared.max().item() <= 1.0


def test_ultralytics_differentiable_letterbox_matches_predict_geometry():
    model, _ = _build_model()
    prepared = torch.ones(1, 3, 300, 640)

    letterboxed = model._differentiable_letterbox(prepared)

    assert letterboxed.shape == (1, 3, 320, 640)
    assert torch.allclose(
        letterboxed[:, :, :10, :],
        torch.full_like(letterboxed[:, :, :10, :], 114 / 255),
    )
    assert torch.all(letterboxed[:, :, 10:310, :] == 1.0)
    assert torch.allclose(
        letterboxed[:, :, 310:, :],
        torch.full_like(letterboxed[:, :, 310:, :], 114 / 255),
    )


def test_ultralytics_end_to_end_objective_uses_one_to_one_class_scores():
    probabilities = torch.tensor(
        [[[0.2, 0.6], [0.4, 0.1]]],
        dtype=torch.float32,
    )
    scores = torch.logit(probabilities).requires_grad_(True)
    output = (
        torch.zeros(1, 2, 6),
        {
            "one2many": {},
            "one2one": {"scores": scores},
        },
    )

    objective = UltralyticsDetectionModel._end_to_end_confidence_objective(output)

    assert objective.item() == pytest.approx(0.5)
    assert objective.requires_grad is True


def test_ultralytics_confidence_objective_normalizes_inference_mode_geometry_cache():
    model, torch_model = _build_model()
    cache_holder = torch.nn.Module()

    with torch.inference_mode():
        cache_holder.anchors = torch.ones(2, 4)
        cache_holder.strides = torch.ones(1, 4)

    torch_model.inference_cache = cache_holder
    assert torch.is_inference(cache_holder.anchors)
    assert torch.is_inference(cache_holder.strides)

    image = Image.new("RGB", (640, 300), color=(192, 192, 192))
    prepared = model.prepare_detection_attack_input(image).requires_grad_(True)

    objective = model.detection_confidence_objective(prepared)
    (gradient,) = torch.autograd.grad(objective, prepared)

    assert not torch.is_inference(cache_holder.anchors)
    assert not torch.is_inference(cache_holder.strides)
    assert torch.isfinite(gradient).all()


def test_ultralytics_end_to_end_objective_propagates_gradient_to_source_image():
    model, torch_model = _build_model()
    image = Image.new("RGB", (640, 300), color=(192, 192, 192))
    prepared = model.prepare_detection_attack_input(image).requires_grad_(True)

    objective = model.detection_confidence_objective(prepared)

    assert objective.ndim == 0
    assert objective.requires_grad is True
    assert torch_model.last_input_shape == (1, 3, 320, 640)

    objective.backward()

    assert prepared.grad is not None
    assert torch.isfinite(prepared.grad).all()
    assert torch.count_nonzero(prepared.grad) > 0


def test_ultralytics_confidence_objective_requires_end_to_end_head():
    model, _ = _build_model(end_to_end=False)
    prepared = torch.zeros(1, 3, 320, 640)

    with pytest.raises(
        NotImplementedError,
        match="does not use an end-to-end detection head",
    ):
        model.detection_confidence_objective(prepared)


@pytest.mark.parametrize(
    ("output", "message"),
    [
        (
            torch.zeros(1, 300, 6),
            "Unexpected Ultralytics end-to-end",
        ),
        (
            (torch.zeros(1, 300, 6), {}),
            "Unexpected Ultralytics end-to-end",
        ),
        (
            (torch.zeros(1, 300, 6), {"one2one": {}}),
            "Unexpected Ultralytics one-to-one score",
        ),
        (
            (
                torch.zeros(1, 300, 6),
                {"one2one": {"scores": torch.zeros(1, 80)}},
            ),
            "Unexpected Ultralytics one-to-one score",
        ),
    ],
)
def test_ultralytics_confidence_objective_rejects_invalid_provider_output(
    output,
    message,
):
    with pytest.raises(ValueError, match=message):
        UltralyticsDetectionModel._end_to_end_confidence_objective(output)
