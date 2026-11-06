"""Contrato A3: referencias fijas, geometría fuente y pérdidas de cajas RF-DETR."""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from ml.models.detection import Detection, DetectionBox, DetectionResult
from ml.models.detection_localization import (
    DetectionLocalizationReference,
    DetectionLocalizationReferences,
)
from ml.models.rf_detr.catalog import build_rf_detr_class_catalog
from ml.models.rf_detr.config import RfDetrDetectionModelConfig
from ml.models.rf_detr.model import RfDetrDetectionModel


class _Processor:
    do_resize = True
    do_rescale = True
    do_normalize = True
    size = {"height": 16, "width": 16}
    image_mean = [0.0, 0.0, 0.0]
    image_std = [1.0, 1.0, 1.0]


class _DifferentiableProvider(torch.nn.Module):
    """Emula el contrato de pérdidas, no el matcher científico de RF-DETR."""

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.config = SimpleNamespace(
            id2label={0: "N/A", 1: "person", 3: "car"},
            num_labels=4,
            bbox_loss_coefficient=5.0,
            giou_loss_coefficient=2.0,
        )
        self.last_labels = None
        self.last_pixels = None
        self.loss_mode = "normal"

    def forward(self, *, pixel_values, pixel_mask, labels):
        self.last_pixels = pixel_values
        self.last_labels = labels
        target = labels[0]["boxes"]
        signal = pixel_values.mean() * self.scale
        # Predicción artificial: depende de TODOS los píxeles fuente.
        predicted = signal * torch.ones_like(target) * 0.25
        loss_bbox = (predicted - target).abs().sum()
        loss_giou = (predicted - target).square().sum()
        if self.loss_mode == "no_bbox":
            return SimpleNamespace(loss_dict={"loss_giou": loss_giou})
        if self.loss_mode == "nan":
            loss_giou = loss_giou * float("nan")
        if self.loss_mode == "detached":
            loss_bbox = loss_bbox.detach()
            loss_giou = loss_giou.detach()
        if self.loss_mode == "detached_giou":
            loss_giou = loss_giou.detach()
        return SimpleNamespace(
            loss_dict={
                "loss_bbox": loss_bbox,
                "loss_giou": loss_giou,
                "loss_ce": signal * 1000,
                "loss_bbox_0": signal * 1000,
                "loss_giou_enc": signal * 1000,
            }
        )


def _model() -> tuple[RfDetrDetectionModel, _DifferentiableProvider]:
    model = RfDetrDetectionModel(
        RfDetrDetectionModelConfig(
            id="rf_detr_nano",
            name="RF-DETR synthetic",
            revision="test-1",
            checkpoint="synthetic/weights",
            checkpoint_revision="rev",
        )
    )
    provider = _DifferentiableProvider().eval()
    model.model = provider
    model.image_processor = _Processor()
    model._id2label = provider.config.id2label
    model._class_catalog = build_rf_detr_class_catalog(
        model_id=model.id, model_name=model.name, id2label=model._id2label
    )
    return model, provider


def _references() -> DetectionLocalizationReferences:
    return DetectionLocalizationReferences.from_supervised_objects(
        source_width=12,
        source_height=8,
        objects=(
            (1, DetectionBox(3, 1, 9, 7)),
            (3, DetectionBox(0, 0, 3, 2)),
        ),
    )


def test_supervised_references_normalize_non_square_source_geometry():
    references = _references()
    labels = RfDetrDetectionModel._localization_target_labels(
        references, device=torch.device("cpu"), dtype=torch.float32
    )
    assert references.source == "supervised"
    assert labels[0]["class_labels"].dtype == torch.long
    assert labels[0]["class_labels"].tolist() == [1, 3]
    assert labels[0]["boxes"].shape == (2, 4)
    assert torch.allclose(
        labels[0]["boxes"],
        torch.tensor([[0.5, 0.5, 0.5, 0.75], [0.125, 0.125, 0.25, 0.25]]),
    )
    assert not labels[0]["boxes"].requires_grad


def test_pseudo_references_are_filtered_and_frozen_from_clean_predictions():
    original = DetectionResult(
        source_width=12,
        source_height=8,
        detections=(
            Detection(class_id=1, confidence=0.9, box=DetectionBox(3, 1, 9, 7)),
            Detection(class_id=3, confidence=0.2, box=DetectionBox(1, 1, 4, 4)),
        ),
    )
    references = DetectionLocalizationReferences.from_detections(
        original, min_confidence=0.5
    )
    assert references.source == "pseudo"
    assert references.objects == (
        DetectionLocalizationReference(1, DetectionBox(3, 1, 9, 7)),
    )
    with pytest.raises(FrozenInstanceError):
        references.objects = ()


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), -0.1, 1.1, True])
def test_pseudo_references_reject_invalid_confidence(threshold):
    with pytest.raises(ValueError, match="min_confidence"):
        DetectionLocalizationReferences.from_detections(
            DetectionResult(source_width=12, source_height=8),
            min_confidence=threshold,
        )


def test_references_reject_invalid_class_ids_and_out_of_bounds_boxes():
    with pytest.raises(TypeError, match="class_id"):
        DetectionLocalizationReference(True, DetectionBox(1, 1, 3, 3))
    with pytest.raises(ValueError, match="geometry"):
        DetectionLocalizationReferences.from_supervised_objects(
            source_width=12,
            source_height=8,
            objects=((1, DetectionBox(1, 1, 13, 7)),),
        )


def test_localization_objective_propagates_gradient_without_updating_weights():
    model, provider = _model()
    prepared = model.prepare_detection_attack_input(
        Image.new("RGB", (12, 8), color=(192, 192, 192))
    ).requires_grad_(True)
    refs = _references()
    snapshot = [parameter.clone().detach() for parameter in provider.parameters()]
    original_refs = refs.objects

    objective = model.detection_localization_objective(prepared, refs)
    (gradient,) = torch.autograd.grad(objective, prepared)

    # Finales L1/GIoU del proveedor; CE y auxiliares quedan excluidos.
    assert objective.ndim == 0
    assert objective.detach().item() == pytest.approx(
        (5 * provider.last_labels[0]["boxes"].sub(provider.last_pixels.mean() * 0.25)
         .abs().sum() +
         2 * provider.last_labels[0]["boxes"].sub(provider.last_pixels.mean() * 0.25)
         .square().sum()).item(),
    )
    assert provider.training is False
    assert provider.last_pixels.shape == (1, 3, 16, 16)
    assert provider.last_labels[0]["class_labels"].tolist() == [1, 3]
    assert gradient.shape == prepared.shape
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0
    assert refs.objects == original_refs
    for before, parameter in zip(snapshot, provider.parameters()):
        assert torch.equal(before, parameter.detach())
        assert parameter.grad is None


def test_localization_objective_rejects_empty_references_and_mismatched_geometry():
    model, _ = _model()
    tensor = torch.rand((1, 3, 8, 12), requires_grad=True)
    empty = DetectionLocalizationReferences.from_supervised_objects(
        source_width=12, source_height=8, objects=()
    )
    with pytest.raises(ValueError, match="at least one"):
        model.detection_localization_objective(tensor, empty)
    with pytest.raises(ValueError, match="geometry"):
        model.detection_localization_objective(tensor, DetectionLocalizationReferences(
            source_width=6, source_height=8, objects=(
                DetectionLocalizationReference(1, DetectionBox(1, 1, 4, 5)),
            ), source="supervised"
        ))


def test_localization_objective_rejects_unknown_and_placeholder_classes():
    model, _ = _model()
    tensor = torch.rand((1, 3, 8, 12), requires_grad=True)
    for class_id in (0, 2, 4):
        refs = DetectionLocalizationReferences.from_supervised_objects(
            source_width=12, source_height=8,
            objects=((class_id, DetectionBox(1, 1, 4, 5)),),
        )
        with pytest.raises(ValueError, match="invalid model class"):
            model.detection_localization_objective(tensor, refs)


@pytest.mark.parametrize(
    ("loss_mode", "message"),
    [
        ("no_bbox", "loss_bbox"),
        ("nan", "finite"),
        ("detached", "not differentiable"),
        ("detached_giou", "not differentiable"),
    ],
)
def test_localization_objective_rejects_invalid_provider_losses(loss_mode, message):
    model, provider = _model()
    provider.loss_mode = loss_mode
    tensor = torch.rand((1, 3, 8, 12), requires_grad=True)
    with pytest.raises((ValueError, RuntimeError), match=message):
        model.detection_localization_objective(tensor, _references())


def test_gradient_ascent_increases_synthetic_localization_error():
    model, provider = _model()
    original = model.prepare_detection_attack_input(
        Image.new("RGB", (12, 8), color=(192, 192, 192))
    ).requires_grad_(True)
    references = _references()

    initial = model.detection_localization_objective(original, references)
    (gradient,) = torch.autograd.grad(initial, original)
    candidate = (original.detach() + 0.005 * gradient.sign()).clamp(0, 1)
    candidate.requires_grad_(True)
    after = model.detection_localization_objective(candidate, references)

    assert after.item() > initial.item()
    assert provider.scale.grad is None
    assert references.objects == _references().objects
