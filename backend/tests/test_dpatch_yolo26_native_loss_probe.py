"""B2 exploratory gate: native YOLO26n loss and ART DPatch, without production changes.

This probe deliberately uses a fixed synthetic reference, not a claim of attack
success. Run with RUN_DPATCH_YOLO26_REAL_SMOKE=1 and the real_model marker.
"""

import os
from collections.abc import Mapping
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from art.attacks.evasion import DPatch
from art.estimators.estimator import BaseEstimator, LossGradientsMixin
from art.estimators.object_detection.object_detector import ObjectDetectorMixin

from ml.models.detection_inference import DetectionEvaluationInferenceConfiguration


def _yolo_batch_from_art_targets(targets, *, wrapper, model_input, source_shape):
    """Map source xyxy boxes to normalized cxcywh in letterboxed model coordinates."""
    _, _, input_height, input_width = model_input.shape
    source_height, source_width = source_shape
    resized_h, resized_w, top, _bottom, left, _right = wrapper._letterbox_geometry(
        source_height=source_height, source_width=source_width
    )
    scale_x = resized_w / source_width
    scale_y = resized_h / source_height
    device = model_input.device
    dtype = model_input.dtype
    all_boxes = []
    all_classes = []
    all_indices = []
    for batch_index, target in enumerate(targets):
        boxes = np.asarray(target["boxes"], dtype=np.float32).reshape(-1, 4)
        labels = np.asarray(target["labels"], dtype=np.int64).reshape(-1)
        if len(boxes) != len(labels):
            raise ValueError("Every bounding box requires one label")
        for (x1, y1, x2, y2), label in zip(boxes, labels):
            x1m, x2m = x1 * scale_x + left, x2 * scale_x + left
            y1m, y2m = y1 * scale_y + top, y2 * scale_y + top
            all_boxes.append(
                [
                    (x1m + x2m) / (2 * input_width),
                    (y1m + y2m) / (2 * input_height),
                    (x2m - x1m) / input_width,
                    (y2m - y1m) / input_height,
                ]
            )
            all_classes.append(int(label))
            all_indices.append(batch_index)
    return {
        "img": model_input,
        "batch_idx": torch.tensor(all_indices, dtype=torch.long, device=device),
        "cls": torch.tensor(all_classes, dtype=dtype, device=device).reshape(-1, 1),
        "bboxes": torch.tensor(all_boxes, dtype=dtype, device=device).reshape(-1, 4),
    }


def _yolo26_one_to_one_loss(predictions, *, criterion, batch):
    """Use the native loss of the deployed YOLO26 one-to-one detection head.

    An inference-optimized head may lack one-to-many weights (cv2/cv3=None),
    leaving ``predictions[1]["one2many"]`` empty. The one-to-one head remains
    available and is the same objective regardless of whether fusion occurred.
    """
    raw = predictions[1] if isinstance(predictions, tuple) else predictions
    one2one = raw.get("one2one") if isinstance(raw, dict) else None
    if not isinstance(one2one, dict) or not all(
        key in one2one for key in ("boxes", "scores", "feats")
    ):
        raise ValueError("YOLO26 did not return differentiable one-to-one predictions")

    components, _details = criterion.loss(one2one, batch)
    if not isinstance(components, torch.Tensor) or components.ndim > 1:
        raise TypeError("YOLO26 one-to-one loss must return tensor components")
    return components.sum()


def _prepare_one_to_one_loss_hyp(criterion, *, defaults=None):
    """Resolve native loss gains from checkpoint options and Ultralytics defaults.

    Inference checkpoints often lack training-only ``box``, ``cls`` and ``dfl``
    options. Use the installed Ultralytics configuration as fallback, rather
    than inventing weights or modifying the detector's ``model.args``.
    Only this probe's criterion receives the completed configuration.
    """
    if defaults is None:
        from ultralytics.utils import DEFAULT_CFG_DICT

        defaults = DEFAULT_CFG_DICT
    hyp = criterion.hyp
    required = ("box", "cls", "dfl")
    original = (
        dict(hyp)
        if isinstance(hyp, Mapping)
        else {key: getattr(hyp, key) for key in required if hasattr(hyp, key)}
    )
    resolved = {}
    for key in required:
        value = original.get(key)
        if value is None:
            value = defaults.get(key)
        if value is None:
            raise ValueError(f"Missing YOLO26 loss gain in checkpoint and defaults: {key}")
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not np.isfinite(value) or value <= 0):
            raise ValueError(f"Invalid YOLO26 loss gain {key}: {value!r}")
        resolved[key] = value

    if isinstance(hyp, Mapping):
        criterion.hyp = SimpleNamespace(**{**original, **resolved})
    elif any(key not in original or original[key] is None for key in required):
        # Do not mutate an attribute-based configuration shared with the model.
        criterion.hyp = SimpleNamespace(**{**vars(hyp), **resolved})
    return criterion


class _Yolo26ArtProbe(BaseEstimator, LossGradientsMixin, ObjectDetectorMixin):
    """Only a feasibility bridge, intentionally not integrated into production."""

    def __init__(self, wrapper, source_shape):
        self.wrapper = wrapper
        self._source_shape = source_shape
        # Use the native end-to-end head's loss even if one-to-many was fused
        # away for inference. Creating this criterion does not train weights.
        native_criterion = wrapper.get_torch_model().init_criterion()
        self._one2one_criterion = _prepare_one_to_one_loss_hyp(native_criterion.one2one)
        super().__init__(model=wrapper.get_torch_model(), clip_values=(0.0, 1.0))

    @property
    def channels_first(self):
        return True

    @property
    def native_label_is_pytorch_format(self):
        return True

    @property
    def input_shape(self):
        return (3, *self._source_shape)

    def fit(self, x, y, **kwargs):
        raise NotImplementedError("This probe must never train detector weights")

    def predict(self, x, **kwargs):
        results = []
        configuration = DetectionEvaluationInferenceConfiguration(score_floor=0.001)
        for item in x:
            rgb = np.transpose(item, (1, 2, 0))
            image = Image.fromarray(np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8), "RGB")
            prediction = self.wrapper.detect_for_evaluation(image, configuration)
            results.append(
                {
                    "boxes": np.array(
                        [[d.box.x_min, d.box.y_min, d.box.x_max, d.box.y_max]
                         for d in prediction.detections], dtype=np.float32
                    ).reshape(-1, 4),
                    "labels": np.array([d.class_id for d in prediction.detections], dtype=np.int64),
                    "scores": np.array([d.confidence for d in prediction.detections], dtype=np.float32),
                }
            )
        return results

    def loss_gradient(self, x, y, **kwargs):
        if x.ndim != 4 or x.shape[1:] != self.input_shape:
            raise ValueError("Expected one or more source RGB CHW images")
        device, dtype = self.wrapper._torch_model_device_and_dtype()
        source = torch.as_tensor(
            np.ascontiguousarray(x), device=device, dtype=dtype
        ).detach().requires_grad_(True)
        model = self.wrapper.get_torch_model()
        if model.training:
            raise AssertionError("Probe requires YOLO in eval mode")
        self.wrapper._normalize_detection_geometry_cache_for_autograd(model)
        model_input = self.wrapper._differentiable_letterbox(source)
        batch = _yolo_batch_from_art_targets(
            y, wrapper=self.wrapper, model_input=model_input, source_shape=self._source_shape
        )
        predictions = model.forward(model_input)
        loss = _yolo26_one_to_one_loss(
            predictions, criterion=self._one2one_criterion, batch=batch
        )
        if loss.ndim != 0 or not loss.requires_grad or not torch.isfinite(loss):
            raise AssertionError("Native YOLO26 loss must be a finite differentiable scalar")
        (gradient,) = torch.autograd.grad(loss, source)
        if gradient is None or not torch.isfinite(gradient).all():
            raise AssertionError("Gradient to source pixels is missing or nonfinite")
        return gradient.detach().cpu().numpy().astype(np.float32)


def test_yolo26_art_targets_follow_letterbox_geometry():
    class _GeometryStub:
        @staticmethod
        def _letterbox_geometry(*, source_height, source_width):
            assert (source_height, source_width) == (100, 200)
            return 100, 200, 50, 50, 0, 0

    dummy_input = torch.zeros((1, 3, 200, 200), dtype=torch.float32)
    refs = [{
        "boxes": np.array([[50, 25, 150, 75]], dtype=np.float32),
        "labels": np.array([7], dtype=np.int64),
    }]
    batch = _yolo_batch_from_art_targets(
        refs, wrapper=_GeometryStub(), model_input=dummy_input, source_shape=(100, 200)
    )
    torch.testing.assert_close(batch["bboxes"], torch.tensor([[0.5, 0.5, 0.5, 0.25]]))
    torch.testing.assert_close(batch["cls"], torch.tensor([[7.0]]))
    torch.testing.assert_close(batch["batch_idx"], torch.tensor([0]))


def test_one_to_one_loss_gains_use_checkpoint_values_without_mutating_args():
    args = {"box": 7.5, "cls": 0.5, "dfl": 1.5, "extra": "unchanged"}
    criterion = SimpleNamespace(hyp=args)

    assert _prepare_one_to_one_loss_hyp(criterion) is criterion
    assert (criterion.hyp.box, criterion.hyp.cls, criterion.hyp.dfl) == (7.5, 0.5, 1.5)
    assert criterion.hyp.extra == "unchanged"
    assert criterion.hyp is not args
    assert args == {"box": 7.5, "cls": 0.5, "dfl": 1.5, "extra": "unchanged"}


def test_one_to_one_loss_gains_accept_attribute_configuration_unchanged():
    hyp = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
    criterion = SimpleNamespace(hyp=hyp)
    _prepare_one_to_one_loss_hyp(criterion)
    assert criterion.hyp is hyp


def test_one_to_one_loss_gains_fall_back_to_official_defaults_without_mutation():
    # A deployed checkpoint can omit all training-only loss coefficients.
    checkpoint_args = {"task": "detect"}
    official_defaults = {"box": 7.5, "cls": 0.5, "dfl": 1.5}
    criterion = SimpleNamespace(hyp=checkpoint_args)
    _prepare_one_to_one_loss_hyp(criterion, defaults=official_defaults)
    assert (criterion.hyp.box, criterion.hyp.cls, criterion.hyp.dfl) == (7.5, 0.5, 1.5)
    assert criterion.hyp.task == "detect"
    assert checkpoint_args == {"task": "detect"}


def test_one_to_one_loss_gains_use_installed_ultralytics_defaults():
    from ultralytics.utils import DEFAULT_CFG_DICT

    criterion = SimpleNamespace(hyp={})
    _prepare_one_to_one_loss_hyp(criterion)
    assert (criterion.hyp.box, criterion.hyp.cls, criterion.hyp.dfl) == tuple(
        DEFAULT_CFG_DICT[key] for key in ("box", "cls", "dfl")
    )


def test_one_to_one_loss_gains_prioritize_checkpoint_over_defaults():
    checkpoint_args = {"box": 3.25, "cls": 0.75}
    criterion = SimpleNamespace(hyp=checkpoint_args)
    _prepare_one_to_one_loss_hyp(
        criterion, defaults={"box": 7.5, "cls": 0.5, "dfl": 1.5}
    )
    assert (criterion.hyp.box, criterion.hyp.cls, criterion.hyp.dfl) == (3.25, 0.75, 1.5)
    assert checkpoint_args == {"box": 3.25, "cls": 0.75}


def test_one_to_one_loss_gains_can_fill_attribute_configuration():
    checkpoint_hyp = SimpleNamespace(cls=0.8, extra="unchanged")
    criterion = SimpleNamespace(hyp=checkpoint_hyp)
    _prepare_one_to_one_loss_hyp(
        criterion, defaults={"box": 7.5, "cls": 0.5, "dfl": 1.5}
    )
    assert (criterion.hyp.box, criterion.hyp.cls, criterion.hyp.dfl) == (7.5, 0.8, 1.5)
    assert criterion.hyp.extra == "unchanged"
    assert vars(checkpoint_hyp) == {"cls": 0.8, "extra": "unchanged"}


@pytest.mark.parametrize(
    ("checkpoint", "defaults", "message"),
    [
        ({}, {"box": 7.5, "cls": 0.5}, "Missing YOLO26 loss gain"),
        ({"box": -1}, {"box": 7.5, "cls": 0.5, "dfl": 1.5}, "Invalid YOLO26 loss gain"),
        ({"cls": float("nan")}, {"box": 7.5, "cls": 0.5, "dfl": 1.5}, "Invalid YOLO26 loss gain"),
    ],
)
def test_one_to_one_loss_gains_reject_missing_or_invalid_configuration(checkpoint, defaults, message):
    criterion = SimpleNamespace(hyp=checkpoint)
    with pytest.raises(ValueError, match=message):
        _prepare_one_to_one_loss_hyp(criterion, defaults=defaults)
    assert criterion.hyp is checkpoint


@pytest.mark.parametrize("one2many", [{}, {"boxes": torch.tensor(0.0)}])
def test_one_to_one_loss_works_with_fused_and_unfused_heads(one2many):
    pixel = torch.tensor(2.0, requires_grad=True)
    one2one = {
        "boxes": pixel,
        "scores": torch.tensor(1.0),
        "feats": [torch.tensor(1.0)],
    }

    class _OneToOneCriterion:
        @staticmethod
        def loss(predictions, batch):
            assert predictions is one2one
            assert batch == {"test": True}
            return torch.stack((predictions["boxes"], 2 * predictions["boxes"])), {}

    result = _yolo26_one_to_one_loss(
        (None, {"one2many": one2many, "one2one": one2one}),
        criterion=_OneToOneCriterion(),
        batch={"test": True},
    )
    assert result.ndim == 0
    torch.testing.assert_close(result, torch.tensor(6.0))
    (gradient,) = torch.autograd.grad(result, pixel)
    torch.testing.assert_close(gradient, torch.tensor(3.0))


def test_one_to_one_loss_requires_raw_detection_predictions():
    with pytest.raises(ValueError, match="one-to-one predictions"):
        _yolo26_one_to_one_loss(
            (None, {"one2many": {}}), criterion=object(), batch={}
        )


def _snapshot_model_parameters(model):
    """Capture post-inference parameters by name, not by list position.

    Ultralytics may fuse Conv/BN during its first public prediction, changing
    parameter values and enumeration. This snapshot must be taken *after* that
    legitimate initialization and *before* any adversarial gradients.
    """
    return {
        name: (parameter, parameter.detach().clone(), parameter.grad, parameter.requires_grad)
        for name, parameter in model.named_parameters()
    }


def _assert_model_parameters_unchanged(model, snapshot):
    """Assert the attack preserves all parameter objects, values and grad state."""
    current = dict(model.named_parameters())
    assert current.keys() == snapshot.keys(), "The attack changed the model parameter set"
    for name, (original, values, prior_grad, requires_grad) in snapshot.items():
        parameter = current[name]
        assert parameter is original, f"The attack replaced parameter {name}"
        assert torch.equal(parameter.detach(), values), f"The attack changed parameter {name}"
        assert parameter.grad is prior_grad, f"The attack changed gradient state of {name}"
        assert parameter.requires_grad == requires_grad, (
            f"The attack changed requires_grad of {name}"
        )


def test_attack_parameter_snapshot_detects_accidental_parameter_updates():
    model = torch.nn.Linear(3, 2)
    snapshot = _snapshot_model_parameters(model)
    _assert_model_parameters_unchanged(model, snapshot)
    with torch.no_grad():
        model.weight.add_(1.0)
    with pytest.raises(AssertionError, match="The attack changed parameter weight"):
        _assert_model_parameters_unchanged(model, snapshot)


def test_attack_parameter_snapshot_detects_parameter_replacement():
    model = torch.nn.Linear(3, 2)
    snapshot = _snapshot_model_parameters(model)
    model.weight = torch.nn.Parameter(model.weight.detach().clone())
    with pytest.raises(AssertionError, match="The attack replaced parameter weight"):
        _assert_model_parameters_unchanged(model, snapshot)


@pytest.mark.real_model
@pytest.mark.skipif(
    os.getenv("RUN_DPATCH_YOLO26_REAL_SMOKE") != "1",
    reason="Requires explicit YOLO26 DPatch native-loss feasibility run",
)
def test_yolo26_native_loss_supports_art_dpatch_without_weight_updates():
    """Execute ART DPatch.generate once with native YOLO loss and fixed test targets."""
    from ml.models.definitions import build_model_registry

    wrapper = build_model_registry().get_model("yolo26n")
    wrapper.load()
    try:
        torch_model = wrapper.get_torch_model()
        torch_model.eval()
        source_height, source_width = 96, 128
        estimator = _Yolo26ArtProbe(wrapper, (source_height, source_width))
        x = np.full((1, 3, source_height, source_width), 0.5, dtype=np.float32)
        # Fixed synthetic references exercise the API; they are NOT COCO annotations.
        target = [{
            "boxes": np.array([[32.0, 24.0, 96.0, 72.0]], dtype=np.float32),
            "labels": np.array([0], dtype=np.int64),
            "scores": np.array([1.0], dtype=np.float32),
        }]
        # The first public prediction may fuse Conv/BN and remove auxiliary
        # detection-head parameters. That optimization is not an attack update.
        preds = estimator.predict(x)
        assert len(preds) == 1
        assert set(preds[0]) == {"boxes", "labels", "scores"}
        assert wrapper.get_torch_model() is torch_model
        weights_after_inference = _snapshot_model_parameters(torch_model)
        grad = estimator.loss_gradient(x, target)
        assert grad.shape == x.shape
        assert np.isfinite(grad).all()
        assert np.count_nonzero(grad) > 0
        _assert_model_parameters_unchanged(torch_model, weights_after_inference)

        attack = DPatch(
            estimator=estimator,
            patch_shape=(3, 12, 12),
            learning_rate=0.02,
            max_iter=1,
            batch_size=1,
            verbose=False,
        )
        patch = attack.generate(x, y=target)
        _assert_model_parameters_unchanged(torch_model, weights_after_inference)
        result = attack.apply_patch(x, random_location=False)
        assert patch.shape == (3, 12, 12)
        assert np.isfinite(patch).all()
        assert np.all((0 <= patch) & (patch <= 1))
        assert result.shape == x.shape
        assert np.isfinite(result).all()
        assert np.array_equal(result[0, :, 12:, :], x[0, :, 12:, :])
        assert np.array_equal(result[0, :, :12, 12:], x[0, :, :12, 12:])
        assert torch_model.training is False
        _assert_model_parameters_unchanged(torch_model, weights_after_inference)
    finally:
        wrapper.unload()
