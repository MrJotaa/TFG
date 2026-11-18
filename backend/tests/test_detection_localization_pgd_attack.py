"""A4: invariantes matemáticos, catálogo y ejecución por imagen de PGD localización."""

from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

from core.cancellation import CancellationToken, OperationCancelled
from core.errors import BadRequestError
from core.task_compatibility import require_attack_model_compatibility
from ml.attacks.detection_localization_pgd import DetectionLocalizationPGDAttack
from ml.attacks.registry import create_attack_runner, get_attack, resolve_attack_parameters
from ml.models.detection import Detection, DetectionBox, DetectionResult
from schemas.attack import AttackParameters, AttackType


class _NonFiniteGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        ctx.shape = value.shape
        return value.mean()

    @staticmethod
    def backward(ctx, gradient):
        return torch.full(ctx.shape, float("inf"), device=gradient.device)


class _FakeRFDETR:
    id = "rf_detr_nano"
    task_type = "detection"

    def __init__(self, *, detections=True, gradient_mode="normal"):
        self.loaded = True
        self.detect_calls = 0
        self.reference_snapshots = []
        self.input_snapshots = []
        self.gradient_mode = gradient_mode
        self.detections = detections
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.cancel_token = None

    def is_loaded(self):
        return self.loaded

    def prepare_detection_attack_input(self, image):
        return to_tensor(image.convert("RGB")).unsqueeze(0)

    def detect(self, image):
        self.detect_calls += 1
        detections = (
            (Detection(1, 0.9, DetectionBox(1, 1, image.width - 1, image.height - 1)),)
            if self.detections else ()
        )
        return DetectionResult(image.width, image.height, detections)

    def detection_localization_objective(self, prepared_input, references):
        self.reference_snapshots.append(references)
        self.input_snapshots.append(prepared_input.detach().clone())
        if self.cancel_token is not None:
            self.cancel_token.cancel()
        if self.gradient_mode == "missing":
            return prepared_input.new_tensor(1.0, requires_grad=True)
        if self.gradient_mode == "nan":
            return prepared_input.mean() * float("nan")
        if self.gradient_mode == "nonfinite_gradient":
            # Gradiente infinito con valor forward finito: caso distinto de NaN loss.
            return _NonFiniteGrad.apply(prepared_input)
        # Ascender implica reducir el valor medio, sin gradiente en los pesos.
        return (1.0 - prepared_input.mean()) * self.scale


def _img(value=128):
    return Image.new("RGB", (12, 8), (value, value, value))


def _params(**kw):
    return SimpleNamespace(epsilon=0.03, alpha=0.01, steps=3, **kw)


def _run(wrapper=None, *, image=None, params=None):
    wrapper = wrapper or _FakeRFDETR()
    return DetectionLocalizationPGDAttack().apply(
        wrapper,
        image or _img(),
        params or _params(),
    )


def test_attack_registry_catalog_is_restricted_to_rfdetr():
    attack = get_attack(AttackType.DETECTION_LOCALIZATION_PGD)
    assert attack.supported_model_ids == ("rf_detr_nano",)
    assert attack.supported_task_types == ("detection",)
    assert attack.revision == "rf-detr-localization-pgd-linf-pseudo-v1"
    assert attack.knowledge_type == "white_box"
    assert attack.requires_exclusive_model_execution
    assert attack.catalog_entry()["supported_model_ids"] == ["rf_detr_nano"]
    require_attack_model_compatibility(attack, _FakeRFDETR())
    with pytest.raises(BadRequestError, match="yolo26n"):
        require_attack_model_compatibility(
            attack, SimpleNamespace(id="yolo26n", task_type="detection")
        )
    resolved = resolve_attack_parameters(attack, AttackParameters())
    assert resolved.model_dump() == {"epsilon": 0.03, "alpha": 0.01, "steps": 10}


def test_attack_increases_synthetic_loss_and_keeps_pseudorefs_fixed():
    model = _FakeRFDETR()
    result = _run(model)
    assert model.detect_calls == 1
    assert len(model.reference_snapshots) == 3
    assert all(ref is model.reference_snapshots[0] for ref in model.reference_snapshots)
    assert model.reference_snapshots[0].source == "pseudo"
    assert len(model.reference_snapshots[0].objects) == 1
    assert model.input_snapshots[1].mean() < model.input_snapshots[0].mean()
    assert result["adversarial_image"].size == (12, 8)
    assert result["perturbation_image"].size == (12, 8)
    assert result["perturbation_linf"] > 0
    assert model.scale.grad is None


@pytest.mark.parametrize("epsilon", [0.0, 0.001, 0.01, 0.03, 0.3])
def test_saved_rgb_respects_exact_linf_budget(epsilon):
    model = _FakeRFDETR()
    image = _img()
    attack = DetectionLocalizationPGDAttack()
    result = attack.apply(model, image, SimpleNamespace(epsilon=epsilon, alpha=0.1, steps=4))
    clean = to_tensor(image)
    adversarial = to_tensor(result["adversarial_image"])
    delta = (adversarial - clean).abs()
    assert delta.max().item() <= epsilon + 1e-7
    assert result["perturbation_linf"] <= epsilon + 1e-4
    assert result["perturbation_l2"] == pytest.approx(torch.linalg.vector_norm(delta).item(), abs=5e-5)
    if epsilon == 0.0:
        assert result["adversarial_image"].tobytes() == image.tobytes()
        assert model.detect_calls == 0


def test_no_detections_preserves_image_and_never_computes_gradient():
    model = _FakeRFDETR(detections=False)
    image = _img()
    result = _run(model, image=image)
    assert result["adversarial_image"].tobytes() == image.tobytes()
    assert result["perturbation_linf"] == 0.0
    assert result["perturbation_l2"] == 0.0
    assert model.detect_calls == 1
    assert model.reference_snapshots == []


def test_repeated_calls_have_independent_references_and_deterministic_output():
    model = _FakeRFDETR()
    attack = DetectionLocalizationPGDAttack()
    a = attack.apply(model, _img(100), _params())
    b = attack.apply(model, _img(100), _params())
    assert a["adversarial_image"].tobytes() == b["adversarial_image"].tobytes()
    assert model.reference_snapshots[0] is not model.reference_snapshots[3]
    assert model.detect_calls == 2


def test_rejects_unloaded_or_incompatible_model():
    model = _FakeRFDETR()
    model.loaded = False
    with pytest.raises(ValueError, match="Model is not loaded"):
        _run(model)
    model.id = "yolo26n"
    with pytest.raises(ValueError, match="no es compatible"):
        _run(model)


@pytest.mark.parametrize("mode,exception_pattern", [
    ("missing", "no input gradient"),
    ("nan", "must be finite"),
    ("nonfinite_gradient", "gradient must be finite"),
])
def test_invalid_objective_or_gradient_is_rejected(mode, exception_pattern):
    model = _FakeRFDETR(gradient_mode=mode)
    with pytest.raises((ValueError, RuntimeError), match=exception_pattern):
        _run(model)


def test_cancellation_before_first_step_and_between_steps():
    attack = DetectionLocalizationPGDAttack()
    model = _FakeRFDETR()
    token = CancellationToken()
    token.cancel()
    runner = create_attack_runner(attack, model, _params(), cancellation_token=token)
    assert runner.requires_exclusive_model_execution
    with pytest.raises(OperationCancelled):
        runner.apply(model, _img(), _params())
    assert model.detect_calls == 0

    token2 = CancellationToken()
    model2 = _FakeRFDETR()
    model2.cancel_token = token2
    runner2 = create_attack_runner(attack, model2, _params(), cancellation_token=token2)
    with pytest.raises(OperationCancelled):
        runner2.apply(model2, _img(), _params())
    assert len(model2.reference_snapshots) == 1
    assert model.scale.grad is None


def test_runner_does_not_store_cancellation_on_singleton():
    attack = DetectionLocalizationPGDAttack()
    token = CancellationToken()
    runner1 = create_attack_runner(attack, _FakeRFDETR(), _params(), token)
    runner2 = create_attack_runner(attack, _FakeRFDETR(), _params())
    assert runner1 is not runner2
    assert runner1.cancellation_token is token
    assert runner2.cancellation_token is None
    assert not hasattr(attack, "cancellation_token")
