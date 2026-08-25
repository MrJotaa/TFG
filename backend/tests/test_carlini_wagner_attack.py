import numpy as np
import pytest
import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

import ml.attacks.carlini_wagner as carlini_wagner_module
from ml.attacks.base import (
    AdversarialAttackResult,
    AdversarialEvaluationResult,
)
from ml.attacks.carlini_wagner import CarliniWagnerAttack
from schemas.attack import AttackParameters


class TinyClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        score = inputs.mean(dim=(1, 2, 3)) * self.scale
        return torch.stack((score, -score, score * 0.5), dim=1)


class FakeClassificationModel:
    id = "tiny-classifier"
    task_type = "classification"

    def __init__(self) -> None:
        self.model = TinyClassifier()

    def is_loaded(self) -> bool:
        return True

    def get_torch_model(self) -> torch.nn.Module:
        return self.model

    def get_attack_input_shape(self) -> tuple[int, ...]:
        return (3, 4, 4)

    def get_class_labels(self) -> tuple[str, ...]:
        return ("class-0", "class-1", "class-2")

    def prepare_attack_input(self, image: Image.Image) -> torch.Tensor:
        return pil_to_tensor(image).float().div(255).unsqueeze(0)

    def normalize_attack_input(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    def forward_attack_logits(self, prepared_input: torch.Tensor) -> torch.Tensor:
        return self.model(prepared_input)


class FakeARTAttack:
    def __init__(self) -> None:
        self.calls = []

    def generate(self, *, x, y):
        self.calls.append((x.copy(), None if y is None else y.copy()))
        return np.clip(x - 0.01, 0.0, 1.0).astype(np.float32)


def _resolved_parameters(**overrides):
    attack = CarliniWagnerAttack()
    return attack.resolve_parameters(AttackParameters(**overrides))


def test_carlini_wagner_parameter_contract_defaults_to_l2():
    parameters = _resolved_parameters()

    assert parameters.model_dump() == {
        "norm": "l2",
        "confidence": 0.0,
        "learning_rate": 0.01,
        "binary_search_steps": 3,
        "max_iter": 10,
    }


def test_carlini_wagner_linf_drops_l2_only_search_parameter():
    parameters = _resolved_parameters(norm="linf", binary_search_steps=7)

    assert parameters.model_dump() == {
        "norm": "linf",
        "confidence": 0.0,
        "learning_rate": 0.01,
        "max_iter": 10,
    }


def test_carlini_wagner_parameter_contract_rejects_unknown_norm():
    attack = CarliniWagnerAttack()

    with pytest.raises(ValueError, match="debe ser uno de"):
        attack.resolve_parameters(AttackParameters(norm="l1"))


def test_carlini_wagner_builds_l2_with_pinned_art_defaults(monkeypatch):
    captured = {}

    def fake_l2(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(carlini_wagner_module, "CarliniL2Method", fake_l2)

    estimator = object()
    CarliniWagnerAttack().build_art_attack(
        estimator=estimator,
        parameters=_resolved_parameters(
            norm="l2",
            confidence=0.5,
            learning_rate=0.02,
            binary_search_steps=4,
            max_iter=25,
        ),
    )

    assert captured == {
        "classifier": estimator,
        "confidence": 0.5,
        "targeted": False,
        "learning_rate": 0.02,
        "max_iter": 25,
        "batch_size": 1,
        "verbose": False,
        "binary_search_steps": 4,
        "initial_const": 0.01,
        "max_halving": 5,
        "max_doubling": 5,
    }


def test_carlini_wagner_builds_linf_with_pinned_art_defaults(monkeypatch):
    captured = {}

    def fake_linf(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(carlini_wagner_module, "CarliniLInfMethod", fake_linf)

    estimator = object()
    CarliniWagnerAttack().build_art_attack(
        estimator=estimator,
        parameters=_resolved_parameters(norm="linf"),
    )

    assert captured == {
        "classifier": estimator,
        "confidence": 0.0,
        "targeted": False,
        "learning_rate": 0.01,
        "max_iter": 10,
        "batch_size": 1,
        "verbose": False,
        "decrease_factor": 0.9,
        "initial_const": 1e-5,
        "largest_const": 20.0,
        "const_factor": 2.0,
    }


def test_carlini_wagner_runner_reuses_art_attack_and_original_class(monkeypatch):
    attack = CarliniWagnerAttack()
    fake_art_attack = FakeARTAttack()
    build_calls = []

    def fake_build_art_attack(*, estimator, parameters):
        build_calls.append((estimator, parameters.model_dump()))
        return fake_art_attack

    monkeypatch.setattr(attack, "build_art_attack", fake_build_art_attack)

    model = FakeClassificationModel()
    parameters = _resolved_parameters(max_iter=2)
    runner = attack.create_runner(model_wrapper=model, parameters=parameters)
    image = Image.new("RGB", (4, 4), color=(128, 128, 128))

    first = runner.apply(
        model_wrapper=model,
        image=image,
        parameters=parameters,
        original_class_index=1,
    )
    second = runner.apply(
        model_wrapper=model,
        image=image,
        parameters=parameters,
        original_class_index=2,
    )

    assert len(build_calls) == 1
    assert len(fake_art_attack.calls) == 2
    assert fake_art_attack.calls[0][1].dtype == np.int64
    assert fake_art_attack.calls[0][1].tolist() == [1]
    assert fake_art_attack.calls[1][1].tolist() == [2]

    for result in (first, second):
        assert isinstance(result, AdversarialAttackResult)
        assert result.clean_model_input.shape == (1, 3, 4, 4)
        assert result.adversarial_model_input.shape == (1, 3, 4, 4)
        assert torch.allclose(
            result.model_space_delta,
            result.adversarial_model_input - result.clean_model_input,
        )
        assert result.perturbation_linf == pytest.approx(0.01, abs=1e-4)
        assert result.adversarial_image.size == (4, 4)
        assert result.perturbation_image.size == (4, 4)


def test_carlini_wagner_evaluation_runner_reuses_prepared_input(monkeypatch):
    attack = CarliniWagnerAttack()
    monkeypatch.setattr(
        attack,
        "build_art_attack",
        lambda **kwargs: FakeARTAttack(),
    )

    model = FakeClassificationModel()
    parameters = _resolved_parameters(max_iter=2)
    runner = attack.create_runner(model_wrapper=model, parameters=parameters)
    image = Image.new("RGB", (4, 4), color=(128, 128, 128))
    prepared_input = model.prepare_attack_input(image).detach()

    monkeypatch.setattr(
        model,
        "prepare_attack_input",
        lambda image: (_ for _ in ()).throw(
            AssertionError("Evaluation must reuse the prepared input")
        ),
    )
    monkeypatch.setattr(
        runner,
        "_tensor_to_image",
        lambda tensor: (_ for _ in ()).throw(
            AssertionError("Evaluation must not build model-space images")
        ),
    )

    result = runner.apply_for_evaluation(
        model_wrapper=model,
        image=image,
        prepared_input=prepared_input,
        parameters=parameters,
        original_class_index=1,
    )

    assert isinstance(result, AdversarialEvaluationResult)
    assert result.clean_model_input is not None
    assert torch.equal(result.clean_model_input, prepared_input)
    assert result.adversarial_model_input.shape == prepared_input.shape


def test_carlini_wagner_runner_rejects_out_of_range_original_class(monkeypatch):
    attack = CarliniWagnerAttack()
    monkeypatch.setattr(
        attack,
        "build_art_attack",
        lambda **kwargs: FakeARTAttack(),
    )

    model = FakeClassificationModel()
    parameters = _resolved_parameters()
    runner = attack.create_runner(model_wrapper=model, parameters=parameters)

    with pytest.raises(ValueError, match="outside the model output range"):
        runner.apply(
            model_wrapper=model,
            image=Image.new("RGB", (4, 4), color=(128, 128, 128)),
            parameters=parameters,
            original_class_index=3,
        )
