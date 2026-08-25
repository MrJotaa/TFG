import pytest

import ml.attacks.deepfool as deepfool_module
from ml.attacks.deepfool import DeepFoolAttack
from schemas.attack import AttackParameters


def _resolved_parameters(**overrides):
    attack = DeepFoolAttack()
    return attack.resolve_parameters(AttackParameters(**overrides))


def test_deepfool_parameter_contract_uses_art_defaults():
    parameters = _resolved_parameters()

    assert parameters.model_dump() == {
        "max_iter": 100,
        "overshoot": 1e-6,
        "nb_grads": 10,
    }


def test_deepfool_parameter_contract_accepts_custom_values():
    parameters = _resolved_parameters(
        max_iter=25,
        overshoot=0.001,
        nb_grads=5,
    )

    assert parameters.model_dump() == {
        "max_iter": 25,
        "overshoot": 0.001,
        "nb_grads": 5,
    }


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ({"max_iter": 0}, "mayor o igual"),
        ({"overshoot": -0.01}, "mayor o igual"),
        ({"nb_grads": 0}, "mayor o igual"),
    ],
)
def test_deepfool_parameter_contract_rejects_invalid_values(parameters, message):
    attack = DeepFoolAttack()

    with pytest.raises(ValueError, match=message):
        attack.resolve_parameters(AttackParameters(**parameters))


def test_deepfool_builds_art_attack_with_explicit_runtime_defaults(monkeypatch):
    captured = {}

    def fake_deepfool(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(deepfool_module, "ARTDeepFool", fake_deepfool)

    estimator = object()
    DeepFoolAttack().build_art_attack(
        estimator=estimator,
        parameters=_resolved_parameters(
            max_iter=30,
            overshoot=0.002,
            nb_grads=7,
        ),
    )

    assert captured == {
        "classifier": estimator,
        "max_iter": 30,
        "epsilon": 0.002,
        "nb_grads": 7,
        "batch_size": 1,
        "verbose": False,
    }


def test_deepfool_is_classification_only_white_box_attack():
    attack = DeepFoolAttack()

    assert attack.knowledge_type == "white_box"
    assert attack.supports_task_type("classification") is True
    assert attack.supports_task_type("detection") is False
