import pytest

import ml.attacks.zoo as zoo_module
from ml.attacks.zoo import ZerothOrderOptimizationAttack
from schemas.attack import AttackParameters


def _resolved_parameters(**overrides):
    attack = ZerothOrderOptimizationAttack()
    return attack.resolve_parameters(AttackParameters(**overrides))


def test_zoo_parameter_contract_uses_art_defaults():
    parameters = _resolved_parameters()

    assert parameters.model_dump() == {
        "confidence": 0.0,
        "learning_rate": 0.01,
        "max_iter": 10,
        "binary_search_steps": 1,
        "initial_const": 0.001,
        "nb_parallel": 128,
        "use_resize": True,
        "variable_h": 0.0001,
    }


def test_zoo_parameter_contract_accepts_custom_values():
    parameters = _resolved_parameters(
        confidence=0.5,
        learning_rate=0.02,
        max_iter=20,
        binary_search_steps=2,
        initial_const=0.01,
        nb_parallel=64,
        use_resize=False,
        variable_h=0.001,
    )

    assert parameters.model_dump() == {
        "confidence": 0.5,
        "learning_rate": 0.02,
        "max_iter": 20,
        "binary_search_steps": 2,
        "initial_const": 0.01,
        "nb_parallel": 64,
        "use_resize": False,
        "variable_h": 0.001,
    }


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ({"confidence": -0.1}, "mayor o igual"),
        ({"learning_rate": 0}, "mayor o igual"),
        ({"max_iter": 0}, "mayor o igual"),
        ({"binary_search_steps": 0}, "mayor o igual"),
        ({"initial_const": 0}, "mayor o igual"),
        ({"nb_parallel": 0}, "mayor o igual"),
        ({"use_resize": "true"}, "debe ser booleano"),
        ({"variable_h": 0}, "mayor o igual"),
    ],
)
def test_zoo_parameter_contract_rejects_invalid_values(parameters, message):
    attack = ZerothOrderOptimizationAttack()

    with pytest.raises(ValueError, match=message):
        attack.resolve_parameters(AttackParameters(**parameters))


def test_zoo_builds_art_attack_with_explicit_runtime_defaults(monkeypatch):
    captured = {}

    def fake_zoo_attack(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(zoo_module, "ARTZooAttack", fake_zoo_attack)

    estimator = object()
    ZerothOrderOptimizationAttack().build_art_attack(
        estimator=estimator,
        parameters=_resolved_parameters(
            confidence=0.5,
            learning_rate=0.02,
            max_iter=20,
            binary_search_steps=2,
            initial_const=0.01,
            nb_parallel=64,
            use_resize=False,
            variable_h=0.001,
        ),
    )

    assert captured == {
        "classifier": estimator,
        "confidence": 0.5,
        "targeted": False,
        "learning_rate": 0.02,
        "max_iter": 20,
        "binary_search_steps": 2,
        "initial_const": 0.01,
        "abort_early": True,
        "use_resize": False,
        "use_importance": True,
        "nb_parallel": 64,
        "batch_size": 1,
        "variable_h": 0.001,
        "verbose": False,
    }


def test_zoo_is_classification_only_black_box_attack():
    attack = ZerothOrderOptimizationAttack()

    assert attack.knowledge_type == "black_box"
    assert attack.supports_task_type("classification") is True
    assert attack.supports_task_type("detection") is False
