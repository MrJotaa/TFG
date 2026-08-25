import numpy as np
import pytest

import ml.attacks.hop_skip_jump as hop_skip_jump_module
from ml.attacks.hop_skip_jump import HopSkipJumpAttack
from schemas.attack import AttackParameters


def _resolved_parameters(**overrides):
    attack = HopSkipJumpAttack()
    return attack.resolve_parameters(AttackParameters(**overrides))


def test_hop_skip_jump_parameter_contract_uses_art_defaults():
    parameters = _resolved_parameters()

    assert parameters.model_dump() == {
        "norm": "l2",
        "max_iter": 50,
        "max_eval": 10000,
        "init_eval": 100,
        "init_size": 100,
    }


def test_hop_skip_jump_parameter_contract_accepts_custom_values():
    parameters = _resolved_parameters(
        norm="linf",
        max_iter=20,
        max_eval=500,
        init_eval=50,
        init_size=25,
    )

    assert parameters.model_dump() == {
        "norm": "linf",
        "max_iter": 20,
        "max_eval": 500,
        "init_eval": 50,
        "init_size": 25,
    }


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ({"norm": "l1"}, "debe ser uno de"),
        ({"max_iter": 0}, "mayor o igual"),
        ({"max_eval": 0}, "mayor o igual"),
        ({"init_eval": 0}, "mayor o igual"),
        ({"init_size": 0}, "mayor o igual"),
    ],
)
def test_hop_skip_jump_parameter_contract_rejects_invalid_values(parameters, message):
    attack = HopSkipJumpAttack()

    with pytest.raises(ValueError, match=message):
        attack.resolve_parameters(AttackParameters(**parameters))


def test_hop_skip_jump_requires_init_eval_not_to_exceed_max_eval():
    attack = HopSkipJumpAttack()

    with pytest.raises(ValueError, match="init_eval.*menor o igual.*max_eval"):
        attack.resolve_parameters(AttackParameters(init_eval=101, max_eval=100))


@pytest.mark.parametrize(
    ("norm", "expected_art_norm"),
    [
        ("l2", 2),
        ("linf", np.inf),
    ],
)
def test_hop_skip_jump_builds_art_attack_with_explicit_runtime_defaults(
    monkeypatch,
    norm,
    expected_art_norm,
):
    captured = {}

    def fake_hop_skip_jump(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(hop_skip_jump_module, "ARTHopSkipJump", fake_hop_skip_jump)

    estimator = object()
    HopSkipJumpAttack().build_art_attack(
        estimator=estimator,
        parameters=_resolved_parameters(
            norm=norm,
            max_iter=20,
            max_eval=500,
            init_eval=50,
            init_size=25,
        ),
    )

    assert captured == {
        "classifier": estimator,
        "batch_size": 64,
        "targeted": False,
        "norm": expected_art_norm,
        "max_iter": 20,
        "max_eval": 500,
        "init_eval": 50,
        "init_size": 25,
        "verbose": False,
    }


def test_hop_skip_jump_is_classification_only_black_box_attack():
    attack = HopSkipJumpAttack()

    assert attack.knowledge_type == "black_box"
    assert attack.supports_task_type("classification") is True
    assert attack.supports_task_type("detection") is False
