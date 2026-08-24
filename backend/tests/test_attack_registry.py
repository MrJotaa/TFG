import pytest

from ml.attacks.carlini_wagner import CarliniWagnerAttack
from ml.attacks.deepfool import DeepFoolAttack
from ml.attacks.detection_confidence import DetectionConfidenceAttack
from ml.attacks.fgsm import FGSMAttack
from ml.attacks.hop_skip_jump import HopSkipJumpAttack
from ml.attacks.pgd import PGDAttack
from ml.attacks.random_noise import RandomNoiseAttack
from ml.attacks.registry import (
    apply_attack_runner_batch_for_evaluation,
    apply_attack_runner_for_evaluation,
    create_attack_runner,
    get_attack,
    get_attack_runner_evaluation_batch_size,
    list_available_attacks,
)
from ml.attacks.zoo import ZerothOrderOptimizationAttack
from schemas.attack import AttackType


@pytest.mark.parametrize(
    ("attack_type", "expected_class"),
    [
        (AttackType.FGSM, FGSMAttack),
        (AttackType.PGD, PGDAttack),
        (AttackType.CARLINI_WAGNER, CarliniWagnerAttack),
        (AttackType.DEEPFOOL, DeepFoolAttack),
        (AttackType.HOP_SKIP_JUMP, HopSkipJumpAttack),
        (AttackType.ZOO, ZerothOrderOptimizationAttack),
        (AttackType.RANDOM_NOISE, RandomNoiseAttack),
        (AttackType.YOLO_CONFIDENCE, DetectionConfidenceAttack),
    ],
)
def test_get_attack_returns_registered_implementation(
    attack_type,
    expected_class,
):
    attack = get_attack(attack_type)

    assert isinstance(attack, expected_class)
    assert attack.id == attack_type.value


def test_list_available_attacks_returns_all_registered_attacks():
    available_attacks = set(list_available_attacks())

    assert available_attacks == {
        AttackType.FGSM,
        AttackType.PGD,
        AttackType.CARLINI_WAGNER,
        AttackType.DEEPFOOL,
        AttackType.HOP_SKIP_JUMP,
        AttackType.ZOO,
        AttackType.RANDOM_NOISE,
        AttackType.YOLO_CONFIDENCE,
    }


def test_create_attack_runner_keeps_legacy_attack_compatible():
    legacy_attack = object()

    assert (
        create_attack_runner(
            attack=legacy_attack,
            model_wrapper=object(),
            parameters=object(),
        )
        is legacy_attack
    )


def test_apply_attack_runner_for_evaluation_uses_optional_fast_path():
    prepared_input = object()
    expected_result = object()

    class OptimizedRunner:
        def apply_for_evaluation(
            self,
            *,
            model_wrapper,
            image,
            prepared_input: object,
            parameters,
            original_class_index,
        ):
            assert model_wrapper == "model"
            assert image == "image"
            assert prepared_input is prepared_input_reference
            assert parameters == "parameters"
            assert original_class_index == 7
            return expected_result

        def apply(self, **kwargs):
            raise AssertionError("Legacy apply must not run when fast path exists")

    prepared_input_reference = prepared_input
    result = apply_attack_runner_for_evaluation(
        OptimizedRunner(),
        model_wrapper="model",
        image="image",
        prepared_input=prepared_input,
        parameters="parameters",
        original_class_index=7,
    )

    assert result is expected_result


def test_apply_attack_runner_for_evaluation_keeps_legacy_runner_compatible():
    calls = []
    expected_result = object()

    class LegacyRunner:
        def apply(
            self,
            *,
            model_wrapper,
            image,
            parameters,
            original_class_index,
        ):
            calls.append(
                (model_wrapper, image, parameters, original_class_index)
            )
            return expected_result

    result = apply_attack_runner_for_evaluation(
        LegacyRunner(),
        model_wrapper="model",
        image="image",
        prepared_input=object(),
        parameters="parameters",
        original_class_index=3,
    )

    assert result is expected_result
    assert calls == [("model", "image", "parameters", 3)]



def test_attack_runner_evaluation_batch_size_defaults_to_one():
    assert get_attack_runner_evaluation_batch_size(object()) == 1

    class BatchedRunner:
        evaluation_micro_batch_size = 4

    assert get_attack_runner_evaluation_batch_size(BatchedRunner()) == 4


def test_apply_attack_runner_batch_for_evaluation_uses_optional_batch_path():
    expected_results = (object(), object())

    class BatchedRunner:
        def apply_many_for_evaluation(
            self,
            *,
            model_wrapper,
            images,
            prepared_inputs,
            parameters,
            original_class_indices,
        ):
            assert model_wrapper == "model"
            assert images == ("image-1", "image-2")
            assert prepared_inputs == ("input-1", "input-2")
            assert parameters == ("params-1", "params-2")
            assert original_class_indices == (1, 2)
            return expected_results

    results = apply_attack_runner_batch_for_evaluation(
        BatchedRunner(),
        model_wrapper="model",
        images=("image-1", "image-2"),
        prepared_inputs=("input-1", "input-2"),
        parameters=("params-1", "params-2"),
        original_class_indices=(1, 2),
    )

    assert results == expected_results

def test_get_unknown_attack_raises_value_error():
    with pytest.raises(ValueError, match="not supported"):
        get_attack("unknown_attack")


@pytest.mark.parametrize(
    (
        "attack",
        "supports_classification",
        "supports_detection",
    ),
    [
        (FGSMAttack(), True, False),
        (PGDAttack(), True, False),
        (CarliniWagnerAttack(), True, False),
        (DeepFoolAttack(), True, False),
        (HopSkipJumpAttack(), True, False),
        (ZerothOrderOptimizationAttack(), True, False),
        (RandomNoiseAttack(), True, True),
        (DetectionConfidenceAttack(), False, True),
    ],
)
def test_attack_task_type_compatibility(
    attack,
    supports_classification,
    supports_detection,
):
    assert attack.supports_task_type("classification") is supports_classification
    assert attack.supports_task_type("detection") is supports_detection


@pytest.mark.parametrize(
    "attack",
    [
        FGSMAttack(),
        PGDAttack(),
        CarliniWagnerAttack(),
        DeepFoolAttack(),
        HopSkipJumpAttack(),
        ZerothOrderOptimizationAttack(),
        RandomNoiseAttack(),
        DetectionConfidenceAttack(),
    ],
)
def test_attacks_reject_unknown_task_type(attack):
    assert attack.supports_task_type("segmentation") is False


def test_registered_attacks_resolve_only_their_own_parameters():
    from schemas.attack import AttackParameters

    fgsm = get_attack(AttackType.FGSM)
    pgd = get_attack(AttackType.PGD)
    carlini_wagner = get_attack(AttackType.CARLINI_WAGNER)
    deepfool = get_attack(AttackType.DEEPFOOL)
    hop_skip_jump = get_attack(AttackType.HOP_SKIP_JUMP)
    zoo = get_attack(AttackType.ZOO)
    random_noise = get_attack(AttackType.RANDOM_NOISE)

    assert fgsm.resolve_parameters(AttackParameters()).model_dump() == {
        "epsilon": 0.03,
    }
    assert pgd.resolve_parameters(AttackParameters()).model_dump() == {
        "epsilon": 0.03,
        "alpha": 0.01,
        "steps": 10,
    }
    assert carlini_wagner.resolve_parameters(AttackParameters()).model_dump() == {
        "norm": "l2",
        "confidence": 0.0,
        "learning_rate": 0.01,
        "binary_search_steps": 3,
        "max_iter": 10,
    }
    assert deepfool.resolve_parameters(AttackParameters()).model_dump() == {
        "max_iter": 100,
        "overshoot": 1e-6,
        "nb_grads": 10,
    }
    assert hop_skip_jump.resolve_parameters(AttackParameters()).model_dump() == {
        "norm": "l2",
        "max_iter": 50,
        "max_eval": 10000,
        "init_eval": 100,
        "init_size": 100,
    }
    assert zoo.resolve_parameters(AttackParameters()).model_dump() == {
        "confidence": 0.0,
        "learning_rate": 0.01,
        "max_iter": 10,
        "binary_search_steps": 1,
        "initial_const": 0.001,
        "nb_parallel": 128,
        "use_resize": True,
        "variable_h": 0.0001,
    }
    assert random_noise.resolve_parameters(AttackParameters()).model_dump() == {
        "epsilon": 0.03,
        "seed": 0,
    }


def test_parameter_resolution_strips_irrelevant_legacy_values():
    from schemas.attack import AttackParameters

    fgsm = get_attack(AttackType.FGSM)
    parameters = fgsm.resolve_parameters(
        AttackParameters(epsilon=0.05, alpha=0.02, steps=30, seed=17)
    )

    assert parameters.model_dump() == {"epsilon": 0.05}


def test_parameter_resolution_rejects_unknown_values():
    from schemas.attack import AttackParameters

    pgd = get_attack(AttackType.PGD)

    with pytest.raises(ValueError, match="Unsupported parameters"):
        pgd.resolve_parameters(AttackParameters(epsilon=0.03, learning_rate_typo=0.01))


def test_attack_catalog_exposes_capabilities_and_parameter_contracts():
    from ml.attacks.registry import list_attack_catalog

    catalog = {item["id"]: item for item in list_attack_catalog()}

    assert catalog["fgsm"]["knowledge_type"] == "white_box"
    assert catalog["fgsm"]["supported_task_types"] == ["classification"]
    assert [parameter["key"] for parameter in catalog["fgsm"]["parameters"]] == [
        "epsilon"
    ]

    assert catalog["random_noise"]["knowledge_type"] == "baseline"
    assert catalog["random_noise"]["supported_task_types"] == [
        "classification",
        "detection",
    ]
    assert [
        parameter["key"] for parameter in catalog["random_noise"]["parameters"]
    ] == ["epsilon", "seed"]

    assert catalog["carlini_wagner"]["knowledge_type"] == "white_box"
    assert catalog["carlini_wagner"]["supported_task_types"] == ["classification"]
    assert [
        parameter["key"] for parameter in catalog["carlini_wagner"]["parameters"]
    ] == [
        "norm",
        "confidence",
        "learning_rate",
        "binary_search_steps",
        "max_iter",
    ]
    assert catalog["carlini_wagner"]["parameters"][0]["choices"] == [
        {"value": "l2", "label": "L2"},
        {"value": "linf", "label": "L∞"},
    ]

    assert catalog["deepfool"]["knowledge_type"] == "white_box"
    assert catalog["deepfool"]["supported_task_types"] == ["classification"]
    assert [parameter["key"] for parameter in catalog["deepfool"]["parameters"]] == [
        "max_iter",
        "overshoot",
        "nb_grads",
    ]

    assert catalog["hop_skip_jump"]["knowledge_type"] == "black_box"
    assert catalog["hop_skip_jump"]["supported_task_types"] == ["classification"]
    assert [
        parameter["key"] for parameter in catalog["hop_skip_jump"]["parameters"]
    ] == ["norm", "max_iter", "max_eval", "init_eval", "init_size"]


def test_zoo_catalog_exposes_black_box_parameter_contract():
    from ml.attacks.registry import list_attack_catalog

    catalog = {item["id"]: item for item in list_attack_catalog()}
    zoo = catalog["zoo"]

    assert zoo["knowledge_type"] == "black_box"
    assert zoo["supported_task_types"] == ["classification"]
    assert [parameter["key"] for parameter in zoo["parameters"]] == [
        "confidence",
        "learning_rate",
        "max_iter",
        "binary_search_steps",
        "initial_const",
        "nb_parallel",
        "use_resize",
        "variable_h",
    ]
