import pytest

from ml.attacks.fgsm import FGSMAttack
from ml.attacks.pgd import PGDAttack
from ml.attacks.random_noise import RandomNoiseAttack
from ml.attacks.registry import get_attack, list_available_attacks
from ml.attacks.yolo_confidence import YOLOConfidenceAttack
from schemas.attack import AttackType


@pytest.mark.parametrize(
    ("attack_type", "expected_class"),
    [
        (AttackType.FGSM, FGSMAttack),
        (AttackType.PGD, PGDAttack),
        (AttackType.RANDOM_NOISE, RandomNoiseAttack),
        (AttackType.YOLO_CONFIDENCE, YOLOConfidenceAttack),
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
        AttackType.RANDOM_NOISE,
        AttackType.YOLO_CONFIDENCE,
    }


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
        (RandomNoiseAttack(), True, True),
        (YOLOConfidenceAttack(), False, True),
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
        RandomNoiseAttack(),
        YOLOConfidenceAttack(),
    ],
)
def test_attacks_reject_unknown_task_type(attack):
    assert attack.supports_task_type("segmentation") is False


def test_registered_attacks_resolve_only_their_own_parameters():
    from schemas.attack import AttackParameters

    fgsm = get_attack(AttackType.FGSM)
    pgd = get_attack(AttackType.PGD)
    random_noise = get_attack(AttackType.RANDOM_NOISE)

    assert fgsm.resolve_parameters(AttackParameters()).model_dump() == {
        "epsilon": 0.03,
    }
    assert pgd.resolve_parameters(AttackParameters()).model_dump() == {
        "epsilon": 0.03,
        "alpha": 0.01,
        "steps": 10,
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
