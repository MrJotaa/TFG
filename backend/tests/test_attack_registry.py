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
