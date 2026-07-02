from ml.attacks.fgsm import FGSMAttack
from ml.attacks.random_noise import RandomNoiseAttack
from ml.attacks.pgd import PGDAttack
from schemas.attack import AttackType
from ml.attacks.yolo_confidence import YOLOConfidenceAttack


_AVAILABLE_ATTACKS = {
    AttackType.FGSM: FGSMAttack(),
    AttackType.RANDOM_NOISE: RandomNoiseAttack(),
    AttackType.PGD: PGDAttack(),
    AttackType.YOLO_CONFIDENCE: YOLOConfidenceAttack(),
}


def get_attack(attack_type: AttackType):
    if attack_type not in _AVAILABLE_ATTACKS:
        raise ValueError(f"Attack '{attack_type}' is not supported")

    return _AVAILABLE_ATTACKS[attack_type]


def list_available_attacks():
    return list(_AVAILABLE_ATTACKS.keys())
