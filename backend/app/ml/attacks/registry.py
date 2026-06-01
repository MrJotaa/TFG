from ml.attacks.fgsm import FGSMAttack
from schemas.attack import AttackType


_AVAILABLE_ATTACKS = {
    AttackType.FGSM: FGSMAttack(),
}


def get_attack(attack_type: AttackType):
    if attack_type not in _AVAILABLE_ATTACKS:
        raise ValueError(f"Attack '{attack_type}' is not supported")

    return _AVAILABLE_ATTACKS[attack_type]


def list_available_attacks():
    return list(_AVAILABLE_ATTACKS.keys())