from ml.attacks.fgsm import FGSMAttack
from ml.attacks.pgd import PGDAttack
from ml.attacks.random_noise import RandomNoiseAttack
from ml.attacks.yolo_confidence import YOLOConfidenceAttack
from schemas.attack import AttackParameters, AttackType

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


def resolve_attack_parameters(attack, parameters: AttackParameters) -> AttackParameters:
    resolver = getattr(attack, "resolve_parameters", None)
    if callable(resolver):
        return resolver(parameters)

    resolved = parameters.model_dump(mode="python")
    for key in ("epsilon", "alpha", "steps", "seed"):
        resolved.setdefault(key, getattr(parameters, key))

    return AttackParameters(**resolved)


def list_available_attacks():
    return list(_AVAILABLE_ATTACKS.keys())


def list_attack_catalog():
    return [attack.catalog_entry() for attack in _AVAILABLE_ATTACKS.values()]
