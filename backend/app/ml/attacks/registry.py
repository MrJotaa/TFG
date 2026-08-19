from inspect import Parameter, signature

from ml.attacks.carlini_wagner import CarliniWagnerAttack
from ml.attacks.deepfool import DeepFoolAttack
from ml.attacks.fgsm import FGSMAttack
from ml.attacks.hop_skip_jump import HopSkipJumpAttack
from ml.attacks.pgd import PGDAttack
from ml.attacks.random_noise import RandomNoiseAttack
from ml.attacks.yolo_confidence import YOLOConfidenceAttack
from ml.attacks.zoo import ZerothOrderOptimizationAttack
from schemas.attack import AttackParameters, AttackType

_AVAILABLE_ATTACKS = {
    AttackType.FGSM: FGSMAttack(),
    AttackType.RANDOM_NOISE: RandomNoiseAttack(),
    AttackType.PGD: PGDAttack(),
    AttackType.CARLINI_WAGNER: CarliniWagnerAttack(),
    AttackType.DEEPFOOL: DeepFoolAttack(),
    AttackType.HOP_SKIP_JUMP: HopSkipJumpAttack(),
    AttackType.ZOO: ZerothOrderOptimizationAttack(),
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


def create_attack_runner(
    attack,
    model_wrapper,
    parameters,
    cancellation_token=None,
):
    factory = getattr(attack, "create_runner", None)
    if not callable(factory):
        return attack

    factory_parameters = signature(factory).parameters
    accepts_cancellation = "cancellation_token" in factory_parameters or any(
        parameter.kind is Parameter.VAR_KEYWORD
        for parameter in factory_parameters.values()
    )
    if accepts_cancellation:
        return factory(
            model_wrapper=model_wrapper,
            parameters=parameters,
            cancellation_token=cancellation_token,
        )
    return factory(model_wrapper=model_wrapper, parameters=parameters)


def list_available_attacks():
    return list(_AVAILABLE_ATTACKS.keys())


def list_attack_catalog():
    return [attack.catalog_entry() for attack in _AVAILABLE_ATTACKS.values()]
