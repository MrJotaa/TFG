from inspect import Parameter, signature

from ml.attacks.carlini_wagner import CarliniWagnerAttack
from ml.attacks.deepfool import DeepFoolAttack
from ml.attacks.detection_confidence import DetectionConfidenceAttack
from ml.attacks.detection_localization_pgd import DetectionLocalizationPGDAttack
from ml.attacks.fgsm import FGSMAttack
from ml.attacks.hop_skip_jump import HopSkipJumpAttack
from ml.attacks.pgd import PGDAttack
from ml.attacks.random_noise import RandomNoiseAttack
from ml.attacks.square_detection import SquareDetectionAttack
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
    AttackType.YOLO_CONFIDENCE: DetectionConfidenceAttack(),
    AttackType.DETECTION_LOCALIZATION_PGD: DetectionLocalizationPGDAttack(),
    AttackType.SQUARE_DETECTION: SquareDetectionAttack(),
}


def get_attack(attack_type: AttackType):
    if attack_type not in _AVAILABLE_ATTACKS:
        raise ValueError(f"El ataque '{attack_type}' no es compatible")

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


def apply_attack_runner_for_evaluation(
    runner,
    *,
    model_wrapper,
    image,
    prepared_input,
    parameters,
    original_class_index=None,
):
    """Use an evaluation-specific capability when a runner exposes it."""
    evaluation_apply = getattr(runner, "apply_for_evaluation", None)
    if callable(evaluation_apply):
        return evaluation_apply(
            model_wrapper=model_wrapper,
            image=image,
            prepared_input=prepared_input,
            parameters=parameters,
            original_class_index=original_class_index,
        )

    return runner.apply(
        model_wrapper=model_wrapper,
        image=image,
        parameters=parameters,
        original_class_index=original_class_index,
    )


def get_attack_runner_evaluation_batch_size(runner) -> int:
    """Return the bounded micro-batch capability declared by an attack runner."""
    batch_size = getattr(runner, "evaluation_micro_batch_size", 1)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("evaluation_micro_batch_size must be an integer")
    if batch_size < 1:
        raise ValueError("evaluation_micro_batch_size must be positive")
    return batch_size


def apply_attack_runner_batch_for_evaluation(
    runner,
    *,
    model_wrapper,
    images,
    prepared_inputs,
    parameters,
    original_class_indices,
):
    """Execute an evaluation micro-batch while retaining legacy fallback."""
    batch_apply = getattr(runner, "apply_many_for_evaluation", None)
    if callable(batch_apply):
        results = batch_apply(
            model_wrapper=model_wrapper,
            images=images,
            prepared_inputs=prepared_inputs,
            parameters=parameters,
            original_class_indices=original_class_indices,
        )
    else:
        results = tuple(
            apply_attack_runner_for_evaluation(
                runner,
                model_wrapper=model_wrapper,
                image=image,
                prepared_input=prepared_input,
                parameters=sample_parameters,
                original_class_index=original_class_index,
            )
            for image, prepared_input, sample_parameters, original_class_index in zip(
                images,
                prepared_inputs,
                parameters,
                original_class_indices,
            )
        )

    results = tuple(results)
    if len(results) != len(images):
        raise ValueError("Evaluation attack batch returned an unexpected result count")
    return results


def list_available_attacks():
    return list(_AVAILABLE_ATTACKS.keys())


def list_attack_catalog():
    return [attack.catalog_entry() for attack in _AVAILABLE_ATTACKS.values()]
