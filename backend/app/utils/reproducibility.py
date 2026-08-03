import hashlib

from schemas.attack import AttackParameters, AttackType


def attack_parameters_for_image(
    *,
    attack_type: AttackType,
    parameters: AttackParameters,
    image_id: str,
) -> AttackParameters:
    """Derive a stable per-image seed without depending on dataset order."""
    if attack_type != AttackType.RANDOM_NOISE:
        return parameters
    digest = hashlib.sha256(
        f"{parameters.seed}\0{image_id}".encode("utf-8")
    ).digest()
    image_seed = int.from_bytes(digest[:4], "big") & 0x7FFFFFFF
    return parameters.model_copy(update={"seed": image_seed})
