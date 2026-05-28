from fastapi import HTTPException

from ml.attacks.fgsm import FGSMAttack
from schemas.attack import (
    AttackComparison,
    AttackRequest,
    AttackResponse,
    AttackType,
)
from services.dataset_service import get_image_from_dataset
from services.model_service import get_active_model_instance
from utils.image_encoding import image_to_base64


_available_attacks = {
    AttackType.FGSM: FGSMAttack(),
}


async def apply_adversarial_attack(request: AttackRequest) -> AttackResponse:
    try:
        image = await get_image_from_dataset(
            dataset_id=request.dataset_id,
            image_id=request.image_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    try:
        model = get_active_model_instance()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if request.attack_type not in _available_attacks:
        raise HTTPException(
            status_code=400,
            detail=f"Attack '{request.attack_type}' is not supported",
        )

    attack = _available_attacks[request.attack_type]

    original_prediction = model.predict(image)

    try:
        attack_result = attack.apply(
            model_wrapper=model,
            image=image,
            parameters=request.parameters,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    adversarial_image = attack_result["adversarial_image"]
    perturbation_image = attack_result["perturbation_image"]

    adversarial_prediction = model.predict(adversarial_image)

    confidence_difference = round(
        adversarial_prediction.confidence - original_prediction.confidence,
        4,
    )

    prediction_changed = (
        original_prediction.prediction != adversarial_prediction.prediction
    )

    comparison = AttackComparison(
        original_label=original_prediction.prediction,
        adversarial_label=adversarial_prediction.prediction,
        original_confidence=original_prediction.confidence,
        adversarial_confidence=adversarial_prediction.confidence,
        confidence_difference=confidence_difference,
        prediction_changed=prediction_changed,
        attack_success=prediction_changed,
        perturbation_linf=attack_result["perturbation_linf"],
        perturbation_l2=attack_result["perturbation_l2"],
    )

    return AttackResponse(
        attack_type=request.attack_type,
        parameters=request.parameters,
        adversarial_image=image_to_base64(adversarial_image),
        perturbation_image=image_to_base64(perturbation_image),
        original_prediction=original_prediction,
        adversarial_prediction=adversarial_prediction,
        comparison=comparison,
    )