from ml.attacks.registry import get_attack
from schemas.attack import (
    AttackComparison,
    AttackRequest,
    AttackResponse,
)
from services.dataset_service import get_image_from_dataset
from services.model_service import get_active_model_instance
from utils.result_files import (
    create_result_directory,
    save_result_image,
    save_result_metadata,
)
from ml.explainability.gradcam import generate_gradcam
from core.task_compatibility import (
    CLASSIFICATION_TASK_TYPE,
    require_attack_task_type,
    require_model_task_type,
)
from core.errors import BadRequestError


async def apply_adversarial_attack(request: AttackRequest) -> AttackResponse:
    image = await get_image_from_dataset(
        dataset_id=request.dataset_id,
        image_id=request.image_id,
    )

    model = get_active_model_instance()

    require_model_task_type(
        model=model,
        expected_task_type=CLASSIFICATION_TASK_TYPE,
        endpoint="/attack",
    )

    try:
        attack = get_attack(request.attack_type)
    except ValueError as error:
        raise BadRequestError(str(error)) from error

    require_attack_task_type(
        attack=attack,
        task_type=model.task_type,
    )

    original_prediction = model.predict(image)

    attack_result = attack.apply(
        model_wrapper=model,
        image=image,
        parameters=request.parameters,
    )

    adversarial_image = attack_result["adversarial_image"]
    perturbation_image = attack_result["perturbation_image"]

    adversarial_prediction = model.predict(adversarial_image)

    original_gradcam_result = generate_gradcam(model, image)
    adversarial_gradcam_result = generate_gradcam(model, adversarial_image)

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

    result_id, result_path = create_result_directory(prefix="attack")

    adversarial_image_url = save_result_image(
        image=adversarial_image,
        result_id=result_id,
        result_path=result_path,
        filename="adversarial.png",
    )

    perturbation_image_url = save_result_image(
        image=perturbation_image,
        result_id=result_id,
        result_path=result_path,
        filename="perturbation.png",
    )

    original_gradcam_image_url = save_result_image(
        image=original_gradcam_result["gradcam_pil_image"],
        result_id=result_id,
        result_path=result_path,
        filename="original_gradcam.png",
    )

    adversarial_gradcam_image_url = save_result_image(
        image=adversarial_gradcam_result["gradcam_pil_image"],
        result_id=result_id,
        result_path=result_path,
        filename="adversarial_gradcam.png",
    )

    save_result_metadata(
        result_path=result_path,
        metadata={
            "result_id": result_id,
            "result_type": "attack",
            "dataset_id": request.dataset_id,
            "image_id": request.image_id,
            "model_id": model.id,
            "attack_type": request.attack_type,
            "parameters": request.parameters.model_dump(),
            "expires_in_hours": 24,
        },
    )

    return AttackResponse(
        result_id=result_id,
        attack_type=request.attack_type,
        parameters=request.parameters,
        adversarial_image_url=adversarial_image_url,
        perturbation_image_url=perturbation_image_url,
        original_gradcam_image_url=original_gradcam_image_url,
        adversarial_gradcam_image_url=adversarial_gradcam_image_url,
        original_prediction=original_prediction,
        adversarial_prediction=adversarial_prediction,
        comparison=comparison,
    )
