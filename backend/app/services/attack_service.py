from core.errors import BadRequestError
from core.task_compatibility import (
    CLASSIFICATION_TASK_TYPE,
    require_attack_task_type,
    require_model_task_type,
)
from ml.attacks.base import AdversarialAttackResult
from ml.attacks.presentation import reconstruct_attack_in_source_geometry
from ml.attacks.registry import get_attack
from ml.explainability.gradcam import (
    generate_gradcam,
    generate_gradcam_from_prepared_input,
)
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

    if isinstance(attack_result, AdversarialAttackResult):
        perturbation_linf = attack_result.perturbation_linf
        perturbation_l2 = attack_result.perturbation_l2
        adversarial_prediction = model.predict_from_prepared_input(
            attack_result.adversarial_model_input
        )

        reconstruction = reconstruct_attack_in_source_geometry(
            source_image=image,
            model_space_delta=attack_result.model_space_delta,
        )
        adversarial_image_to_persist = reconstruction.adversarial_image
        perturbation_image_to_persist = (
            reconstruction.perturbation_visualization
        )
        del reconstruction

        original_gradcam_result = generate_gradcam_from_prepared_input(
            model_wrapper=model,
            model_input=attack_result.clean_model_input,
            visualization_image=image.copy().convert("RGB"),
        )
        adversarial_gradcam_result = generate_gradcam_from_prepared_input(
            model_wrapper=model,
            model_input=attack_result.adversarial_model_input,
            visualization_image=adversarial_image_to_persist,
        )
    else:
        adversarial_image = attack_result["adversarial_image"]
        adversarial_image_to_persist = adversarial_image
        perturbation_image_to_persist = attack_result["perturbation_image"]
        perturbation_linf = attack_result["perturbation_linf"]
        perturbation_l2 = attack_result["perturbation_l2"]
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
        perturbation_linf=perturbation_linf,
        perturbation_l2=perturbation_l2,
    )

    result_id, result_path = create_result_directory(prefix="attack")

    adversarial_image_url = save_result_image(
        image=adversarial_image_to_persist,
        result_id=result_id,
        result_path=result_path,
        filename="adversarial.png",
    )

    perturbation_image_url = save_result_image(
        image=perturbation_image_to_persist,
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
