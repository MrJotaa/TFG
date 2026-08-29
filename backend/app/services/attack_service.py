import asyncio

from core.cancellation import CancellationToken
from core.errors import BadRequestError
from core.task_compatibility import (
    CLASSIFICATION_TASK_TYPE,
    require_attack_task_type,
    require_model_task_type,
)
from ml.attacks.base import AdversarialAttackResult
from ml.attacks.presentation import reconstruct_attack_in_source_geometry
from ml.attacks.registry import (
    create_attack_runner,
    get_attack,
    resolve_attack_parameters,
)
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
    save_result_payload,
)


def _raise_if_cancelled(cancellation_token: CancellationToken | None) -> None:
    if cancellation_token is not None:
        cancellation_token.raise_if_cancelled()


async def apply_adversarial_attack(
    request: AttackRequest,
    *,
    cancellation_token: CancellationToken | None = None,
) -> AttackResponse:
    try:
        attack = get_attack(request.attack_type)
        request.parameters = resolve_attack_parameters(attack, request.parameters)
    except ValueError as error:
        raise BadRequestError(str(error)) from error

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

    require_attack_task_type(
        attack=attack,
        task_type=model.task_type,
    )

    _raise_if_cancelled(cancellation_token)
    original_prediction = await asyncio.to_thread(model.predict, image)
    _raise_if_cancelled(cancellation_token)

    attack_runner = create_attack_runner(
        attack=attack,
        model_wrapper=model,
        parameters=request.parameters,
        cancellation_token=cancellation_token,
    )
    _raise_if_cancelled(cancellation_token)
    attack_result = await asyncio.to_thread(
        attack_runner.apply,
        model_wrapper=model,
        image=image,
        parameters=request.parameters,
    )
    _raise_if_cancelled(cancellation_token)

    if isinstance(attack_result, AdversarialAttackResult):
        perturbation_linf = attack_result.perturbation_linf
        perturbation_l2 = attack_result.perturbation_l2
        adversarial_prediction = await asyncio.to_thread(
            model.predict_from_prepared_input,
            attack_result.adversarial_model_input,
        )
        _raise_if_cancelled(cancellation_token)

        reconstruction = reconstruct_attack_in_source_geometry(
            source_image=image,
            model_space_delta=attack_result.model_space_delta,
        )
        adversarial_image_to_persist = reconstruction.adversarial_image
        perturbation_image_to_persist = reconstruction.perturbation_visualization
        del reconstruction

        _raise_if_cancelled(cancellation_token)
        original_gradcam_result = await asyncio.to_thread(
            generate_gradcam_from_prepared_input,
            model_wrapper=model,
            model_input=attack_result.clean_model_input,
            visualization_image=image.copy().convert("RGB"),
        )
        _raise_if_cancelled(cancellation_token)
        adversarial_gradcam_result = await asyncio.to_thread(
            generate_gradcam_from_prepared_input,
            model_wrapper=model,
            model_input=attack_result.adversarial_model_input,
            visualization_image=adversarial_image_to_persist,
        )
        _raise_if_cancelled(cancellation_token)
    else:
        adversarial_image = attack_result["adversarial_image"]
        adversarial_image_to_persist = adversarial_image
        perturbation_image_to_persist = attack_result["perturbation_image"]
        perturbation_linf = attack_result["perturbation_linf"]
        perturbation_l2 = attack_result["perturbation_l2"]
        adversarial_prediction = await asyncio.to_thread(
            model.predict, adversarial_image
        )
        _raise_if_cancelled(cancellation_token)

        original_gradcam_result = await asyncio.to_thread(
            generate_gradcam, model, image
        )
        _raise_if_cancelled(cancellation_token)
        adversarial_gradcam_result = await asyncio.to_thread(
            generate_gradcam,
            model,
            adversarial_image,
        )
        _raise_if_cancelled(cancellation_token)

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

    _raise_if_cancelled(cancellation_token)

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

    response = AttackResponse(
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

    save_result_metadata(
        result_path=result_path,
        metadata={
            "result_id": result_id,
            "result_type": "attack",
            "task_type": "classification",
            "dataset_id": request.dataset_id,
            "dataset_revision": int(image.info.get("dataset_revision", 0)),
            "image_id": request.image_id,
            "model_id": model.id,
            "model_name": getattr(model, "name", model.id),
            "model_revision": getattr(model, "revision", model.id),
            "attack_type": request.attack_type.value,
            "attack_revision": getattr(attack, "revision", None),
            "parameters": request.parameters.model_dump(mode="json"),
            "pipeline_version": "classification-individual-analysis-v1",
            "expires_in_hours": 24,
        },
    )
    save_result_payload(
        result_path=result_path,
        payload=response.model_dump(mode="json"),
    )
    return response
