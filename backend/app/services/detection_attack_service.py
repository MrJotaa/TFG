from core.cancellation import CancellationToken
from core.errors import BadRequestError
from core.task_compatibility import (
    DETECTION_TASK_TYPE,
    require_attack_task_type,
    require_model_task_type,
)
from ml.attacks.registry import (
    create_attack_runner,
    get_attack,
    resolve_attack_parameters,
)
from ml.explainability.gradcam import generate_detection_eigencam
from ml.models.detection_visualization import draw_detections_on_image
from schemas.detection import to_detection_predictions
from schemas.detection_attack import (
    DetectionAttackComparison,
    DetectionAttackRequest,
    DetectionAttackResponse,
)
from services.dataset_service import get_image_from_dataset
from services.detection_model_execution import (
    run_detection_attack_operation,
    run_detection_model_operation,
)
from services.model_service import get_active_model_instance
from utils.detection_metrics import (
    calculate_detection_comparison,
)
from utils.result_files import (
    create_result_directory,
    save_result_image,
    save_result_metadata,
)


async def apply_detection_attack(
    request: DetectionAttackRequest,
    *,
    cancellation_token: CancellationToken | None = None,
) -> DetectionAttackResponse:
    try:
        attack = get_attack(request.attack_type)
        request.parameters = resolve_attack_parameters(attack, request.parameters)
    except ValueError as error:
        raise BadRequestError(str(error)) from error

    image = await get_image_from_dataset(
        dataset_id=request.dataset_id,
        image_id=request.image_id,
    )

    model = await run_detection_model_operation(get_active_model_instance)

    require_model_task_type(
        model=model,
        expected_task_type=DETECTION_TASK_TYPE,
        endpoint="/detect/attack",
    )

    require_attack_task_type(
        attack=attack,
        task_type=model.task_type,
    )

    if cancellation_token is not None:
        cancellation_token.raise_if_cancelled()
    original_detection_result, original_annotated_image = (
        await run_detection_model_operation(
            model.detect_with_visualization,
            image,
        )
    )

    attack_runner = create_attack_runner(
        attack=attack,
        model_wrapper=model,
        parameters=request.parameters,
        cancellation_token=cancellation_token,
    )
    if cancellation_token is not None:
        cancellation_token.raise_if_cancelled()
    attack_result = await run_detection_attack_operation(
        attack_runner,
        model_wrapper=model,
        image=image,
        parameters=request.parameters,
    )
    if cancellation_token is not None:
        cancellation_token.raise_if_cancelled()

    adversarial_image = attack_result["adversarial_image"]
    perturbation_image = attack_result["perturbation_image"]

    adversarial_detection_result, adversarial_annotated_image = (
        await run_detection_model_operation(
            model.detect_with_visualization,
            adversarial_image,
        )
    )
    if cancellation_token is not None:
        cancellation_token.raise_if_cancelled()

    original_detections = original_detection_result.detections
    adversarial_detections = adversarial_detection_result.detections

    original_explanation_image = None
    adversarial_explanation_image = None
    supports_explainability = getattr(
        model,
        "supports_detection_explainability",
        None,
    )
    if callable(supports_explainability) and supports_explainability():
        original_heatmap = await run_detection_model_operation(
            generate_detection_eigencam,
            model,
            image,
        )
        if cancellation_token is not None:
            cancellation_token.raise_if_cancelled()
        original_explanation_image = draw_detections_on_image(
            image=original_heatmap,
            detections=original_detections,
        )

        adversarial_heatmap = await run_detection_model_operation(
            generate_detection_eigencam,
            model,
            adversarial_image,
        )
        if cancellation_token is not None:
            cancellation_token.raise_if_cancelled()
        adversarial_explanation_image = draw_detections_on_image(
            image=adversarial_heatmap,
            detections=adversarial_detections,
        )

    metrics = calculate_detection_comparison(
        original_detections=original_detections,
        adversarial_detections=adversarial_detections,
    )

    comparison = DetectionAttackComparison(
        original_detection_count=metrics.original_count,
        adversarial_detection_count=metrics.adversarial_count,
        detections_difference=metrics.detections_difference,
        average_original_confidence=round(metrics.average_original_confidence, 4),
        average_adversarial_confidence=round(
            metrics.average_adversarial_confidence,
            4,
        ),
        average_confidence_difference=round(
            metrics.average_confidence_difference,
            4,
        ),
        attack_success=metrics.attack_success,
        perturbation_linf=attack_result["perturbation_linf"],
        perturbation_l2=attack_result["perturbation_l2"],
    )

    if cancellation_token is not None:
        cancellation_token.raise_if_cancelled()

    result_id, result_path = create_result_directory(prefix="detection_attack")

    original_annotated_image_url = save_result_image(
        image=original_annotated_image,
        result_id=result_id,
        result_path=result_path,
        filename="original_annotated.png",
    )

    adversarial_image_url = save_result_image(
        image=adversarial_image,
        result_id=result_id,
        result_path=result_path,
        filename="adversarial.png",
    )

    adversarial_annotated_image_url = save_result_image(
        image=adversarial_annotated_image,
        result_id=result_id,
        result_path=result_path,
        filename="adversarial_annotated.png",
    )

    perturbation_image_url = save_result_image(
        image=perturbation_image,
        result_id=result_id,
        result_path=result_path,
        filename="perturbation.png",
    )

    original_explanation_image_url = None
    adversarial_explanation_image_url = None
    if (
        original_explanation_image is not None
        and adversarial_explanation_image is not None
    ):
        original_explanation_image_url = save_result_image(
            image=original_explanation_image,
            result_id=result_id,
            result_path=result_path,
            filename="original_explanation.png",
        )
        adversarial_explanation_image_url = save_result_image(
            image=adversarial_explanation_image,
            result_id=result_id,
            result_path=result_path,
            filename="adversarial_explanation.png",
        )

    metadata = {
        "result_id": result_id,
        "result_type": "detection_attack",
        "dataset_id": request.dataset_id,
        "image_id": request.image_id,
        "model_id": model.id,
        "model_name": model.name,
        "attack_type": request.attack_type,
        "parameters": request.parameters.model_dump(),
        "original_detection_count": metrics.original_count,
        "adversarial_detection_count": metrics.adversarial_count,
        "expires_in_hours": 24,
    }
    if original_explanation_image_url and adversarial_explanation_image_url:
        metadata.update(
            {
                "explanation_type": "eigencam",
                "original_explanation_image": "original_explanation.png",
                "adversarial_explanation_image": "adversarial_explanation.png",
            }
        )

    save_result_metadata(
        result_path=result_path,
        metadata=metadata,
    )

    return DetectionAttackResponse(
        result_id=result_id,
        attack_type=request.attack_type,
        parameters=request.parameters,
        original_annotated_image_url=original_annotated_image_url,
        adversarial_image_url=adversarial_image_url,
        adversarial_annotated_image_url=adversarial_annotated_image_url,
        perturbation_image_url=perturbation_image_url,
        original_detections=to_detection_predictions(original_detections),
        adversarial_detections=to_detection_predictions(adversarial_detections),
        comparison=comparison,
        original_explanation_image_url=original_explanation_image_url,
        adversarial_explanation_image_url=adversarial_explanation_image_url,
    )
