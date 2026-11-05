import asyncio

from core.cancellation import CancellationToken
from core.errors import BadRequestError
from core.task_compatibility import (
    CLASSIFICATION_TASK_TYPE,
    DETECTION_TASK_TYPE,
    require_model_task_type,
)
from ml.explainability.gradcam import (
    generate_clean_gradcam,
    generate_detection_eigencam,
)
from ml.models.detection_visualization import draw_detections_on_image
from schemas.detection import to_detection_predictions
from schemas.explainability import (
    DetectionExplainRequest,
    DetectionExplainResponse,
    GradCamResponse,
)
from services.dataset_service import get_image_from_dataset
from services.model_service import get_active_model_instance
from utils.result_files import (
    create_result_directory,
    save_result_image,
    save_result_metadata,
)


async def generate_gradcam_for_image(
    dataset_id: str,
    image_id: str,
    *,
    cancellation_token: CancellationToken | None = None,
    owner_id: str | None = None,
) -> GradCamResponse:
    image = await get_image_from_dataset(
        dataset_id=dataset_id,
        image_id=image_id,
        owner_id=owner_id,
    )

    model = get_active_model_instance()

    require_model_task_type(
        model=model,
        expected_task_type=CLASSIFICATION_TASK_TYPE,
        endpoint="/explain/gradcam",
    )

    source_image = image.copy().convert("RGB")
    model_input = model.preprocess(image)
    if cancellation_token is not None:
        cancellation_token.raise_if_cancelled()
    result = await asyncio.to_thread(
        generate_clean_gradcam,
        model_wrapper=model,
        model_input=model_input,
        source_image=source_image,
    )
    if cancellation_token is not None:
        cancellation_token.raise_if_cancelled()

    result_id, result_path = create_result_directory(prefix="gradcam")

    gradcam_image_url = save_result_image(
        image=result["gradcam_pil_image"],
        result_id=result_id,
        result_path=result_path,
        filename="gradcam.png",
    )

    save_result_metadata(
        result_path=result_path,
        metadata={
            "owner_id": owner_id,
            "result_id": result_id,
            "result_type": "gradcam",
            "dataset_id": dataset_id,
            "image_id": image_id,
            "model_id": model.id,
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "expires_in_hours": 24,
        },
    )

    return GradCamResponse(
        result_id=result_id,
        prediction=result["prediction"],
        confidence=result["confidence"],
        gradcam_image_url=gradcam_image_url,
    )


async def generate_detection_explainability(
    request: DetectionExplainRequest,
    *,
    cancellation_token: CancellationToken | None = None,
    owner_id: str | None = None,
) -> DetectionExplainResponse:
    image = await get_image_from_dataset(
        dataset_id=request.dataset_id,
        image_id=request.image_id,
        owner_id=owner_id,
    )

    model = get_active_model_instance()

    require_model_task_type(
        model=model,
        expected_task_type=DETECTION_TASK_TYPE,
        endpoint="/explain/detection",
    )

    if cancellation_token is not None:
        cancellation_token.raise_if_cancelled()
    detection_result = await asyncio.to_thread(model.detect, image)
    if cancellation_token is not None:
        cancellation_token.raise_if_cancelled()
    detections = detection_result.detections

    if not detections:
        raise BadRequestError("No se encontraron detecciones en la imagen seleccionada.")

    heatmap_image = await asyncio.to_thread(generate_detection_eigencam, model, image)
    if cancellation_token is not None:
        cancellation_token.raise_if_cancelled()

    explanation_image = draw_detections_on_image(
        image=heatmap_image,
        detections=detections,
    )

    result_id, result_path = create_result_directory(prefix="detection_explain")

    explanation_image_url = save_result_image(
        image=explanation_image,
        result_id=result_id,
        result_path=result_path,
        filename="detection_explanation.png",
    )

    save_result_metadata(
        result_path=result_path,
        metadata={
            "owner_id": owner_id,
            "result_id": result_id,
            "result_type": "detection_eigencam",
            "dataset_id": request.dataset_id,
            "image_id": request.image_id,
            "model_id": model.id,
            "model_name": model.name,
            "detections_count": len(detections),
            "explanation_type": "eigencam",
            "expires_in_hours": 24,
        },
    )

    return DetectionExplainResponse(
        result_id=result_id,
        model_id=model.id,
        model_name=model.name,
        explanation_type="eigencam",
        explanation_image_url=explanation_image_url,
        detections=to_detection_predictions(detections),
    )
