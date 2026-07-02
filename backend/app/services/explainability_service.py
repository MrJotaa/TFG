from fastapi import HTTPException

from ml.explainability.gradcam import (
    draw_detections_on_image,
    generate_detection_eigencam,
    generate_gradcam,
)
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
from core.task_compatibility import (
    CLASSIFICATION_TASK_TYPE,
    DETECTION_TASK_TYPE,
    require_model_task_type,
)


async def generate_gradcam_for_image(dataset_id: str, image_id: str) -> GradCamResponse:
    try:
        image = await get_image_from_dataset(dataset_id, image_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    try:
        model = get_active_model_instance()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    require_model_task_type(
        model=model,
        expected_task_type=CLASSIFICATION_TASK_TYPE,
        endpoint="/explain/gradcam",
    )

    result = generate_gradcam(model, image)

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
) -> DetectionExplainResponse:
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

    require_model_task_type(
        model=model,
        expected_task_type=DETECTION_TASK_TYPE,
        endpoint="/explain/detection",
    )

    detection_response = model.detect(image)
    detections = detection_response.detections

    if not detections:
        raise HTTPException(
            status_code=400,
            detail="No detections found in the selected image.",
        )

    heatmap_image = generate_detection_eigencam(model, image)

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
        detections=detections,
    )
