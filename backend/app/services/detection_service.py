from fastapi import HTTPException

from schemas.detection import DetectionRequest, DetectionResponse
from services.dataset_service import get_image_from_dataset
from services.model_service import get_active_model_instance
from utils.result_files import (
    create_result_directory,
    save_result_image,
    save_result_metadata,
)


async def detect_objects(request: DetectionRequest) -> DetectionResponse:
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

    if getattr(model, "task_type", None) != "detection":
        raise HTTPException(
            status_code=400,
            detail=(
                f"Active model '{model.id}' is not a detection model. "
                "Select a detection model before calling /detect."
            ),
        )

    detection_response, annotated_image = model.detect_with_visualization(image)

    result_id, result_path = create_result_directory(prefix="detection")

    annotated_image_url = save_result_image(
        image=annotated_image,
        result_id=result_id,
        result_path=result_path,
        filename="annotated.png",
    )

    save_result_metadata(
        result_path=result_path,
        metadata={
            "result_id": result_id,
            "result_type": "detection",
            "dataset_id": request.dataset_id,
            "image_id": request.image_id,
            "model_id": model.id,
            "model_name": model.name,
            "detections_count": len(detection_response.detections),
            "expires_in_hours": 24,
        },
    )

    detection_response.annotated_image_url = annotated_image_url

    return detection_response