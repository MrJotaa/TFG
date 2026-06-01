from fastapi import HTTPException

from ml.explainability.gradcam import generate_gradcam
from schemas.explainability import GradCamResponse
from services.dataset_service import get_image_from_dataset
from services.model_service import get_active_model_instance
from utils.result_files import (
    create_result_directory,
    save_result_image,
    save_result_metadata,
)


async def generate_gradcam_for_image(dataset_id: str, image_id: str) -> GradCamResponse:
    try:
        image = await get_image_from_dataset(dataset_id, image_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    try:
        model = get_active_model_instance()
        result = generate_gradcam(model, image)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

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