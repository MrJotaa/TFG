from fastapi import HTTPException

from ml.explainability.gradcam import generate_gradcam
from schemas.explainability import GradCamResponse
from services.dataset_service import get_image_from_dataset
from services.model_service import get_active_model_instance


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

    return GradCamResponse(**result)