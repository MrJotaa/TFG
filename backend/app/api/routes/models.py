from fastapi import APIRouter

from schemas.models import ModelInfo
from services.model_service import (
    get_active_model,
    list_models,
    select_model,
)


router = APIRouter()


@router.get("/models", response_model=list[ModelInfo])
async def get_models() -> list[ModelInfo]:
    return list_models()


@router.get("/models/active", response_model=ModelInfo)
async def get_current_model() -> ModelInfo:
    return get_active_model()


@router.post(
    "/models/{model_id}/select",
    response_model=ModelInfo,
)
async def select_current_model(model_id: str) -> ModelInfo:
    return select_model(model_id)
