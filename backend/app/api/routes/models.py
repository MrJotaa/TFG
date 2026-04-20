from fastapi import APIRouter, HTTPException

from schemas.models import ModelInfo
from services.model_service import get_active_model, list_models, select_model

router = APIRouter()


@router.get("/models", response_model=list[ModelInfo])
async def get_models() -> list[ModelInfo]:
    return list_models()


@router.get("/models/active", response_model=ModelInfo)
async def get_current_model() -> ModelInfo:
    try:
        return get_active_model()
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/models/{model_id}/select", response_model=ModelInfo)
async def select_current_model(model_id: str) -> ModelInfo:
    try:
        return select_model(model_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))