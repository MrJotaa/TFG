from fastapi import APIRouter

from schemas.evaluation import EvaluationRequest, EvaluationResponse
from services.evaluation_service import evaluate_dataset

router = APIRouter()


@router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate(request: EvaluationRequest) -> EvaluationResponse:
    return await evaluate_dataset(request)