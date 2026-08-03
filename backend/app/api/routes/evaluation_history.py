from fastapi import APIRouter, Depends, Query

from api.dependencies.auth import require_current_user
from schemas.evaluation_history import (
    EvaluationCompatibilityRequest,
    EvaluationCompatibilityResponse,
    EvaluationHistoryPage,
    EvaluationTaskType,
    StoredEvaluationResponse,
)
from schemas.users import UserResponse
from services.evaluation_history_service import (
    find_compatible_evaluation,
    get_stored_evaluation,
    list_evaluation_history,
)

router = APIRouter()


@router.post(
    "/evaluations/compatibility",
    response_model=EvaluationCompatibilityResponse,
)
async def check_evaluation_compatibility(
    request: EvaluationCompatibilityRequest,
    current_user: UserResponse = Depends(require_current_user),
) -> EvaluationCompatibilityResponse:
    return await find_compatible_evaluation(
        request=request,
        owner_id=current_user.id,
    )


@router.get("/evaluations", response_model=EvaluationHistoryPage)
async def get_evaluation_history(
    limit: int = Query(default=10, ge=1, le=50),
    offset: int = Query(default=0, ge=0),
    task_type: EvaluationTaskType | None = Query(default=None),
    current_user: UserResponse = Depends(require_current_user),
) -> EvaluationHistoryPage:
    return await list_evaluation_history(
        owner_id=current_user.id,
        limit=limit,
        offset=offset,
        task_type=task_type,
    )


@router.get(
    "/evaluations/{evaluation_id}",
    response_model=StoredEvaluationResponse,
)
async def get_evaluation_history_result(
    evaluation_id: str,
    current_user: UserResponse = Depends(require_current_user),
) -> StoredEvaluationResponse:
    return await get_stored_evaluation(
        evaluation_id=evaluation_id,
        owner_id=current_user.id,
    )
