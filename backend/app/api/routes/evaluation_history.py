from fastapi import APIRouter, Depends, Query, Response, status

from api.dependencies.auth import require_current_user
from schemas.evaluation_history import (
    EvaluationCompatibilityRequest,
    EvaluationCompatibilityResponse,
    EvaluationHistoryItem,
    EvaluationHistoryPage,
    EvaluationPreservationUpdate,
    EvaluationTaskType,
    StoredEvaluationResponse,
)
from schemas.users import UserResponse
from services.evaluation_history_service import (
    delete_evaluation_result,
    find_compatible_evaluation,
    get_stored_evaluation,
    list_evaluation_history,
    update_evaluation_preservation,
)

router = APIRouter()


@router.post(
    "/evaluations/compatibility",
)
async def check_evaluation_compatibility(
    request: EvaluationCompatibilityRequest,
    current_user: UserResponse = Depends(require_current_user),
) -> EvaluationCompatibilityResponse:
    return await find_compatible_evaluation(
        request=request,
        owner_id=current_user.id,
    )


@router.get("/evaluations")
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
)
async def get_evaluation_history_result(
    evaluation_id: str,
    current_user: UserResponse = Depends(require_current_user),
) -> StoredEvaluationResponse:
    return await get_stored_evaluation(
        evaluation_id=evaluation_id,
        owner_id=current_user.id,
    )


@router.patch(
    "/evaluations/{evaluation_id}/preservation",
)
async def change_evaluation_preservation(
    evaluation_id: str,
    update: EvaluationPreservationUpdate,
    current_user: UserResponse = Depends(require_current_user),
) -> EvaluationHistoryItem:
    return await update_evaluation_preservation(
        evaluation_id=evaluation_id,
        owner_id=current_user.id,
        update=update,
        retention_seconds=current_user.result_retention_seconds,
    )


@router.delete(
    "/evaluations/{evaluation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_evaluation_history_result(
    evaluation_id: str,
    current_user: UserResponse = Depends(require_current_user),
) -> Response:
    await delete_evaluation_result(
        evaluation_id=evaluation_id,
        owner_id=current_user.id,
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)
