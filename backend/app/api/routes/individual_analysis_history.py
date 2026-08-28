from fastapi import APIRouter, Depends, Query, Response, status

from api.dependencies.auth import require_current_user
from schemas.individual_analysis_history import (
    IndividualAnalysisCompatibilityRequest,
    IndividualAnalysisCompatibilityResponse,
    IndividualAnalysisHistoryItem,
    IndividualAnalysisHistoryPage,
    IndividualAnalysisSaveRequest,
    IndividualAnalysisTaskType,
    StoredIndividualAnalysisResponse,
)
from schemas.users import UserResponse
from services.individual_analysis_history_service import (
    delete_individual_analysis,
    find_compatible_individual_analysis,
    get_stored_individual_analysis,
    list_individual_analysis_history,
    save_individual_analysis,
)

router = APIRouter()


@router.post(
    "/individual-analyses/compatibility",
)
async def check_individual_analysis_compatibility(
    request: IndividualAnalysisCompatibilityRequest,
    current_user: UserResponse = Depends(require_current_user),
) -> IndividualAnalysisCompatibilityResponse:
    return await find_compatible_individual_analysis(
        request=request,
        owner_id=current_user.id,
    )


@router.post(
    "/individual-analyses",
    status_code=status.HTTP_201_CREATED,
)
async def persist_individual_analysis(
    request: IndividualAnalysisSaveRequest,
    current_user: UserResponse = Depends(require_current_user),
) -> IndividualAnalysisHistoryItem:
    return await save_individual_analysis(request=request, owner_id=current_user.id)


@router.get(
    "/individual-analyses",
)
async def get_individual_analysis_history(
    limit: int = Query(default=10, ge=1, le=50),
    offset: int = Query(default=0, ge=0),
    task_type: IndividualAnalysisTaskType | None = Query(default=None),
    current_user: UserResponse = Depends(require_current_user),
) -> IndividualAnalysisHistoryPage:
    return await list_individual_analysis_history(
        owner_id=current_user.id,
        limit=limit,
        offset=offset,
        task_type=task_type,
    )


@router.get(
    "/individual-analyses/{analysis_id}",
)
async def get_individual_analysis(
    analysis_id: str,
    current_user: UserResponse = Depends(require_current_user),
) -> StoredIndividualAnalysisResponse:
    return await get_stored_individual_analysis(
        analysis_id=analysis_id,
        owner_id=current_user.id,
    )


@router.delete(
    "/individual-analyses/{analysis_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def remove_individual_analysis(
    analysis_id: str,
    current_user: UserResponse = Depends(require_current_user),
) -> Response:
    await delete_individual_analysis(analysis_id=analysis_id, owner_id=current_user.id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
