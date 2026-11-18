from fastapi import APIRouter, Depends, Request

from api.dependencies.auth import require_current_user
from schemas.detection import DetectionRequest, DetectionResponse
from schemas.users import UserResponse
from services.detection_service import detect_objects
from services.request_cancellation import run_cancellable_request

router = APIRouter()


@router.post("/detect")
async def detect_image(
    request: DetectionRequest,
    http_request: Request,
    current_user: UserResponse = Depends(require_current_user),
) -> DetectionResponse:
    return await run_cancellable_request(
        http_request,
        lambda cancellation_token: detect_objects(
            request,
            owner_id=current_user.id,
            cancellation_token=cancellation_token,
        ),
    )
