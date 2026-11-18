from fastapi import APIRouter, Depends, Request

from api.dependencies.auth import require_current_user
from schemas.detection_attack import DetectionAttackRequest, DetectionAttackResponse
from schemas.users import UserResponse
from services.detection_attack_service import apply_detection_attack
from services.request_cancellation import run_cancellable_request

router = APIRouter()


@router.post("/detect/attack")
async def attack_detection_image(
    request: DetectionAttackRequest,
    http_request: Request,
    current_user: UserResponse = Depends(require_current_user),
) -> DetectionAttackResponse:
    return await run_cancellable_request(
        http_request,
        lambda cancellation_token: apply_detection_attack(
            request,
            owner_id=current_user.id,
            cancellation_token=cancellation_token,
        ),
    )
