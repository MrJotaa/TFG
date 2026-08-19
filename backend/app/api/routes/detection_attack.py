from fastapi import APIRouter, Request

from schemas.detection_attack import DetectionAttackRequest, DetectionAttackResponse
from services.detection_attack_service import apply_detection_attack
from services.request_cancellation import run_cancellable_request

router = APIRouter()


@router.post("/detect/attack", response_model=DetectionAttackResponse)
async def attack_detection_image(
    request: DetectionAttackRequest,
    http_request: Request,
) -> DetectionAttackResponse:
    return await run_cancellable_request(
        http_request,
        lambda cancellation_token: apply_detection_attack(
            request,
            cancellation_token=cancellation_token,
        ),
    )
