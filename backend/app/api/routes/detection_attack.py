from fastapi import APIRouter

from schemas.detection_attack import DetectionAttackRequest, DetectionAttackResponse
from services.detection_attack_service import apply_detection_attack

router = APIRouter()


@router.post("/detect/attack", response_model=DetectionAttackResponse)
async def attack_detection_image(
    request: DetectionAttackRequest,
) -> DetectionAttackResponse:
    return await apply_detection_attack(request)