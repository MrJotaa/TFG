from fastapi import APIRouter

from schemas.attack import AttackRequest, AttackResponse
from services.attack_service import apply_adversarial_attack

router = APIRouter()


@router.post("/attack", response_model=AttackResponse)
async def attack(request: AttackRequest) -> AttackResponse:
    return await apply_adversarial_attack(request)
