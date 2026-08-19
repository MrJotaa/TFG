from fastapi import APIRouter

from ml.attacks.registry import list_attack_catalog
from schemas.attack import AttackCatalogItem, AttackRequest, AttackResponse
from services.attack_service import apply_adversarial_attack

router = APIRouter()


@router.get("/attacks", response_model=list[AttackCatalogItem])
async def get_attacks() -> list[AttackCatalogItem]:
    return list_attack_catalog()


@router.post("/attack", response_model=AttackResponse)
async def attack(request: AttackRequest) -> AttackResponse:
    return await apply_adversarial_attack(request)
