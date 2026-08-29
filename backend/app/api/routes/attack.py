from fastapi import APIRouter, Request

from ml.attacks.registry import list_attack_catalog
from schemas.attack import AttackCatalogItem, AttackRequest, AttackResponse
from services.attack_service import apply_adversarial_attack
from services.request_cancellation import run_cancellable_request

router = APIRouter()


@router.get("/attacks")
async def get_attacks() -> list[AttackCatalogItem]:
    return list_attack_catalog()


@router.post("/attack")
async def attack(
    request: AttackRequest,
    http_request: Request,
) -> AttackResponse:
    return await run_cancellable_request(
        http_request,
        lambda cancellation_token: apply_adversarial_attack(
            request,
            cancellation_token=cancellation_token,
        ),
    )
