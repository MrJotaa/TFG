from fastapi import APIRouter, Depends, Request

from api.dependencies.auth import require_current_user
from ml.attacks.registry import list_attack_catalog
from schemas.attack import AttackCatalogItem, AttackRequest, AttackResponse
from schemas.users import UserResponse
from services.attack_service import apply_adversarial_attack
from services.request_cancellation import run_cancellable_request

router = APIRouter()


@router.get("/attacks")
async def get_attacks(
    _current_user: UserResponse = Depends(require_current_user),
) -> list[AttackCatalogItem]:
    return list_attack_catalog()


@router.post("/attack")
async def attack(
    request: AttackRequest,
    http_request: Request,
    current_user: UserResponse = Depends(require_current_user),
) -> AttackResponse:
    return await run_cancellable_request(
        http_request,
        lambda cancellation_token: apply_adversarial_attack(
            request,
            owner_id=current_user.id,
            cancellation_token=cancellation_token,
        ),
    )
