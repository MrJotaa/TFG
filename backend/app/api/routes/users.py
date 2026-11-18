from typing import Annotated

from fastapi import APIRouter, Depends

from api.dependencies.auth import require_current_user
from core.errors import NotFoundError
from schemas.datasets import DatasetResponse
from schemas.users import (
    UserCreate,
    UserDeleteRequest,
    UserResponse,
    UserRetentionUpdate,
    UserUpdate,
)
from services.dataset_service import get_datasets_by_user
from services.user_service import (
    create_user,
    delete_user_with_password,
    get_user,
    update_user,
    update_user_retention,
)

router = APIRouter()


async def _require_own_user(
    user_id: str,
    current_user: UserResponse,
) -> None:
    # Conserva la validación de IDs y el 404 de usuarios inexistentes.
    # Un usuario existente pero ajeno se trata también como no encontrado.
    user = await get_user(user_id)
    if user.id != current_user.id:
        raise NotFoundError("Usuario no encontrado")


@router.patch(
    "/users/me/retention",
)
async def update_current_user_retention(
    retention: UserRetentionUpdate,
    current_user: Annotated[UserResponse, Depends(require_current_user)],
) -> UserResponse:
    return await update_user_retention(current_user.id, retention)


@router.get("/users")
async def get_users(
    current_user: Annotated[UserResponse, Depends(require_current_user)],
) -> list[UserResponse]:
    # Compatibilidad con la ruta antigua sin permitir enumerar cuentas.
    return [current_user]


@router.post("/users")
async def register_user(
    user: UserCreate,
    _current_user: Annotated[UserResponse, Depends(require_current_user)],
) -> UserResponse:
    return await create_user(user)


@router.get(
    "/users/{user_id}",
)
async def get_user_by_id(
    user_id: str,
    current_user: Annotated[UserResponse, Depends(require_current_user)],
) -> UserResponse:
    await _require_own_user(user_id, current_user)
    return current_user


@router.patch("/users/me")
async def update_current_user(
    user: UserUpdate,
    current_user: Annotated[UserResponse, Depends(require_current_user)],
) -> UserResponse:
    return await update_user(current_user.id, user)


@router.patch(
    "/users/{user_id}",
)
async def update_user_by_id(
    user_id: str,
    user: UserUpdate,
    current_user: Annotated[UserResponse, Depends(require_current_user)],
) -> UserResponse:
    await _require_own_user(user_id, current_user)
    return await update_user(current_user.id, user)


@router.delete("/users/me")
async def delete_current_user_account(
    delete_request: UserDeleteRequest,
    current_user: Annotated[UserResponse, Depends(require_current_user)],
) -> dict:
    await delete_user_with_password(
        user_id=current_user.id,
        password=delete_request.password,
    )

    return {
        "message": "User account deleted successfully",
    }


@router.delete("/users/{user_id}")
async def delete_user_by_id(
    user_id: str,
    delete_request: UserDeleteRequest,
    current_user: Annotated[UserResponse, Depends(require_current_user)],
) -> dict:
    await _require_own_user(user_id, current_user)
    await delete_user_with_password(
        user_id=current_user.id,
        password=delete_request.password,
    )

    return {
        "message": "User deleted successfully",
    }


@router.get(
    "/users/{user_id}/datasets",
)
async def get_user_datasets(
    user_id: str,
    current_user: Annotated[UserResponse, Depends(require_current_user)],
) -> list[DatasetResponse]:
    await _require_own_user(user_id, current_user)
    return await get_datasets_by_user(current_user.id)
