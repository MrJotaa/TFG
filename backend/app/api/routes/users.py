from typing import Annotated

from fastapi import APIRouter, Depends

from api.dependencies.auth import require_current_user
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
    delete_user,
    delete_user_with_password,
    get_user,
    list_users,
    update_user,
    update_user_retention,
)

router = APIRouter()


@router.patch(
    "/users/me/retention",
)
async def update_current_user_retention(
    retention: UserRetentionUpdate,
    current_user: Annotated[UserResponse, Depends(require_current_user)],
) -> UserResponse:
    return await update_user_retention(current_user.id, retention)


@router.get("/users")
async def get_users() -> list[UserResponse]:
    return await list_users()


@router.post("/users")
async def register_user(
    user: UserCreate,
) -> UserResponse:
    return await create_user(user)


@router.get(
    "/users/{user_id}",
)
async def get_user_by_id(
    user_id: str,
) -> UserResponse:
    return await get_user(user_id)


@router.patch(
    "/users/{user_id}",
)
async def update_user_by_id(
    user_id: str,
    user: UserUpdate,
) -> UserResponse:
    return await update_user(user_id, user)


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
) -> dict:
    await delete_user(user_id)

    return {
        "message": "User deleted successfully",
    }


@router.get(
    "/users/{user_id}/datasets",
)
async def get_user_datasets(
    user_id: str,
) -> list[DatasetResponse]:
    return await get_datasets_by_user(user_id)
