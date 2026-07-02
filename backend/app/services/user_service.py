import os
import shutil

from bson import ObjectId

from core.errors import BadRequestError, NotFoundError
from db.collections import users_collection
from schemas.auth import SignupRequest
from schemas.users import UserCreate, UserResponse, UserUpdate
from services.dataset_service import (
    UPLOADS_DATASETS_PATH,
    delete_dataset,
)
from utils.security import hash_password, verify_password


INVALID_USER_ID_MESSAGE = "Invalid user id"
USER_NOT_FOUND_MESSAGE = "User not found"
USER_ALREADY_EXISTS_MESSAGE = "User with this email already exists"


def user_to_response(user: dict) -> UserResponse:
    return UserResponse(
        id=str(user["_id"]),
        name=user["name"],
        email=user["email"],
    )


def _validate_user_id(user_id: str) -> ObjectId:
    if not ObjectId.is_valid(user_id):
        raise BadRequestError(INVALID_USER_ID_MESSAGE)

    return ObjectId(user_id)


async def list_users() -> list[UserResponse]:
    users = []

    async for user in users_collection.find():
        users.append(user_to_response(user))

    return users


async def create_user(user_data: UserCreate) -> UserResponse:
    existing_user = await users_collection.find_one(
        {"email": user_data.email}
    )

    if existing_user:
        raise BadRequestError(USER_ALREADY_EXISTS_MESSAGE)

    user_doc = {
        "name": user_data.name,
        "email": user_data.email,
        "password": hash_password(user_data.password),
        "datasets": [],
    }

    result = await users_collection.insert_one(user_doc)

    created_user = await users_collection.find_one(
        {"_id": result.inserted_id}
    )

    return user_to_response(created_user)


async def get_user(user_id: str) -> UserResponse:
    object_id = _validate_user_id(user_id)

    user = await users_collection.find_one({"_id": object_id})

    if not user:
        raise NotFoundError(USER_NOT_FOUND_MESSAGE)

    return user_to_response(user)


async def update_user(
    user_id: str,
    user_data: UserUpdate,
) -> UserResponse:
    object_id = _validate_user_id(user_id)

    user = await users_collection.find_one({"_id": object_id})

    if not user:
        raise NotFoundError(USER_NOT_FOUND_MESSAGE)

    update_data = {}

    if user_data.name is not None:
        update_data["name"] = user_data.name

    if user_data.email is not None:
        update_data["email"] = user_data.email

    if user_data.password is not None:
        update_data["password"] = hash_password(user_data.password)

    if update_data:
        await users_collection.update_one(
            {"_id": object_id},
            {"$set": update_data},
        )

    updated_user = await users_collection.find_one(
        {"_id": object_id}
    )

    if not updated_user:
        raise NotFoundError(USER_NOT_FOUND_MESSAGE)

    return user_to_response(updated_user)


async def delete_user(user_id: str) -> None:
    object_id = _validate_user_id(user_id)

    user = await users_collection.find_one({"_id": object_id})

    if not user:
        raise NotFoundError(USER_NOT_FOUND_MESSAGE)

    dataset_ids = user.get("datasets", [])

    for dataset_id in dataset_ids:
        await delete_dataset(str(dataset_id))

    user_datasets_path = os.path.join(
        UPLOADS_DATASETS_PATH,
        f"user_{user_id}",
    )

    if os.path.exists(user_datasets_path):
        shutil.rmtree(
            user_datasets_path,
            ignore_errors=True,
        )

    result = await users_collection.delete_one(
        {"_id": object_id}
    )

    if result.deleted_count == 0:
        raise NotFoundError(USER_NOT_FOUND_MESSAGE)


async def get_user_document_by_email(email: str) -> dict | None:
    return await users_collection.find_one({"email": email})


async def get_user_document_by_id(user_id: str) -> dict | None:
    if not ObjectId.is_valid(user_id):
        return None

    return await users_collection.find_one(
        {"_id": ObjectId(user_id)}
    )


async def create_user_from_signup(
    signup_data: SignupRequest,
) -> UserResponse:
    user_data = UserCreate(
        name=signup_data.name,
        email=signup_data.email,
        password=signup_data.password,
    )

    return await create_user(user_data)


async def delete_user_with_password(
    user_id: str,
    password: str,
) -> None:
    object_id = _validate_user_id(user_id)

    user = await users_collection.find_one({"_id": object_id})

    if not user:
        raise NotFoundError(USER_NOT_FOUND_MESSAGE)

    hashed_password = user.get("password")

    if not hashed_password:
        raise BadRequestError("Invalid password")

    if not verify_password(password, hashed_password):
        raise BadRequestError("Invalid password")

    await delete_user(user_id)