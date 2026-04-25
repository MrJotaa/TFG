from bson import ObjectId

from db.collections import users_collection
from schemas.users import UserCreate, UserResponse, UserUpdate
from utils.security import hash_password


def user_to_response(user: dict) -> UserResponse:
    return UserResponse(
        id=str(user["_id"]),
        name=user["name"],
        email=user["email"],
    )


async def list_users() -> list[UserResponse]:
    users = []

    async for user in users_collection.find():
        users.append(user_to_response(user))

    return users


async def create_user(user_data: UserCreate) -> UserResponse:
    existing_user = await users_collection.find_one({"email": user_data.email})

    if existing_user:
        raise ValueError(f"User with email {user_data.email} already exists")

    user_doc = {
        "name": user_data.name,
        "email": user_data.email,
        "password": hash_password(user_data.password),
        "datasets": [],
    }

    result = await users_collection.insert_one(user_doc)

    created_user = await users_collection.find_one({"_id": result.inserted_id})

    return user_to_response(created_user)


async def get_user(user_id: str) -> UserResponse:
    if not ObjectId.is_valid(user_id):
        raise ValueError(f"Invalid user ID: {user_id}")

    user = await users_collection.find_one({"_id": ObjectId(user_id)})

    if not user:
        raise ValueError(f"User with id {user_id} not found")

    return user_to_response(user)


async def update_user(user_id: str, user_data: UserUpdate) -> UserResponse:
    if not ObjectId.is_valid(user_id):
        raise ValueError(f"Invalid user ID: {user_id}")

    update_data = {}

    if user_data.name is not None:
        update_data["name"] = user_data.name

    if user_data.email is not None:
        update_data["email"] = user_data.email

    if user_data.password is not None:
        update_data["hashed_password"] = hash_password(user_data.password)

    if update_data:
        await users_collection.update_one({"_id": ObjectId(user_id)}, {"$set": update_data})

    user = await users_collection.find_one({"_id": ObjectId(user_id)})

    if not user:
        raise ValueError(f"User with id {user_id} not found")

    return user_to_response(user)


async def delete_user(user_id: str) -> None:
    if not ObjectId.is_valid(user_id):
        raise ValueError(f"Invalid user ID: {user_id}")

    result = await users_collection.delete_one({"_id": ObjectId(user_id)})

    if result.deleted_count == 0:
        raise ValueError(f"User with id {user_id} not found")