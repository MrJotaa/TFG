from schemas.users import UserCreate, UserResponse, UserUpdate


_users: dict[int, dict] = {}
_next_user_id = 1


def list_users() -> list[UserResponse]:
    return [
        UserResponse(id=user_id, name=user["name"], email=user["email"])
        for user_id, user in _users.items()
    ]


def create_user(user_data: UserCreate) -> UserResponse:
    global _next_user_id

    user_id = _next_user_id
    _next_user_id += 1

    _users[user_id] = {
        "name": user_data.name,
        "email": user_data.email,
        "password": user_data.password,
        "datasets": [],
    }

    return UserResponse(
        id=user_id,
        name=user_data.name,
        email=user_data.email,
    )


def get_user(user_id: int) -> UserResponse:
    if user_id not in _users:
        raise ValueError(f"User with id {user_id} not found")

    user = _users[user_id]
    return UserResponse(
        id=user_id,
        name=user["name"],
        email=user["email"],
    )


def update_user(user_id: int, user_data: UserUpdate) -> UserResponse:
    if user_id not in _users:
        raise ValueError(f"User with id {user_id} not found")

    stored_user = _users[user_id]

    if user_data.name is not None:
        stored_user["name"] = user_data.name

    if user_data.email is not None:
        stored_user["email"] = user_data.email

    if user_data.password is not None:
        stored_user["password"] = user_data.password

    return UserResponse(
        id=user_id,
        name=stored_user["name"],
        email=stored_user["email"],
    )


def delete_user(user_id: int) -> None:
    if user_id not in _users:
        raise ValueError(f"User with id {user_id} not found")

    del _users[user_id]