from core.errors import UnauthorizedError
from schemas.auth import AuthResponse, LoginRequest, SignupRequest
from schemas.users import UserResponse
from services.user_service import (
    create_user_from_signup,
    get_user_document_by_email,
    get_user_document_by_id,
    user_to_response,
)
from utils.security import (
    create_access_token,
    create_artifact_access_token,
    decode_access_token,
    verify_password,
)

INVALID_CREDENTIALS_MESSAGE = "Correo electrónico o contraseña no válidos"
INVALID_TOKEN_MESSAGE = "Token no válido o caducado"


def create_auth_response(user: UserResponse) -> AuthResponse:
    token = create_access_token(_token_identity(user))

    return AuthResponse(
        access_token=token,
        user=user,
    )


def create_artifact_token(user: UserResponse) -> str:
    return create_artifact_access_token(_token_identity(user))


def _token_identity(user: UserResponse) -> dict:
    return {
        "sub": user.id,
        "email": user.email,
    }


async def signup(signup_data: SignupRequest) -> AuthResponse:
    user = await create_user_from_signup(signup_data)

    return create_auth_response(user)


async def login(login_data: LoginRequest) -> AuthResponse:
    user_document = await get_user_document_by_email(login_data.email)

    if not user_document:
        raise UnauthorizedError(INVALID_CREDENTIALS_MESSAGE)

    hashed_password = user_document.get("password")

    if not hashed_password:
        raise UnauthorizedError(INVALID_CREDENTIALS_MESSAGE)

    if not verify_password(login_data.password, hashed_password):
        raise UnauthorizedError(INVALID_CREDENTIALS_MESSAGE)

    user = user_to_response(user_document)

    return create_auth_response(user)


async def get_current_user(token: str) -> UserResponse:
    payload = _decode_token(token)
    if payload.get("scope") == "artifacts":
        raise UnauthorizedError(INVALID_TOKEN_MESSAGE)
    return await _user_from_token_payload(payload)


async def get_artifact_current_user(token: str) -> UserResponse:
    payload = _decode_token(token)
    if payload.get("scope") != "artifacts":
        raise UnauthorizedError(INVALID_TOKEN_MESSAGE)
    return await _user_from_token_payload(payload)


def _decode_token(token: str) -> dict:
    try:
        return decode_access_token(token)
    except ValueError as error:
        raise UnauthorizedError(INVALID_TOKEN_MESSAGE) from error


async def _user_from_token_payload(payload: dict) -> UserResponse:
    user_id = payload.get("sub")

    if not user_id:
        raise UnauthorizedError("Contenido del token no válido")

    user_document = await get_user_document_by_id(user_id)

    if not user_document:
        raise UnauthorizedError("Usuario no encontrado")

    return user_to_response(user_document)
