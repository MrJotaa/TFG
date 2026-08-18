from typing import Literal

from pydantic import BaseModel, EmailStr

ResultRetentionSeconds = Literal[3600, 7200, 86400, 172800, 604800]
DEFAULT_RESULT_RETENTION_SECONDS = 86400


class UserCreate(BaseModel):
    name: str
    email: EmailStr
    password: str


class UserUpdate(BaseModel):
    name: str | None = None
    email: EmailStr | None = None
    password: str | None = None


class UserRetentionUpdate(BaseModel):
    result_retention_seconds: ResultRetentionSeconds


class UserResponse(BaseModel):
    id: str
    name: str
    email: EmailStr
    result_retention_seconds: ResultRetentionSeconds = DEFAULT_RESULT_RETENTION_SECONDS


class UserDeleteRequest(BaseModel):
    password: str
