import os
import time

from bson import ObjectId
from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse

from api.dependencies.auth import require_current_user
from core.errors import GoneError, NotFoundError
from db.collections import datasets_collection
from schemas.users import UserResponse
from services.evaluation_history_service import (
    evaluation_artifact_belongs_to_user,
    evaluation_artifact_is_accessible,
)
from services.individual_analysis_history_service import (
    individual_analysis_artifact_is_accessible,
)
from utils.result_files import (
    RESULTS_BASE_PATH,
    read_result_expiration,
    read_result_metadata,
)

router = APIRouter()
_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"})
_NOT_FOUND = "Artefacto de resultado no encontrado"


async def _temporary_result_belongs_to_user(result_id: str, owner_id: str) -> bool:
    """Authorize temporary artifacts, including older results without owner metadata.

    Older artifacts can only be recovered while their source dataset still
    exists and is owned by the authenticated user. No anonymous fallback.
    """
    try:
        metadata = read_result_metadata(result_id)
    except (OSError, ValueError, TypeError):
        return False

    metadata_owner = metadata.get("owner_id")
    if metadata_owner is not None:
        return str(metadata_owner) == owner_id

    dataset_id = metadata.get("dataset_id")
    if not isinstance(dataset_id, str) or not ObjectId.is_valid(dataset_id):
        return False
    dataset = await datasets_collection.find_one(
        {"_id": ObjectId(dataset_id)}, {"owner_id": 1}
    )
    return dataset is not None and str(dataset.get("owner_id")) == owner_id


@router.get("/results/{result_id}/{filename}")
async def get_temporary_result_artifact(
    result_id: str,
    filename: str,
    current_user: UserResponse = Depends(require_current_user),
) -> FileResponse:
    if (
        result_id in {".", ".."}
        or os.path.basename(result_id) != result_id
        or os.path.basename(filename) != filename
        or filename.startswith(".")
        or os.path.splitext(filename)[1].lower() not in _IMAGE_EXTENSIONS
    ):
        raise NotFoundError(_NOT_FOUND)

    result_path = os.path.join(RESULTS_BASE_PATH, result_id)
    artifact_path = os.path.join(result_path, filename)
    if not os.path.isfile(artifact_path):
        raise NotFoundError(_NOT_FOUND)

    if result_id.startswith(("evaluation_", "detection_evaluation_")):
        # Persisted evaluations have their own owner and retention in MongoDB.
        owned = await evaluation_artifact_belongs_to_user(
            result_id, current_user.id
        )
        if not owned:
            raise NotFoundError(_NOT_FOUND)
        accessible = await evaluation_artifact_is_accessible(
            result_id, owner_id=current_user.id
        )
    else:
        saved = False
        if result_id.startswith(("attack_", "detection_attack_")):
            saved = await individual_analysis_artifact_is_accessible(
                result_id, owner_id=current_user.id
            )
        if not saved and not await _temporary_result_belongs_to_user(
            result_id, current_user.id
        ):
            raise NotFoundError(_NOT_FOUND)
        accessible = saved or read_result_expiration(result_path) > time.time()

    if not accessible:
        raise GoneError("El artefacto de resultado ha caducado")

    return FileResponse(
        artifact_path,
        headers={
            "Cache-Control": "private, no-store, max-age=0",
            "Pragma": "no-cache",
        },
    )
