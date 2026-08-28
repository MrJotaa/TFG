import os
import time

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse

from api.dependencies.auth import require_artifact_current_user
from core.errors import GoneError, NotFoundError
from schemas.users import UserResponse
from services.evaluation_history_service import evaluation_artifact_is_accessible
from services.individual_analysis_history_service import (
    individual_analysis_artifact_is_accessible,
)
from services.result_artifact_access_service import temporary_result_belongs_to_user
from utils.result_files import RESULTS_BASE_PATH, read_result_expiration

router = APIRouter()


@router.get("/results/{result_id}/{filename}")
async def get_temporary_result_artifact(
    result_id: str,
    filename: str,
    current_user: UserResponse = Depends(require_artifact_current_user),
) -> FileResponse:
    if (
        os.path.basename(result_id) != result_id
        or os.path.basename(filename) != filename
    ):
        raise NotFoundError("Artefacto de resultado no encontrado")

    result_path = os.path.join(RESULTS_BASE_PATH, result_id)

    if result_id.startswith(("evaluation_", "detection_evaluation_")):
        accessible = await evaluation_artifact_is_accessible(
            result_id,
            owner_id=current_user.id,
        )
    elif result_id.startswith(("attack_", "detection_attack_")):
        accessible = await individual_analysis_artifact_is_accessible(
            result_id,
            owner_id=current_user.id,
        )
        if not accessible:
            accessible = (
                await temporary_result_belongs_to_user(result_id, current_user.id)
                and read_result_expiration(result_path) > time.time()
            )
    else:
        accessible = (
            await temporary_result_belongs_to_user(result_id, current_user.id)
            and read_result_expiration(result_path) > time.time()
        )
    if not accessible:
        raise GoneError("El artefacto de resultado ha caducado o no está disponible")

    artifact_path = os.path.join(result_path, filename)
    if not os.path.isfile(artifact_path):
        raise NotFoundError("Artefacto de resultado no encontrado")

    return FileResponse(
        artifact_path,
        headers={
            "Cache-Control": "private, no-store, max-age=0",
            "Pragma": "no-cache",
        },
    )
