import os
import time

from fastapi import APIRouter
from fastapi.responses import FileResponse

from core.errors import GoneError, NotFoundError
from services.evaluation_history_service import evaluation_artifact_is_accessible
from services.individual_analysis_history_service import (
    individual_analysis_artifact_is_accessible,
)
from utils.result_files import RESULTS_BASE_PATH, read_result_expiration

router = APIRouter()


@router.get("/results/{result_id}/{filename}")
async def get_temporary_result_artifact(
    result_id: str,
    filename: str,
) -> FileResponse:
    if (
        os.path.basename(result_id) != result_id
        or os.path.basename(filename) != filename
    ):
        raise NotFoundError("Artefacto de resultado no encontrado")

    result_path = os.path.join(RESULTS_BASE_PATH, result_id)
    artifact_path = os.path.join(result_path, filename)
    if not os.path.isfile(artifact_path):
        raise NotFoundError("Artefacto de resultado no encontrado")

    if result_id.startswith(("evaluation_", "detection_evaluation_")):
        accessible = await evaluation_artifact_is_accessible(result_id)
    elif result_id.startswith(("attack_", "detection_attack_")):
        accessible = await individual_analysis_artifact_is_accessible(result_id)
        if not accessible:
            accessible = read_result_expiration(result_path) > time.time()
    else:
        accessible = read_result_expiration(result_path) > time.time()
    if not accessible:
        raise GoneError("El artefacto de resultado ha caducado")

    return FileResponse(
        artifact_path,
        headers={
            "Cache-Control": "private, no-store, max-age=0",
            "Pragma": "no-cache",
        },
    )
