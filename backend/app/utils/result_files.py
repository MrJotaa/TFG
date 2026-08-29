import json
import os
import shutil
import time
import uuid
from datetime import datetime, timezone

from PIL import Image

RESULTS_BASE_PATH = "uploads/results"
RESULTS_TTL_SECONDS = 24 * 60 * 60  # 24 horas
_METADATA_FILENAME = "metadata.json"
_INVALID_RESULT_IDENTITY = "Result identity is invalid"


def ensure_results_directory() -> (
    None
):  # Crea el directorio base de resultados si no existe
    os.makedirs(RESULTS_BASE_PATH, exist_ok=True)


def cleanup_old_results(
    ttl_seconds: int = RESULTS_TTL_SECONDS,
    *,
    protected_result_ids: set[str] | None = None,
) -> None:
    """
    Elimina directorios de resultados temporales antiguos.
    """
    ensure_results_directory()

    now = time.time()
    protected_result_ids = protected_result_ids or set()

    for entry in os.listdir(RESULTS_BASE_PATH):
        entry_path = os.path.join(RESULTS_BASE_PATH, entry)

        if not os.path.isdir(entry_path):
            continue

        if entry in protected_result_ids:
            continue

        expires_at = read_result_expiration(entry_path, ttl_seconds=ttl_seconds)
        if expires_at <= now:
            shutil.rmtree(entry_path, ignore_errors=True)


def read_result_expiration(
    result_path: str,
    *,
    ttl_seconds: int = RESULTS_TTL_SECONDS,
) -> float:
    metadata_path = os.path.join(result_path, _METADATA_FILENAME)
    try:
        with open(metadata_path, encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)
        explicit_expiration = metadata.get("expires_at")
        if isinstance(explicit_expiration, str):
            normalized = explicit_expiration.replace("Z", "+00:00")
            expires_at = datetime.fromisoformat(normalized)
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)
            return expires_at.timestamp()
        created_at_value = metadata.get("created_at")
        if isinstance(created_at_value, str):
            normalized = created_at_value.replace("Z", "+00:00")
            created_at = datetime.fromisoformat(normalized)
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
            return created_at.timestamp() + ttl_seconds
    except (OSError, TypeError, ValueError):
        pass
    try:
        return os.path.getmtime(result_path) + ttl_seconds
    except OSError:
        return 0.0


def create_result_directory(
    prefix: str = "result",
    *,
    result_id: str | None = None,
) -> tuple[str, str]:
    """
    Crea un directorio único para almacenar resultados visuales generados.

    Returns:
        result_id: identificador único del resultado
        result_path: ruta física del directorio
    """
    result_id = result_id or f"{prefix}_{uuid.uuid4()}"
    if os.path.basename(result_id) != result_id:
        raise ValueError(_INVALID_RESULT_IDENTITY)
    result_path = os.path.join(RESULTS_BASE_PATH, result_id)

    os.makedirs(result_path, exist_ok=True)

    return result_id, result_path


def save_result_image(
    image: Image.Image,
    result_id: str,
    result_path: str,
    filename: str,
) -> str:
    """
    Guarda una imagen PIL como PNG dentro del directorio de resultados
    y devuelve su URL pública.
    """
    image_path = os.path.join(result_path, filename)
    image.save(image_path, format="PNG")

    return f"/results/{result_id}/{filename}"


def save_result_metadata(
    result_path: str,
    metadata: dict,
) -> str:
    metadata_path = os.path.join(result_path, _METADATA_FILENAME)

    metadata_with_timestamp = {
        **metadata,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata_with_timestamp, file, indent=2)

    return metadata_path


def save_result_payload(
    result_path: str,
    payload: dict,
    *,
    filename: str = "analysis.json",
) -> str:
    os.makedirs(result_path, exist_ok=True)
    payload_path = os.path.join(result_path, filename)
    temporary_path = f"{payload_path}.{uuid.uuid4().hex}.tmp"
    try:
        with open(temporary_path, "w", encoding="utf-8") as file:
            json.dump(
                payload,
                file,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
        os.replace(temporary_path, payload_path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
    return payload_path


def read_result_metadata(result_id: str) -> dict:
    if not isinstance(result_id, str) or os.path.basename(result_id) != result_id:
        raise ValueError(_INVALID_RESULT_IDENTITY)
    metadata_path = os.path.join(RESULTS_BASE_PATH, result_id, _METADATA_FILENAME)
    with open(metadata_path, encoding="utf-8") as file:
        return json.load(file)


def read_result_payload(
    result_id: str,
    *,
    filename: str = "analysis.json",
) -> dict:
    if not isinstance(result_id, str) or os.path.basename(result_id) != result_id:
        raise ValueError(_INVALID_RESULT_IDENTITY)
    payload_path = os.path.join(RESULTS_BASE_PATH, result_id, filename)
    with open(payload_path, encoding="utf-8") as file:
        return json.load(file)
