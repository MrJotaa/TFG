import asyncio
import json
import logging
import os
import shutil
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pydantic import ValidationError

from core.errors import BadRequestError, ForbiddenError, GoneError, NotFoundError
from db.collections import detection_evaluations_collection
from schemas.detection_evaluation import (
    DetectionEvaluationHistoryItem,
    DetectionEvaluationRequest,
    DetectionEvaluationResponse,
)
from utils.result_files import RESULTS_TTL_SECONDS

DETECTION_EVALUATION_RESULT_PREFIX = "detection_evaluation_"
DETECTION_EVALUATION_ARTIFACT_FILENAME = "evaluation.json"
DETECTION_EVALUATION_NOT_FOUND = "Detection evaluation not found"
DETECTION_EVALUATION_STORAGE_SCHEMA_VERSION = 1
DETECTION_EVALUATIONS_BASE_PATH = os.getenv(
    "DETECTION_EVALUATIONS_BASE_PATH",
    "uploads/evaluations",
)
logger = logging.getLogger(__name__)


def new_detection_evaluation_id() -> str:
    return f"{DETECTION_EVALUATION_RESULT_PREFIX}{uuid.uuid4()}"


async def persist_detection_evaluation(
    *,
    response: DetectionEvaluationResponse,
    request: DetectionEvaluationRequest,
    owner_id: str,
) -> None:
    """Persist a validated response without duplicating its large payload in MongoDB."""

    _validate_result_id(response.result_id)
    result_path = _result_path(response.result_id)
    created_at = datetime.now(timezone.utc)
    expires_at = created_at + timedelta(seconds=RESULTS_TTL_SECONDS)

    await asyncio.to_thread(cleanup_expired_detection_evaluation_artifacts)
    await asyncio.to_thread(
        _write_response_artifact,
        result_path,
        response,
    )
    index_created = False
    try:
        await detection_evaluations_collection.insert_one(
            {
                "_id": response.result_id,
                "result_type": "detection_evaluation",
                "storage_schema_version": (DETECTION_EVALUATION_STORAGE_SCHEMA_VERSION),
                "status": "completed",
                "owner_id": owner_id,
                "dataset_id": response.dataset_id,
                "dataset_revision": response.dataset_revision,
                "annotation_snapshot_id": response.annotation_snapshot_id,
                "model_id": response.model_id,
                "model_name": response.model_name,
                "attack_type": response.attack_type.value,
                "parameters": request.parameters.model_dump(mode="json"),
                "supervised_configuration": (
                    request.supervised_configuration.model_dump(mode="json")
                ),
                "summary": response.summary.model_dump(mode="json"),
                "supervised_available": bool(
                    response.supervised is not None and response.supervised.available
                ),
                "supervised_reason": (
                    response.supervised.reason.value
                    if response.supervised is not None
                    and response.supervised.reason is not None
                    else None
                ),
                "metrics_schema_version": response.metrics_schema_version,
                "delta_convention": response.delta_convention,
                "artifact_filename": DETECTION_EVALUATION_ARTIFACT_FILENAME,
                "created_at": created_at,
                "expires_at": expires_at,
            }
        )
        index_created = True
    finally:
        if not index_created:
            await asyncio.to_thread(
                _remove_result_directory_safely,
                result_path,
                "detection_evaluation_artifact_cleanup_failed",
            )


async def list_detection_evaluations(
    *,
    owner_id: str,
    limit: int,
    offset: int,
) -> list[DetectionEvaluationHistoryItem]:
    cursor = (
        detection_evaluations_collection.find(
            {
                "owner_id": owner_id,
                "status": "completed",
                "expires_at": {"$gt": datetime.now(timezone.utc)},
            }
        )
        .sort([("created_at", -1), ("_id", -1)])
        .skip(offset)
        .limit(limit)
    )
    return [_history_item(document) async for document in cursor]


async def get_detection_evaluation(
    *,
    evaluation_id: str,
    owner_id: str,
) -> DetectionEvaluationResponse:
    _validate_result_id(evaluation_id)
    document = await detection_evaluations_collection.find_one({"_id": evaluation_id})
    if document is None or document.get("result_type") != "detection_evaluation":
        raise NotFoundError(DETECTION_EVALUATION_NOT_FOUND)
    if document.get("owner_id") != owner_id:
        raise ForbiddenError("Detection evaluation belongs to another user")
    if document.get("status") != "completed":
        raise BadRequestError("Detection evaluation is not complete")
    if document.get("storage_schema_version") != (
        DETECTION_EVALUATION_STORAGE_SCHEMA_VERSION
    ):
        raise BadRequestError("Detection evaluation storage version is unsupported")

    expires_at = _utc_datetime(document.get("expires_at"), "expires_at")
    if expires_at <= datetime.now(timezone.utc):
        raise GoneError("Detection evaluation has expired")

    result_path = _result_path(evaluation_id)
    artifact_filename = document.get("artifact_filename")
    if artifact_filename != DETECTION_EVALUATION_ARTIFACT_FILENAME:
        raise BadRequestError("Detection evaluation index is invalid")
    artifact_path = result_path / artifact_filename
    if not artifact_path.is_file():
        raise GoneError("Detection evaluation artifact is no longer available")

    try:
        payload = await asyncio.to_thread(_read_json, artifact_path)
        response = DetectionEvaluationResponse.model_validate(payload)
    except (OSError, UnicodeError, ValueError, ValidationError) as error:
        raise BadRequestError("Detection evaluation artifact is invalid") from error
    if response.result_id != evaluation_id:
        raise BadRequestError("Detection evaluation artifact identity is invalid")
    return response


def _history_item(document: dict) -> DetectionEvaluationHistoryItem:
    try:
        if document.get("storage_schema_version") != (
            DETECTION_EVALUATION_STORAGE_SCHEMA_VERSION
        ):
            raise ValueError("unsupported storage schema version")
        return DetectionEvaluationHistoryItem(
            result_id=document["_id"],
            dataset_id=document["dataset_id"],
            dataset_revision=document["dataset_revision"],
            annotation_snapshot_id=document.get("annotation_snapshot_id"),
            model_id=document["model_id"],
            model_name=document["model_name"],
            attack_type=document["attack_type"],
            parameters=document["parameters"],
            supervised_configuration=document["supervised_configuration"],
            summary=document["summary"],
            supervised_available=document["supervised_available"],
            supervised_reason=document.get("supervised_reason"),
            metrics_schema_version=document["metrics_schema_version"],
            delta_convention=document["delta_convention"],
            created_at=_utc_datetime(document.get("created_at"), "created_at"),
            expires_at=_utc_datetime(document.get("expires_at"), "expires_at"),
        )
    except (KeyError, TypeError, ValueError, ValidationError) as error:
        raise BadRequestError("Detection evaluation index is invalid") from error


def _validate_result_id(result_id: str) -> None:
    if not _is_valid_result_id(result_id):
        raise NotFoundError(DETECTION_EVALUATION_NOT_FOUND)


def _is_valid_result_id(result_id) -> bool:
    if not isinstance(result_id, str) or not result_id.startswith(
        DETECTION_EVALUATION_RESULT_PREFIX
    ):
        return False
    suffix = result_id.removeprefix(DETECTION_EVALUATION_RESULT_PREFIX)
    try:
        uuid.UUID(suffix)
    except (AttributeError, TypeError, ValueError):
        return False
    return True


def _result_path(result_id: str) -> Path:
    return Path(DETECTION_EVALUATIONS_BASE_PATH) / result_id


def cleanup_expired_detection_evaluation_artifacts() -> None:
    base_path = Path(DETECTION_EVALUATIONS_BASE_PATH)
    base_path.mkdir(parents=True, exist_ok=True)
    expiration_timestamp = datetime.now(timezone.utc).timestamp() - (
        RESULTS_TTL_SECONDS
    )
    for entry in base_path.iterdir():
        if not entry.is_dir() or not _is_valid_result_id(entry.name):
            continue
        try:
            if entry.stat().st_mtime <= expiration_timestamp:
                _remove_result_directory(entry)
        except OSError:
            logger.exception(
                "expired_detection_evaluation_artifact_cleanup_failed result_id=%s",
                entry.name,
            )


def _write_response_artifact(
    result_path: Path,
    response: DetectionEvaluationResponse,
) -> None:
    result_path.mkdir(parents=True, exist_ok=False)
    artifact_path = result_path / DETECTION_EVALUATION_ARTIFACT_FILENAME
    temporary_path = artifact_path.with_name(
        f"{artifact_path.name}.{uuid.uuid4().hex}.tmp"
    )
    artifact_written = False
    try:
        try:
            with temporary_path.open("w", encoding="utf-8") as artifact_file:
                json.dump(
                    response.model_dump(mode="json"),
                    artifact_file,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                )
            os.replace(temporary_path, artifact_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
        artifact_written = True
    finally:
        if not artifact_written:
            _remove_result_directory_safely(
                result_path,
                "detection_evaluation_artifact_rollback_failed",
            )


def _read_json(path: Path):
    with path.open(encoding="utf-8") as artifact_file:
        return json.load(artifact_file)


def _remove_result_directory(result_path: Path) -> None:
    if result_path.is_dir():
        shutil.rmtree(result_path)


def _remove_result_directory_safely(
    result_path: Path,
    log_event: str,
) -> None:
    try:
        _remove_result_directory(result_path)
    except OSError:
        logger.exception("%s result_id=%s", log_event, result_path.name)


def _utc_datetime(value, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError(f"{field_name} must be a datetime")
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
