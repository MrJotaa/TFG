import asyncio
import hashlib
import json
import logging
import os
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from bson import ObjectId
from pydantic import ValidationError
from pymongo.errors import DuplicateKeyError

from core.errors import (
    BadRequestError,
    ConflictError,
    ForbiddenError,
    GoneError,
    NotFoundError,
)
from db.collections import datasets_collection, evaluations_collection
from ml.attacks.registry import get_attack, resolve_attack_parameters
from schemas.detection_evaluation import (
    DetectionEvaluationRequest,
    DetectionEvaluationResponse,
)
from schemas.evaluation import EvaluationRequest, EvaluationResponse
from schemas.evaluation_history import (
    EvaluationCompatibilityRequest,
    EvaluationCompatibilityResponse,
    EvaluationHistoryItem,
    EvaluationHistoryPage,
    EvaluationPreservationUpdate,
    EvaluationTaskType,
    StoredEvaluationResponse,
)
from services.evaluation_cancellation import request_evaluation_cancellation
from services.model_service import get_active_model, get_active_model_revision
from utils.detection_evaluation_artifacts import (
    remove_detection_evaluation_visual_artifacts,
)
from utils.result_files import RESULTS_BASE_PATH

logger = logging.getLogger(__name__)
_EVALUATION_NOT_FOUND = "Evaluación no encontrada"
_EVALUATION_WRONG_OWNER = "La evaluación pertenece a otro usuario"

EVALUATION_STORAGE_SCHEMA_VERSION = 2
CLASSIFICATION_PIPELINE_VERSION = "classification-evaluation-v2"
DETECTION_PIPELINE_VERSION = "detection-evaluation-v3-yolo26-end2end-tide1"
CLASSIFICATION_ARTIFACT_FILENAME = "evaluation.json"
DETECTION_ARTIFACT_FILENAME = "evaluation.json"
DETECTION_EVALUATIONS_BASE_PATH = Path(
    os.getenv("DETECTION_EVALUATIONS_BASE_PATH", "uploads/evaluations")
)


@dataclass(frozen=True, slots=True)
class EvaluationExecution:
    result_id: str
    owner_id: str
    task_type: EvaluationTaskType
    signature: str
    dataset_id: str
    dataset_name: str
    dataset_revision: int
    annotation_snapshot_id: str | None
    model_id: str
    model_name: str
    attack_type: str
    parameters: dict
    supervised_configuration: dict | None
    retention_seconds: int
    created_at: datetime


async def find_compatible_evaluation(
    *,
    request: EvaluationCompatibilityRequest,
    owner_id: str,
) -> EvaluationCompatibilityResponse:
    identity = await _build_identity(
        request=request,
        owner_id=owner_id,
        task_type=request.task_type,
    )
    document = await _find_compatible_document(
        owner_id=owner_id,
        signature=identity["signature"],
    )
    return EvaluationCompatibilityResponse(
        match=_history_item(document) if document is not None else None
    )


async def begin_evaluation_execution(
    *,
    request: EvaluationRequest | DetectionEvaluationRequest,
    owner_id: str,
    task_type: EvaluationTaskType,
    retention_seconds: int,
) -> EvaluationExecution:
    identity = await _build_identity(
        request=request,
        owner_id=owner_id,
        task_type=task_type,
    )
    compatible = await _find_compatible_document(
        owner_id=owner_id,
        signature=identity["signature"],
    )
    if compatible is not None:
        if compatible.get("status") == "running":
            raise ConflictError(
                {
                    "code": "evaluation_in_progress",
                    "message": "Ya hay una evaluación idéntica en curso",
                    "result_id": compatible["_id"],
                }
            )
        if not request.force_recompute:
            compatible_expires_at = compatible.get("expires_at")
            raise ConflictError(
                {
                    "code": "compatible_evaluation_exists",
                    "message": "Ya hay una evaluación compatible disponible",
                    "result_id": compatible["_id"],
                    "expires_at": (
                        _utc_datetime(compatible_expires_at, "expires_at").isoformat()
                        if compatible_expires_at is not None
                        else None
                    ),
                    "preserved": bool(compatible.get("preserved")),
                }
            )

    result_id = _new_result_id(task_type)
    created_at = datetime.now(timezone.utc)
    execution = EvaluationExecution(
        result_id=result_id,
        owner_id=owner_id,
        task_type=task_type,
        signature=identity["signature"],
        dataset_id=identity["dataset_id"],
        dataset_name=identity["dataset_name"],
        dataset_revision=identity["dataset_revision"],
        annotation_snapshot_id=identity["annotation_snapshot_id"],
        model_id=identity["model_id"],
        model_name=identity["model_name"],
        attack_type=identity["attack_type"],
        parameters=identity["parameters"],
        supervised_configuration=identity["supervised_configuration"],
        retention_seconds=retention_seconds,
        created_at=created_at,
    )
    try:
        await evaluations_collection.insert_one(
            {
                "_id": result_id,
                "result_type": f"{task_type}_evaluation",
                "task_type": task_type,
                "storage_schema_version": EVALUATION_STORAGE_SCHEMA_VERSION,
                "status": "running",
                "owner_id": owner_id,
                "signature": identity["signature"],
                "pipeline_version": identity["pipeline_version"],
                "dataset_id": identity["dataset_id"],
                "dataset_name": identity["dataset_name"],
                "dataset_revision": identity["dataset_revision"],
                "dataset_fingerprint": identity["dataset_fingerprint"],
                "annotation_snapshot_id": identity["annotation_snapshot_id"],
                "model_id": identity["model_id"],
                "model_name": identity["model_name"],
                "model_revision": identity["model_revision"],
                "attack_type": identity["attack_type"],
                "attack_revision": identity.get("attack_revision"),
                "parameters": identity["parameters"],
                "supervised_configuration": identity["supervised_configuration"],
                "retention_seconds": retention_seconds,
                "preserved": False,
                "created_at": created_at,
            }
        )
    except DuplicateKeyError as error:
        raise ConflictError(
            {
                "code": "evaluation_in_progress",
                "message": "Ya hay una evaluación idéntica en curso",
            }
        ) from error
    return execution


async def complete_evaluation_execution(
    *,
    execution: EvaluationExecution,
    response: EvaluationResponse | DetectionEvaluationResponse,
) -> datetime:
    if response.result_id != execution.result_id:
        raise ValueError("Evaluation result identity does not match its execution")
    artifact_path = _artifact_path(execution.result_id, execution.task_type)
    await asyncio.to_thread(_write_json_atomically, artifact_path, response)

    completed_at = datetime.now(timezone.utc)
    expires_at = completed_at + timedelta(seconds=execution.retention_seconds)
    response_payload = response.model_dump(mode="json")
    update = {
        "status": "completed",
        "completed_at": completed_at,
        "expires_at": expires_at,
        "artifact_filename": artifact_path.name,
        "summary": response_payload.get("summary"),
        "metrics_schema_version": response_payload.get("metrics_schema_version"),
    }
    if execution.task_type == "detection":
        supervised = response_payload.get("supervised")
        update.update(
            {
                "supervised_available": bool(
                    supervised is not None and supervised.get("available")
                ),
                "supervised_reason": (
                    supervised.get("reason") if supervised is not None else None
                ),
                "delta_convention": response_payload.get("delta_convention"),
            }
        )

    result = await evaluations_collection.update_one(
        {
            "_id": execution.result_id,
            "owner_id": execution.owner_id,
            "status": "running",
        },
        {"$set": update},
    )
    if result.matched_count == 0:
        await asyncio.to_thread(_remove_evaluation_artifacts, execution.result_id)
        raise ConflictError("La ejecución de la evaluación ya no está activa")
    return expires_at


async def abort_evaluation_execution(execution: EvaluationExecution) -> None:
    await evaluations_collection.delete_one(
        {
            "_id": execution.result_id,
            "owner_id": execution.owner_id,
        }
    )
    await asyncio.to_thread(_remove_evaluation_artifacts, execution.result_id)


async def list_evaluation_history(
    *,
    owner_id: str,
    limit: int,
    offset: int,
    task_type: EvaluationTaskType | None = None,
) -> EvaluationHistoryPage:
    query = {
        "owner_id": owner_id,
        "$or": [
            {"status": "running"},
            {"status": "completed", "preserved": True},
            {
                "status": "completed",
                "preserved": {"$ne": True},
                "expires_at": {"$gt": datetime.now(timezone.utc)},
            },
        ],
    }
    if task_type is not None:
        query["task_type"] = task_type
    total = await evaluations_collection.count_documents(query)
    cursor = (
        evaluations_collection.find(query)
        .sort([("created_at", -1), ("_id", -1)])
        .skip(offset)
        .limit(limit)
    )
    items = [_history_item(document) async for document in cursor]
    return EvaluationHistoryPage(
        items=items,
        total=total,
        limit=limit,
        offset=offset,
    )


async def get_stored_evaluation(
    *,
    evaluation_id: str,
    owner_id: str,
) -> StoredEvaluationResponse:
    document = await _get_accessible_document(evaluation_id, owner_id)
    task_type = _document_task_type(document)
    artifact_path = _artifact_path(evaluation_id, task_type)
    if not artifact_path.is_file():
        raise GoneError("El artefacto de la evaluación ya no está disponible")
    try:
        payload = await asyncio.to_thread(_read_json, artifact_path)
        if task_type == "classification":
            evaluation = EvaluationResponse.model_validate(payload)
        else:
            evaluation = DetectionEvaluationResponse.model_validate(payload)
    except (OSError, ValueError, ValidationError) as error:
        raise BadRequestError("El artefacto de la evaluación no es válido") from error
    if evaluation.result_id != evaluation_id:
        raise BadRequestError("La identidad del artefacto de la evaluación no es válida")
    return StoredEvaluationResponse(task_type=task_type, evaluation=evaluation)


async def get_evaluation_history_item(
    *, evaluation_id: str, owner_id: str
) -> EvaluationHistoryItem:
    return _history_item(await _get_accessible_document(evaluation_id, owner_id))


async def update_evaluation_preservation(
    *,
    evaluation_id: str,
    owner_id: str,
    update: EvaluationPreservationUpdate,
    retention_seconds: int,
) -> EvaluationHistoryItem:
    document = await _get_owned_completed_document(evaluation_id, owner_id)
    if update.preserved:
        if not _artifact_path(evaluation_id, _document_task_type(document)).is_file():
            raise GoneError("El artefacto de la evaluación ya no está disponible")
        values = {"preserved": True, "expires_at": None}
    else:
        values = {
            "preserved": False,
            "retention_seconds": retention_seconds,
            "expires_at": datetime.now(timezone.utc)
            + timedelta(seconds=retention_seconds),
        }
    await evaluations_collection.update_one({"_id": evaluation_id}, {"$set": values})
    document.update(values)
    return _history_item(document)


async def cancel_running_evaluation(*, evaluation_id: str, owner_id: str) -> None:
    await _get_owned_running_document(evaluation_id, owner_id)
    active_execution = request_evaluation_cancellation(evaluation_id)
    result = await evaluations_collection.delete_one(
        {
            "_id": evaluation_id,
            "owner_id": owner_id,
            "status": "running",
        }
    )
    if result.deleted_count == 0:
        raise BadRequestError("La evaluación ya no está en curso")

    if not active_execution:
        try:
            await asyncio.to_thread(_remove_evaluation_artifacts, evaluation_id)
        except OSError:
            logger.exception(
                "cancelled_evaluation_artifact_cleanup_failed result_id=%s",
                evaluation_id,
            )


async def delete_evaluation_result(*, evaluation_id: str, owner_id: str) -> None:
    await _get_owned_completed_document(evaluation_id, owner_id)
    await _delete_matching_evaluations({"_id": evaluation_id, "owner_id": owner_id})


async def delete_evaluations_for_owner(owner_id: str) -> None:
    await _delete_matching_evaluations({"owner_id": owner_id})


async def delete_evaluations_for_dataset(dataset_id: str) -> None:
    await _delete_matching_evaluations({"dataset_id": dataset_id})


async def cleanup_expired_evaluations() -> set[str]:
    now = datetime.now(timezone.utc)
    query = {
        "$or": [
            {
                "status": "completed",
                "preserved": {"$ne": True},
                "expires_at": {"$lte": now},
            },
            {
                "status": "running",
                "created_at": {"$lte": now - timedelta(days=1)},
            },
        ]
    }
    await _delete_matching_evaluations(query)
    active_ids = set()
    async for document in evaluations_collection.find(
        {
            "$or": [
                {"status": "running"},
                {"status": "completed", "preserved": True},
                {
                    "status": "completed",
                    "preserved": {"$ne": True},
                    "expires_at": {"$gt": now},
                },
            ]
        },
        {"_id": 1},
    ):
        active_ids.add(document["_id"])
    return active_ids


def cleanup_orphaned_evaluation_artifacts(
    active_result_ids: set[str],
    *,
    orphan_ttl_seconds: int = 86400,
) -> None:
    cutoff = datetime.now(timezone.utc).timestamp() - orphan_ttl_seconds
    for base_path in (DETECTION_EVALUATIONS_BASE_PATH, Path(RESULTS_BASE_PATH)):
        base_path.mkdir(parents=True, exist_ok=True)
        for entry in base_path.iterdir():
            if (
                not entry.is_dir()
                or not entry.name.startswith("detection_evaluation_")
                or entry.name in active_result_ids
            ):
                continue
            try:
                if entry.stat().st_mtime <= cutoff:
                    shutil.rmtree(entry)
            except OSError:
                logger.exception(
                    "orphaned_evaluation_cleanup_failed result_id=%s",
                    entry.name,
                )


async def evaluation_artifact_belongs_to_user(result_id: str, owner_id: str) -> bool:
    """The database owns the identity of a persisted evaluation."""
    document = await evaluations_collection.find_one(
        {"_id": result_id, "owner_id": owner_id}, {"_id": 1}
    )
    return document is not None


async def evaluation_artifact_is_accessible(
    result_id: str, *, owner_id: str | None = None
) -> bool:
    query = {"_id": result_id}
    if owner_id is not None:
        query["owner_id"] = owner_id
    document = await evaluations_collection.find_one(query)
    if document is None or document.get("status") != "completed":
        return False
    if document.get("preserved") is True:
        return True
    try:
        return _utc_datetime(document.get("expires_at"), "expires_at") > datetime.now(
            timezone.utc
        )
    except ValueError:
        return False


async def _build_identity(
    *, request, owner_id: str, task_type: EvaluationTaskType
) -> dict:
    if not ObjectId.is_valid(request.dataset_id):
        raise BadRequestError("Identificador de conjunto de datos no válido")
    dataset = await datasets_collection.find_one({"_id": ObjectId(request.dataset_id)})
    if dataset is None:
        raise NotFoundError("Conjunto de datos no encontrado")
    if str(dataset.get("owner_id")) != owner_id:
        raise ForbiddenError("El conjunto de datos pertenece a otro usuario")

    model = get_active_model()

    dataset_revision = int(dataset.get("dataset_revision", 0))
    dataset_fingerprint = _dataset_fingerprint(dataset)
    try:
        attack = get_attack(request.attack_type)
        resolved_parameters = resolve_attack_parameters(attack, request.parameters)
    except ValueError as error:
        raise BadRequestError(str(error)) from error
    parameters = resolved_parameters.model_dump(mode="json")
    attack_revision = attack.revision
    supervised_configuration = None
    if task_type == "detection":
        configuration = getattr(request, "supervised_configuration", None)
        if configuration is None:
            from schemas.detection_metrics import DetectionSupervisedConfiguration

            configuration = DetectionSupervisedConfiguration()
        supervised_configuration = configuration.model_dump(mode="json")
    model_revision = get_active_model_revision()
    pipeline_version = (
        CLASSIFICATION_PIPELINE_VERSION
        if task_type == "classification"
        else DETECTION_PIPELINE_VERSION
    )
    signature_payload = {
        "owner_id": owner_id,
        "task_type": task_type,
        "dataset_id": request.dataset_id,
        "dataset_revision": dataset_revision,
        "dataset_fingerprint": dataset_fingerprint,
        "model_id": model.id,
        "model_revision": model_revision,
        "attack_type": request.attack_type.value,
        "parameters": parameters,
        "supervised_configuration": supervised_configuration,
        "pipeline_version": pipeline_version,
    }
    if attack_revision is not None:
        signature_payload["attack_revision"] = attack_revision

    signature = hashlib.sha256(
        json.dumps(
            signature_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    return {
        **signature_payload,
        "signature": signature,
        "dataset_name": dataset.get("name", request.dataset_id),
        "annotation_snapshot_id": (
            str(dataset["active_annotation_snapshot_id"])
            if dataset.get("active_annotation_snapshot_id") is not None
            else None
        ),
        "model_name": model.name,
    }


def _dataset_fingerprint(dataset: dict) -> str:
    manifest = {
        "dataset_id": str(dataset["_id"]),
        "revision": int(dataset.get("dataset_revision", 0)),
        "annotation_snapshot_id": (
            str(dataset["active_annotation_snapshot_id"])
            if dataset.get("active_annotation_snapshot_id") is not None
            else None
        ),
        "images": [
            {
                "id": image.get("id"),
                "stored_filename": image.get("stored_filename"),
                "width": image.get("width"),
                "height": image.get("height"),
                "label": image.get("label"),
                "annotation_review_status": image.get("annotation_review_status"),
            }
            for image in dataset.get("images", [])
        ],
    }
    return hashlib.sha256(
        json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


async def _find_compatible_document(*, owner_id: str, signature: str) -> dict | None:
    now = datetime.now(timezone.utc)
    running = await evaluations_collection.find_one(
        {"owner_id": owner_id, "signature": signature, "status": "running"}
    )
    if running is not None:
        return running
    cursor = evaluations_collection.find(
        {
            "owner_id": owner_id,
            "signature": signature,
            "status": "completed",
            "$or": [
                {"preserved": True},
                {"preserved": {"$ne": True}, "expires_at": {"$gt": now}},
            ],
        }
    ).sort([("completed_at", -1), ("_id", -1)])
    async for document in cursor:
        if _artifact_path(document["_id"], _document_task_type(document)).is_file():
            return document
    return None


async def _get_owned_running_document(evaluation_id: str, owner_id: str) -> dict:
    if not _valid_result_id(evaluation_id):
        raise NotFoundError(_EVALUATION_NOT_FOUND)
    document = await evaluations_collection.find_one({"_id": evaluation_id})
    if document is None:
        raise NotFoundError(_EVALUATION_NOT_FOUND)
    if document.get("owner_id") != owner_id:
        raise ForbiddenError(_EVALUATION_WRONG_OWNER)
    if document.get("status") != "running":
        raise BadRequestError("La evaluación no está en curso")
    return document


async def _get_owned_completed_document(evaluation_id: str, owner_id: str) -> dict:
    if not _valid_result_id(evaluation_id):
        raise NotFoundError(_EVALUATION_NOT_FOUND)
    document = await evaluations_collection.find_one({"_id": evaluation_id})
    if document is None:
        raise NotFoundError(_EVALUATION_NOT_FOUND)
    if document.get("owner_id") != owner_id:
        raise ForbiddenError(_EVALUATION_WRONG_OWNER)
    if document.get("status") != "completed":
        raise BadRequestError("La evaluación no está completada")
    return document


async def _get_accessible_document(evaluation_id: str, owner_id: str) -> dict:
    if not _valid_result_id(evaluation_id):
        raise NotFoundError(_EVALUATION_NOT_FOUND)
    document = await evaluations_collection.find_one({"_id": evaluation_id})
    if document is None:
        raise NotFoundError(_EVALUATION_NOT_FOUND)
    if document.get("owner_id") != owner_id:
        raise ForbiddenError(_EVALUATION_WRONG_OWNER)
    if document.get("status") != "completed":
        raise BadRequestError("La evaluación no está completada")
    if document.get("preserved") is not True:
        expires_at = _utc_datetime(document.get("expires_at"), "expires_at")
        if expires_at <= datetime.now(timezone.utc):
            raise GoneError("La evaluación ha caducado")
    return document


def _history_item(document: dict) -> EvaluationHistoryItem:
    try:
        created_at = _utc_datetime(document.get("created_at"), "created_at")
        expires_at = (
            _utc_datetime(document.get("expires_at"), "expires_at")
            if document.get("expires_at") is not None
            else None
        )
        retention_seconds = document.get("retention_seconds")
        if retention_seconds is None and expires_at is not None:
            retention_seconds = max(0, int((expires_at - created_at).total_seconds()))
        task_type = _document_task_type(document)
        preserved = bool(document.get("preserved"))
        artifact_available = (
            document.get("status") == "completed"
            and _artifact_path(document["_id"], task_type).is_file()
        )
        exact_reuse_available = bool(
            document.get("status") == "completed"
            and document.get("signature")
            and document.get("storage_schema_version")
            == EVALUATION_STORAGE_SCHEMA_VERSION
            and (
                preserved
                or (
                    expires_at is not None
                    and expires_at > datetime.now(timezone.utc)
                )
            )
            and artifact_available
        )
        return EvaluationHistoryItem(
            result_id=document["_id"],
            task_type=task_type,
            status=document["status"],
            dataset_id=document["dataset_id"],
            dataset_name=document.get("dataset_name", document["dataset_id"]),
            dataset_revision=int(document.get("dataset_revision", 0)),
            model_id=document["model_id"],
            model_name=document.get("model_name", document["model_id"]),
            attack_type=document["attack_type"],
            parameters=document.get("parameters", {}),
            supervised_configuration=document.get("supervised_configuration"),
            summary=document.get("summary"),
            created_at=created_at,
            completed_at=(
                _utc_datetime(document.get("completed_at"), "completed_at")
                if document.get("completed_at") is not None
                else None
            ),
            expires_at=expires_at,
            retention_seconds=retention_seconds,
            preserved=preserved,
            artifact_available=artifact_available,
            exact_reuse_available=exact_reuse_available,
        )
    except (KeyError, TypeError, ValueError, ValidationError) as error:
        raise BadRequestError("El índice del historial de evaluaciones no es válido") from error


def _document_task_type(document: dict) -> EvaluationTaskType:
    task_type = document.get("task_type")
    if task_type in {"classification", "detection"}:
        return task_type
    if document.get("result_type") == "detection_evaluation":
        return "detection"
    if document.get("result_type") in {"evaluation", "classification_evaluation"}:
        return "classification"
    raise BadRequestError("El índice del historial de evaluaciones no es válido")


def _new_result_id(task_type: EvaluationTaskType) -> str:
    prefix = "evaluation" if task_type == "classification" else "detection_evaluation"
    return f"{prefix}_{uuid.uuid4()}"


def _valid_result_id(result_id) -> bool:
    if not isinstance(result_id, str) or os.path.basename(result_id) != result_id:
        return False
    for prefix in ("evaluation_", "detection_evaluation_"):
        if result_id.startswith(prefix):
            try:
                uuid.UUID(result_id.removeprefix(prefix))
                return True
            except ValueError:
                return False
    return False


def _artifact_path(result_id: str, task_type: EvaluationTaskType) -> Path:
    if task_type == "classification":
        return Path(RESULTS_BASE_PATH) / result_id / CLASSIFICATION_ARTIFACT_FILENAME
    return DETECTION_EVALUATIONS_BASE_PATH / result_id / DETECTION_ARTIFACT_FILENAME


def _write_json_atomically(path: Path, response) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as artifact_file:
            json.dump(
                response.model_dump(mode="json"),
                artifact_file,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _read_json(path: Path):
    with path.open(encoding="utf-8") as artifact_file:
        return json.load(artifact_file)


async def _delete_matching_evaluations(query: dict) -> None:
    documents = [document async for document in evaluations_collection.find(query)]
    for document in documents:
        result_id = document.get("_id")
        if not isinstance(result_id, str):
            continue
        try:
            await asyncio.to_thread(_remove_evaluation_artifacts, result_id)
        except OSError:
            logger.exception(
                "evaluation_artifact_cleanup_failed result_id=%s", result_id
            )
            raise
        await evaluations_collection.delete_one({"_id": result_id})


def _remove_evaluation_artifacts(result_id: str) -> None:
    for path in (
        Path(RESULTS_BASE_PATH) / result_id,
        DETECTION_EVALUATIONS_BASE_PATH / result_id,
    ):
        if path.is_dir():
            shutil.rmtree(path)
    if result_id.startswith("detection_evaluation_"):
        remove_detection_evaluation_visual_artifacts(result_id)


def _utc_datetime(value, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError(f"{field_name} must be a datetime")
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
