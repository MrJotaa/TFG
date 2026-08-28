import asyncio
import hashlib
import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from bson import ObjectId
from pydantic import ValidationError
from pymongo.errors import DuplicateKeyError

from core.errors import BadRequestError, ForbiddenError, GoneError, NotFoundError
from db.collections import datasets_collection, individual_analyses_collection
from ml.attacks.registry import get_attack, resolve_attack_parameters
from schemas.attack import AttackResponse
from schemas.detection_attack import DetectionAttackResponse
from schemas.individual_analysis_history import (
    IndividualAnalysisCompatibilityRequest,
    IndividualAnalysisCompatibilityResponse,
    IndividualAnalysisHistoryItem,
    IndividualAnalysisIdentity,
    IndividualAnalysisHistoryPage,
    IndividualAnalysisSaveRequest,
    IndividualAnalysisTaskType,
    StoredIndividualAnalysisResponse,
)
from services.model_service import get_active_model, get_active_model_revision
from utils.result_files import (
    RESULTS_BASE_PATH,
    read_result_metadata,
    read_result_payload,
)

ANALYSIS_STORAGE_SCHEMA_VERSION = 1
ANALYSIS_ARTIFACT_FILENAME = "analysis.json"
PIPELINE_VERSIONS = {
    "classification": "classification-individual-analysis-v1",
    "detection": "detection-individual-analysis-v1",
}


async def save_individual_analysis(
    *,
    request: IndividualAnalysisSaveRequest,
    owner_id: str,
) -> IndividualAnalysisHistoryItem:
    metadata, analysis = _load_temporary_analysis(request.result_id, request.task_type)
    dataset, image_metadata = await _validate_dataset_context(
        dataset_id=metadata.get("dataset_id"),
        image_id=metadata.get("image_id"),
        owner_id=owner_id,
    )
    try:
        dataset_revision = int(metadata["dataset_revision"])
    except (KeyError, TypeError, ValueError) as error:
        raise BadRequestError(
            "El análisis temporal no contiene una revisión de dataset válida"
        ) from error
    signature = _signature_from_metadata(
        metadata=metadata,
        owner_id=owner_id,
    )
    existing = await individual_analyses_collection.find_one(
        {"owner_id": owner_id, "signature": signature}
    )
    if existing is not None:
        if _source_artifact_exists(existing):
            return _history_item(existing)
        await individual_analyses_collection.delete_one({"_id": existing["_id"]})

    created_at = _parse_metadata_datetime(metadata.get("created_at"))
    saved_at = datetime.now(timezone.utc)
    document = {
        "_id": f"individual_analysis_{uuid.uuid4()}",
        "result_type": "individual_analysis",
        "storage_schema_version": ANALYSIS_STORAGE_SCHEMA_VERSION,
        "owner_id": owner_id,
        "signature": signature,
        "task_type": request.task_type,
        "source_result_id": request.result_id,
        "dataset_id": str(dataset["_id"]),
        "dataset_name": dataset.get("name", str(dataset["_id"])),
        "dataset_revision": dataset_revision,
        "image_id": metadata["image_id"],
        "image_filename": image_metadata.get("filename", metadata["image_id"]),
        "model_id": metadata["model_id"],
        "model_name": metadata.get("model_name", metadata["model_id"]),
        "model_revision": metadata.get("model_revision"),
        "attack_type": metadata["attack_type"],
        "attack_revision": metadata.get("attack_revision"),
        "parameters": metadata.get("parameters", {}),
        "pipeline_version": metadata["pipeline_version"],
        "summary": _analysis_summary(request.task_type, analysis),
        "explanation_available": _explanation_available(request.task_type, analysis),
        "created_at": created_at,
        "saved_at": saved_at,
    }
    try:
        await individual_analyses_collection.insert_one(document)
    except DuplicateKeyError:
        existing = await individual_analyses_collection.find_one(
            {"owner_id": owner_id, "signature": signature}
        )
        if existing is None:
            raise
        return _history_item(existing)
    return _history_item(document)


async def find_compatible_individual_analysis(
    *,
    request: IndividualAnalysisCompatibilityRequest,
    owner_id: str,
) -> IndividualAnalysisCompatibilityResponse:
    identity = await _build_current_identity(request=request, owner_id=owner_id)
    document = await individual_analyses_collection.find_one(
        {"owner_id": owner_id, "signature": identity["signature"]}
    )
    if document is not None and not _source_artifact_exists(document):
        document = None
    return IndividualAnalysisCompatibilityResponse(
        identity=IndividualAnalysisIdentity.model_validate(identity),
        match=_history_item(document) if document is not None else None,
    )


async def list_individual_analysis_history(
    *,
    owner_id: str,
    limit: int,
    offset: int,
    task_type: IndividualAnalysisTaskType | None = None,
) -> IndividualAnalysisHistoryPage:
    query = {"owner_id": owner_id}
    if task_type is not None:
        query["task_type"] = task_type
    total = await individual_analyses_collection.count_documents(query)
    cursor = (
        individual_analyses_collection.find(query)
        .sort([("saved_at", -1), ("_id", -1)])
        .skip(offset)
        .limit(limit)
    )
    return IndividualAnalysisHistoryPage(
        items=[_history_item(document) async for document in cursor],
        total=total,
        limit=limit,
        offset=offset,
    )


async def get_stored_individual_analysis(
    *, analysis_id: str, owner_id: str
) -> StoredIndividualAnalysisResponse:
    document = await _get_owned_document(analysis_id=analysis_id, owner_id=owner_id)
    source_result_id = document["source_result_id"]
    try:
        payload = await asyncio.to_thread(read_result_payload, source_result_id)
        analysis = _validate_analysis_payload(document["task_type"], payload)
    except (OSError, ValueError, ValidationError) as error:
        raise GoneError("El análisis guardado ya no está disponible") from error
    return StoredIndividualAnalysisResponse(
        item=_history_item(document),
        analysis=analysis,
    )


async def delete_individual_analysis(*, analysis_id: str, owner_id: str) -> None:
    document = await _get_owned_document(analysis_id=analysis_id, owner_id=owner_id)
    source_result_id = document.get("source_result_id")
    if isinstance(source_result_id, str):
        await asyncio.to_thread(_remove_result_directory, source_result_id)
    await individual_analyses_collection.delete_one({"_id": analysis_id})


async def delete_individual_analyses_for_owner(owner_id: str) -> None:
    await _delete_matching({"owner_id": owner_id})


async def delete_individual_analyses_for_dataset(dataset_id: str) -> None:
    await _delete_matching({"dataset_id": dataset_id})


async def delete_individual_analyses_for_image(
    dataset_id: str,
    image_id: str,
) -> None:
    await _delete_matching({"dataset_id": dataset_id, "image_id": image_id})


async def individual_analysis_artifact_is_accessible(
    result_id: str,
    *,
    owner_id: str,
) -> bool:
    document = await individual_analyses_collection.find_one(
        {"source_result_id": result_id, "owner_id": owner_id}
    )
    return document is not None


async def protected_individual_result_ids() -> set[str]:
    protected = set()
    async for document in individual_analyses_collection.find(
        {},
        {"source_result_id": 1},
    ):
        source_result_id = document.get("source_result_id")
        if isinstance(source_result_id, str):
            protected.add(source_result_id)
    return protected


async def _build_current_identity(
    *, request: IndividualAnalysisCompatibilityRequest, owner_id: str
) -> dict:
    dataset, image_metadata = await _validate_dataset_context(
        dataset_id=request.dataset_id,
        image_id=request.image_id,
        owner_id=owner_id,
    )
    model = get_active_model()
    if model.task_type != request.task_type:
        raise BadRequestError(
            "El modelo activo no corresponde al tipo de análisis solicitado"
        )
    try:
        attack = get_attack(request.attack_type)
        parameters = resolve_attack_parameters(attack, request.parameters).model_dump(
            mode="json"
        )
    except ValueError as error:
        raise BadRequestError(str(error)) from error
    metadata = {
        "task_type": request.task_type,
        "dataset_id": request.dataset_id,
        "dataset_name": dataset.get("name", request.dataset_id),
        "dataset_revision": int(dataset.get("dataset_revision", 0)),
        "image_id": request.image_id,
        "image_filename": image_metadata.get("filename", request.image_id),
        "model_id": model.id,
        "model_name": model.name,
        "model_revision": get_active_model_revision(),
        "attack_type": request.attack_type.value,
        "attack_revision": attack.revision,
        "parameters": parameters,
        "pipeline_version": PIPELINE_VERSIONS[request.task_type],
    }
    return {**metadata, "signature": _signature_from_metadata(metadata, owner_id)}


async def _validate_dataset_context(
    *, dataset_id: str | None, image_id: str | None, owner_id: str
) -> tuple[dict, dict]:
    if not isinstance(dataset_id, str) or not ObjectId.is_valid(dataset_id):
        raise BadRequestError("Identificador de conjunto de datos no válido")
    dataset = await datasets_collection.find_one({"_id": ObjectId(dataset_id)})
    if dataset is None:
        raise NotFoundError("Conjunto de datos no encontrado")
    if str(dataset.get("owner_id")) != owner_id:
        raise ForbiddenError("El conjunto de datos pertenece a otro usuario")
    image_metadata = next(
        (image for image in dataset.get("images", []) if image.get("id") == image_id),
        None,
    )
    if image_metadata is None:
        raise NotFoundError("Imagen no encontrada en el conjunto de datos")
    return dataset, image_metadata


def _load_temporary_analysis(
    result_id: str, task_type: IndividualAnalysisTaskType
) -> tuple[dict, AttackResponse | DetectionAttackResponse]:
    expected_prefix = (
        "attack_" if task_type == "classification" else "detection_attack_"
    )
    if not _valid_result_id(result_id, expected_prefix):
        raise NotFoundError("Análisis temporal no encontrado")
    try:
        metadata = read_result_metadata(result_id)
        payload = read_result_payload(result_id)
        analysis = _validate_analysis_payload(task_type, payload)
    except (OSError, ValueError, ValidationError, json.JSONDecodeError) as error:
        raise GoneError("El análisis temporal ya no está disponible") from error
    if metadata.get("task_type") != task_type:
        raise BadRequestError("El tipo del análisis temporal no coincide")
    if analysis.result_id != result_id:
        raise BadRequestError("La identidad del análisis temporal no es válida")
    return metadata, analysis


def _validate_analysis_payload(task_type: IndividualAnalysisTaskType, payload: dict):
    if task_type == "classification":
        return AttackResponse.model_validate(payload)
    return DetectionAttackResponse.model_validate(payload)


def _signature_from_metadata(metadata: dict, owner_id: str) -> str:
    required = (
        "task_type",
        "dataset_id",
        "dataset_revision",
        "image_id",
        "model_id",
        "model_revision",
        "attack_type",
        "parameters",
        "pipeline_version",
    )
    missing = [key for key in required if metadata.get(key) is None]
    if missing:
        raise BadRequestError(
            "El análisis temporal no contiene identidad suficiente para guardarse"
        )
    payload = {"owner_id": owner_id, **{key: metadata.get(key) for key in required}}
    if metadata.get("attack_revision") is not None:
        payload["attack_revision"] = metadata["attack_revision"]
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _analysis_summary(task_type: IndividualAnalysisTaskType, analysis) -> dict:
    if task_type == "classification":
        return {
            "original_label": analysis.original_prediction.prediction,
            "adversarial_label": analysis.adversarial_prediction.prediction,
            "original_confidence": analysis.original_prediction.confidence,
            "adversarial_confidence": analysis.adversarial_prediction.confidence,
            "attack_success": analysis.comparison.attack_success,
        }
    return {
        "original_detection_count": len(analysis.original_detections),
        "adversarial_detection_count": len(analysis.adversarial_detections),
        "average_original_confidence": analysis.comparison.average_original_confidence,
        "average_adversarial_confidence": (
            analysis.comparison.average_adversarial_confidence
        ),
        "attack_success": analysis.comparison.attack_success,
    }


def _explanation_available(task_type: IndividualAnalysisTaskType, analysis) -> bool:
    if task_type == "classification":
        return bool(
            analysis.original_gradcam_image_url
            and analysis.adversarial_gradcam_image_url
        )
    return bool(
        analysis.original_explanation_image_url
        and analysis.adversarial_explanation_image_url
    )


def _history_item(document: dict) -> IndividualAnalysisHistoryItem:
    try:
        return IndividualAnalysisHistoryItem(
            analysis_id=document["_id"],
            source_result_id=document["source_result_id"],
            task_type=document["task_type"],
            dataset_id=document["dataset_id"],
            dataset_name=document.get("dataset_name", document["dataset_id"]),
            dataset_revision=int(document.get("dataset_revision", 0)),
            image_id=document["image_id"],
            image_filename=document.get("image_filename", document["image_id"]),
            model_id=document["model_id"],
            model_name=document.get("model_name", document["model_id"]),
            model_revision=document.get("model_revision"),
            attack_type=document["attack_type"],
            attack_revision=document.get("attack_revision"),
            parameters=document.get("parameters", {}),
            summary=document.get("summary", {}),
            explanation_available=bool(document.get("explanation_available")),
            pipeline_version=document.get("pipeline_version"),
            created_at=_utc_datetime(document["created_at"]),
            saved_at=_utc_datetime(document["saved_at"]),
        )
    except (KeyError, TypeError, ValueError, ValidationError) as error:
        raise BadRequestError(
            "El índice del historial de análisis no es válido"
        ) from error


async def _get_owned_document(*, analysis_id: str, owner_id: str) -> dict:
    if not _valid_analysis_id(analysis_id):
        raise NotFoundError("Análisis guardado no encontrado")
    document = await individual_analyses_collection.find_one({"_id": analysis_id})
    if document is None:
        raise NotFoundError("Análisis guardado no encontrado")
    if document.get("owner_id") != owner_id:
        raise ForbiddenError("El análisis guardado pertenece a otro usuario")
    return document


async def _delete_matching(query: dict) -> None:
    documents = [
        document async for document in individual_analyses_collection.find(query)
    ]
    for document in documents:
        source_result_id = document.get("source_result_id")
        if isinstance(source_result_id, str):
            await asyncio.to_thread(_remove_result_directory, source_result_id)
        await individual_analyses_collection.delete_one({"_id": document["_id"]})


def _source_artifact_exists(document: dict) -> bool:
    source_result_id = document.get("source_result_id")
    return isinstance(source_result_id, str) and (
        Path(RESULTS_BASE_PATH) / source_result_id / ANALYSIS_ARTIFACT_FILENAME
    ).is_file()


def _remove_result_directory(result_id: str) -> None:
    path = Path(RESULTS_BASE_PATH) / result_id
    if path.is_dir():
        shutil.rmtree(path)


def _valid_result_id(result_id: str, prefix: str) -> bool:
    if not isinstance(result_id, str) or os.path.basename(result_id) != result_id:
        return False
    if not result_id.startswith(prefix):
        return False
    try:
        uuid.UUID(result_id.removeprefix(prefix))
        return True
    except ValueError:
        return False


def _valid_analysis_id(analysis_id: str) -> bool:
    prefix = "individual_analysis_"
    if not isinstance(analysis_id, str) or not analysis_id.startswith(prefix):
        return False
    try:
        uuid.UUID(analysis_id.removeprefix(prefix))
        return True
    except ValueError:
        return False


def _parse_metadata_datetime(value) -> datetime:
    if not isinstance(value, str):
        raise BadRequestError(
            "El análisis temporal no tiene una fecha de creación válida"
        )
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise BadRequestError(
            "El análisis temporal no tiene una fecha de creación válida"
        ) from error
    return _utc_datetime(parsed)


def _utc_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
