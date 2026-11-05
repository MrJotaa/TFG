import asyncio
import json
import os
from io import BytesIO

import pytest
from bson import ObjectId
from fastapi import UploadFile
from pymongo import AsyncMongoClient, MongoClient
from pymongo.errors import DuplicateKeyError, PyMongoError
from starlette.datastructures import Headers

import services.dataset_annotation_service as dataset_annotation_service
from db.migrations import (
    downgrade_coco_annotations_schema,
    upgrade_coco_annotations_schema,
)
from schemas.users import UserResponse


def _sync_database():
    mongo_uri = (
        os.environ.get("MONGO_URI")
        or os.environ.get("MONGODB_URI")
        or os.environ.get("MONGO_URL")
        or "mongodb://mongo:27017"
    )
    client = MongoClient(mongo_uri)
    return client, client[os.environ["MONGO_DB_NAME"]]


def _coco_payload(
    *,
    image_names=("image-1.png",),
    with_annotation=True,
):
    images = [
        {
            "id": index + 1,
            "file_name": image_name,
            "width": 32,
            "height": 32,
        }
        for index, image_name in enumerate(image_names)
    ]
    categories = [
        {
            "id": 7,
            "name": "cat",
            "supercategory": "animal",
        },
        {
            "id": 8,
            "name": "unused",
        },
    ]
    annotations = (
        [
            {
                "id": 101,
                "image_id": 1,
                "category_id": 7,
                "bbox": [1.123456789, 2.5, 10.25, 12.125],
                "area": 124.296875,
                "iscrowd": 0,
            }
        ]
        if with_annotation
        else []
    )
    return {
        "images": images,
        "categories": categories,
        "annotations": annotations,
    }


def _upload_coco(client, dataset_id, auth_headers, payload, filename="coco.json"):
    content = (
        payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
    )
    return client.post(
        f"/datasets/{dataset_id}/annotations/coco",
        headers=auth_headers,
        files={
            "file": (
                filename,
                content,
                "application/json",
            )
        },
    )


def _service_upload(payload):
    return UploadFile(
        BytesIO(json.dumps(payload).encode("utf-8")),
        filename="coco.json",
        headers=Headers({"content-type": "application/json"}),
    )


def _annotation_test_user(owner_id: ObjectId) -> UserResponse:
    return UserResponse(
        id=str(owner_id),
        name="Snapshot Test",
        email="snapshot@example.com",
    )


async def _seed_annotation_snapshot(
    database,
    *,
    dataset_id: ObjectId,
    owner_id: ObjectId,
    snapshot_id: str,
    dataset_revision: int,
):
    await database.datasets.insert_one(
        {
            "_id": dataset_id,
            "name": "Concurrent annotations",
            "owner_id": owner_id,
            "image_count": 1,
            "dataset_revision": dataset_revision,
            "active_annotation_snapshot_id": snapshot_id,
            "annotation_snapshot_format": "coco",
            "images": [
                {
                    "id": "image-1",
                    "filename": "image-1.png",
                    "original_file_name": "image-1.png",
                    "path": "/trusted/image-1.png",
                    "width": 32,
                    "height": 32,
                    "annotation_review_status": "reviewed_with_objects",
                }
            ],
        }
    )
    await database.annotation_categories.insert_one(
        {
            "_id": f"{snapshot_id}-category",
            "dataset_id": dataset_id,
            "snapshot_id": snapshot_id,
            "coco_category_id": 70,
            "name": f"{snapshot_id}-cat",
        }
    )
    await database.object_annotations.insert_one(
        {
            "_id": f"{snapshot_id}-annotation",
            "dataset_id": dataset_id,
            "snapshot_id": snapshot_id,
            "image_id": "image-1",
            "category_id": f"{snapshot_id}-category",
            "coco_annotation_id": 700,
            "bbox_x1": 1.0,
            "bbox_y1": 2.0,
            "bbox_x2": 11.0,
            "bbox_y2": 14.0,
            "area": 120.0,
            "iscrowd": 0,
        }
    )


async def _seed_preparing_snapshot(database, dataset_id, snapshot_id):
    await database.annotation_categories.insert_one(
        {
            "_id": f"{snapshot_id}-category",
            "dataset_id": dataset_id,
            "snapshot_id": snapshot_id,
            "coco_category_id": 80,
            "name": f"{snapshot_id}-cat",
        }
    )
    await database.object_annotations.insert_one(
        {
            "_id": f"{snapshot_id}-annotation",
            "dataset_id": dataset_id,
            "snapshot_id": snapshot_id,
            "image_id": "image-1",
            "category_id": f"{snapshot_id}-category",
            "coco_annotation_id": 800,
            "bbox_x1": 2.0,
            "bbox_y1": 3.0,
            "bbox_x2": 12.0,
            "bbox_y2": 15.0,
            "area": 120.0,
            "iscrowd": 0,
        }
    )


class _CoordinatedDatasetCollection:
    def __init__(self, collection, prepared, allow_activation):
        self._collection = collection
        self._prepared = prepared
        self._allow_activation = allow_activation

    async def find_one(self, *args, **kwargs):
        return await self._collection.find_one(*args, **kwargs)

    async def update_one(self, query, update):
        task = asyncio.current_task()
        if task is not None and task.get_name() == "import-b":
            self._prepared.set()
            await self._allow_activation.wait()
        return await self._collection.update_one(query, update)


class _FailingAnnotationCollection:
    def __init__(self, collection):
        self._collection = collection

    async def insert_many(self, documents, *, ordered):
        raise PyMongoError("simulated annotation persistence failure")

    async def delete_many(self, query):
        return await self._collection.delete_many(query)

    async def aggregate(self, pipeline):
        return await self._collection.aggregate(pipeline)


class _CountingDatasetCollection:
    def __init__(self, collection, *, change_revision_on_verification=False):
        self._collection = collection
        self._change_revision_on_verification = change_revision_on_verification
        self.find_one_calls = 0
        self.find_one_calls_by_task = {}

    async def find_one(self, *args, **kwargs):
        self.find_one_calls += 1
        task = asyncio.current_task()
        task_name = task.get_name() if task is not None else None
        task_call_count = self.find_one_calls_by_task.get(task_name, 0) + 1
        self.find_one_calls_by_task[task_name] = task_call_count
        document = await self._collection.find_one(*args, **kwargs)
        if (
            document is not None
            and self._change_revision_on_verification
            and task_call_count > 1
        ):
            document["dataset_revision"] = (
                document.get("dataset_revision", 0) + task_call_count - 1
            )
        return document

    async def update_one(self, *args, **kwargs):
        return await self._collection.update_one(*args, **kwargs)


class _CountingCategoryCollection:
    def __init__(self, collection):
        self._collection = collection
        self.find_calls = 0

    def find(self, *args, **kwargs):
        self.find_calls += 1
        return self._collection.find(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._collection, name)


class _ControlledAnnotationCollection:
    def __init__(self, collection, *, read_paused=None, allow_read=None):
        self._collection = collection
        self._read_paused = read_paused
        self._allow_read = allow_read
        self.aggregate_calls = 0
        self.aggregate_calls_by_task = {}

    async def insert_many(self, *args, **kwargs):
        return await self._collection.insert_many(*args, **kwargs)

    async def delete_many(self, *args, **kwargs):
        return await self._collection.delete_many(*args, **kwargs)

    async def aggregate(self, *args, **kwargs):
        self.aggregate_calls += 1
        task = asyncio.current_task()
        task_name = task.get_name() if task is not None else None
        task_call_count = self.aggregate_calls_by_task.get(task_name, 0) + 1
        self.aggregate_calls_by_task[task_name] = task_call_count
        if (
            task_name == "coverage-get"
            and task_call_count == 1
            and self._read_paused is not None
            and self._allow_read is not None
        ):
            self._read_paused.set()
            await self._allow_read.wait()
        return await self._collection.aggregate(*args, **kwargs)


def test_stable_coverage_builder_preserves_states_and_category_distribution():
    coverage = dataset_annotation_service._build_annotation_coverage(
        "dataset-1",
        [
            {"annotation_review_status": "reviewed_with_objects"},
            {"annotation_review_status": "reviewed_empty"},
            {"annotation_review_status": "unreviewed"},
        ],
        [
            {
                "_id": "category-unused",
                "coco_category_id": 8,
                "name": "unused",
            },
            {
                "_id": "category-cat",
                "coco_category_id": 7,
                "name": "cat",
            },
        ],
        {"category-cat": 2},
        has_annotation_snapshot=True,
    )

    assert coverage.model_dump() == {
        "dataset_id": "dataset-1",
        "has_annotation_snapshot": True,
        "total_images": 3,
        "reviewed_images": 2,
        "unreviewed_images": 1,
        "reviewed_empty_images": 1,
        "reviewed_with_objects_images": 1,
        "coverage_percentage": 66.67,
        "total_objects": 2,
        "total_categories": 2,
        "objects_by_category": [
            {
                "category_id": "category-cat",
                "coco_category_id": 7,
                "name": "cat",
                "object_count": 2,
            },
            {
                "category_id": "category-unused",
                "coco_category_id": 8,
                "name": "unused",
                "object_count": 0,
            },
        ],
    }


def test_stable_coverage_builder_handles_empty_dataset():
    coverage = dataset_annotation_service._build_annotation_coverage(
        "empty-dataset",
        [],
        [],
        {},
        has_annotation_snapshot=False,
    )

    assert coverage.model_dump() == {
        "dataset_id": "empty-dataset",
        "has_annotation_snapshot": False,
        "total_images": 0,
        "reviewed_images": 0,
        "unreviewed_images": 0,
        "reviewed_empty_images": 0,
        "reviewed_with_objects_images": 0,
        "coverage_percentage": 0.0,
        "total_objects": 0,
        "total_categories": 0,
        "objects_by_category": [],
    }


def test_new_dataset_annotation_coverage_is_fully_unreviewed(
    client,
    created_dataset,
    auth_headers,
):
    response = client.get(
        f"/datasets/{created_dataset['id']}/annotations/coverage",
        headers=auth_headers,
    )

    assert response.status_code == 200
    assert response.json() == {
        "dataset_id": created_dataset["id"],
        "has_annotation_snapshot": False,
        "total_images": 1,
        "reviewed_images": 0,
        "unreviewed_images": 1,
        "reviewed_empty_images": 0,
        "reviewed_with_objects_images": 0,
        "coverage_percentage": 0.0,
        "total_objects": 0,
        "total_categories": 0,
        "objects_by_category": [],
    }


@pytest.mark.asyncio
async def test_annotation_coverage_read_without_concurrency_uses_single_attempt(
    monkeypatch,
):
    mongo_uri = (
        os.environ.get("MONGO_URI")
        or os.environ.get("MONGODB_URI")
        or os.environ.get("MONGO_URL")
        or "mongodb://mongo:27017"
    )
    mongo_client = AsyncMongoClient(mongo_uri)
    database = mongo_client[os.environ["MONGO_DB_NAME"]]
    dataset_id = ObjectId()
    owner_id = ObjectId()
    await _seed_annotation_snapshot(
        database,
        dataset_id=dataset_id,
        owner_id=owner_id,
        snapshot_id="snapshot-stable",
        dataset_revision=3,
    )

    dataset_collection = _CountingDatasetCollection(database.datasets)
    category_collection = _CountingCategoryCollection(database.annotation_categories)
    annotation_collection = _ControlledAnnotationCollection(database.object_annotations)
    monkeypatch.setattr(
        dataset_annotation_service,
        "datasets_collection",
        dataset_collection,
    )
    monkeypatch.setattr(
        dataset_annotation_service,
        "annotation_categories_collection",
        category_collection,
    )
    monkeypatch.setattr(
        dataset_annotation_service,
        "object_annotations_collection",
        annotation_collection,
    )

    coverage = await dataset_annotation_service.get_annotation_coverage(
        str(dataset_id),
        _annotation_test_user(owner_id),
    )

    assert coverage.total_images == 1
    assert coverage.reviewed_with_objects_images == 1
    assert coverage.total_categories == 1
    assert coverage.total_objects == 1
    assert dataset_collection.find_one_calls == 2
    assert category_collection.find_calls == 1
    assert annotation_collection.aggregate_calls == 1
    await mongo_client.close()


@pytest.mark.asyncio
async def test_annotation_coverage_read_retries_when_active_snapshot_changes(
    monkeypatch,
):
    mongo_uri = (
        os.environ.get("MONGO_URI")
        or os.environ.get("MONGODB_URI")
        or os.environ.get("MONGO_URL")
        or "mongodb://mongo:27017"
    )
    mongo_client = AsyncMongoClient(mongo_uri)
    database = mongo_client[os.environ["MONGO_DB_NAME"]]
    dataset_id = ObjectId()
    owner_id = ObjectId()
    snapshot_a = "snapshot-a"
    initial_revision = 5
    await _seed_annotation_snapshot(
        database,
        dataset_id=dataset_id,
        owner_id=owner_id,
        snapshot_id=snapshot_a,
        dataset_revision=initial_revision,
    )
    await database.datasets.update_one(
        {"_id": dataset_id},
        {
            "$set": {"image_count": 2},
            "$push": {
                "images": {
                    "id": "image-2",
                    "filename": "image-2.png",
                    "original_file_name": "image-2.png",
                    "path": "/trusted/image-2.png",
                    "width": 32,
                    "height": 32,
                    "annotation_review_status": "unreviewed",
                }
            },
        },
    )

    read_paused = asyncio.Event()
    allow_read = asyncio.Event()
    dataset_collection = _CountingDatasetCollection(database.datasets)
    category_collection = _CountingCategoryCollection(database.annotation_categories)
    annotation_collection = _ControlledAnnotationCollection(
        database.object_annotations,
        read_paused=read_paused,
        allow_read=allow_read,
    )
    monkeypatch.setattr(
        dataset_annotation_service,
        "datasets_collection",
        dataset_collection,
    )
    monkeypatch.setattr(
        dataset_annotation_service,
        "annotation_categories_collection",
        category_collection,
    )
    monkeypatch.setattr(
        dataset_annotation_service,
        "object_annotations_collection",
        annotation_collection,
    )

    current_user = _annotation_test_user(owner_id)
    coverage_task = asyncio.create_task(
        dataset_annotation_service.get_annotation_coverage(
            str(dataset_id),
            current_user,
        ),
        name="coverage-get",
    )

    try:
        await asyncio.wait_for(read_paused.wait(), timeout=3)
        replacement_payload = _coco_payload(
            image_names=("image-1.png", "image-2.png"),
            with_annotation=False,
        )
        import_response = await dataset_annotation_service.import_coco_annotations(
            str(dataset_id),
            _service_upload(replacement_payload),
            current_user,
        )
        active_dataset = await database.datasets.find_one({"_id": dataset_id})
        snapshot_b = active_dataset["active_annotation_snapshot_id"]

        assert snapshot_b != snapshot_a
        assert active_dataset["dataset_revision"] == initial_revision + 1
        assert (
            await database.annotation_categories.count_documents(
                {"dataset_id": dataset_id, "snapshot_id": snapshot_a}
            )
            == 0
        )
        assert (
            await database.object_annotations.count_documents(
                {"dataset_id": dataset_id, "snapshot_id": snapshot_a}
            )
            == 0
        )

        allow_read.set()
        coverage = await asyncio.wait_for(coverage_task, timeout=3)
    finally:
        allow_read.set()
        if not coverage_task.done():
            await asyncio.gather(coverage_task, return_exceptions=True)

    assert coverage.model_dump() == import_response.coverage.model_dump()
    assert coverage.reviewed_images == 2
    assert coverage.reviewed_empty_images == 2
    assert coverage.reviewed_with_objects_images == 0
    assert coverage.unreviewed_images == 0
    assert coverage.coverage_percentage == 100.0
    assert coverage.total_objects == 0
    assert coverage.total_categories == 2
    assert {item.name for item in coverage.objects_by_category} == {
        "cat",
        "unused",
    }
    assert dataset_collection.find_one_calls_by_task["coverage-get"] == 3
    assert annotation_collection.aggregate_calls_by_task["coverage-get"] == 2
    assert category_collection.find_calls == 2
    assert (
        await database.annotation_categories.count_documents(
            {"dataset_id": dataset_id, "snapshot_id": snapshot_b}
        )
        == 2
    )
    assert (
        await database.object_annotations.count_documents(
            {"dataset_id": dataset_id, "snapshot_id": snapshot_b}
        )
        == 0
    )
    await mongo_client.close()


@pytest.mark.asyncio
async def test_annotation_coverage_read_stops_after_retry_limit(monkeypatch):
    mongo_uri = (
        os.environ.get("MONGO_URI")
        or os.environ.get("MONGODB_URI")
        or os.environ.get("MONGO_URL")
        or "mongodb://mongo:27017"
    )
    mongo_client = AsyncMongoClient(mongo_uri)
    database = mongo_client[os.environ["MONGO_DB_NAME"]]
    dataset_id = ObjectId()
    owner_id = ObjectId()
    await _seed_annotation_snapshot(
        database,
        dataset_id=dataset_id,
        owner_id=owner_id,
        snapshot_id="snapshot-changing",
        dataset_revision=10,
    )

    dataset_collection = _CountingDatasetCollection(
        database.datasets,
        change_revision_on_verification=True,
    )
    category_collection = _CountingCategoryCollection(database.annotation_categories)
    annotation_collection = _ControlledAnnotationCollection(database.object_annotations)
    monkeypatch.setattr(
        dataset_annotation_service,
        "datasets_collection",
        dataset_collection,
    )
    monkeypatch.setattr(
        dataset_annotation_service,
        "annotation_categories_collection",
        category_collection,
    )
    monkeypatch.setattr(
        dataset_annotation_service,
        "object_annotations_collection",
        annotation_collection,
    )

    _prepared_arg_1 = str(dataset_id)
    _prepared_arg_2 = _annotation_test_user(owner_id)

    with pytest.raises(
        dataset_annotation_service.ConflictError,
        match="cambió repetidamente",
    ):
        await dataset_annotation_service.get_annotation_coverage(
            _prepared_arg_1,
            _prepared_arg_2,
        )

    max_attempts = dataset_annotation_service.ANNOTATION_COVERAGE_MAX_READ_ATTEMPTS
    assert dataset_collection.find_one_calls == max_attempts + 1
    assert category_collection.find_calls == max_attempts
    assert annotation_collection.aggregate_calls == max_attempts
    unchanged_dataset = await database.datasets.find_one({"_id": dataset_id})
    assert unchanged_dataset["dataset_revision"] == 10
    assert unchanged_dataset["active_annotation_snapshot_id"] == "snapshot-changing"
    await mongo_client.close()


def test_import_coco_snapshot_updates_coverage_states_and_canonical_records(
    client,
    created_dataset,
    auth_headers,
    image_file_factory,
):
    dataset_id = created_dataset["id"]
    add_response = client.post(
        f"/datasets/{dataset_id}/images",
        headers=auth_headers,
        files=[
            image_file_factory("image-2.png"),
            image_file_factory("image-3.png"),
        ],
    )
    assert add_response.status_code == 200

    response = _upload_coco(
        client,
        dataset_id,
        auth_headers,
        _coco_payload(image_names=("image-1.png", "image-2.png")),
    )

    assert response.status_code == 200
    data = response.json()
    assert data["format"] == "coco"
    assert data["replaced_existing_snapshot"] is False

    dataset_response = client.get(f"/datasets/{dataset_id}", headers=auth_headers)
    assert dataset_response.status_code == 200
    assert dataset_response.json()["task_type"] == "detection"

    coverage = data["coverage"]
    assert coverage == {
        "dataset_id": dataset_id,
        "has_annotation_snapshot": True,
        "total_images": 3,
        "reviewed_images": 2,
        "unreviewed_images": 1,
        "reviewed_empty_images": 1,
        "reviewed_with_objects_images": 1,
        "coverage_percentage": 66.67,
        "total_objects": 1,
        "total_categories": 2,
        "objects_by_category": [
            {
                "category_id": coverage["objects_by_category"][0]["category_id"],
                "coco_category_id": 7,
                "name": "cat",
                "object_count": 1,
            },
            {
                "category_id": coverage["objects_by_category"][1]["category_id"],
                "coco_category_id": 8,
                "name": "unused",
                "object_count": 0,
            },
        ],
    }
    persisted_coverage = client.get(
        f"/datasets/{dataset_id}/annotations/coverage",
        headers=auth_headers,
    )
    assert persisted_coverage.status_code == 200
    assert persisted_coverage.json() == coverage

    images_response = client.get(f"/datasets/{dataset_id}/images", headers=auth_headers)
    assert images_response.status_code == 200
    statuses = {
        image["filename"]: image["annotation_review_status"]
        for image in images_response.json()
    }
    assert statuses == {
        "image-1.png": "reviewed_with_objects",
        "image-2.png": "reviewed_empty",
        "image-3.png": "unreviewed",
    }

    mongo_client, database = _sync_database()
    try:
        dataset = database.datasets.find_one({"_id": ObjectId(dataset_id)})
        snapshot_id = dataset["active_annotation_snapshot_id"]
        category = database.annotation_categories.find_one(
            {
                "dataset_id": ObjectId(dataset_id),
                "snapshot_id": snapshot_id,
                "coco_category_id": 7,
            }
        )
        annotation = database.object_annotations.find_one(
            {
                "dataset_id": ObjectId(dataset_id),
                "snapshot_id": snapshot_id,
                "coco_annotation_id": 101,
            }
        )
    finally:
        mongo_client.close()

    assert annotation["category_id"] == category["_id"]
    assert annotation["bbox_x1"] == 1.123456789
    assert annotation["bbox_y1"] == 2.5
    assert annotation["bbox_x2"] == pytest.approx(11.373456789)
    assert annotation["bbox_y2"] == 14.625
    assert annotation["area"] == 124.296875
    assert annotation["iscrowd"] == 0


def test_reimport_is_idempotent_and_replacement_removes_old_snapshot(
    client,
    created_dataset,
    auth_headers,
):
    dataset_id = created_dataset["id"]
    initial_payload = _coco_payload()

    first = _upload_coco(client, dataset_id, auth_headers, initial_payload)
    second = _upload_coco(client, dataset_id, auth_headers, initial_payload)

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["replaced_existing_snapshot"] is True

    replacement_payload = _coco_payload(with_annotation=False)
    replacement_payload["categories"] = []
    replacement = _upload_coco(
        client,
        dataset_id,
        auth_headers,
        replacement_payload,
    )
    assert replacement.status_code == 200
    assert replacement.json()["coverage"]["total_objects"] == 0
    assert replacement.json()["coverage"]["total_categories"] == 0
    assert replacement.json()["coverage"]["reviewed_empty_images"] == 1

    mongo_client, database = _sync_database()
    try:
        dataset_object_id = ObjectId(dataset_id)
        dataset = database.datasets.find_one({"_id": dataset_object_id})
        snapshot_id = dataset["active_annotation_snapshot_id"]
        category_count = database.annotation_categories.count_documents(
            {"dataset_id": dataset_object_id}
        )
        annotation_count = database.object_annotations.count_documents(
            {"dataset_id": dataset_object_id}
        )
        snapshot_ids = database.object_annotations.distinct(
            "snapshot_id",
            {"dataset_id": dataset_object_id},
        )
    finally:
        mongo_client.close()

    assert category_count == 0
    assert annotation_count == 0
    assert snapshot_ids == []
    assert snapshot_id


@pytest.mark.asyncio
async def test_concurrent_import_responses_remain_bound_to_their_own_snapshots(
    monkeypatch,
):
    mongo_uri = (
        os.environ.get("MONGO_URI")
        or os.environ.get("MONGODB_URI")
        or os.environ.get("MONGO_URL")
        or "mongodb://mongo:27017"
    )
    mongo_client = AsyncMongoClient(mongo_uri)
    database = mongo_client[os.environ["MONGO_DB_NAME"]]
    dataset_id = ObjectId()
    owner_id = ObjectId()
    old_snapshot_id = "snapshot-before-a"
    initial_revision = 4
    await _seed_annotation_snapshot(
        database,
        dataset_id=dataset_id,
        owner_id=owner_id,
        snapshot_id=old_snapshot_id,
        dataset_revision=initial_revision,
    )
    await database.datasets.update_one(
        {"_id": dataset_id},
        {
            "$set": {"image_count": 2},
            "$push": {
                "images": {
                    "id": "image-2",
                    "filename": "image-2.png",
                    "original_file_name": "image-2.png",
                    "path": "/trusted/image-2.png",
                    "width": 32,
                    "height": 32,
                    "annotation_review_status": "unreviewed",
                }
            },
        },
    )

    b_prepared = asyncio.Event()
    allow_b_activation = asyncio.Event()
    a_cleanup_started = asyncio.Event()
    allow_a_cleanup = asyncio.Event()
    a_cleanup_finished = asyncio.Event()
    allow_a_response = asyncio.Event()
    cleanup_calls = []

    monkeypatch.setattr(
        dataset_annotation_service,
        "datasets_collection",
        _CoordinatedDatasetCollection(
            database.datasets,
            b_prepared,
            allow_b_activation,
        ),
    )
    monkeypatch.setattr(
        dataset_annotation_service,
        "annotation_categories_collection",
        database.annotation_categories,
    )
    monkeypatch.setattr(
        dataset_annotation_service,
        "object_annotations_collection",
        database.object_annotations,
    )

    original_cleanup = dataset_annotation_service._cleanup_snapshot

    async def coordinated_cleanup(cleanup_dataset_id, cleanup_snapshot_id):
        task = asyncio.current_task()
        cleanup_calls.append(
            (
                task.get_name() if task is not None else None,
                cleanup_dataset_id,
                cleanup_snapshot_id,
            )
        )
        assert isinstance(cleanup_snapshot_id, str)
        if task is not None and task.get_name() == "import-a":
            a_cleanup_started.set()
            await allow_a_cleanup.wait()
        await original_cleanup(cleanup_dataset_id, cleanup_snapshot_id)
        if task is not None and task.get_name() == "import-a":
            a_cleanup_finished.set()
            await allow_a_response.wait()

    monkeypatch.setattr(
        dataset_annotation_service,
        "_cleanup_snapshot",
        coordinated_cleanup,
    )

    payload_a = _coco_payload(image_names=("image-1.png", "image-2.png"))
    payload_a["categories"][0]["name"] = "cat-a"
    payload_b = _coco_payload()
    payload_b["categories"] = [
        {
            "id": 9,
            "name": "dog-b",
            "supercategory": "animal",
        }
    ]
    payload_b["annotations"] = [
        {
            "id": 201,
            "image_id": 1,
            "category_id": 9,
            "bbox": [1, 2, 4, 5],
            "area": 20,
            "iscrowd": 0,
        },
        {
            "id": 202,
            "image_id": 1,
            "category_id": 9,
            "bbox": [8, 9, 6, 7],
            "area": 42,
            "iscrowd": 0,
        },
    ]
    current_user = _annotation_test_user(owner_id)
    import_a = asyncio.create_task(
        dataset_annotation_service.import_coco_annotations(
            str(dataset_id),
            _service_upload(payload_a),
            current_user,
        ),
        name="import-a",
    )
    import_b = None

    try:
        await asyncio.wait_for(a_cleanup_started.wait(), timeout=3)
        after_a_activation = await database.datasets.find_one({"_id": dataset_id})
        snapshot_a = after_a_activation["active_annotation_snapshot_id"]
        assert snapshot_a != old_snapshot_id

        import_b = asyncio.create_task(
            dataset_annotation_service.import_coco_annotations(
                str(dataset_id),
                _service_upload(payload_b),
                current_user,
            ),
            name="import-b",
        )
        await asyncio.wait_for(b_prepared.wait(), timeout=3)

        prepared_categories = [
            document
            async for document in database.annotation_categories.find(
                {"dataset_id": dataset_id}
            )
        ]
        prepared_snapshot_ids = {
            document["snapshot_id"] for document in prepared_categories
        }
        snapshot_b_candidates = prepared_snapshot_ids - {
            old_snapshot_id,
            snapshot_a,
        }
        assert len(snapshot_b_candidates) == 1
        snapshot_b = snapshot_b_candidates.pop()
        assert (
            await database.annotation_categories.count_documents(
                {
                    "dataset_id": dataset_id,
                    "snapshot_id": snapshot_b,
                }
            )
            == 1
        )
        assert (
            await database.object_annotations.count_documents(
                {
                    "dataset_id": dataset_id,
                    "snapshot_id": snapshot_b,
                }
            )
            == 2
        )

        assert cleanup_calls == [
            ("import-a", dataset_id, old_snapshot_id),
        ]
        allow_a_cleanup.set()
        await asyncio.wait_for(a_cleanup_finished.wait(), timeout=3)
        assert (
            await database.annotation_categories.count_documents(
                {
                    "dataset_id": dataset_id,
                    "snapshot_id": snapshot_b,
                }
            )
            == 1
        )
        assert (
            await database.object_annotations.count_documents(
                {
                    "dataset_id": dataset_id,
                    "snapshot_id": snapshot_b,
                }
            )
            == 2
        )

        allow_b_activation.set()
        response_b = await asyncio.wait_for(import_b, timeout=3)

        assert not import_a.done()
        assert (
            await database.annotation_categories.count_documents(
                {
                    "dataset_id": dataset_id,
                    "snapshot_id": snapshot_a,
                }
            )
            == 0
        )
        assert (
            await database.object_annotations.count_documents(
                {
                    "dataset_id": dataset_id,
                    "snapshot_id": snapshot_a,
                }
            )
            == 0
        )
        active_while_a_is_paused = await database.datasets.find_one({"_id": dataset_id})
        assert active_while_a_is_paused["active_annotation_snapshot_id"] == snapshot_b
        public_coverage = await dataset_annotation_service.get_annotation_coverage(
            str(dataset_id),
            current_user,
        )

        allow_a_response.set()
        response_a = await asyncio.wait_for(import_a, timeout=3)
    finally:
        allow_a_cleanup.set()
        allow_a_response.set()
        allow_b_activation.set()
        pending_imports = [
            task
            for task in (import_a, import_b)
            if task is not None and not task.done()
        ]
        if pending_imports:
            await asyncio.gather(*pending_imports, return_exceptions=True)

    final_dataset = await database.datasets.find_one({"_id": dataset_id})
    final_categories = [
        document
        async for document in database.annotation_categories.find(
            {"dataset_id": dataset_id}
        )
    ]
    final_annotations = [
        document
        async for document in database.object_annotations.find(
            {"dataset_id": dataset_id}
        )
    ]

    assert response_a.coverage.model_dump() == {
        "dataset_id": str(dataset_id),
        "has_annotation_snapshot": True,
        "total_images": 2,
        "reviewed_images": 2,
        "unreviewed_images": 0,
        "reviewed_empty_images": 1,
        "reviewed_with_objects_images": 1,
        "coverage_percentage": 100.0,
        "total_objects": 1,
        "total_categories": 2,
        "objects_by_category": [
            {
                "category_id": response_a.coverage.objects_by_category[0].category_id,
                "coco_category_id": 7,
                "name": "cat-a",
                "object_count": 1,
            },
            {
                "category_id": response_a.coverage.objects_by_category[1].category_id,
                "coco_category_id": 8,
                "name": "unused",
                "object_count": 0,
            },
        ],
    }
    expected_b_coverage = {
        "dataset_id": str(dataset_id),
        "has_annotation_snapshot": True,
        "total_images": 2,
        "reviewed_images": 1,
        "unreviewed_images": 1,
        "reviewed_empty_images": 0,
        "reviewed_with_objects_images": 1,
        "coverage_percentage": 50.0,
        "total_objects": 2,
        "total_categories": 1,
        "objects_by_category": [
            {
                "category_id": response_b.coverage.objects_by_category[0].category_id,
                "coco_category_id": 9,
                "name": "dog-b",
                "object_count": 2,
            }
        ],
    }
    assert response_b.coverage.model_dump() == expected_b_coverage
    assert public_coverage.model_dump() == expected_b_coverage
    assert final_dataset["active_annotation_snapshot_id"] == snapshot_b
    assert final_dataset["dataset_revision"] == initial_revision + 2
    assert len(final_categories) == 1
    assert len(final_annotations) == 2
    assert {document["snapshot_id"] for document in final_categories} == {snapshot_b}
    assert {document["snapshot_id"] for document in final_annotations} == {snapshot_b}
    assert len({document["_id"] for document in final_categories}) == 1
    assert len({document["_id"] for document in final_annotations}) == 2
    assert cleanup_calls == [
        ("import-a", dataset_id, old_snapshot_id),
        ("import-b", dataset_id, snapshot_a),
    ]
    await mongo_client.close()


@pytest.mark.asyncio
async def test_partial_annotation_write_failure_cleans_only_prepared_snapshot(
    monkeypatch,
):
    mongo_uri = (
        os.environ.get("MONGO_URI")
        or os.environ.get("MONGODB_URI")
        or os.environ.get("MONGO_URL")
        or "mongodb://mongo:27017"
    )
    mongo_client = AsyncMongoClient(mongo_uri)
    database = mongo_client[os.environ["MONGO_DB_NAME"]]
    dataset_id = ObjectId()
    owner_id = ObjectId()
    old_snapshot_id = "snapshot-active"
    other_preparing_snapshot_id = "snapshot-other-import"
    initial_revision = 9
    await _seed_annotation_snapshot(
        database,
        dataset_id=dataset_id,
        owner_id=owner_id,
        snapshot_id=old_snapshot_id,
        dataset_revision=initial_revision,
    )
    await _seed_preparing_snapshot(
        database,
        dataset_id,
        other_preparing_snapshot_id,
    )

    monkeypatch.setattr(
        dataset_annotation_service,
        "datasets_collection",
        database.datasets,
    )
    monkeypatch.setattr(
        dataset_annotation_service,
        "annotation_categories_collection",
        database.annotation_categories,
    )
    monkeypatch.setattr(
        dataset_annotation_service,
        "object_annotations_collection",
        _FailingAnnotationCollection(database.object_annotations),
    )

    cleanup_calls = []
    original_cleanup = dataset_annotation_service._cleanup_snapshot

    async def recording_cleanup(cleanup_dataset_id, cleanup_snapshot_id):
        cleanup_calls.append((cleanup_dataset_id, cleanup_snapshot_id))
        await original_cleanup(cleanup_dataset_id, cleanup_snapshot_id)

    monkeypatch.setattr(
        dataset_annotation_service,
        "_cleanup_snapshot",
        recording_cleanup,
    )
    current_user = _annotation_test_user(owner_id)

    _prepared_arg_1_2 = str(dataset_id)
    _prepared_arg_2_2 = _service_upload(_coco_payload())

    with pytest.raises(
        PyMongoError,
        match="simulated annotation persistence failure",
    ):
        await dataset_annotation_service.import_coco_annotations(
            _prepared_arg_1_2,
            _prepared_arg_2_2,
            current_user,
        )

    assert len(cleanup_calls) == 1
    cleanup_dataset_id, failed_snapshot_id = cleanup_calls[0]
    assert cleanup_dataset_id == dataset_id
    assert isinstance(failed_snapshot_id, str)
    assert failed_snapshot_id not in {
        old_snapshot_id,
        other_preparing_snapshot_id,
    }
    assert (
        await database.annotation_categories.count_documents(
            {
                "dataset_id": dataset_id,
                "snapshot_id": failed_snapshot_id,
            }
        )
        == 0
    )
    assert (
        await database.object_annotations.count_documents(
            {
                "dataset_id": dataset_id,
                "snapshot_id": failed_snapshot_id,
            }
        )
        == 0
    )

    unchanged_dataset = await database.datasets.find_one({"_id": dataset_id})
    assert unchanged_dataset["active_annotation_snapshot_id"] == old_snapshot_id
    assert unchanged_dataset["dataset_revision"] == initial_revision
    for preserved_snapshot_id in (
        old_snapshot_id,
        other_preparing_snapshot_id,
    ):
        assert (
            await database.annotation_categories.count_documents(
                {
                    "dataset_id": dataset_id,
                    "snapshot_id": preserved_snapshot_id,
                }
            )
            == 1
        )
        assert (
            await database.object_annotations.count_documents(
                {
                    "dataset_id": dataset_id,
                    "snapshot_id": preserved_snapshot_id,
                }
            )
            == 1
        )

    coverage = await dataset_annotation_service.get_annotation_coverage(
        str(dataset_id),
        current_user,
    )
    assert coverage.has_annotation_snapshot is True
    assert coverage.reviewed_with_objects_images == 1
    assert coverage.total_categories == 1
    assert coverage.total_objects == 1
    assert coverage.objects_by_category[0].name == f"{old_snapshot_id}-cat"
    await mongo_client.close()


@pytest.mark.parametrize(
    "invalid_payload",
    [
        b"{bad-json",
        {
            **_coco_payload(),
            "annotations": [
                *_coco_payload()["annotations"],
                *_coco_payload()["annotations"],
            ],
        },
        {
            **_coco_payload(),
            "annotations": [
                {
                    **_coco_payload()["annotations"][0],
                    "category_id": 999,
                }
            ],
        },
        {
            **_coco_payload(),
            "annotations": [
                {
                    **_coco_payload()["annotations"][0],
                    "bbox": [30, 30, 5, 5],
                }
            ],
        },
        {
            **_coco_payload(),
            "images": [
                {
                    **_coco_payload()["images"][0],
                    "file_name": "../../image-1.png",
                }
            ],
        },
    ],
)
def test_invalid_import_preserves_previous_snapshot(
    client,
    created_dataset,
    auth_headers,
    invalid_payload,
):
    dataset_id = created_dataset["id"]
    valid_response = _upload_coco(
        client,
        dataset_id,
        auth_headers,
        _coco_payload(),
    )
    assert valid_response.status_code == 200

    mongo_client, database = _sync_database()
    dataset_object_id = ObjectId(dataset_id)
    before = database.datasets.find_one({"_id": dataset_object_id})
    previous_snapshot = before["active_annotation_snapshot_id"]

    invalid_response = _upload_coco(
        client,
        dataset_id,
        auth_headers,
        invalid_payload,
    )
    assert invalid_response.status_code == 400
    detail = invalid_response.json()["detail"]
    assert detail["code"] == "invalid_coco_annotations"
    assert detail["snapshot_applied"] is False

    after = database.datasets.find_one({"_id": dataset_object_id})
    active_categories = database.annotation_categories.count_documents(
        {
            "dataset_id": dataset_object_id,
            "snapshot_id": previous_snapshot,
        }
    )
    active_annotations = database.object_annotations.count_documents(
        {
            "dataset_id": dataset_object_id,
            "snapshot_id": previous_snapshot,
        }
    )
    mongo_client.close()

    assert after["active_annotation_snapshot_id"] == previous_snapshot
    assert active_categories == 2
    assert active_annotations == 1


def test_annotation_endpoints_require_owner(
    client,
    created_dataset,
    auth_headers,
):
    dataset_id = created_dataset["id"]

    unauthenticated = client.get(f"/datasets/{dataset_id}/annotations/coverage")
    assert unauthenticated.status_code in {401, 403}

    signup = client.post(
        "/auth/signup",
        json={
            "name": "Other User",
            "email": "other@example.com",
            "password": "secure-password-123",
        },
    )
    assert signup.status_code == 200
    other_headers = {
        "Authorization": f"Bearer {signup.json()['access_token']}",
    }

    forbidden_get = client.get(
        f"/datasets/{dataset_id}/annotations/coverage",
        headers=other_headers,
    )
    forbidden_import = _upload_coco(
        client,
        dataset_id,
        other_headers,
        _coco_payload(),
    )
    assert forbidden_get.status_code == 403
    assert forbidden_import.status_code == 403

    missing = client.get(
        "/datasets/507f1f77bcf86cd799439011/annotations/coverage",
        headers=auth_headers,
    )
    assert missing.status_code == 404


def test_annotation_upload_rejects_type_size_and_unsafe_name(
    client,
    created_dataset,
    auth_headers,
    monkeypatch,
):
    dataset_id = created_dataset["id"]

    wrong_type = client.post(
        f"/datasets/{dataset_id}/annotations/coco",
        headers=auth_headers,
        files={"file": ("coco.txt", b"{}", "text/plain")},
    )
    assert wrong_type.status_code == 400

    unsafe_name = _upload_coco(
        client,
        dataset_id,
        auth_headers,
        _coco_payload(),
        filename="../coco.json",
    )
    assert unsafe_name.status_code == 400

    monkeypatch.setattr(
        dataset_annotation_service,
        "COCO_MAX_UPLOAD_BYTES",
        10,
    )
    oversized = _upload_coco(
        client,
        dataset_id,
        auth_headers,
        _coco_payload(),
    )
    assert oversized.status_code == 400
    assert oversized.json()["detail"]["field"] == "size"


def test_deleting_image_and_dataset_removes_annotation_records(
    client,
    created_dataset,
    auth_headers,
):
    dataset_id = created_dataset["id"]
    image_id = created_dataset["images"][0]["id"]
    assert (
        _upload_coco(
            client,
            dataset_id,
            auth_headers,
            _coco_payload(),
        ).status_code
        == 200
    )

    delete_image = client.delete(
        f"/datasets/{dataset_id}/images/{image_id}", headers=auth_headers
    )
    assert delete_image.status_code == 200

    mongo_client, database = _sync_database()
    dataset_object_id = ObjectId(dataset_id)
    assert (
        database.object_annotations.count_documents(
            {
                "dataset_id": dataset_object_id,
                "image_id": image_id,
            }
        )
        == 0
    )

    delete_dataset = client.delete(f"/datasets/{dataset_id}", headers=auth_headers)
    assert delete_dataset.status_code == 200
    assert (
        database.annotation_categories.count_documents(
            {"dataset_id": dataset_object_id}
        )
        == 0
    )
    assert (
        database.object_annotations.count_documents({"dataset_id": dataset_object_id})
        == 0
    )
    mongo_client.close()


@pytest.mark.asyncio
async def test_migration_marks_existing_images_unreviewed_and_is_reversible():
    mongo_uri = (
        os.environ.get("MONGO_URI")
        or os.environ.get("MONGODB_URI")
        or os.environ.get("MONGO_URL")
        or "mongodb://mongo:27017"
    )
    mongo_client = AsyncMongoClient(mongo_uri)
    database = mongo_client[os.environ["MONGO_DB_NAME"]]
    dataset_id = ObjectId()
    await database.datasets.insert_one(
        {
            "_id": dataset_id,
            "name": "Legacy",
            "owner_id": ObjectId(),
            "image_count": 1,
            "images": [
                {
                    "id": "legacy-image",
                    "filename": "legacy.png",
                    "label": "free label",
                }
            ],
        }
    )

    await upgrade_coco_annotations_schema(database)
    migrated = await database.datasets.find_one({"_id": dataset_id})

    assert migrated["dataset_revision"] == 0
    assert migrated["images"][0]["annotation_review_status"] == "unreviewed"
    assert migrated["images"][0]["original_file_name"] == "legacy.png"
    assert migrated["images"][0]["label"] == "free label"

    await downgrade_coco_annotations_schema(database)
    downgraded = await database.datasets.find_one({"_id": dataset_id})
    assert "dataset_revision" not in downgraded
    assert "annotation_review_status" not in downgraded["images"][0]
    assert "original_file_name" not in downgraded["images"][0]
    assert downgraded["images"][0]["label"] == "free label"

    await upgrade_coco_annotations_schema(database)
    await mongo_client.close()


@pytest.mark.asyncio
async def test_category_and_annotation_unique_indexes_are_enforced():
    mongo_uri = (
        os.environ.get("MONGO_URI")
        or os.environ.get("MONGODB_URI")
        or os.environ.get("MONGO_URL")
        or "mongodb://mongo:27017"
    )
    mongo_client = AsyncMongoClient(mongo_uri)
    database = mongo_client[os.environ["MONGO_DB_NAME"]]
    await upgrade_coco_annotations_schema(database)

    dataset_id = ObjectId()
    category = {
        "_id": "category-1",
        "dataset_id": dataset_id,
        "snapshot_id": "snapshot-1",
        "coco_category_id": 1,
        "name": "cat",
    }
    await database.annotation_categories.insert_one(category)

    with pytest.raises(DuplicateKeyError):
        await database.annotation_categories.insert_one(
            {
                **category,
                "_id": "category-2",
            }
        )

    annotation = {
        "_id": "annotation-1",
        "dataset_id": dataset_id,
        "snapshot_id": "snapshot-1",
        "image_id": "image-1",
        "category_id": "category-1",
        "coco_annotation_id": 1,
        "bbox_x1": 0.1,
        "bbox_y1": 0.2,
        "bbox_x2": 1.3,
        "bbox_y2": 2.4,
        "area": 2.64,
        "iscrowd": 0,
    }
    await database.object_annotations.insert_one(annotation)

    with pytest.raises(DuplicateKeyError):
        await database.object_annotations.insert_one(
            {
                **annotation,
                "_id": "annotation-2",
            }
        )
    await mongo_client.close()


def test_classification_dataset_rejects_coco_import(
    client,
    registered_user,
    auth_headers,
    image_file_factory,
):
    user_id = registered_user["user"]["id"]
    created = client.post(
        "/datasets",
        headers=auth_headers,
        data={
            "user_id": user_id,
            "name": "Classification Dataset",
            "task_type": "classification",
        },
        files=[image_file_factory("image.png")],
    )
    assert created.status_code == 200

    response = client.post(
        f"/datasets/{created.json()['id']}/annotations/coco",
        headers=auth_headers,
        files={
            "file": (
                "annotations.json",
                b"{}",
                "application/json",
            )
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Las anotaciones COCO solo están disponibles para conjuntos de datos de detección"
    )
