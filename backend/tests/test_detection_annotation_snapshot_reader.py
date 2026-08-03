import builtins
from dataclasses import FrozenInstanceError

import pytest
from bson import ObjectId
from pymongo.errors import PyMongoError

import services.detection_annotation_snapshot as snapshot_reader
from services.detection_annotation_snapshot import (
    DetectionAnnotationSnapshotConflictError,
    DetectionAnnotationSnapshotError,
    DetectionAnnotationSnapshotReadError,
    read_detection_annotation_snapshot,
)


def _dataset(
    dataset_id,
    *,
    revision=1,
    snapshot_id="snapshot-a",
):
    document = {
        "_id": dataset_id,
        "dataset_revision": revision,
        "images": [
            {
                "id": "image-empty",
                "filename": "empty.png",
                "width": 100,
                "height": 80,
                "annotation_review_status": "reviewed_empty",
            },
            {
                "id": "image-object",
                "filename": "object.png",
                "width": 100,
                "height": 80,
                "annotation_review_status": "reviewed_with_objects",
            },
            {
                "id": "image-unreviewed",
                "filename": "unreviewed.png",
                "width": 100,
                "height": 80,
                "annotation_review_status": "unreviewed",
            },
        ],
    }
    if snapshot_id is not None:
        document["active_annotation_snapshot_id"] = snapshot_id
    return document


def _category(snapshot_id):
    return {
        "_id": f"{snapshot_id}-category",
        "dataset_id": ObjectId(),
        "snapshot_id": snapshot_id,
        "coco_category_id": 91,
        "name": "person",
    }


def _annotation(snapshot_id):
    return {
        "_id": f"{snapshot_id}-annotation",
        "dataset_id": ObjectId(),
        "snapshot_id": snapshot_id,
        "image_id": "image-object",
        "category_id": f"{snapshot_id}-category",
        "coco_annotation_id": 1001,
        "bbox_x1": 1.25,
        "bbox_y1": 2.5,
        "bbox_x2": 31.75,
        "bbox_y2": 42.125,
        "area": 1208.5625,
        "iscrowd": 0,
    }


class _AsyncCursor:
    def __init__(self, documents):
        self.documents = list(documents)

    def sort(self, *args):
        return self

    def __aiter__(self):
        async def iterate():
            for document in self.documents:
                yield document

        return iterate()


class _DatasetCollection:
    def __init__(self, documents):
        self.documents = list(documents)
        self.queries = []

    async def find_one(self, query):
        self.queries.append(query)
        if not self.documents:
            raise AssertionError("unexpected dataset read")
        return self.documents.pop(0)


class _SnapshotCollection:
    def __init__(self, documents_by_snapshot):
        self.documents_by_snapshot = documents_by_snapshot
        self.queries = []

    def find(self, query):
        self.queries.append(query)
        return _AsyncCursor(self.documents_by_snapshot.get(query["snapshot_id"], ()))


def _install_snapshot_documents(
    monkeypatch,
    *,
    dataset,
    categories,
    annotations,
):
    monkeypatch.setattr(
        snapshot_reader,
        "datasets_collection",
        _DatasetCollection([dataset, dataset]),
    )
    monkeypatch.setattr(
        snapshot_reader,
        "annotation_categories_collection",
        _SnapshotCollection({"snapshot-a": categories}),
    )
    monkeypatch.setattr(
        snapshot_reader,
        "object_annotations_collection",
        _SnapshotCollection({"snapshot-a": annotations}),
    )


@pytest.mark.asyncio
async def test_snapshot_reader_filters_every_collection_by_dataset_and_snapshot(
    monkeypatch,
):
    dataset_id = ObjectId()
    dataset = _dataset(dataset_id)
    datasets = _DatasetCollection([dataset, dataset])
    categories = _SnapshotCollection({"snapshot-a": [_category("snapshot-a")]})
    annotations = _SnapshotCollection({"snapshot-a": [_annotation("snapshot-a")]})
    monkeypatch.setattr(snapshot_reader, "datasets_collection", datasets)
    monkeypatch.setattr(
        snapshot_reader,
        "annotation_categories_collection",
        categories,
    )
    monkeypatch.setattr(
        snapshot_reader,
        "object_annotations_collection",
        annotations,
    )

    snapshot = await read_detection_annotation_snapshot(str(dataset_id))

    expected_filter = {
        "dataset_id": dataset_id,
        "snapshot_id": "snapshot-a",
    }
    assert categories.queries == [expected_filter]
    assert annotations.queries == [expected_filter]
    assert snapshot.snapshot_id == "snapshot-a"
    assert snapshot.dataset_revision == 1
    assert snapshot.categories[0].internal_id == "snapshot-a-category"
    assert snapshot.annotations[0].category_internal_id == "snapshot-a-category"
    assert snapshot.annotations[0].object_internal_id == "snapshot-a-annotation"
    assert snapshot.reviewed_image_ids == {"image-empty", "image-object"}


@pytest.mark.asyncio
async def test_snapshot_reader_retries_without_mixing_replaced_snapshots(
    monkeypatch,
):
    dataset_id = ObjectId()
    dataset_a = _dataset(dataset_id, revision=1, snapshot_id="snapshot-a")
    dataset_b = _dataset(dataset_id, revision=2, snapshot_id="snapshot-b")
    datasets = _DatasetCollection([dataset_a, dataset_b, dataset_b])
    categories = _SnapshotCollection(
        {
            "snapshot-a": [_category("snapshot-a")],
            "snapshot-b": [_category("snapshot-b")],
        }
    )
    annotations = _SnapshotCollection(
        {
            "snapshot-a": [_annotation("snapshot-a")],
            "snapshot-b": [_annotation("snapshot-b")],
        }
    )
    monkeypatch.setattr(snapshot_reader, "datasets_collection", datasets)
    monkeypatch.setattr(
        snapshot_reader,
        "annotation_categories_collection",
        categories,
    )
    monkeypatch.setattr(
        snapshot_reader,
        "object_annotations_collection",
        annotations,
    )

    snapshot = await read_detection_annotation_snapshot(str(dataset_id))

    assert snapshot.snapshot_id == "snapshot-b"
    assert snapshot.dataset_revision == 2
    assert [item.internal_id for item in snapshot.categories] == ["snapshot-b-category"]
    assert [item.category_internal_id for item in snapshot.annotations] == [
        "snapshot-b-category"
    ]
    assert [query["snapshot_id"] for query in categories.queries] == [
        "snapshot-a",
        "snapshot-b",
    ]


@pytest.mark.asyncio
async def test_snapshot_reader_raises_typed_conflict_after_bounded_retries(
    monkeypatch,
):
    dataset_id = ObjectId()
    datasets = _DatasetCollection(
        [
            _dataset(dataset_id, revision=1, snapshot_id="snapshot-a"),
            _dataset(dataset_id, revision=2, snapshot_id="snapshot-b"),
            _dataset(dataset_id, revision=3, snapshot_id="snapshot-c"),
            _dataset(dataset_id, revision=4, snapshot_id="snapshot-d"),
        ]
    )
    categories = _SnapshotCollection(
        {
            snapshot_id: [_category(snapshot_id)]
            for snapshot_id in ("snapshot-a", "snapshot-b", "snapshot-c")
        }
    )
    annotations = _SnapshotCollection(
        {
            snapshot_id: [_annotation(snapshot_id)]
            for snapshot_id in ("snapshot-a", "snapshot-b", "snapshot-c")
        }
    )
    monkeypatch.setattr(snapshot_reader, "datasets_collection", datasets)
    monkeypatch.setattr(
        snapshot_reader,
        "annotation_categories_collection",
        categories,
    )
    monkeypatch.setattr(
        snapshot_reader,
        "object_annotations_collection",
        annotations,
    )

    with pytest.raises(
        DetectionAnnotationSnapshotConflictError,
        match="changed repeatedly",
    ):
        await read_detection_annotation_snapshot(
            str(dataset_id),
            max_attempts=3,
        )

    assert len(categories.queries) == 3
    assert len(annotations.queries) == 3


@pytest.mark.asyncio
async def test_snapshot_reader_without_active_snapshot_skips_annotation_queries(
    monkeypatch,
):
    dataset_id = ObjectId()
    dataset = _dataset(dataset_id, snapshot_id=None)
    datasets = _DatasetCollection([dataset, dataset])
    categories = _SnapshotCollection({})
    annotations = _SnapshotCollection({})
    monkeypatch.setattr(snapshot_reader, "datasets_collection", datasets)
    monkeypatch.setattr(
        snapshot_reader,
        "annotation_categories_collection",
        categories,
    )
    monkeypatch.setattr(
        snapshot_reader,
        "object_annotations_collection",
        annotations,
    )

    snapshot = await read_detection_annotation_snapshot(str(dataset_id))

    assert snapshot.has_active_snapshot is False
    assert snapshot.categories == ()
    assert snapshot.annotations == ()
    assert categories.queries == []
    assert annotations.queries == []


@pytest.mark.asyncio
async def test_materialized_snapshot_is_immutable(monkeypatch):
    dataset_id = ObjectId()
    dataset = _dataset(dataset_id)
    monkeypatch.setattr(
        snapshot_reader,
        "datasets_collection",
        _DatasetCollection([dataset, dataset]),
    )
    monkeypatch.setattr(
        snapshot_reader,
        "annotation_categories_collection",
        _SnapshotCollection({"snapshot-a": [_category("snapshot-a")]}),
    )
    monkeypatch.setattr(
        snapshot_reader,
        "object_annotations_collection",
        _SnapshotCollection({"snapshot-a": [_annotation("snapshot-a")]}),
    )

    snapshot = await read_detection_annotation_snapshot(str(dataset_id))

    assert isinstance(snapshot.images, tuple)
    assert isinstance(snapshot.categories, tuple)
    assert isinstance(snapshot.annotations, tuple)
    with pytest.raises(FrozenInstanceError):
        snapshot.snapshot_id = "mutated"


@pytest.mark.asyncio
async def test_snapshot_reader_does_not_read_original_coco_json(monkeypatch):
    dataset_id = ObjectId()
    dataset = _dataset(dataset_id)
    monkeypatch.setattr(
        snapshot_reader,
        "datasets_collection",
        _DatasetCollection([dataset, dataset]),
    )
    monkeypatch.setattr(
        snapshot_reader,
        "annotation_categories_collection",
        _SnapshotCollection({"snapshot-a": [_category("snapshot-a")]}),
    )
    monkeypatch.setattr(
        snapshot_reader,
        "object_annotations_collection",
        _SnapshotCollection({"snapshot-a": [_annotation("snapshot-a")]}),
    )

    def forbidden_open(*args, **kwargs):
        raise AssertionError("the canonical snapshot reader must not open files")

    monkeypatch.setattr(builtins, "open", forbidden_open)

    snapshot = await read_detection_annotation_snapshot(str(dataset_id))

    assert snapshot.snapshot_id == "snapshot-a"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("document_type", "field_name"),
    [
        ("image", "filename"),
        ("category", "name"),
        ("annotation", "bbox_x1"),
    ],
)
async def test_snapshot_reader_wraps_missing_persisted_fields(
    monkeypatch,
    document_type,
    field_name,
):
    dataset_id = ObjectId()
    dataset = _dataset(dataset_id)
    category = _category("snapshot-a")
    annotation = _annotation("snapshot-a")
    if document_type == "image":
        del dataset["images"][0][field_name]
    elif document_type == "category":
        del category[field_name]
    else:
        del annotation[field_name]
    _install_snapshot_documents(
        monkeypatch,
        dataset=dataset,
        categories=[category],
        annotations=[annotation],
    )

    with pytest.raises(
        DetectionAnnotationSnapshotError,
        match=f"missing required field '{field_name}'",
    ) as captured:
        await read_detection_annotation_snapshot(str(dataset_id))

    assert isinstance(captured.value.__cause__, KeyError)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("document_type", "field_name", "invalid_value"),
    [
        ("dataset", "images", {"not": "a sequence"}),
        ("image", "width", "100"),
        ("image", "annotation_review_status", "invalid"),
        ("category", "coco_category_id", "91"),
        ("category", "supercategory", 7),
        ("category", "dataset_id", "not-an-object-id"),
        ("category", "snapshot_id", 7),
        ("annotation", "iscrowd", True),
        ("annotation", "area", "400"),
    ],
)
async def test_snapshot_reader_rejects_invalid_persisted_field_types(
    monkeypatch,
    document_type,
    field_name,
    invalid_value,
):
    dataset_id = ObjectId()
    dataset = _dataset(dataset_id)
    category = _category("snapshot-a")
    annotation = _annotation("snapshot-a")
    if document_type == "dataset":
        dataset[field_name] = invalid_value
    elif document_type == "image":
        dataset["images"][0][field_name] = invalid_value
    elif document_type == "category":
        category[field_name] = invalid_value
    else:
        annotation[field_name] = invalid_value
    _install_snapshot_documents(
        monkeypatch,
        dataset=dataset,
        categories=[category],
        annotations=[annotation],
    )

    with pytest.raises(DetectionAnnotationSnapshotError):
        await read_detection_annotation_snapshot(str(dataset_id))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field_name", "invalid_reference", "message"),
    [
        ("image_id", "missing-image", "unknown image"),
        ("category_id", "missing-category", "unknown category"),
    ],
)
async def test_snapshot_reader_rejects_incoherent_persisted_references(
    monkeypatch,
    field_name,
    invalid_reference,
    message,
):
    dataset_id = ObjectId()
    dataset = _dataset(dataset_id)
    annotation = _annotation("snapshot-a")
    annotation[field_name] = invalid_reference
    _install_snapshot_documents(
        monkeypatch,
        dataset=dataset,
        categories=[_category("snapshot-a")],
        annotations=[annotation],
    )

    with pytest.raises(DetectionAnnotationSnapshotError, match=message):
        await read_detection_annotation_snapshot(str(dataset_id))


@pytest.mark.asyncio
async def test_snapshot_reader_wraps_database_errors_with_cause(monkeypatch):
    dataset_id = ObjectId()
    provider_error = PyMongoError("database unavailable")

    class _FailingDatasetCollection:
        async def find_one(self, query):
            raise provider_error

    monkeypatch.setattr(
        snapshot_reader,
        "datasets_collection",
        _FailingDatasetCollection(),
    )

    with pytest.raises(DetectionAnnotationSnapshotReadError) as captured:
        await read_detection_annotation_snapshot(str(dataset_id))

    assert captured.value.__cause__ is provider_error
