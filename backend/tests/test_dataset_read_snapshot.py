import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from bson import ObjectId
from PIL import Image

import services.dataset_service as dataset_service


class _CountingImages(list):
    def __init__(self, values):
        super().__init__(values)
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        return super().__iter__()


@pytest.mark.asyncio
async def test_dataset_read_snapshot_reuses_one_document_and_image_index(
    monkeypatch,
    tmp_path,
):
    dataset_id = str(ObjectId())
    images = []
    for index, color in enumerate(((255, 0, 0), (0, 255, 0)), start=1):
        image_id = f"image-{index}"
        image_path = tmp_path / f"{image_id}.png"
        Image.new("RGB", (4, 3), color=color).save(image_path)
        images.append(
            {
                "id": image_id,
                "filename": image_path.name,
                "stored_filename": image_path.name,
                "path": str(image_path),
                "label": None,
                "width": 4,
                "height": 3,
                "annotation_review_status": "unreviewed",
            }
        )

    counted_images = _CountingImages(images)
    dataset_document = {
        "_id": ObjectId(dataset_id),
        "name": "benchmark-dataset",
        "owner_id": ObjectId(),
        "image_count": len(images),
        "images": counted_images,
    }
    find_one = AsyncMock(return_value=dataset_document)
    monkeypatch.setattr(
        dataset_service,
        "datasets_collection",
        SimpleNamespace(find_one=find_one),
    )

    with dataset_service.dataset_read_snapshot(dataset_id):
        dataset = await dataset_service.get_dataset(dataset_id)
        first = await dataset_service.get_image_from_dataset(dataset_id, "image-1")
        second = await dataset_service.get_image_from_dataset(dataset_id, "image-2")

    assert dataset.image_count == 2
    assert first.size == (4, 3)
    assert second.size == (4, 3)
    assert find_one.await_count == 1
    assert counted_images.iterations == 2


@pytest.mark.asyncio
async def test_dataset_read_snapshot_is_scoped_and_does_not_become_global_cache(
    monkeypatch,
    tmp_path,
):
    dataset_id = str(ObjectId())
    image_path = tmp_path / "image.png"
    Image.new("RGB", (2, 2), color=(10, 20, 30)).save(image_path)
    dataset_document = {
        "_id": ObjectId(dataset_id),
        "name": "dataset",
        "owner_id": ObjectId(),
        "image_count": 1,
        "images": [
            {
                "id": "image-1",
                "filename": "image.png",
                "stored_filename": "image.png",
                "path": str(image_path),
                "label": None,
                "width": 2,
                "height": 2,
                "annotation_review_status": "unreviewed",
            }
        ],
    }
    find_one = AsyncMock(return_value=dataset_document)
    monkeypatch.setattr(
        dataset_service,
        "datasets_collection",
        SimpleNamespace(find_one=find_one),
    )

    with dataset_service.dataset_read_snapshot(dataset_id):
        await dataset_service.get_dataset(dataset_id)
        await dataset_service.get_image_from_dataset(dataset_id, "image-1")

    await dataset_service.get_image_from_dataset(dataset_id, "image-1")

    assert find_one.await_count == 2


@pytest.mark.asyncio
async def test_dataset_read_snapshot_is_isolated_between_concurrent_tasks(
    monkeypatch,
    tmp_path,
):
    documents = {}
    for index, color in enumerate(((200, 10, 10), (10, 200, 10)), start=1):
        dataset_id = str(ObjectId())
        image_path = tmp_path / f"concurrent-{index}.png"
        Image.new("RGB", (3, 3), color=color).save(image_path)
        documents[dataset_id] = {
            "_id": ObjectId(dataset_id),
            "name": f"dataset-{index}",
            "owner_id": ObjectId(),
            "image_count": 1,
            "images": [
                {
                    "id": "image-1",
                    "filename": image_path.name,
                    "stored_filename": image_path.name,
                    "path": str(image_path),
                    "label": None,
                    "width": 3,
                    "height": 3,
                    "annotation_review_status": "unreviewed",
                }
            ],
        }

    async def find_one(query):
        await asyncio.sleep(0)
        return documents[str(query["_id"])]

    collection = SimpleNamespace(find_one=AsyncMock(side_effect=find_one))
    monkeypatch.setattr(dataset_service, "datasets_collection", collection)

    async def load(dataset_id):
        with dataset_service.dataset_read_snapshot(dataset_id):
            dataset = await dataset_service.get_dataset(dataset_id)
            await asyncio.sleep(0)
            image = await dataset_service.get_image_from_dataset(
                dataset_id,
                "image-1",
            )
            return dataset.id, image.getpixel((0, 0))

    results = await asyncio.gather(*(load(dataset_id) for dataset_id in documents))

    assert {dataset_id for dataset_id, _ in results} == set(documents)
    assert collection.find_one.await_count == 2
