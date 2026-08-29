from types import SimpleNamespace

import pytest
from bson import ObjectId

import services.dataset_service as dataset_service
import services.individual_analysis_history_service as history_service


class _Collection:
    def __init__(self, documents=None):
        self.documents = list(documents or [])
        self.updates = []
        self.deletes = []

    def find(self, query, *_args):
        matching = [
            document
            for document in self.documents
            if all(document.get(key) == value for key, value in query.items())
        ]

        class _Cursor:
            def __aiter__(self):
                self._iterator = iter(matching)
                return self

            async def __anext__(self):
                try:
                    return next(self._iterator)
                except StopIteration as error:
                    raise StopAsyncIteration from error

        return _Cursor()

    async def delete_one(self, query):
        self.documents = [
            document
            for document in self.documents
            if not all(document.get(key) == value for key, value in query.items())
        ]
        return SimpleNamespace(deleted_count=1)

    async def update_one(self, query, update):
        self.updates.append((query, update))
        return SimpleNamespace(modified_count=1)

    async def delete_many(self, query):
        self.deletes.append(query)
        return SimpleNamespace(deleted_count=0)


@pytest.mark.asyncio
async def test_delete_individual_analyses_for_image_removes_only_matching_artifacts(
    monkeypatch,
    tmp_path,
):
    matching_result = "attack_00000000-0000-4000-8000-000000000301"
    retained_result = "attack_00000000-0000-4000-8000-000000000302"
    analyses = _Collection(
        [
            {
                "_id": "individual_analysis_00000000-0000-4000-8000-000000000401",
                "dataset_id": "dataset-1",
                "image_id": "image-1",
                "source_result_id": matching_result,
            },
            {
                "_id": "individual_analysis_00000000-0000-4000-8000-000000000402",
                "dataset_id": "dataset-1",
                "image_id": "image-2",
                "source_result_id": retained_result,
            },
        ]
    )
    for result_id in (matching_result, retained_result):
        result_path = tmp_path / result_id
        result_path.mkdir()
        (result_path / "analysis.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(history_service, "individual_analyses_collection", analyses)
    monkeypatch.setattr(history_service, "RESULTS_BASE_PATH", str(tmp_path))

    await history_service.delete_individual_analyses_for_image(
        "dataset-1",
        "image-1",
    )

    assert [document["image_id"] for document in analyses.documents] == ["image-2"]
    assert not (tmp_path / matching_result).exists()
    assert (tmp_path / retained_result).is_dir()


@pytest.mark.asyncio
async def test_dataset_image_deletion_cascades_to_saved_individual_analysis(
    monkeypatch,
    tmp_path,
):
    dataset_id = str(ObjectId())
    image_id = "image-1"
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"image")
    object_id = ObjectId(dataset_id)
    dataset = {
        "_id": object_id,
        "images": [{"id": image_id, "path": str(image_path)}],
    }
    datasets = _Collection()
    annotations = _Collection()
    cascade_calls = []

    async def fake_get_dataset_document(received_dataset_id):
        assert received_dataset_id == dataset_id
        return object_id, dataset

    async def fake_delete_individual_analyses(received_dataset_id, received_image_id):
        cascade_calls.append((received_dataset_id, received_image_id))

    monkeypatch.setattr(
        dataset_service,
        "_get_dataset_document",
        fake_get_dataset_document,
    )
    monkeypatch.setattr(dataset_service, "datasets_collection", datasets)
    monkeypatch.setattr(
        dataset_service,
        "object_annotations_collection",
        annotations,
    )
    monkeypatch.setattr(
        history_service,
        "delete_individual_analyses_for_image",
        fake_delete_individual_analyses,
    )

    await dataset_service.delete_image_from_dataset(dataset_id, image_id)

    assert not image_path.exists()
    assert cascade_calls == [(dataset_id, image_id)]
