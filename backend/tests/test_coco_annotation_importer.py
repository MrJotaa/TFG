import json

import pytest

from services.annotation_importers.base import DatasetImageIdentity
from services.annotation_importers.coco import (
    CocoAnnotationImporter,
    CocoImportValidationError,
)


def _dataset_image(
    image_id: str = "dataset-image-1",
    file_name: str = "nested/image.png",
    width: int = 100,
    height: int = 80,
) -> DatasetImageIdentity:
    return DatasetImageIdentity(
        image_id=image_id,
        original_file_name=file_name,
        stored_path=f"/trusted/{image_id}.png",
        width=width,
        height=height,
    )


def _valid_coco() -> dict:
    return {
        "images": [
            {
                "id": 10,
                "file_name": "nested/image.png",
                "width": 100,
                "height": 80,
            }
        ],
        "categories": [
            {
                "id": 3,
                "name": "vehicle",
                "supercategory": "transport",
            }
        ],
        "annotations": [
            {
                "id": 21,
                "image_id": 10,
                "category_id": 3,
                "bbox": [1.123456789, 2.25, 10.333333333, 20.75],
                "area": 214.41666665975,
                "iscrowd": 1,
            }
        ],
    }


def _parse(payload: dict, images=None):
    return CocoAnnotationImporter().parse(
        json.dumps(payload).encode(),
        tuple(images or [_dataset_image()]),
    )


def test_coco_importer_creates_canonical_snapshot_with_full_precision():
    snapshot = _parse(_valid_coco())

    assert len(snapshot.categories) == 1
    assert snapshot.categories[0].external_id == 3
    assert snapshot.categories[0].name == "vehicle"
    assert snapshot.categories[0].supercategory == "transport"

    assert len(snapshot.annotations) == 1
    annotation = snapshot.annotations[0]
    assert annotation.external_id == 21
    assert annotation.image_id == "dataset-image-1"
    assert annotation.category_internal_id == snapshot.categories[0].internal_id
    assert annotation.bbox_x1 == 1.123456789
    assert annotation.bbox_y1 == 2.25
    assert annotation.bbox_x2 == pytest.approx(11.456790122)
    assert annotation.bbox_y2 == 23.0
    assert annotation.area == 214.41666665975
    assert annotation.iscrowd == 1


def test_coco_importer_marks_images_present_without_objects_as_reviewed():
    payload = _valid_coco()
    payload["annotations"] = []

    snapshot = _parse(payload)

    assert snapshot.reviewed_image_ids == {"dataset-image-1"}
    assert snapshot.annotations == ()


def test_coco_importer_uses_unique_basename_fallback():
    payload = _valid_coco()
    payload["images"][0]["file_name"] = "different/tree/image.png"

    snapshot = _parse(payload)

    assert snapshot.annotations[0].image_id == "dataset-image-1"


@pytest.mark.parametrize(
    ("mutate", "expected_section"),
    [
        (lambda payload: payload.update(images="invalid"), "images"),
        (
            lambda payload: payload["categories"].append({**payload["categories"][0]}),
            "categories",
        ),
        (
            lambda payload: payload["images"].append({**payload["images"][0]}),
            "images",
        ),
        (
            lambda payload: payload["annotations"].append(
                {**payload["annotations"][0]}
            ),
            "annotations",
        ),
        (
            lambda payload: payload["annotations"][0].update(image_id=999),
            "annotations",
        ),
        (
            lambda payload: payload["annotations"][0].update(category_id=999),
            "annotations",
        ),
        (
            lambda payload: payload["images"][0].update(width=0),
            "images",
        ),
        (
            lambda payload: payload["images"][0].update(width=99),
            "images",
        ),
        (
            lambda payload: payload["annotations"][0].update(bbox=[1, 2, 0, 4]),
            "annotations",
        ),
        (
            lambda payload: payload["annotations"][0].update(
                bbox=[1, 2, float("nan"), 4]
            ),
            "annotations",
        ),
        (
            lambda payload: payload["annotations"][0].update(bbox=[95, 2, 10, 4]),
            "annotations",
        ),
        (
            lambda payload: payload["annotations"][0].update(area=-1),
            "annotations",
        ),
        (
            lambda payload: payload["annotations"][0].update(iscrowd=2),
            "annotations",
        ),
        (
            lambda payload: payload["images"][0].update(file_name="../../etc/passwd"),
            "images",
        ),
    ],
)
def test_coco_importer_rejects_invalid_payloads(mutate, expected_section):
    payload = _valid_coco()
    mutate(payload)

    with pytest.raises(CocoImportValidationError) as error:
        _parse(payload)

    assert error.value.detail["section"] == expected_section
    assert error.value.detail["snapshot_applied"] is False


def test_coco_importer_rejects_malformed_json():
    with pytest.raises(CocoImportValidationError) as error:
        CocoAnnotationImporter().parse(
            b"{not-json",
            (_dataset_image(),),
        )

    assert error.value.detail["section"] == "root"
    assert error.value.detail["snapshot_applied"] is False


def test_coco_importer_rejects_unmatched_dataset_image():
    payload = _valid_coco()
    payload["images"][0]["file_name"] = "missing.png"

    with pytest.raises(CocoImportValidationError, match="could not be matched"):
        _parse(payload)


def test_coco_importer_rejects_ambiguous_basename():
    payload = _valid_coco()
    payload["images"][0]["file_name"] = "image.png"
    images = [
        _dataset_image("one", "a/image.png"),
        _dataset_image("two", "b/image.png"),
    ]

    with pytest.raises(CocoImportValidationError, match="ambiguous"):
        _parse(payload, images)


def test_coco_importer_rejects_same_dataset_image_associated_twice():
    payload = _valid_coco()
    payload["images"].append(
        {
            "id": 11,
            "file_name": "image.png",
            "width": 100,
            "height": 80,
        }
    )

    with pytest.raises(CocoImportValidationError, match="more than once"):
        _parse(payload)
