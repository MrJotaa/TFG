from pymongo import ASCENDING, DESCENDING, IndexModel
from pymongo.errors import OperationFailure

from db.collections import (
    annotation_categories_collection,
    datasets_collection,
    detection_evaluations_collection,
    object_annotations_collection,
)


def _collections(database=None):
    if database is None:
        return (
            datasets_collection,
            annotation_categories_collection,
            object_annotations_collection,
        )
    return (
        database["datasets"],
        database["annotation_categories"],
        database["object_annotations"],
    )


async def upgrade_coco_annotations_schema(database=None) -> None:
    """
    Migración aditiva e idempotente.

    Las imágenes existentes reciben ``unreviewed`` sin reinterpretar su etiqueta
    libre. ``$mergeObjects`` conserva estados ya escritos en ejecuciones futuras.
    """

    datasets, categories, annotations = _collections(database)

    await datasets.update_many(
        {},
        [
            {
                "$set": {
                    "dataset_revision": {
                        "$ifNull": ["$dataset_revision", 0],
                    },
                    "images": {
                        "$map": {
                            "input": {"$ifNull": ["$images", []]},
                            "as": "image",
                            "in": {
                                "$mergeObjects": [
                                    {
                                        "annotation_review_status": "unreviewed",
                                        "original_file_name": "$$image.filename",
                                    },
                                    "$$image",
                                ]
                            },
                        }
                    },
                }
            }
        ],
    )

    await datasets.create_indexes(
        [
            IndexModel([("owner_id", ASCENDING)], name="datasets_owner_id"),
            IndexModel(
                [("active_annotation_snapshot_id", ASCENDING)],
                name="datasets_active_annotation_snapshot",
                sparse=True,
            ),
        ]
    )
    await categories.create_indexes(
        [
            IndexModel(
                [
                    ("dataset_id", ASCENDING),
                    ("snapshot_id", ASCENDING),
                    ("coco_category_id", ASCENDING),
                ],
                unique=True,
                name="category_external_id_per_snapshot",
            ),
            IndexModel(
                [
                    ("dataset_id", ASCENDING),
                    ("snapshot_id", ASCENDING),
                    ("name", ASCENDING),
                ],
                unique=True,
                name="category_name_per_snapshot",
            ),
        ]
    )
    await annotations.create_indexes(
        [
            IndexModel(
                [
                    ("dataset_id", ASCENDING),
                    ("snapshot_id", ASCENDING),
                    ("coco_annotation_id", ASCENDING),
                ],
                unique=True,
                name="annotation_external_id_per_snapshot",
            ),
            IndexModel(
                [
                    ("dataset_id", ASCENDING),
                    ("snapshot_id", ASCENDING),
                    ("image_id", ASCENDING),
                ],
                name="annotations_by_image",
            ),
            IndexModel(
                [
                    ("dataset_id", ASCENDING),
                    ("snapshot_id", ASCENDING),
                    ("category_id", ASCENDING),
                ],
                name="annotations_by_category",
            ),
        ]
    )


async def downgrade_coco_annotations_schema(database=None) -> None:
    """Revierte los campos/colecciones añadidos sin tocar imágenes o etiquetas."""

    datasets, categories, annotations = _collections(database)

    await categories.drop()
    await annotations.drop()
    await datasets.update_many(
        {},
        [
            {
                "$set": {
                    "images": {
                        "$map": {
                            "input": {"$ifNull": ["$images", []]},
                            "as": "image",
                            "in": {
                                "$arrayToObject": {
                                    "$filter": {
                                        "input": {"$objectToArray": "$$image"},
                                        "as": "field",
                                        "cond": {
                                            "$not": [
                                                {
                                                    "$in": [
                                                        "$$field.k",
                                                        [
                                                            "annotation_review_status",
                                                            "original_file_name",
                                                        ],
                                                    ],
                                                }
                                            ]
                                        },
                                    }
                                }
                            },
                        }
                    }
                }
            },
            {
                "$unset": [
                    "active_annotation_snapshot_id",
                    "annotation_snapshot_format",
                    "annotation_updated_at",
                    "dataset_revision",
                ]
            },
        ],
    )


async def upgrade_detection_evaluations_schema(database=None) -> None:
    """Create the indexes used by unified temporary evaluation history."""

    evaluations = (
        detection_evaluations_collection
        if database is None
        else database["detection_evaluations"]
    )
    # The previous TTL index could delete the Mongo document before the
    # application removed its filesystem artifacts. Expiry is now enforced by
    # reads and a periodic cascade cleanup, so this becomes a normal index.
    try:
        index_information = await evaluations.index_information()
        expiration_index = index_information.get("detection_evaluations_expiration")
        if expiration_index and "expireAfterSeconds" in expiration_index:
            await evaluations.drop_index("detection_evaluations_expiration")
    except (AttributeError, OperationFailure):
        # Lightweight test doubles and a first startup may not expose the index.
        pass

    if hasattr(evaluations, "update_many"):
        await evaluations.update_many(
            {
                "result_type": "detection_evaluation",
                "task_type": {"$exists": False},
            },
            {"$set": {"task_type": "detection"}},
        )

    await evaluations.create_indexes(
        [
            IndexModel(
                [
                    ("owner_id", ASCENDING),
                    ("created_at", DESCENDING),
                ],
                name="detection_evaluations_by_owner",
            ),
            IndexModel(
                [("expires_at", ASCENDING)],
                name="detection_evaluations_expiration",
                sparse=True,
            ),
            IndexModel(
                [
                    ("owner_id", ASCENDING),
                    ("signature", ASCENDING),
                    ("status", ASCENDING),
                    ("completed_at", DESCENDING),
                ],
                name="evaluations_compatibility_lookup",
                sparse=True,
            ),
            IndexModel(
                [("owner_id", ASCENDING), ("signature", ASCENDING)],
                name="one_running_evaluation_per_signature",
                unique=True,
                partialFilterExpression={
                    "status": "running",
                    "signature": {"$type": "string"},
                },
            ),
        ]
    )
