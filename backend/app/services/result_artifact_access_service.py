from bson import ObjectId

from db.collections import datasets_collection
from utils.result_files import read_result_metadata


async def temporary_result_belongs_to_user(result_id: str, owner_id: str) -> bool:
    try:
        metadata = read_result_metadata(result_id)
    except (OSError, TypeError, ValueError):
        return False

    dataset_id = metadata.get("dataset_id")
    if not isinstance(dataset_id, str) or not ObjectId.is_valid(dataset_id):
        return False

    dataset = await datasets_collection.find_one(
        {"_id": ObjectId(dataset_id)},
        {"owner_id": 1},
    )
    return dataset is not None and str(dataset.get("owner_id")) == owner_id
