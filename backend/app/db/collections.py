from db.database import get_database

db = get_database()

users_collection = db["users"]
datasets_collection = db["datasets"]
annotation_categories_collection = db["annotation_categories"]
object_annotations_collection = db["object_annotations"]
