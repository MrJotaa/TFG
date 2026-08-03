from db.database import get_database

db = get_database()

users_collection = db["users"]
datasets_collection = db["datasets"]
annotation_categories_collection = db["annotation_categories"]
object_annotations_collection = db["object_annotations"]
detection_evaluations_collection = db["detection_evaluations"]
# The existing collection is kept so previously persisted detection results
# remain recoverable. New classification and detection evaluations share it.
evaluations_collection = detection_evaluations_collection
