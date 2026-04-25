from db.database import get_database

db = get_database()

users_collection = db["users"]
datasets_collection = db["datasets"]