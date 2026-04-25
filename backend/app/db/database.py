import os

from dotenv import load_dotenv
from pymongo import AsyncMongoClient

load_dotenv()

MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "tfg_adversarial_ai")

client = AsyncMongoClient(MONGO_URL)
database = client[MONGO_DB_NAME]


def get_database():
    return database