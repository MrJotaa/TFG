from fastapi import FastAPI

from api.routes.predict import router as predict_router
from api.routes.models import router as models_router
from api.routes.users import router as users_router
from api.routes.datasets import router as datasets_router

from db.database import get_database

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "API running"}

app.include_router(predict_router, tags=["Inference"])
app.include_router(models_router, tags=["Models"])
app.include_router(users_router, tags=["Users"])
app.include_router(datasets_router, tags=["Datasets"])

@app.get("/health/db")
async def health_db():
    db = get_database()
    await db.command("ping")
    return {"status": "MongoDB connected"}