from fastapi import FastAPI

from api.routes.predict import router as predict_router
from api.routes.models import router as models_router
from api.routes.users import router as users_router

from db.database import get_database

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "API running"}

app.include_router(predict_router)
app.include_router(models_router)
app.include_router(users_router)

@app.get("/health/db")
async def health_db():
    db = get_database()
    await db.command("ping")
    return {"status": "MongoDB connected"}