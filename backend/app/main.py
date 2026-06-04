from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.routes.predict import router as predict_router
from api.routes.models import router as models_router
from api.routes.users import router as users_router
from api.routes.datasets import router as datasets_router
from api.routes.explainability import router as explainability_router
from api.routes.attack import router as attack_router
from api.routes.evaluate import router as evaluate_router
from api.routes.auth import router as auth_router

from db.database import get_database

app = FastAPI()

app.add_middleware( # Para permitir solicitudes desde el frontend en desarrollo
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/results", StaticFiles(directory="uploads/results"), name="results")

@app.get("/")
async def root():
    return {"message": "API running"}

app.include_router(predict_router, tags=["Inference"])
app.include_router(models_router, tags=["Models"])
app.include_router(auth_router, tags=["Auth"])
app.include_router(users_router, tags=["Users"])
app.include_router(datasets_router, tags=["Datasets"])
app.include_router(explainability_router, tags=["Explainability"])
app.include_router(attack_router, tags=["Attack"])
app.include_router(evaluate_router, tags=["Evaluation"])

@app.get("/health/db")
async def health_db():
    db = get_database()
    await db.command("ping")
    return {"status": "MongoDB connected"}
