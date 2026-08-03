import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.attack import router as attack_router
from api.routes.auth import router as auth_router
from api.routes.datasets import router as datasets_router
from api.routes.detect import router as detect_router
from api.routes.detection_attack import router as detection_attack_router
from api.routes.detection_evaluate import router as detection_evaluate_router
from api.routes.evaluate import router as evaluate_router
from api.routes.evaluation_history import router as evaluation_history_router
from api.routes.explainability import router as explainability_router
from api.routes.models import router as models_router
from api.routes.predict import router as predict_router
from api.routes.results import router as results_router
from api.routes.users import router as users_router
from core.error_handlers import register_error_handlers
from db.database import get_database
from db.migrations import (
    upgrade_coco_annotations_schema,
    upgrade_detection_evaluations_schema,
)
from services.result_cleanup_service import (
    cleanup_temporary_results_once,
    run_result_cleanup_loop,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await upgrade_coco_annotations_schema()
    await upgrade_detection_evaluations_schema()
    await cleanup_temporary_results_once()
    cleanup_task = asyncio.create_task(run_result_cleanup_loop())
    try:
        yield
    finally:
        cleanup_task.cancel()
        await asyncio.gather(cleanup_task, return_exceptions=True)


app = FastAPI(lifespan=lifespan)

register_error_handlers(app)

app.add_middleware(  # Para permitir solicitudes desde el frontend en desarrollo
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "API running"}


app.include_router(predict_router, tags=["Inference"])
app.include_router(detect_router, tags=["Detection"])
app.include_router(detection_attack_router, tags=["Detection Attack"])
app.include_router(detection_evaluate_router, tags=["Detection Evaluation"])
app.include_router(models_router, tags=["Models"])
app.include_router(auth_router, tags=["Auth"])
app.include_router(users_router, tags=["Users"])
app.include_router(datasets_router, tags=["Datasets"])
app.include_router(explainability_router, tags=["Explainability"])
app.include_router(attack_router, tags=["Attack"])
app.include_router(evaluate_router, tags=["Evaluation"])
app.include_router(evaluation_history_router, tags=["Evaluation History"])
app.include_router(results_router, tags=["Temporary Results"])


@app.get("/health/db")
async def health_db():
    db = get_database()
    await db.command("ping")
    return {"status": "MongoDB connected"}
