from fastapi import FastAPI

from api.routes.predict import router as predict_router
from api.routes.models import router as models_router

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "API running"}

app.include_router(predict_router)
app.include_router(models_router)