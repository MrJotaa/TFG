from fastapi import FastAPI

from api.routes.predict import router as predict_router

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "API running"}

app.include_router(predict_router)