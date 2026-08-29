from fastapi import APIRouter, Request

from schemas.detection import DetectionRequest, DetectionResponse
from services.detection_service import detect_objects
from services.request_cancellation import run_cancellable_request

router = APIRouter()


@router.post("/detect")
async def detect_image(
    request: DetectionRequest,
    http_request: Request,
) -> DetectionResponse:
    return await run_cancellable_request(
        http_request,
        lambda cancellation_token: detect_objects(
            request,
            cancellation_token=cancellation_token,
        ),
    )
