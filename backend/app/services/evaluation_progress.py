import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Literal, TypeVar

from pydantic import BaseModel
from starlette.exceptions import HTTPException

from core.cancellation import CancellationToken, OperationCancelled
from core.errors import ApplicationError

EvaluationProgressPhase = Literal["images", "metrics"]


@dataclass(frozen=True, slots=True)
class EvaluationProgress:
    processed: int
    total: int
    phase: EvaluationProgressPhase = "images"

    def __post_init__(self) -> None:
        if self.total < 0:
            raise ValueError("total must not be negative")
        if self.processed < 0 or self.processed > self.total:
            raise ValueError("processed must be between zero and total")


EvaluationProgressCallback = Callable[[EvaluationProgress], None]
TResponse = TypeVar("TResponse", bound=BaseModel)


def report_evaluation_progress(
    callback: EvaluationProgressCallback | None,
    *,
    processed: int,
    total: int,
    phase: EvaluationProgressPhase = "images",
) -> None:
    if callback is None:
        return
    callback(EvaluationProgress(processed=processed, total=total, phase=phase))


def _json_line(payload: dict) -> bytes:
    return (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")


def _stream_error_payload(
    error: ApplicationError | HTTPException,
) -> dict:
    detail = error.detail
    if isinstance(detail, dict):
        message = str(
            detail.get("message") or detail.get("code") or "Evaluation failed"
        )
    else:
        message = str(detail)

    return {
        "type": "error",
        "status": error.status_code,
        "message": message,
        "detail": detail,
    }


async def stream_evaluation(
    runner: Callable[[EvaluationProgressCallback, CancellationToken], Awaitable[TResponse]],
) -> AsyncIterator[bytes]:
    """Stream progress events and the final Pydantic evaluation response as NDJSON."""
    queue: asyncio.Queue[dict | BaseException] = asyncio.Queue()
    cancellation_token = CancellationToken()

    def on_progress(progress: EvaluationProgress) -> None:
        queue.put_nowait(
            {
                "type": "progress",
                "processed": progress.processed,
                "total": progress.total,
                "phase": progress.phase,
            }
        )

    async def run_evaluation() -> None:
        try:
            response = await runner(on_progress, cancellation_token)
        except OperationCancelled:
            queue.put_nowait({"type": "cancelled"})
        except (ApplicationError, HTTPException) as error:
            queue.put_nowait(_stream_error_payload(error))
        else:
            queue.put_nowait(
                {
                    "type": "result",
                    "data": response.model_dump(mode="json"),
                }
            )

    def propagate_unexpected_failure(
        completed_task: asyncio.Task[None],
    ) -> None:
        if completed_task.cancelled():
            return

        error = completed_task.exception()
        if error is not None:
            queue.put_nowait(error)

    task = asyncio.create_task(run_evaluation())
    task.add_done_callback(propagate_unexpected_failure)

    try:
        while True:
            event = await queue.get()

            if isinstance(event, BaseException):
                raise event

            yield _json_line(event)
            if event["type"] in {"result", "error", "cancelled"}:
                break
    finally:
        cancellation_token.cancel()
        with suppress(asyncio.CancelledError):
            await task
