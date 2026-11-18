import asyncio
import json

import pytest
from pydantic import BaseModel

from core.cancellation import OperationCancelled
from core.errors import BadRequestError
from services.evaluation_progress import EvaluationProgress, stream_evaluation


class _Response(BaseModel):
    result_id: str


def _decode(events):
    return [json.loads(event.decode("utf-8")) for event in events]


def test_evaluation_progress_validates_bounds():
    with pytest.raises(ValueError, match="between zero and total"):
        EvaluationProgress(processed=2, total=1)


@pytest.mark.asyncio
async def test_stream_evaluation_emits_progress_then_result():
    async def runner(report, cancellation_token):
        report(EvaluationProgress(processed=0, total=2))
        report(EvaluationProgress(processed=1, total=2))
        report(EvaluationProgress(processed=2, total=2, phase="metrics"))
        return _Response(result_id="evaluation-1")

    events = _decode([event async for event in stream_evaluation(runner)])

    assert events == [
        {
            "type": "progress",
            "processed": 0,
            "total": 2,
            "phase": "images",
        },
        {
            "type": "progress",
            "processed": 1,
            "total": 2,
            "phase": "images",
        },
        {
            "type": "progress",
            "processed": 2,
            "total": 2,
            "phase": "metrics",
        },
        {"type": "result", "data": {"result_id": "evaluation-1"}},
    ]


@pytest.mark.asyncio
async def test_stream_evaluation_serializes_controlled_errors():
    async def runner(report, cancellation_token):
        report(EvaluationProgress(processed=0, total=1))
        raise BadRequestError(
            {"code": "invalid_evaluation", "message": "Evaluation is invalid"}
        )

    events = _decode([event async for event in stream_evaluation(runner)])

    assert events[0]["type"] == "progress"
    assert events[1] == {
        "type": "error",
        "status": 400,
        "message": "Evaluation is invalid",
        "detail": {
            "code": "invalid_evaluation",
            "message": "Evaluation is invalid",
        },
    }


@pytest.mark.asyncio
async def test_stream_evaluation_serializes_cooperative_cancellation():
    async def runner(report, cancellation_token):
        cancellation_token.cancel()
        raise OperationCancelled("Operation cancelled")

    events = _decode([event async for event in stream_evaluation(runner)])

    assert events == [{"type": "cancelled"}]


@pytest.mark.asyncio
async def test_stream_evaluation_cancels_worker_when_consumer_closes_stream():
    worker_stopped = asyncio.Event()

    async def runner(report, cancellation_token):
        report(EvaluationProgress(processed=0, total=1))
        try:
            while True:
                await asyncio.sleep(0.001)
                cancellation_token.raise_if_cancelled()
        finally:
            worker_stopped.set()

    stream = stream_evaluation(runner)
    first_event = _decode([await anext(stream)])
    assert first_event == [
        {"type": "progress", "processed": 0, "total": 1, "phase": "images"}
    ]

    await stream.aclose()

    assert worker_stopped.is_set()
