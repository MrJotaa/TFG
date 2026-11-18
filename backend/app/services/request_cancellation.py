from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import TypeVar

from fastapi import Request

from core.cancellation import CancellationToken, OperationCancelled
from core.errors import ClientDisconnectedError

T = TypeVar("T")


async def run_cancellable_request(
    request: Request,
    operation: Callable[[CancellationToken], Awaitable[T]],
    *,
    poll_interval_seconds: float = 0.1,
) -> T:
    """Cancel cooperative work when the HTTP client disconnects."""
    cancellation_token = CancellationToken()
    task = asyncio.create_task(operation(cancellation_token))

    try:
        while not task.done():
            done, _ = await asyncio.wait({task}, timeout=poll_interval_seconds)
            if task in done:
                return task.result()

            if await request.is_disconnected():
                cancellation_token.cancel()
                try:
                    return await task
                except OperationCancelled as error:
                    raise ClientDisconnectedError() from error

        return task.result()
    finally:
        if not task.done():
            cancellation_token.cancel()
            with suppress(OperationCancelled, asyncio.CancelledError):
                await task
