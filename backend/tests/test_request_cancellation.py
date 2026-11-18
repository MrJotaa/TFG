import asyncio

import pytest

from core.errors import ClientDisconnectedError
from services.request_cancellation import run_cancellable_request


class _DisconnectedRequest:
    async def is_disconnected(self):
        return True


class _ConnectedRequest:
    async def is_disconnected(self):
        return False


@pytest.mark.asyncio
async def test_cancellable_request_returns_completed_result():
    async def operation(cancellation_token):
        cancellation_token.raise_if_cancelled()
        return "done"

    result = await run_cancellable_request(
        _ConnectedRequest(),
        operation,
        poll_interval_seconds=0.001,
    )

    assert result == "done"


@pytest.mark.asyncio
async def test_cancellable_request_propagates_disconnect_to_worker():
    cancellation_observed = asyncio.Event()

    async def operation(cancellation_token):
        while True:
            await asyncio.sleep(0.001)
            if cancellation_token.is_cancelled:
                cancellation_observed.set()
                cancellation_token.raise_if_cancelled()

    _prepared_arg_1 = _DisconnectedRequest()

    with pytest.raises(ClientDisconnectedError) as exc_info:
        await run_cancellable_request(
            _prepared_arg_1,
            operation,
            poll_interval_seconds=0.001,
        )

    assert exc_info.value.status_code == 499
    assert cancellation_observed.is_set()
