import asyncio
import threading
import time

import pytest

from ml.attacks.detection_confidence import DetectionConfidenceAttack
from ml.attacks.random_noise import RandomNoiseAttack
from services.detection_model_execution import (
    run_detection_attack_operation,
    run_detection_model_operation,
)


@pytest.mark.asyncio
async def test_detection_model_operations_are_serialized_per_event_loop():
    state_lock = threading.Lock()
    active_operations = 0
    max_active_operations = 0

    def blocking_operation(value: int) -> int:
        nonlocal active_operations, max_active_operations
        with state_lock:
            active_operations += 1
            max_active_operations = max(max_active_operations, active_operations)
        time.sleep(0.02)
        with state_lock:
            active_operations -= 1
        return value

    results = await asyncio.gather(
        run_detection_model_operation(blocking_operation, 1),
        run_detection_model_operation(blocking_operation, 2),
    )

    assert results == [1, 2]
    assert max_active_operations == 1


@pytest.mark.asyncio
async def test_detection_model_operation_does_not_block_event_loop():
    started = threading.Event()
    release = threading.Event()

    def blocking_operation() -> str:
        started.set()
        release.wait(timeout=1.0)
        return "done"

    task = asyncio.create_task(run_detection_model_operation(blocking_operation))
    for _ in range(100):
        if started.is_set():
            break
        await asyncio.sleep(0.001)

    assert started.is_set()

    heartbeat = False

    async def mark_heartbeat() -> None:
        nonlocal heartbeat
        await asyncio.sleep(0)
        heartbeat = True

    await mark_heartbeat()
    assert heartbeat is True

    release.set()
    assert await task == "done"


class _Runner:
    def __init__(self, *, exclusive: bool):
        self.requires_exclusive_model_execution = exclusive
        self.thread_ids = []

    def apply(self, *, value: int) -> int:
        self.thread_ids.append(threading.get_ident())
        return value


@pytest.mark.asyncio
async def test_detection_attack_operation_supports_opt_in_exclusive_access():
    exclusive_runner = _Runner(exclusive=True)
    nonexclusive_runner = _Runner(exclusive=False)

    assert await run_detection_attack_operation(exclusive_runner, value=1) == 1
    assert await run_detection_attack_operation(nonexclusive_runner, value=2) == 2

    main_thread_id = threading.get_ident()
    assert exclusive_runner.thread_ids[0] != main_thread_id
    assert nonexclusive_runner.thread_ids[0] != main_thread_id


def test_only_model_dependent_detection_attack_requires_exclusive_access():
    assert DetectionConfidenceAttack.requires_exclusive_model_execution is True
    assert RandomNoiseAttack.requires_exclusive_model_execution is False
