import asyncio
from collections.abc import Callable
from threading import Lock
from typing import ParamSpec, TypeVar
from weakref import WeakKeyDictionary

P = ParamSpec("P")
T = TypeVar("T")

_loop_locks: WeakKeyDictionary = WeakKeyDictionary()
_loop_locks_guard = Lock()


def _get_detection_model_lock() -> asyncio.Lock:
    """Return the execution lock associated with the current event loop."""
    loop = asyncio.get_running_loop()
    with _loop_locks_guard:
        lock = _loop_locks.get(loop)
        if lock is None:
            lock = asyncio.Lock()
            _loop_locks[loop] = lock
    return lock


async def run_detection_model_operation(
    operation: Callable[P, T],
    /,
    *args: P.args,
    **kwargs: P.kwargs,
) -> T:
    """
    Run one operation against the shared active detector without blocking FastAPI.

    The application intentionally keeps a single active model instance. Detection
    libraries may keep mutable inference state, so concurrent thread access to the
    same instance is serialized while the synchronous work itself runs outside the
    event loop.
    """
    async with _get_detection_model_lock():
        return await asyncio.to_thread(operation, *args, **kwargs)


async def run_detection_attack_operation(
    runner,
    /,
    *args,
    **kwargs,
):
    """Run an attack, locking the detector only when the runner requires it."""
    operation = runner.apply
    if getattr(runner, "requires_exclusive_model_execution", False):
        return await run_detection_model_operation(operation, *args, **kwargs)
    return await asyncio.to_thread(operation, *args, **kwargs)
