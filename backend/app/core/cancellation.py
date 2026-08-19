from __future__ import annotations

from threading import Event


class OperationCancelled(Exception):
    """Raised by cooperative long-running work after cancellation is requested."""


class CancellationToken:
    """Thread-safe cooperative cancellation token shared with worker threads."""

    def __init__(self) -> None:
        self._event = Event()

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self) -> None:
        self._event.set()

    def raise_if_cancelled(self) -> None:
        if self.is_cancelled:
            raise OperationCancelled("Operation cancelled")
