from threading import Lock

from core.cancellation import CancellationToken

_tokens: dict[str, CancellationToken] = {}
_lock = Lock()


def register_evaluation_cancellation(
    evaluation_id: str,
    cancellation_token: CancellationToken,
) -> None:
    with _lock:
        _tokens[evaluation_id] = cancellation_token


def unregister_evaluation_cancellation(
    evaluation_id: str,
    cancellation_token: CancellationToken,
) -> None:
    with _lock:
        if _tokens.get(evaluation_id) is cancellation_token:
            _tokens.pop(evaluation_id, None)


def request_evaluation_cancellation(evaluation_id: str) -> bool:
    with _lock:
        cancellation_token = _tokens.get(evaluation_id)

    if cancellation_token is None:
        return False

    cancellation_token.cancel()
    return True
