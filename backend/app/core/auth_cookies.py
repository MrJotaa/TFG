import os

ARTIFACT_SESSION_COOKIE_NAME = "artifact_session"
ARTIFACT_SESSION_COOKIE_PATH = "/results"


def artifact_cookie_secure() -> bool:
    return os.getenv("ARTIFACT_COOKIE_SECURE", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
