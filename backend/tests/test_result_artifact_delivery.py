import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path


def test_temporary_result_artifact_requires_authentication(client):
    result_id = f"gradcam_{uuid.uuid4()}"
    result_path = Path("uploads/results") / result_id
    result_path.mkdir(parents=True, exist_ok=False)
    (result_path / "gradcam.png").write_bytes(b"artifact")
    (result_path / "metadata.json").write_text(
        json.dumps(
            {
                "result_id": result_id,
                "result_type": "gradcam",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        ),
        encoding="utf-8",
    )

    try:
        client.cookies.clear()
        response = client.get(f"/results/{result_id}/gradcam.png")

        assert response.status_code == 401
    finally:
        shutil.rmtree(result_path, ignore_errors=True)
