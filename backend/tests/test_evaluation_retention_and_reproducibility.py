import os
from datetime import datetime, timedelta, timezone

import pytest

from schemas.attack import AttackParameters, AttackType
from schemas.users import UserRetentionUpdate
from utils.reproducibility import attack_parameters_for_image
from utils.result_files import cleanup_old_results, read_result_expiration


@pytest.mark.parametrize("seconds", [3600, 7200, 86400, 172800, 604800])
def test_supported_result_retention_values(seconds):
    assert UserRetentionUpdate(
        result_retention_seconds=seconds
    ).result_retention_seconds == seconds


@pytest.mark.parametrize("seconds", [0, 60, 90000, 604801])
def test_unsupported_result_retention_values_are_rejected(seconds):
    with pytest.raises(ValueError):
        UserRetentionUpdate(result_retention_seconds=seconds)


def test_random_noise_seed_is_stable_per_image_and_changes_between_images():
    parameters = AttackParameters(seed=1234)
    first = attack_parameters_for_image(
        attack_type=AttackType.RANDOM_NOISE,
        parameters=parameters,
        image_id="image-a",
    )
    repeated = attack_parameters_for_image(
        attack_type=AttackType.RANDOM_NOISE,
        parameters=parameters,
        image_id="image-a",
    )
    second = attack_parameters_for_image(
        attack_type=AttackType.RANDOM_NOISE,
        parameters=parameters,
        image_id="image-b",
    )
    assert first.seed == repeated.seed
    assert first.seed != second.seed
    assert parameters.seed == 1234


def test_cleanup_respects_explicit_expiration_and_protected_ids(
    tmp_path, monkeypatch
):
    from utils import result_files

    monkeypatch.setattr(result_files, "RESULTS_BASE_PATH", str(tmp_path))
    expired = tmp_path / "expired"
    protected = tmp_path / "protected"
    active = tmp_path / "active"
    for path in (expired, protected, active):
        path.mkdir()

    now = datetime.now(timezone.utc)
    (expired / "metadata.json").write_text(
        '{"expires_at":"%s"}'
        % (now - timedelta(seconds=1)).isoformat(),
        encoding="utf-8",
    )
    (protected / "metadata.json").write_text(
        '{"expires_at":"%s"}'
        % (now - timedelta(seconds=1)).isoformat(),
        encoding="utf-8",
    )
    (active / "metadata.json").write_text(
        '{"expires_at":"%s"}'
        % (now + timedelta(days=7)).isoformat(),
        encoding="utf-8",
    )

    cleanup_old_results(protected_result_ids={"protected"})

    assert not expired.exists()
    assert protected.exists()
    assert active.exists()
    assert read_result_expiration(str(active)) > now.timestamp()
    assert os.path.isdir(active)
