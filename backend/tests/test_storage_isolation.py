"""Regression checks for the filesystem boundaries of the pytest process."""

import os
from pathlib import Path

from services import (
    dataset_service,
    detection_evaluation_persistence as detection_persistence,
    evaluation_history_service,
)
from utils import result_files


def test_test_storage_never_points_into_application_uploads():
    application_uploads = (Path(__file__).resolve().parents[1] / "uploads").resolve()
    configured_paths = (
        dataset_service.UPLOADS_DATASETS_PATH,
        result_files.RESULTS_BASE_PATH,
        evaluation_history_service.DETECTION_EVALUATIONS_BASE_PATH,
        detection_persistence.DETECTION_EVALUATIONS_BASE_PATH,
    )

    for configured_path in configured_paths:
        path = Path(configured_path).resolve()
        assert path != application_uploads
        assert application_uploads not in path.parents
        assert path.is_dir()


def test_all_storage_modules_use_the_test_directories():
    assert dataset_service.UPLOADS_DATASETS_PATH == os.environ["UPLOADS_DATASETS_PATH"]
    assert result_files.RESULTS_BASE_PATH == os.environ["RESULTS_BASE_PATH"]
    assert (
        evaluation_history_service.DETECTION_EVALUATIONS_BASE_PATH
        == Path(os.environ["DETECTION_EVALUATIONS_BASE_PATH"])
    )
    assert (
        detection_persistence.DETECTION_EVALUATIONS_BASE_PATH
        == os.environ["DETECTION_EVALUATIONS_BASE_PATH"]
    )
    assert Path(result_files.RESULTS_BASE_PATH).parent == Path(
        dataset_service.UPLOADS_DATASETS_PATH
    ).parent
