import json
import logging

import torch

import services.evaluation_profiling as evaluation_profiling


def test_evaluation_profile_accumulates_stages_and_builds_stable_payload(monkeypatch):
    profile = evaluation_profiling.EvaluationProfile(
        task_type="classification",
        dataset_id="dataset-1",
        attack_type="fgsm",
        started_at=10.0,
    )
    profile.add("attack", 0.4)
    profile.add("attack", 0.6)
    profile.add("dataset_load", 0.25)
    monkeypatch.setattr(evaluation_profiling, "perf_counter", lambda: 12.0)

    payload = profile.payload(
        model_id="model-1",
        model_name="Model",
        total_images=4,
        device=torch.device("cpu"),
        extra={"attack_steps": 3},
    )

    assert payload == {
        "profile_schema_version": 1,
        "task_type": "classification",
        "dataset_id": "dataset-1",
        "model_id": "model-1",
        "model_name": "Model",
        "attack_type": "fgsm",
        "images": 4,
        "device": "cpu",
        "throughput_images_per_second": 2.0,
        "total_seconds": 2.0,
        "stages_seconds": {
            "attack": 1.0,
            "dataset_load": 0.25,
        },
        "attack_steps": 3,
    }


def test_evaluation_profile_rejects_negative_stage_duration():
    profile = evaluation_profiling.EvaluationProfile(
        task_type="detection",
        dataset_id="dataset-1",
        attack_type="random_noise",
    )

    try:
        profile.add("attack", -0.01)
    except ValueError as error:
        assert str(error) == "profile stage duration must not be negative"
    else:
        raise AssertionError("negative profile duration must be rejected")


def test_get_model_device_reads_exposed_torch_module():
    class Model:
        def get_torch_model(self):
            return torch.nn.Linear(2, 1)

    assert evaluation_profiling.get_model_device(Model()) == torch.device("cpu")




def test_get_model_device_never_breaks_evaluation_for_diagnostic_failures():
    class Model:
        def get_torch_model(self):
            raise RuntimeError("diagnostic access unavailable")

    assert evaluation_profiling.get_model_device(Model()) is None


def test_log_evaluation_profile_emits_machine_readable_json(caplog):
    logger = logging.getLogger("test_evaluation_profile")
    caplog.set_level(logging.INFO, logger=logger.name)

    evaluation_profiling.log_evaluation_profile(
        logger,
        {"profile_schema_version": 1, "task_type": "detection"},
    )

    message = caplog.records[-1].getMessage()
    prefix = "evaluation_profile "
    assert message.startswith(prefix)
    assert json.loads(message[len(prefix) :]) == {
        "profile_schema_version": 1,
        "task_type": "detection",
    }
