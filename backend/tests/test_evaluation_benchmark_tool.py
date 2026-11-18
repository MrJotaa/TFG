import json
from types import SimpleNamespace

import pytest

import tools.benchmark_evaluations as benchmark_tool
from tools.benchmark_evaluations import BenchmarkRun, _comparison, _summary


def _run(seconds: float, throughput: float, response_bytes: int) -> BenchmarkRun:
    return BenchmarkRun(
        kind="measured",
        index=1,
        result_id="evaluation-1",
        total_images=10,
        elapsed_seconds=seconds,
        throughput_images_per_second=throughput,
        response_bytes=response_bytes,
        progress_events=12,
        time_to_first_progress_seconds=0.1,
        time_to_metrics_phase_seconds=seconds - 0.1,
        resolved_parameters={"epsilon": 0.03},
        effective_supervised_configuration=None,
    )


def _configuration() -> dict:
    return {
        "task": "classification",
        "dataset_id": "dataset-1",
        "model": {"id": "resnet18", "loaded": True},
        "attack": "fgsm",
        "parameters": {"epsilon": 0.03},
        "supervised_configuration": None,
        "force_recompute": True,
        "warmup_runs": 1,
        "measured_runs": 3,
    }


def test_benchmark_summary_uses_only_measured_run_values():
    summary = _summary(
        [
            _run(2.0, 5.0, 1000),
            _run(4.0, 2.5, 1200),
            _run(3.0, 10 / 3, 1100),
        ]
    )

    assert summary["runs"] == 3
    assert summary["mean_seconds"] == 3.0
    assert summary["median_seconds"] == 3.0
    assert summary["min_seconds"] == 2.0
    assert summary["max_seconds"] == 4.0
    assert summary["mean_response_bytes"] == 1100.0


def test_benchmark_comparison_rejects_different_scenarios(tmp_path):
    baseline_path = tmp_path / "baseline.json"
    baseline_configuration = _configuration()
    baseline_configuration["dataset_id"] = "dataset-other"
    baseline_path.write_text(
        json.dumps(
            {
                "configuration": baseline_configuration,
                "summary": {"mean_seconds": 10.0},
            }
        ),
        encoding="utf-8",
    )

    _prepared_arg_2 = _configuration()

    with pytest.raises(RuntimeError, match="dataset_id"):
        _comparison(
            {"mean_seconds": 8.0},
            _prepared_arg_2,
            baseline_path,
        )


def test_benchmark_comparison_reports_percent_change(tmp_path):
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps(
            {
                "configuration": _configuration(),
                "summary": {
                    "mean_seconds": 10.0,
                    "median_seconds": 10.0,
                    "mean_throughput_images_per_second": 1.0,
                    "mean_response_bytes": 1000.0,
                },
            }
        ),
        encoding="utf-8",
    )

    comparison = _comparison(
        {
            "mean_seconds": 8.0,
            "median_seconds": 8.0,
            "mean_throughput_images_per_second": 1.25,
            "mean_response_bytes": 1000.0,
        },
        _configuration(),
        baseline_path,
    )

    metrics = comparison["metrics"]
    assert metrics["mean_seconds"]["percent_change"] == -20.0
    assert metrics["mean_throughput_images_per_second"]["percent_change"] == 25.0
    assert metrics["mean_response_bytes"]["percent_change"] == 0.0


def test_benchmark_payload_always_forces_real_recomputation():
    args = SimpleNamespace(
        task="classification",
        dataset_id="dataset-1",
        attack="fgsm",
        parameters={"epsilon": 0.03},
    )

    assert benchmark_tool._evaluation_payload(args) == {
        "dataset_id": "dataset-1",
        "attack_type": "fgsm",
        "parameters": {"epsilon": 0.03},
        "force_recompute": True,
    }


def test_benchmark_run_parses_stream_result_and_resolved_parameters(monkeypatch):
    events = [
        {"type": "progress", "processed": 0, "total": 2, "phase": "images"},
        {"type": "progress", "processed": 2, "total": 2, "phase": "metrics"},
        {
            "type": "result",
            "data": {
                "result_id": "evaluation-1",
                "parameters": {"epsilon": 0.03},
                "summary": {"total_images": 2},
            },
        },
    ]
    lines = [
        (json.dumps(event, separators=(",", ":")) + "\n").encode("utf-8")
        for event in events
    ]

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def __iter__(self):
            return iter(lines)

    captured_request = None

    def fake_urlopen(request, timeout):
        nonlocal captured_request
        captured_request = request
        assert timeout == 30.0
        return FakeResponse()

    monkeypatch.setattr(benchmark_tool, "urlopen", fake_urlopen)
    args = SimpleNamespace(
        task="classification",
        base_url="http://127.0.0.1:8000",
        timeout=30.0,
    )

    run = benchmark_tool._run_once(
        args=args,
        token="token-1",
        payload={"force_recompute": True},
        kind="measured",
        index=1,
    )

    assert captured_request is not None
    assert captured_request.full_url.endswith("/evaluate/stream")
    assert captured_request.get_header("Authorization") == "Bearer token-1"
    assert run.result_id == "evaluation-1"
    assert run.total_images == 2
    assert run.progress_events == 2
    assert run.resolved_parameters == {"epsilon": 0.03}
    assert run.effective_supervised_configuration is None
