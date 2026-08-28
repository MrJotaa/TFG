#!/usr/bin/env python3
"""Run reproducible end-to-end benchmarks against the evaluation stream API."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

_APPLICATION_JSON = "application/json"


@dataclass(frozen=True, slots=True)
class BenchmarkRun:
    kind: str
    index: int
    result_id: str
    total_images: int
    elapsed_seconds: float
    throughput_images_per_second: float
    response_bytes: int
    progress_events: int
    time_to_first_progress_seconds: float | None
    time_to_metrics_phase_seconds: float | None
    resolved_parameters: dict
    effective_supervised_configuration: dict | None


def _repository_metadata() -> dict:
    repository_root = Path(__file__).resolve().parents[3]
    try:
        commit = subprocess.run(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(repository_root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit or None, "dirty": bool(status.strip())}


def _json_argument(value: str, argument_name: str) -> dict:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(
            f"{argument_name} must contain valid JSON: {error}"
        ) from error
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError(f"{argument_name} must be a JSON object")
    return parsed


def _request_json(
    *,
    base_url: str,
    method: str,
    path: str,
    payload: dict | None = None,
    token: str | None = None,
    timeout: float,
) -> dict:
    body = None
    headers = {"Accept": _APPLICATION_JSON}
    if payload is not None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        headers["Content-Type"] = _APPLICATION_JSON
    if token:
        headers["Authorization"] = f"Bearer {token}"

    request = Request(
        f"{base_url.rstrip('/')}{path}",
        data=body,
        headers=headers,
        method=method,
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"HTTP {error.code} for {method} {path}: {detail}"
        ) from error
    except URLError as error:
        raise RuntimeError(f"Could not reach {base_url}: {error.reason}") from error


def _resolve_token(args: argparse.Namespace) -> str:
    token = os.getenv(args.token_env)
    if token:
        return token

    if not args.email:
        raise RuntimeError(
            f"Set {args.token_env} with a bearer token or provide --email and "
            f"{args.password_env}."
        )

    password = os.getenv(args.password_env)
    if not password:
        raise RuntimeError(
            f"--email requires the password in environment variable "
            f"{args.password_env}."
        )

    response = _request_json(
        base_url=args.base_url,
        method="POST",
        path="/auth/login",
        payload={"email": args.email, "password": password},
        timeout=args.timeout,
    )
    token = response.get("access_token")
    if not isinstance(token, str) or not token:
        raise RuntimeError("Login response did not contain an access token")
    return token


def _select_model(args: argparse.Namespace) -> dict:
    if args.model_id:
        _request_json(
            base_url=args.base_url,
            method="POST",
            path=f"/models/{quote(args.model_id, safe='')}/select",
            timeout=args.timeout,
        )
    return _request_json(
        base_url=args.base_url,
        method="GET",
        path="/models/active",
        timeout=args.timeout,
    )


def _evaluation_payload(args: argparse.Namespace) -> dict:
    payload = {
        "dataset_id": args.dataset_id,
        "attack_type": args.attack,
        "parameters": args.parameters,
        # A performance run must exercise the pipeline instead of reusing history.
        "force_recompute": True,
    }
    if args.task == "detection":
        payload["supervised_configuration"] = args.supervised_configuration
    return payload


def _run_once(
    *,
    args: argparse.Namespace,
    token: str,
    payload: dict,
    kind: str,
    index: int,
) -> BenchmarkRun:
    endpoint = (
        "/evaluate/stream"
        if args.task == "classification"
        else "/detect/evaluate/stream"
    )
    request = Request(
        f"{args.base_url.rstrip('/')}{endpoint}",
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={
            "Accept": "application/x-ndjson",
            "Authorization": f"Bearer {token}",
            "Content-Type": _APPLICATION_JSON,
        },
        method="POST",
    )
    (
        terminal_event,
        elapsed_seconds,
        response_bytes,
        progress_events,
        first_progress_seconds,
        metrics_phase_seconds,
    ) = _read_evaluation_stream(
        request=request,
        endpoint=endpoint,
        timeout=args.timeout,
    )
    data = _validated_result_payload(terminal_event)
    total_images = data["summary"]["total_images"]
    throughput = total_images / elapsed_seconds if elapsed_seconds > 0 else 0.0

    return BenchmarkRun(
        kind=kind,
        index=index,
        result_id=data["result_id"],
        total_images=total_images,
        elapsed_seconds=round(elapsed_seconds, 6),
        throughput_images_per_second=round(throughput, 6),
        response_bytes=response_bytes,
        progress_events=progress_events,
        time_to_first_progress_seconds=_round_optional(first_progress_seconds),
        time_to_metrics_phase_seconds=_round_optional(metrics_phase_seconds),
        resolved_parameters=data["parameters"],
        effective_supervised_configuration=(
            _effective_supervised_configuration(data)
            if args.task == "detection"
            else None
        ),
    )


def _read_evaluation_stream(
    *,
    request: Request,
    endpoint: str,
    timeout: float,
) -> tuple[dict | None, float, int, int, float | None, float | None]:
    started_at = perf_counter()
    response_bytes = 0
    progress_events = 0
    first_progress_seconds = None
    metrics_phase_seconds = None
    terminal_event = None

    try:
        with urlopen(request, timeout=timeout) as response:
            for raw_line in response:
                response_bytes += len(raw_line)
                if not raw_line.strip():
                    continue
                event = json.loads(raw_line)
                event_type = event.get("type")
                elapsed = perf_counter() - started_at
                if event_type == "progress":
                    (
                        progress_events,
                        first_progress_seconds,
                        metrics_phase_seconds,
                    ) = _record_progress_event(
                        event=event,
                        elapsed=elapsed,
                        progress_events=progress_events,
                        first_progress_seconds=first_progress_seconds,
                        metrics_phase_seconds=metrics_phase_seconds,
                    )
                elif event_type in {"result", "error", "cancelled"}:
                    terminal_event = event
                    break
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"HTTP {error.code} while benchmarking {endpoint}: {detail}"
        ) from error
    except URLError as error:
        raise RuntimeError(
            f"Connection failed while benchmarking {endpoint}: {error.reason}"
        ) from error

    return (
        terminal_event,
        perf_counter() - started_at,
        response_bytes,
        progress_events,
        first_progress_seconds,
        metrics_phase_seconds,
    )


def _record_progress_event(
    *,
    event: dict,
    elapsed: float,
    progress_events: int,
    first_progress_seconds: float | None,
    metrics_phase_seconds: float | None,
) -> tuple[int, float | None, float | None]:
    progress_events += 1
    if first_progress_seconds is None:
        first_progress_seconds = elapsed
    if event.get("phase") == "metrics" and metrics_phase_seconds is None:
        metrics_phase_seconds = elapsed
    return progress_events, first_progress_seconds, metrics_phase_seconds


def _validated_result_payload(terminal_event: dict | None) -> dict:
    if terminal_event is None:
        raise RuntimeError("Evaluation stream ended without a terminal event")
    if terminal_event.get("type") != "result":
        raise RuntimeError(
            f"Evaluation did not complete successfully: {terminal_event}"
        )

    data = terminal_event.get("data")
    if not isinstance(data, dict):
        raise RuntimeError("Evaluation result event did not contain an object payload")
    summary = data.get("summary")
    if not isinstance(summary, dict):
        raise RuntimeError("Evaluation result did not contain a summary")
    if not isinstance(summary.get("total_images"), int):
        raise RuntimeError("Evaluation summary did not contain total_images")
    if not isinstance(data.get("result_id"), str):
        raise RuntimeError("Evaluation result did not contain result_id")
    if not isinstance(data.get("parameters"), dict):
        raise RuntimeError("Evaluation result did not contain resolved parameters")
    return data


def _effective_supervised_configuration(data: dict) -> dict:
    supervised = data.get("supervised")
    if not isinstance(supervised, dict):
        raise RuntimeError("Detection result did not contain supervised metrics")
    configuration = supervised.get("configuration")
    if not isinstance(configuration, dict):
        raise RuntimeError(
            "Detection result did not contain effective supervised configuration"
        )
    return configuration


def _round_optional(value: float | None) -> float | None:
    return round(value, 6) if value is not None else None


def _summary(runs: list[BenchmarkRun]) -> dict:
    elapsed = [run.elapsed_seconds for run in runs]
    throughput = [run.throughput_images_per_second for run in runs]
    response_bytes = [run.response_bytes for run in runs]
    if not elapsed:
        raise ValueError("at least one measured run is required")

    return {
        "runs": len(runs),
        "mean_seconds": round(statistics.fmean(elapsed), 6),
        "median_seconds": round(statistics.median(elapsed), 6),
        "min_seconds": round(min(elapsed), 6),
        "max_seconds": round(max(elapsed), 6),
        "stdev_seconds": round(statistics.stdev(elapsed), 6)
        if len(elapsed) > 1
        else 0.0,
        "mean_throughput_images_per_second": round(statistics.fmean(throughput), 6),
        "median_throughput_images_per_second": round(statistics.median(throughput), 6),
        "mean_response_bytes": round(statistics.fmean(response_bytes), 2),
    }


def _percent_change(current: float, baseline: float) -> float | None:
    if baseline == 0:
        return None
    return round((current - baseline) / baseline * 100.0, 3)


def _comparison(
    current_summary: dict,
    current_configuration: dict,
    baseline_path: Path,
) -> dict:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_summary = baseline.get("summary")
    if not isinstance(baseline_summary, dict):
        raise RuntimeError(f"{baseline_path} does not contain a benchmark summary")
    baseline_configuration = baseline.get("configuration")
    if not isinstance(baseline_configuration, dict):
        raise RuntimeError(f"{baseline_path} does not contain benchmark configuration")

    comparable_keys = (
        "task",
        "dataset_id",
        "attack",
        "parameters",
        "supervised_configuration",
    )
    mismatches = [
        key
        for key in comparable_keys
        if baseline_configuration.get(key) != current_configuration.get(key)
    ]
    baseline_model = baseline_configuration.get("model") or {}
    current_model = current_configuration.get("model") or {}
    if baseline_model.get("id") != current_model.get("id"):
        mismatches.append("model.id")
    if mismatches:
        raise RuntimeError(
            "Benchmark comparison configuration differs for: " + ", ".join(mismatches)
        )

    keys = (
        "mean_seconds",
        "median_seconds",
        "mean_throughput_images_per_second",
        "mean_response_bytes",
    )
    changes = {}
    for key in keys:
        current_value = current_summary.get(key)
        baseline_value = baseline_summary.get(key)
        if not isinstance(current_value, (int, float)) or not isinstance(
            baseline_value, (int, float)
        ):
            continue
        changes[key] = {
            "baseline": baseline_value,
            "current": current_value,
            "percent_change": _percent_change(current_value, baseline_value),
        }
    return {"baseline_file": str(baseline_path), "metrics": changes}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark a classification or detection dataset evaluation using "
            "warm-up runs followed by measured runs."
        )
    )
    parser.add_argument(
        "--base-url", default="http://127.0.0.1:8000", help="Backend base URL"
    )
    parser.add_argument(
        "--task", choices=("classification", "detection"), required=True
    )
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--model-id", help="Select this model before benchmarking")
    parser.add_argument("--attack", required=True, help="AttackType API value")
    parser.add_argument(
        "--parameters",
        default="{}",
        help="Attack parameters as a JSON object",
    )
    parser.add_argument(
        "--supervised-configuration",
        default="{}",
        help="Detection supervised configuration as a JSON object",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--label", default="benchmark")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--compare-to",
        type=Path,
        help="Optional previous benchmark JSON to compare with this run",
    )
    parser.add_argument(
        "--token-env",
        default="TFG_BENCHMARK_TOKEN",
        help="Environment variable containing a bearer token",
    )
    parser.add_argument(
        "--email",
        help="Login email used when the token environment variable is absent",
    )
    parser.add_argument(
        "--password-env",
        default="TFG_BENCHMARK_PASSWORD",
        help="Environment variable containing the login password",
    )
    args = parser.parse_args()

    if args.warmup < 0:
        parser.error("--warmup must be zero or greater")
    if args.runs < 1:
        parser.error("--runs must be at least one")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    try:
        args.parameters = _json_argument(args.parameters, "--parameters")
        args.supervised_configuration = _json_argument(
            args.supervised_configuration,
            "--supervised-configuration",
        )
    except argparse.ArgumentTypeError as error:
        parser.error(str(error))
    return args


def main() -> int:
    args = _parse_args()
    try:
        token = _resolve_token(args)
        active_model = _select_model(args)
        expected_task = active_model.get("task_type")
        if expected_task != args.task:
            raise RuntimeError(
                f"Active model task is {expected_task!r}, expected {args.task!r}"
            )

        payload = _evaluation_payload(args)
        warmups = []
        measured = []
        total_iterations = args.warmup + args.runs
        for position in range(total_iterations):
            is_warmup = position < args.warmup
            kind = "warmup" if is_warmup else "measured"
            index = position + 1 if is_warmup else position - args.warmup + 1
            print(
                f"[{kind} {index}/{args.warmup if is_warmup else args.runs}] "
                "running evaluation...",
                flush=True,
            )
            run = _run_once(
                args=args,
                token=token,
                payload=payload,
                kind=kind,
                index=index,
            )
            print(
                f"  {run.elapsed_seconds:.3f}s, "
                f"{run.throughput_images_per_second:.3f} images/s, "
                f"{run.response_bytes} response bytes",
                flush=True,
            )
            (warmups if is_warmup else measured).append(run)

        summary = _summary(measured)
        resolved_parameters = measured[0].resolved_parameters
        effective_supervised_configuration = measured[
            0
        ].effective_supervised_configuration
        if any(run.resolved_parameters != resolved_parameters for run in measured):
            raise RuntimeError("Resolved attack parameters changed between runs")
        if any(
            run.effective_supervised_configuration != effective_supervised_configuration
            for run in measured
        ):
            raise RuntimeError(
                "Effective supervised configuration changed between runs"
            )

        configuration = {
            "task": args.task,
            "dataset_id": args.dataset_id,
            "model": active_model,
            "attack": args.attack,
            "parameters": resolved_parameters,
            "requested_parameters": args.parameters,
            "supervised_configuration": effective_supervised_configuration,
            "requested_supervised_configuration": (
                args.supervised_configuration if args.task == "detection" else None
            ),
            "force_recompute": True,
            "warmup_runs": args.warmup,
            "measured_runs": args.runs,
        }
        output = {
            "benchmark_schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "label": args.label,
            "environment": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "base_url": args.base_url,
                "repository": _repository_metadata(),
            },
            "configuration": configuration,
            "warmup_runs": [asdict(run) for run in warmups],
            "measured_runs": [asdict(run) for run in measured],
            "summary": summary,
        }
        if args.compare_to is not None:
            output["comparison"] = _comparison(
                summary,
                configuration,
                args.compare_to,
            )

        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(output, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        print(f"Saved benchmark to {args.output}")
        return 0
    except (RuntimeError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"benchmark failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
