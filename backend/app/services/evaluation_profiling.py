import json
from dataclasses import dataclass, field
from time import perf_counter

import torch

EVALUATION_PROFILE_SCHEMA_VERSION = 1


def get_model_device(model) -> torch.device | None:
    """Return the underlying torch device when a model exposes one."""
    get_torch_model = getattr(model, "get_torch_model", None)
    if not callable(get_torch_model):
        return None

    try:
        torch_model = get_torch_model()
    except (RuntimeError, ValueError):
        # A model may legitimately not expose a torch module to diagnostics.
        return None

    if not isinstance(torch_model, torch.nn.Module):
        return None

    parameter = next(torch_model.parameters(), None)
    if parameter is not None:
        return parameter.device

    buffer = next(torch_model.buffers(), None)
    return buffer.device if buffer is not None else None


def synchronize_device(device: torch.device | None) -> None:
    """Synchronize CUDA so a wall-clock stage includes completed GPU work."""
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)


def start_profile_stage(device: torch.device | None = None) -> float:
    synchronize_device(device)
    return perf_counter()


def finish_profile_stage(
    started_at: float,
    device: torch.device | None = None,
) -> float:
    synchronize_device(device)
    return perf_counter() - started_at


@dataclass(slots=True)
class EvaluationProfile:
    """Collect stable, implementation-neutral evaluation timing stages."""

    task_type: str
    dataset_id: str
    attack_type: str
    started_at: float = field(default_factory=perf_counter)
    stage_seconds: dict[str, float] = field(default_factory=dict)

    def add(self, stage: str, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("profile stage duration must not be negative")
        self.stage_seconds[stage] = self.stage_seconds.get(stage, 0.0) + seconds

    def payload(
        self,
        *,
        model_id: str,
        model_name: str | None,
        total_images: int,
        device: torch.device | None,
        extra: dict | None = None,
    ) -> dict:
        total_seconds = perf_counter() - self.started_at
        throughput = total_images / total_seconds if total_seconds > 0 else 0.0
        payload = {
            "profile_schema_version": EVALUATION_PROFILE_SCHEMA_VERSION,
            "task_type": self.task_type,
            "dataset_id": self.dataset_id,
            "model_id": model_id,
            "model_name": model_name,
            "attack_type": self.attack_type,
            "images": total_images,
            "device": str(device) if device is not None else "unknown",
            "throughput_images_per_second": round(throughput, 6),
            "total_seconds": round(total_seconds, 6),
            "stages_seconds": {
                stage: round(seconds, 6)
                for stage, seconds in sorted(self.stage_seconds.items())
            },
        }
        if extra:
            payload.update(extra)
        return payload


def log_evaluation_profile(logger, payload: dict) -> None:
    """Emit one machine-readable line that can be archived with benchmarks."""
    logger.info(
        "evaluation_profile %s",
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
    )
