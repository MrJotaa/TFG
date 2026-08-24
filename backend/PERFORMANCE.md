# Evaluation performance baseline

This document defines the repeatable performance protocol for classification and
detection dataset evaluations. The goal is to compare commits on the same
machine without changing the scientific evaluation contract.

## What is measured

Every completed evaluation emits one machine-readable backend log line starting
with `evaluation_profile`. Its JSON payload has `profile_schema_version=1` and
contains the common stages below:

- `dataset_load`: initial dataset read.
- `setup`: model/attack setup and task-specific evaluation context.
- `image_load`: accumulated image loading time.
- `attack`: accumulated adversarial attack time.
- `original_inference`: accumulated clean-image inference time.
- `adversarial_inference`: accumulated adversarial-image inference time.
- `artifact_persistence`: accumulated per-image artifact writing time.
- `evaluation_persistence`: final evaluation/history persistence.

Classification also reports `metrics`. Detection reports `per_image_metrics`,
`summary_metrics`, and `supervised_metrics` separately so COCO/TIDE-style work
can be distinguished from the detector and attack hot path.

CUDA-backed stages synchronize the device at timing boundaries. This makes the
wall-clock numbers include GPU work that would otherwise still be queued when a
stage returns.

## Reproducible end-to-end benchmark

Run the benchmark tool from the repository root while the backend is running.
The tool uses the streaming endpoint, always sends `force_recompute=true`, can
select the model before measurement, performs configurable warm-up runs, and
stores every measured run plus an aggregate summary as JSON. When Git is
available it also records the commit and whether the working tree is dirty.

Authentication can use an existing bearer token:

```bash
export TFG_BENCHMARK_TOKEN='<token>'
```

or a login without putting the password in shell history:

```bash
export TFG_BENCHMARK_PASSWORD='<password>'
```

### Classification example

```bash
python backend/app/tools/benchmark_evaluations.py \
  --task classification \
  --dataset-id '<dataset-id>' \
  --model-id resnet18 \
  --attack fgsm \
  --parameters '{"epsilon":0.03}' \
  --email '<email>' \
  --warmup 1 \
  --runs 3 \
  --label before-dataset-io \
  --output benchmarks/classification-before.json
```

### Detection example

```bash
python backend/app/tools/benchmark_evaluations.py \
  --task detection \
  --dataset-id '<dataset-id>' \
  --model-id yolo26n \
  --attack yolo_confidence \
  --parameters '{}' \
  --supervised-configuration '{"score_floor":0.001}' \
  --email '<email>' \
  --warmup 1 \
  --runs 3 \
  --label before-dataset-io \
  --output benchmarks/detection-before.json
```

Use the attack and model IDs exposed by the current API. Keep dataset, model,
attack parameters, supervised configuration, hardware, and container settings
identical between measurements.

## Comparing a later commit

Pass the earlier JSON to `--compare-to`:

```bash
python backend/app/tools/benchmark_evaluations.py \
  <same arguments as baseline> \
  --label after-dataset-io \
  --output benchmarks/classification-after.json \
  --compare-to benchmarks/classification-before.json
```

The tool rejects comparisons when task, dataset, model ID, attack, attack
parameters, or supervised configuration differ. The comparison reports percent
change for mean/median wall time, mean throughput, and response size.

For elapsed time, a negative percent change is an improvement. For throughput,
a positive percent change is an improvement.

## Benchmark discipline

1. Use the same machine and GPU power/performance settings.
2. Do not benchmark while another evaluation or GPU-heavy task is running.
3. Use at least one warm-up run for real neural models.
4. Prefer three or more measured runs; increase the count when variance is high.
5. Keep `force_recompute=true`; cached evaluation reuse is not pipeline speed.
6. Archive the benchmark JSON and the corresponding `evaluation_profile` log
   lines with the commit/issue notes.
7. Re-run functional tests before accepting a performance change. A faster
   result is not acceptable if predictions, perturbations, metrics, artifacts,
   progress, cancellation, or persistence semantics change.

The benchmark runner intentionally has no third-party Python dependencies so it
can be executed from the host against the Docker Compose backend.
