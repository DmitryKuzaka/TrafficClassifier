"""Time local model calls separately from the time needed to observe packets."""

import argparse
import platform
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier

from traffic_classifier.ml.baseline import load_forest, summarize_prefixes
from traffic_classifier.ml.dataset import (
    DEFAULT_BUNDLE,
    MAX_PACKETS,
    Bundle,
    FlowDataset,
    load_bundle,
)
from traffic_classifier.ml.forest_study import file_digest
from traffic_classifier.ml.model_bundle import ModelBundle, load_model_bundle
from traffic_classifier.ml.prefix_study import DEFAULT_OUTPUT, write_report
from traffic_classifier.vnat_groups import PREFIX_LENGTHS

THREADS = 1
BATCH_SIZES = (1, 32)
WARMUP = 20
REPEATS = 100


def percentiles(values) -> dict:
    return {"median_ms": float(np.median(values)), "p90_ms": float(np.quantile(values, 0.9))}


def time_calls(call: Callable[[], object]) -> dict:
    for _ in range(WARMUP):
        call()
    durations = []
    for _ in range(REPEATS):
        start = perf_counter_ns()
        call()
        durations.append((perf_counter_ns() - start) / 1_000_000)
    return percentiles(durations)


def benchmark_batch(
    cnn: ModelBundle, forest: RandomForestClassifier, batch: Bundle, n: int
) -> dict:
    dataset = FlowDataset(batch, n)
    return {
        "cnn": time_calls(lambda: cnn.logits(dataset.features, dataset.mask).argmax(dim=1)),
        "random_forest": time_calls(lambda: forest.predict(summarize_prefixes(batch, n))),
    }


def run_benchmark(output: Path) -> dict:
    torch.set_num_threads(THREADS)
    validation = load_bundle(DEFAULT_BUNDLE, "validation")
    eligible = np.flatnonzero(validation.mask[:, MAX_PACKETS - 1])
    if len(eligible) < max(BATCH_SIZES):
        raise ValueError("Validation has too few full-length flows for the benchmark batch.")
    batches = {}
    for batch_size in BATCH_SIZES:
        indices = eligible[:batch_size]
        batches[batch_size] = replace(
            validation,
            features=validation.features[indices],
            mask=validation.mask[indices],
            labels=validation.labels[indices],
            flow_ids=validation.flow_ids[indices],
        )

    results = {}
    for n in PREFIX_LENGTHS:
        cnn = load_model_bundle(output / "cnn" / f"n{n}" / "model.pt", expected_n=n)
        forest_path = output / "forest" / f"n{n}.joblib"
        forest = load_forest(forest_path, validation, n)["estimator"]
        forest.set_params(n_jobs=THREADS)
        calls = {
            str(size): benchmark_batch(cnn, forest, batch, n) for size, batch in batches.items()
        }
        gaps_ms = validation.features[eligible, 2, :n]
        results[str(n)] = {
            "cnn_model_id": cnn.model_id,
            "forest_sha256": file_digest(forest_path),
            "prediction_call": calls,
            "observe_packets": percentiles(gaps_ms.sum(axis=1, dtype=np.float64)),
        }
    report = {
        "system": platform.system(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torch_threads": THREADS,
        "forest_jobs": THREADS,
        "warmup_calls": WARMUP,
        "measured_calls": REPEATS,
        "manifest_sha256": validation.manifest_sha256,
        "batch_flow_ids": {str(size): batch.flow_ids.tolist() for size, batch in batches.items()},
        "observation_split": "validation",
        "observation_flows_at_least_64": len(eligible),
        "prediction_scope": "CPU call on prepared tensors, including normalization or feature aggregation.",
        "observe_scope": "Historical time from first to Nth packet on fixed validation flows.",
        "lengths": results,
    }
    write_report(output / "timing.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_benchmark(args.output)
    print(args.output / "timing.json")


if __name__ == "__main__":
    main()
