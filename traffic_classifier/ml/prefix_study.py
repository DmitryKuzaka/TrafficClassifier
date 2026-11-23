"""Compare packet prefix lengths on the fixed validation split."""

import argparse
import json
from pathlib import Path

import numpy as np

from traffic_classifier.ml.dataset import (
    DEFAULT_BUNDLE,
    MAX_PACKETS,
    Bundle,
    FlowDataset,
    load_bundle,
)
from traffic_classifier.ml.experiment import run_experiment
from traffic_classifier.ml.metrics import classification_metrics
from traffic_classifier.ml.model_bundle import ModelBundle, export_model_bundle, load_model_bundle
from traffic_classifier.ml.normalization import DEFAULT_DIR as NORMALIZERS
from traffic_classifier.ml.training import BATCH_SIZE
from traffic_classifier.vnat_groups import CLASSES, PREFIX_LENGTHS

DEFAULT_OUTPUT = Path("artifacts/prefix-study-v1")


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def score_prefixes(bundle: Bundle, n: int, predicted: np.ndarray) -> dict:
    lengths = bundle.mask.sum(axis=1)
    long = lengths >= MAX_PACKETS
    coverage = {}
    for index, label in enumerate(CLASSES):
        class_lengths = lengths[bundle.labels == index]
        count = len(class_lengths)
        short = int((class_lengths < n).sum())
        coverage[label] = {
            "flows": count,
            "short": short,
            "short_fraction": short / count if count else None,
            "at_least_64": int((class_lengths >= MAX_PACKETS).sum()),
        }
    return {
        "all": classification_metrics(bundle.labels, predicted),
        "at_least_64": classification_metrics(bundle.labels[long], predicted[long])
        if long.any()
        else None,
        "coverage": coverage,
        "short_fraction": float((lengths < n).mean()),
    }


def predict_cnn(model: ModelBundle, data: Bundle) -> np.ndarray:
    normalizer = model.normalizer
    if (normalizer.import_id, normalizer.manifest_sha256) != (data.import_id, data.manifest_sha256):
        raise ValueError("CNN bundle and packet data refer to different splits.")
    dataset = FlowDataset(data, normalizer.n)
    return np.concatenate(
        [
            model.logits(x, mask).argmax(dim=1).numpy()
            for x, mask in zip(
                dataset.features.split(BATCH_SIZE), dataset.mask.split(BATCH_SIZE), strict=True
            )
        ]
    )


def run_cnn_study(output: Path) -> dict:
    report_path = output / "cnn-validation.json"
    if report_path.exists():
        raise ValueError(
            f"CNN study already completed: {report_path}. Choose a new output directory."
        )
    validation = load_bundle(DEFAULT_BUNDLE, "validation")
    report = {
        "split": validation.split,
        "import_id": validation.import_id,
        "manifest_sha256": validation.manifest_sha256,
        "lengths": {},
    }
    for n in PREFIX_LENGTHS:
        training = run_experiment(n, output / "cnn")
        directory = output / "cnn" / f"n{n}"
        model_path = directory / "model.pt"
        export_model_bundle(
            directory / "checkpoint.pt", NORMALIZERS / f"n{n}.json", model_path, expected_n=n
        )
        model = load_model_bundle(model_path, expected_n=n)
        metrics = score_prefixes(validation, n, predict_cnn(model, validation))
        recorded = {key: value for key, value in training["validation"].items() if key != "loss"}
        if metrics["all"] != recorded:
            raise ValueError(f"Reloaded CNN predictions differ from the chosen epoch for N={n}.")
        report["lengths"][str(n)] = {
            "cnn": {
                "model_id": model.model_id,
                "best_epoch": training["best_epoch"],
                "validation_loss": training["validation"]["loss"],
                "epochs_run": len(training["history"]),
                "loop_seconds": sum(row["seconds"] for row in training["history"]),
                "config": training["config"],
                "metrics": metrics,
            }
        }
    write_report(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run_cnn_study(parser.parse_args().output)


if __name__ == "__main__":
    main()
