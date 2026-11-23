"""Add comparable Random Forest results to the validation study."""

import argparse
import json
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np

from traffic_classifier.ml.baseline import load_forest, summarize_prefixes, train_forest
from traffic_classifier.ml.dataset import DEFAULT_BUNDLE, load_bundle
from traffic_classifier.ml.prefix_study import DEFAULT_OUTPUT, score_prefixes, write_report
from traffic_classifier.vnat_groups import CLASSES, PREFIX_LENGTHS


def file_digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def run_forest_study(output: Path) -> dict:
    report_path = output / "validation.json"
    if report_path.exists():
        raise ValueError(f"Forest study already completed: {report_path}. Choose a new study.")
    cnn_report = output / "cnn-validation.json"
    report = json.loads(cnn_report.read_text(encoding="utf-8"))
    train = load_bundle(DEFAULT_BUNDLE, "train")
    validation = load_bundle(DEFAULT_BUNDLE, "validation")
    if (report["split"], report["import_id"], report["manifest_sha256"]) != (
        validation.split,
        validation.import_id,
        validation.manifest_sha256,
    ):
        raise ValueError("CNN report is from a different validation split.")
    report["cnn_report_sha256"] = file_digest(cnn_report)
    report["train_majority_index"] = int(np.bincount(train.labels, minlength=len(CLASSES)).argmax())
    directory = output / "forest"
    directory.mkdir(parents=True, exist_ok=True)
    for n in PREFIX_LENGTHS:
        path = directory / f"n{n}.joblib"
        start = perf_counter()
        artifact = train_forest(train, n)
        fit_seconds = perf_counter() - start
        joblib.dump(artifact, path, compress=3)
        forest = load_forest(path, validation, n)
        predictions = forest["estimator"].predict(summarize_prefixes(validation, n))
        metrics = score_prefixes(validation, n, predictions)
        entry = report["lengths"][str(n)]
        if entry["cnn"]["metrics"]["coverage"] != metrics["coverage"]:
            raise ValueError(
                f"Random Forest and CNN evaluated different packet coverage for N={n}."
            )
        entry["random_forest"] = {
            "parameters": forest["parameters"],
            "sha256": file_digest(path),
            "fit_seconds": fit_seconds,
            "metrics": metrics,
        }
        print(
            f"N={n}: CNN={entry['cnn']['metrics']['all']['macro_f1']:.4f}; "
            f"Random Forest={metrics['all']['macro_f1']:.4f}",
            flush=True,
        )
    write_report(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run_forest_study(parser.parse_args().output)


if __name__ == "__main__":
    main()
