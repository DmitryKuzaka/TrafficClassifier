"""Freeze the final comparison, then evaluate the fixed held-out captures."""

import argparse
import csv
import json
from hashlib import sha256
from io import StringIO
from pathlib import Path

import numpy as np
import torch

from traffic_classifier.ml import (
    baseline,
    cnn,
    dataset,
    metrics,
    model_bundle,
    normalization,
    prefix_study,
)
from traffic_classifier.ml.baseline import load_forest, summarize_prefixes
from traffic_classifier.ml.dataset import DEFAULT_BUNDLE, load_bundle
from traffic_classifier.ml.experiment import THREADS
from traffic_classifier.ml.forest_study import file_digest
from traffic_classifier.ml.model_bundle import load_model_bundle
from traffic_classifier.ml.prefix_study import DEFAULT_OUTPUT, predict_cnn, score_prefixes
from traffic_classifier.ml.vnat_split import DEFAULT_MANIFEST, read_split, save_if_unchanged
from traffic_classifier.vnat_groups import CLASSES, PREFIX_LENGTHS

DEFAULT_PROTOCOL = Path("docs/prefix-study-protocol.json")


def build_protocol(output: Path) -> dict:
    path = output / "validation.json"
    validation = json.loads(path.read_text(encoding="utf-8"))
    if validation["split"] != "validation" or validation["manifest_sha256"] != file_digest(
        DEFAULT_MANIFEST
    ):
        raise ValueError("The comparison does not use the fixed validation split.")
    test_rows = [row for row in read_split(DEFAULT_MANIFEST) if row.split == "test"]
    if validation["import_id"] != test_rows[0].import_id:
        raise ValueError("The comparison and manifest use different imports.")
    candidates = {}
    for n in PREFIX_LENGTHS:
        entry = validation["lengths"][str(n)]
        cnn_path = output / "cnn" / f"n{n}" / "model.pt"
        forest_path = output / "forest" / f"n{n}.joblib"
        model = load_model_bundle(cnn_path, expected_n=n)
        if model.model_id != entry["cnn"]["model_id"]:
            raise ValueError(f"CNN file differs from validation for N={n}.")
        if file_digest(forest_path) != entry["random_forest"]["sha256"]:
            raise ValueError(f"Random Forest file differs from validation for N={n}.")
        candidates[str(n)] = {
            "cnn_model_id": model.model_id,
            "cnn_sha256": file_digest(cnn_path),
            "cnn_best_epoch": entry["cnn"]["best_epoch"],
            "cnn_validation_macro_f1": entry["cnn"]["metrics"]["all"]["macro_f1"],
            "forest_sha256": entry["random_forest"]["sha256"],
            "forest_validation_macro_f1": entry["random_forest"]["metrics"]["all"]["macro_f1"],
        }
    selected = max(
        PREFIX_LENGTHS, key=lambda n: (candidates[str(n)]["cnn_validation_macro_f1"], -n)
    )
    evaluation_sources = [Path(__file__)]
    evaluation_sources.extend(
        Path(module.__file__)
        for module in (baseline, cnn, dataset, metrics, model_bundle, normalization, prefix_study)
    )
    return {
        "version": 1,
        "import_id": validation["import_id"],
        "manifest_sha256": validation["manifest_sha256"],
        "validation_report_sha256": file_digest(path),
        "evaluation_code_sha256": {
            source.name: sha256(source.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
            for source in evaluation_sources
        },
        "selected_cnn_n": selected,
        "selection_rule": "Highest full validation macro-F1; smaller N breaks ties.",
        "class_order": list(CLASSES),
        "train_majority_index": validation["train_majority_index"],
        "lengths": candidates,
        "test_rules": {
            "methods": ["cnn", "random_forest", "train_majority"],
            "all": "Every test flow; use min(packet_count, N) real packets.",
            "at_least_64": "The same flows with at least 64 packets for every N and method.",
            "metric": "Macro-F1 and per-class precision, recall, F1, support and confusion matrix.",
            "missing_class": "A slice missing any target class has null macro-F1.",
            "flows": len(test_rows),
            "capture_groups": sorted({row.group_id for row in test_rows}),
            "support_by_class": {
                label: sum(row.label == label for row in test_rows) for label in CLASSES
            },
        },
    }


def write_fixed_json(path: Path, data: dict) -> None:
    save_if_unchanged(path, json.dumps(data, indent=2, allow_nan=False) + "\n")


def evaluate_test(output: Path, protocol_path: Path) -> dict:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol != build_protocol(output):
        raise ValueError(
            "Current models, validation results or metrics differ from the frozen protocol."
        )

    torch.set_num_threads(THREADS)
    data = load_bundle(DEFAULT_BUNDLE, "test")
    rows = [row for row in read_split(DEFAULT_MANIFEST) if row.split == "test"]
    buffer = StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(
        [
            "n",
            "flow_id",
            "capture_group",
            "total_packets",
            "observed_packets",
            "label",
            "cnn",
            "random_forest",
            "train_majority",
        ]
    )
    results = {}
    for n in PREFIX_LENGTHS:
        cnn = load_model_bundle(output / "cnn" / f"n{n}" / "model.pt", expected_n=n)
        forest = load_forest(output / "forest" / f"n{n}.joblib", data, n)
        predictions = {
            "cnn": predict_cnn(cnn, data),
            "random_forest": forest["estimator"].predict(summarize_prefixes(data, n)),
            "train_majority": np.full(len(rows), protocol["train_majority_index"]),
        }
        results[str(n)] = {
            method: score_prefixes(data, n, values) for method, values in predictions.items()
        }
        for index, row in enumerate(rows):
            writer.writerow(
                [
                    n,
                    row.flow_id,
                    row.group_id,
                    row.packet_count,
                    min(row.packet_count, n),
                    row.label,
                    *(CLASSES[values[index]] for values in predictions.values()),
                ]
            )
        print(
            f"N={n}: CNN={results[str(n)]['cnn']['all']['macro_f1']:.4f}; "
            f"Random Forest={results[str(n)]['random_forest']['all']['macro_f1']:.4f}",
            flush=True,
        )
    report = {
        "split": data.split,
        "protocol_sha256": file_digest(protocol_path),
        "manifest_sha256": data.manifest_sha256,
        "selected_cnn_n": protocol["selected_cnn_n"],
        "lengths": results,
    }
    save_if_unchanged(output / "test-predictions.csv", buffer.getvalue())
    write_fixed_json(output / "test.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "test"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()
    if args.command == "freeze":
        protocol = build_protocol(args.output)
        write_fixed_json(args.protocol, protocol)
        print(f"Frozen CNN N={protocol['selected_cnn_n']}: {args.protocol}")
    else:
        evaluate_test(args.output, args.protocol)


if __name__ == "__main__":
    main()
