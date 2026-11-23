"""Inspect mistakes and capture-level results of the chosen CNN."""

import argparse
import json
from pathlib import Path

import pandas as pd

from traffic_classifier.ml.metrics import classification_metrics
from traffic_classifier.ml.prefix_study import DEFAULT_OUTPUT, write_report
from traffic_classifier.vnat_groups import CLASSES, PREFIX_LENGTHS

LABEL_INDEX = {label: index for index, label in enumerate(CLASSES)}
METHODS = ("cnn", "random_forest")


def score_rows(rows: pd.DataFrame) -> dict | None:
    if rows.empty:
        return None
    labels = rows["label"].map(LABEL_INDEX)
    return {
        method: classification_metrics(labels, rows[method].map(LABEL_INDEX)) for method in METHODS
    }


def analyze_errors(output: Path) -> dict:
    test = json.loads((output / "test.json").read_text(encoding="utf-8"))
    n = test["selected_cnn_n"]
    frame = pd.read_csv(output / "test-predictions.csv")
    selected = frame.loc[frame["n"].eq(n)]
    expected = test["lengths"][str(n)]
    if (
        selected["flow_id"].duplicated().any()
        or not selected["total_packets"].clip(upper=n).eq(selected["observed_packets"]).all()
        or score_rows(selected) != {method: expected[method]["all"] for method in METHODS}
    ):
        raise ValueError("Selected predictions do not match the fixed test summary.")

    lengths = selected["total_packets"]
    long_limit = max(PREFIX_LENGTHS)
    subsets = {
        "shorter_than_n": selected.loc[lengths.lt(n)],
        "n_to_63": selected.loc[lengths.ge(n) & lengths.lt(long_limit)],
        "at_least_64": selected.loc[lengths.ge(long_limit)],
    }
    mistakes = selected.loc[selected["cnn"].ne(selected["label"])]
    examples = {}
    for label in CLASSES:
        rows = mistakes.loc[mistakes["label"].eq(label)]
        examples[label] = {
            "count": len(rows),
            "predicted_as": {str(k): int(v) for k, v in rows["cnn"].value_counts().items()},
            "examples": rows[
                ["flow_id", "capture_group", "total_packets", "observed_packets", "cnn"]
            ]
            .head(3)
            .to_dict(orient="records"),
        }
    captures = {
        capture: {
            "flows": len(rows),
            "true_class": rows["label"].iat[0],
            "cnn_correct": int(rows["cnn"].eq(rows["label"]).sum()),
            "forest_correct": int(rows["random_forest"].eq(rows["label"]).sum()),
        }
        for capture, rows in selected.groupby("capture_group", sort=True)
    }
    report = {
        "n": n,
        "protocol_sha256": test["protocol_sha256"],
        "by_length": {key: score_rows(rows) for key, rows in subsets.items()},
        "by_capture": captures,
        "cnn_errors": examples,
    }
    write_report(output / "error-analysis.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = analyze_errors(args.output)
    print(f"Analyzed N={report['n']}: {args.output / 'error-analysis.json'}")


if __name__ == "__main__":
    main()
