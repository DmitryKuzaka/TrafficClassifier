"""Evaluate the baseline on the fixed validation split."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from traffic_classifier.vnat_groups import CLASSES

LABELS = tuple(range(len(CLASSES)))
DEFAULT_REPORT = Path("artifacts/baseline-validation-v1.json")


def classification_metrics(y_true, y_pred) -> dict:
    truth = np.asarray(y_true)
    predicted = np.asarray(y_pred)
    if truth.ndim != 1 or truth.shape != predicted.shape or not len(truth):
        raise ValueError("Truth and predictions must be equally sized, nonempty vectors.")
    if not np.isin(truth, LABELS).all() or not np.isin(predicted, LABELS).all():
        raise ValueError("An unknown class index was found in evaluation.")
    precision, recall, f1, support = precision_recall_fscore_support(
        truth,
        predicted,
        labels=LABELS,
        zero_division=0,
    )
    return {
        "macro_f1": float(f1.mean()) if support.all() else None,
        "accuracy": float(np.mean(truth == predicted)),
        "support": len(truth),
        "by_class": {
            label: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(CLASSES)
        },
        "class_order": list(CLASSES),
        "confusion_matrix": confusion_matrix(truth, predicted, labels=LABELS).tolist(),
    }


def main() -> None:
    from traffic_classifier.ml.baseline import DEFAULT_DIR, load_forest, summarize_prefixes
    from traffic_classifier.ml.dataset import DEFAULT_BUNDLE, load_bundle
    from traffic_classifier.ml.vnat_split import DEFAULT_MANIFEST
    from traffic_classifier.vnat_groups import PREFIX_LENGTHS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, choices=PREFIX_LENGTHS, default=32)
    parser.add_argument("--packages", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--models", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    train = load_bundle(args.packages, "train", args.manifest)
    validation = load_bundle(args.packages, "validation", args.manifest)
    artifact = load_forest(args.models / f"n{args.n}.joblib", validation, args.n)
    features = summarize_prefixes(validation, args.n)
    majority = int(np.argmax(np.bincount(train.labels, minlength=len(CLASSES))))
    forest = classification_metrics(validation.labels, artifact["estimator"].predict(features))
    constant = classification_metrics(validation.labels, np.full(len(validation.labels), majority))
    if forest["macro_f1"] is None:
        raise ValueError("Validation has lost a target class.")
    output = {
        "split": "validation",
        "n": args.n,
        "import_id": validation.import_id,
        "manifest_sha256": validation.manifest_sha256,
        "random_forest": {"parameters": artifact["parameters"], "metrics": forest},
        "majority": {"class": CLASSES[majority], "metrics": constant},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Validation ({len(validation.labels)} flows), N={args.n}")
    print(f"Random Forest macro-F1: {forest['macro_f1']:.4f}")
    print(f"Constant {CLASSES[majority]} macro-F1: {constant['macro_f1']:.4f}")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
