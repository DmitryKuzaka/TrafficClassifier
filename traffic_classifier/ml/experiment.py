"""Train a packet CNN with a fixed validation-based checkpoint rule."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from torch import nn

from traffic_classifier.ml.cnn import ARCHITECTURE, PacketCNN
from traffic_classifier.ml.dataset import make_loader
from traffic_classifier.ml.training import (
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    evaluate_epoch,
    load_training_inputs,
    train_epoch,
)
from traffic_classifier.ml.vnat_split import SEED
from traffic_classifier.vnat_groups import CLASSES, PREFIX_LENGTHS

DEFAULT_DIR = Path("artifacts/cnn-experiments-v1")
MAX_EPOCHS = 20
PATIENCE = 5
THREADS = 2


def plot_history(history: list[dict], best_epoch: int, path: Path) -> None:
    fig = Figure(figsize=(9, 3.5), layout="constrained")
    FigureCanvasAgg(fig)
    loss_plot, f1_plot = fig.subplots(1, 2)
    epochs = [row["epoch"] for row in history]
    loss_plot.plot(epochs, [row["train_loss"] for row in history], label="Train, weighted")
    loss_plot.plot(epochs, [row["validation_loss"] for row in history], label="Validation")
    f1_plot.plot(epochs, [row["train_macro_f1"] for row in history], label="Train")
    f1_plot.plot(epochs, [row["validation_macro_f1"] for row in history], label="Validation")
    for axis, label in ((loss_plot, "Cross-entropy"), (f1_plot, "Macro-F1")):
        axis.set(xlabel="Epoch", ylabel=label)
        axis.axvline(best_epoch, color="grey", linestyle="--", label="Selected epoch")
        axis.legend()
    fig.savefig(path, dpi=150)


def run_experiment(n: int, output: Path = DEFAULT_DIR) -> dict:
    torch.set_num_threads(THREADS)
    torch.manual_seed(SEED)
    train, validation, normalizer = load_training_inputs(n)
    classes = torch.bincount(train.labels, minlength=len(CLASSES)).float()
    if not classes.gt(0).all():
        raise ValueError("Training lacks a target class.")
    weights = classes.rsqrt()
    weights *= classes.sum() / torch.dot(weights, classes)
    config = {
        "n": n,
        "seed": SEED,
        "device": "cpu",
        "threads": THREADS,
        "maximum_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "batch_size": BATCH_SIZE,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "train_class_counts": classes.int().tolist(),
        "train_inverse_sqrt_weights": weights.tolist(),
        "torch_version": str(torch.__version__),
        "checkpoint_metric": "validation_macro_f1",
        "tie_break": "earlier_epoch",
    }
    model = PacketCNN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss(weight=weights, reduction="none")
    train_loader = make_loader(train, BATCH_SIZE, shuffle=True)
    val_loader = make_loader(validation, BATCH_SIZE)
    history = []
    best_score = -1.0
    best_epoch = 0
    best_state = None
    best_validation = None
    for epoch in range(1, MAX_EPOCHS + 1):
        start = perf_counter()
        train_metrics = train_epoch(model, train_loader, optimizer, loss_fn)
        metrics = evaluate_epoch(model, val_loader)
        score = metrics["macro_f1"]
        if score is None:
            raise ValueError("Validation lacks a target class.")
        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "validation_loss": metrics["loss"],
            "train_macro_f1": train_metrics["macro_f1"],
            "validation_macro_f1": score,
            "seconds": perf_counter() - start,
        }
        history.append(record)
        print(
            f"N={n} epoch={epoch:02}: train loss={record['train_loss']:.4f}; validation macro-F1={score:.4f}",
            flush=True,
        )
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = {
                key: parameter.detach().clone() for key, parameter in model.state_dict().items()
            }
            best_validation = metrics
        if epoch - best_epoch >= PATIENCE:
            break

    folder = output / f"n{n}"
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "architecture": ARCHITECTURE,
        "config": config,
        "best_epoch": best_epoch,
        "state_dict": best_state,
        "import_id": normalizer.import_id,
        "manifest_sha256": normalizer.manifest_sha256,
    }
    torch.save(checkpoint, folder / "checkpoint.pt")
    report = {
        "config": config,
        "import_id": normalizer.import_id,
        "manifest_sha256": normalizer.manifest_sha256,
        "best_epoch": best_epoch,
        "validation": best_validation,
        "history": history,
    }
    (folder / "history.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    plot_history(history, best_epoch, folder / "curves.png")
    print(f"N={n}: selected epoch {best_epoch}, validation macro-F1={best_score:.4f}; {folder}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, choices=PREFIX_LENGTHS, default=32)
    parser.add_argument("--output", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args()
    run_experiment(args.n, args.output)


if __name__ == "__main__":
    main()
