"""Check whether the CNN can learn a small balanced subset of train."""

import json
from pathlib import Path
from time import perf_counter

import torch
from torch import nn

from traffic_classifier.ml.cnn import PacketCNN
from traffic_classifier.ml.dataset import DEFAULT_BUNDLE, load_bundle, make_loader
from traffic_classifier.ml.normalization import DEFAULT_DIR, load_normalizer
from traffic_classifier.ml.training import (
    LEARNING_RATE,
    WEIGHT_DECAY,
    classwise_subset,
    evaluate_epoch,
    normalized_dataset,
    train_epoch,
)
from traffic_classifier.ml.vnat_split import SEED

N = 32
PER_CLASS = 12
EPOCHS = 120
OUTPUT = Path("artifacts/cnn-tiny-check-v1.json")


def main() -> None:
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    bundle = load_bundle(DEFAULT_BUNDLE, "train")
    normalizer = load_normalizer(DEFAULT_DIR / f"n{N}.json", bundle)
    if normalizer.n != N:
        raise ValueError("Normalizer has an unexpected packet count.")
    subset = classwise_subset(normalized_dataset(bundle, normalizer), PER_CLASS)
    loader = make_loader(subset, len(subset))
    model = PacketCNN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    before = evaluate_epoch(model, loader)
    start = perf_counter()
    for _ in range(EPOCHS):
        train_epoch(model, loader, optimizer, loss_fn)
    seconds_per_epoch = (perf_counter() - start) / EPOCHS
    after = evaluate_epoch(model, loader)
    if after["loss"] >= before["loss"] * 0.8:
        raise RuntimeError(
            "Tiny training did not reduce loss sufficiently; inspect packet flow and labels."
        )
    report = {
        "purpose": "diagnostic on the same train examples",
        "n": N,
        "seed": SEED,
        "epochs": EPOCHS,
        "per_class": PER_CLASS,
        "seconds_per_epoch": seconds_per_epoch,
        "before": before,
        "after": after,
        "flow_ids": subset.dataset.flow_ids[subset.indices].tolist(),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"Train-only diagnostic: loss {before['loss']:.4f} -> {after['loss']:.4f}; {seconds_per_epoch:.4f} s/epoch."
    )
    print(f"Report: {OUTPUT}")


if __name__ == "__main__":
    main()
