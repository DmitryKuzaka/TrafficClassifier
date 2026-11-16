"""Training and evaluation loops for the packet CNN."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from traffic_classifier.ml.cnn import PacketCNN
from traffic_classifier.ml.dataset import (
    DEFAULT_BUNDLE,
    Bundle,
    FlowDataset,
    load_bundle,
    make_loader,
)
from traffic_classifier.ml.metrics import classification_metrics
from traffic_classifier.ml.normalization import DEFAULT_DIR, Normalizer, load_normalizer
from traffic_classifier.ml.vnat_split import DEFAULT_MANIFEST, SEED
from traffic_classifier.vnat_groups import CLASSES, PREFIX_LENGTHS

LEARNING_RATE = 0.002
WEIGHT_DECAY = 0.0001
BATCH_SIZE = 128


def normalized_dataset(bundle: Bundle, normalizer: Normalizer) -> FlowDataset:
    if (bundle.import_id, bundle.manifest_sha256) != (
        normalizer.import_id,
        normalizer.manifest_sha256,
    ):
        raise ValueError("Normalization settings refer to a different data split.")
    dataset = FlowDataset(bundle, normalizer.n)
    dataset.features = normalizer.transform(dataset.features, dataset.mask)
    return dataset


def load_training_inputs(
    n: int,
    packages: Path = DEFAULT_BUNDLE,
    manifest: Path = DEFAULT_MANIFEST,
    normalization: Path = DEFAULT_DIR,
) -> tuple[FlowDataset, FlowDataset, Normalizer]:
    train = load_bundle(packages, "train", manifest)
    validation = load_bundle(packages, "validation", manifest)
    normalizer = load_normalizer(normalization / f"n{n}.json", train)
    if normalizer.n != n:
        raise ValueError("Normalizer has an unexpected packet count.")
    return (
        normalized_dataset(train, normalizer),
        normalized_dataset(validation, normalizer),
        normalizer,
    )


def train_epoch(
    model: PacketCNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.CrossEntropyLoss,
) -> dict:
    if criterion.reduction != "none":
        raise ValueError("Training requires an unreduced loss for each example.")
    model.train()
    loss_sum = 0.0
    targets, predicted = [], []
    for x, mask, labels in loader:
        optimizer.zero_grad(set_to_none=True)
        logits = model(x, mask)
        loss = criterion(logits, labels).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss.")
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * len(labels)
        targets.append(labels)
        predicted.append(logits.detach().argmax(dim=1))
    return summarize_epoch(loss_sum, targets, predicted)


@torch.inference_mode()
def evaluate_epoch(model: PacketCNN, loader: DataLoader) -> dict:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    loss_sum = 0.0
    targets, predicted = [], []
    for x, mask, labels in loader:
        logits = model(x, mask)
        loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite evaluation loss.")
        loss_sum += loss.item()
        targets.append(labels)
        predicted.append(logits.argmax(dim=1))
    return summarize_epoch(loss_sum, targets, predicted)


def summarize_epoch(
    loss_sum: float, targets: list[torch.Tensor], predicted: list[torch.Tensor]
) -> dict:
    if not targets:
        raise ValueError("Cannot score an empty epoch.")
    true = torch.cat(targets).numpy()
    prediction = torch.cat(predicted).numpy()
    return {"loss": loss_sum / len(true), **classification_metrics(true, prediction)}


def classwise_subset(dataset: FlowDataset, per_class: int) -> Subset:
    indices = []
    for label in range(len(CLASSES)):
        rows = torch.nonzero(dataset.labels == label, as_tuple=True)[0][:per_class]
        if len(rows) != per_class:
            raise ValueError("Insufficient training records for the diagnostic run.")
        indices.extend(rows.tolist())
    return Subset(dataset, indices)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, choices=PREFIX_LENGTHS, default=32)
    args = parser.parse_args()

    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    train, validation, _ = load_training_inputs(args.n)
    loader = make_loader(classwise_subset(train, 8), 12, shuffle=True)
    model = PacketCNN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    start = perf_counter()
    train_metrics = train_epoch(model, loader, optimizer, nn.CrossEntropyLoss(reduction="none"))
    trained = [parameter.detach().clone() for parameter in model.parameters()]
    if not any(not torch.equal(a, b) for a, b in zip(before, trained, strict=True)):
        raise RuntimeError("Training did not change the model weights.")
    validation_metrics = evaluate_epoch(model, make_loader(validation, BATCH_SIZE))
    if any(not torch.equal(a, b) for a, b in zip(trained, model.parameters(), strict=True)):
        raise RuntimeError("Evaluation changed the model weights.")
    report = {
        "purpose": "loop verification",
        "epoch": 1,
        "n": args.n,
        "seed": SEED,
        "device": "cpu",
        "learning_rate": LEARNING_RATE,
        "elapsed_seconds": perf_counter() - start,
        "train": train_metrics,
        "validation": validation_metrics,
    }
    path = Path("artifacts/cnn-loop-check-v1.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"Train loss={train_metrics['loss']:.4f}; validation loss={validation_metrics['loss']:.4f}; {path}"
    )


if __name__ == "__main__":
    main()
