"""Build fixed-length packet tensors from the saved VNAT split."""

import argparse
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from traffic_classifier.clickhouse_client import connect
from traffic_classifier.dataset_report import completed_import_id
from traffic_classifier.ml.packets import InvalidPackets, clean_packets
from traffic_classifier.ml.vnat_split import DEFAULT_MANIFEST, GROUPS, SEED, prefix_rows, read_split
from traffic_classifier.vnat_groups import CLASSES, PREFIX_LENGTHS

MAX_PACKETS = max(PREFIX_LENGTHS)
DEFAULT_BUNDLE = Path("artifacts/vnat-prefixes-v1")


def encode_prefix(offsets_us, sizes, directions, n: int) -> tuple[np.ndarray, np.ndarray]:
    if n not in PREFIX_LENGTHS:
        raise ValueError(f"N must be one of {PREFIX_LENGTHS}.")
    packets = clean_packets(offsets_us[:n], sizes[:n], directions[:n])
    length = len(packets.sizes)
    features = np.zeros((3, n), dtype=np.float32)
    mask = np.zeros(n, dtype=np.bool_)
    mask[:length] = True
    features[0, :length] = packets.sizes
    features[1, :length] = packets.directions
    gaps_us = np.diff(packets.offsets_us, prepend=0)
    features[2, :length] = gaps_us / 1000.0
    return features, mask


@dataclass
class Bundle:
    features: np.ndarray
    mask: np.ndarray
    labels: np.ndarray
    flow_ids: np.ndarray
    split: str
    manifest_sha256: str
    import_id: str


def export_bundle(manifest_path: Path, bundle_path: Path) -> None:
    manifest = read_split(manifest_path)
    manifest_digest = sha256(manifest_path.read_bytes()).hexdigest()
    import_id = manifest[0].import_id
    expected = {row.flow_id: row for row in manifest}
    records = {split: [] for split in GROUPS}
    with connect() as client:
        completed_import_id(client, import_id)
        for flow_id, group, label, count, times, sizes, directions in prefix_rows(
            client, import_id
        ):
            row = expected.pop(flow_id, None)
            try:
                x, mask = encode_prefix(times, sizes, directions, MAX_PACKETS)
            except InvalidPackets:
                if row is not None:
                    raise ValueError(f"Saved flow has become invalid: {flow_id}") from None
                continue
            if row is None or (row.group_id, row.label, row.packet_count) != (group, label, count):
                raise ValueError(f"ClickHouse row does not match the fixed split: {flow_id}")
            if int(mask.sum()) != min(count, MAX_PACKETS):
                raise ValueError(f"Unexpected prefix length for {flow_id}")
            records[row.split].append((x, mask, CLASSES.index(label), flow_id))
    if expected:
        raise ValueError(f"Missing {len(expected)} flows from the fixed manifest.")

    bundle_path.mkdir(parents=True, exist_ok=True)
    for split, rows in records.items():
        save_bundle(bundle_path / f"{split}.npz", rows, split, manifest_digest, import_id)


def save_bundle(path: Path, rows: list, split: str, manifest_digest: str, import_id: str) -> None:
    features, masks, labels, flow_ids = zip(*rows, strict=True)
    with NamedTemporaryFile(dir=path.parent, prefix=".vnat-prefixes-", delete=False) as file:
        temporary = Path(file.name)
        try:
            np.savez_compressed(
                file,
                features=np.stack(features),
                mask=np.stack(masks),
                labels=np.asarray(labels, dtype=np.int64),
                flow_ids=np.asarray(flow_ids),
                split=split,
                manifest_sha256=manifest_digest,
                import_id=import_id,
                class_names=np.asarray(CLASSES),
            )
            file.close()
            temporary.replace(path)
        finally:
            file.close()
            temporary.unlink(missing_ok=True)


def load_bundle(directory: Path, split: str, manifest_path: Path = DEFAULT_MANIFEST) -> Bundle:
    if split not in GROUPS:
        raise ValueError(f"Unsupported split: {split}")
    with np.load(directory / f"{split}.npz", allow_pickle=False) as data:
        if tuple(data["class_names"]) != CLASSES:
            raise ValueError("Unexpected class mapping in the packet bundle.")
        bundle = Bundle(
            data["features"],
            data["mask"],
            data["labels"],
            data["flow_ids"],
            str(data["split"].item()),
            str(data["manifest_sha256"].item()),
            str(data["import_id"].item()),
        )
    rows = len(bundle.labels)
    if (
        bundle.features.shape != (rows, 3, MAX_PACKETS)
        or bundle.mask.shape != (rows, MAX_PACKETS)
        or bundle.flow_ids.shape != (rows,)
        or bundle.labels.shape != (rows,)
        or bundle.features.dtype != np.float32
        or bundle.mask.dtype != np.bool_
        or bundle.labels.dtype != np.int64
    ):
        raise ValueError("Unexpected array shapes in the packet bundle.")
    if (
        bundle.split != split
        or bundle.manifest_sha256 != sha256(manifest_path.read_bytes()).hexdigest()
    ):
        raise ValueError("Packet bundle does not match the requested split and manifest.")
    if not rows or (bundle.labels < 0).any() or (bundle.labels >= len(CLASSES)).any():
        raise ValueError("Empty packet bundle or invalid class labels.")
    if not bundle.mask[:, 0].all() or (~bundle.mask[:, :-1] & bundle.mask[:, 1:]).any():
        raise ValueError("The packet mask must be a contiguous nonempty prefix.")
    expected = [row for row in read_split(manifest_path) if row.split == split]
    if (
        not expected
        or bundle.import_id != expected[0].import_id
        or bundle.flow_ids.tolist() != [row.flow_id for row in expected]
        or bundle.labels.tolist() != [CLASSES.index(row.label) for row in expected]
        or bundle.mask.sum(axis=1).tolist()
        != [min(row.packet_count, MAX_PACKETS) for row in expected]
    ):
        raise ValueError("Packet bundle rows differ from the fixed manifest.")
    return bundle


class FlowDataset(Dataset):
    def __init__(self, bundle: Bundle, n: int):
        if n not in PREFIX_LENGTHS:
            raise ValueError(f"Unsupported N: {n}")
        self.features = torch.from_numpy(np.ascontiguousarray(bundle.features[:, :, :n]))
        self.mask = torch.from_numpy(np.ascontiguousarray(bundle.mask[:, :n]))
        self.labels = torch.from_numpy(bundle.labels)
        self.flow_ids = bundle.flow_ids

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.features[index], self.mask[index], self.labels[index]


def make_loader(dataset: Dataset, batch_size: int, *, shuffle: bool = False) -> DataLoader:
    generator = torch.Generator().manual_seed(SEED)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--n", type=int, choices=PREFIX_LENGTHS, default=32)
    args = parser.parse_args()
    export_bundle(args.manifest, args.output)
    print(f"Saved separate split files in {args.output}")
    for split in GROUPS:
        dataset = FlowDataset(load_bundle(args.output, split, args.manifest), args.n)
        short = int((dataset.mask.sum(dim=1) < args.n).sum())
        print(f"{split}: {len(dataset)} flows, {short} below {args.n} packets")
    train = FlowDataset(load_bundle(args.output, "train", args.manifest), args.n)
    x, mask, labels = next(iter(make_loader(train, 4)))
    print(
        f"First train batch: x={tuple(x.shape)}, mask={tuple(mask.shape)}, y={tuple(labels.shape)}"
    )


if __name__ == "__main__":
    main()
