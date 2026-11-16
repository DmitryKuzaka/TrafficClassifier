"""Record the fixed capture-group splits for the audited VNAT import."""

import argparse
import csv
import json
from collections import Counter
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from hashlib import sha256
from io import StringIO
from pathlib import Path

from clickhouse_connect.driver.client import Client

from traffic_classifier.clickhouse_client import connect
from traffic_classifier.dataset_report import completed_import_id
from traffic_classifier.ml.packets import InvalidPackets, clean_packets
from traffic_classifier.vnat import TRAFFIC_CLASSES
from traffic_classifier.vnat_groups import CLASSES

DEFAULT_MANIFEST = Path("artifacts/vnat-split-v1.csv")
SEED = 42
GROUPS = {
    "train": (
        "netflix:capture1",
        "vimeo:capture1",
        "youtube:capture1",
        "voip:capture1",
        "rsync:newcapture1",
        "scp:long_capture1",
        "sftp:capture3",
        "sftp:newcapture1",
    ),
    "validation": (
        "youtube:capture2",
        "youtube:capture3",
        "voip:capture2",
        "rsync:capture1",
        "scp:newcapture1",
        "sftp:capture1",
    ),
    "test": (
        "netflix:capture2",
        "youtube:capture4",
        "voip:capture3",
        "scp:capture1",
        "sftp:capture2",
        "sftp:newcapture2",
    ),
}

PREFIX_QUERY = """
    SELECT toString(flow_id), capture_group, traffic_class, packet_count,
           arraySlice(arrival_offsets_us, 1, 64),
           arraySlice(packet_sizes_bytes, 1, 64),
           arraySlice(directions, 1, 64)
    FROM flows WHERE import_id = {import_id:String}
    ORDER BY flow_id
"""


@dataclass(frozen=True)
class SplitRow:
    import_id: str
    flow_id: str
    group_id: str
    label: str
    split: str
    seed: int
    packet_count: int


def prefix_rows(client: Client, import_id: str) -> Iterator[tuple]:
    with client.query_row_block_stream(PREFIX_QUERY, parameters={"import_id": import_id}) as blocks:
        for block in blocks:
            yield from block


def assign_groups() -> dict[str, str]:
    assignments = {}
    for split, groups in GROUPS.items():
        for group in groups:
            if group in assignments:
                raise ValueError(f"Capture group appears twice: {group}")
            assignments[group] = split
    return assignments


def validate_splits(rows: list[SplitRow]) -> None:
    seen_flows = {row.flow_id for row in rows}
    if len(seen_flows) != len(rows):
        raise ValueError("Duplicate flow_id in split manifest.")
    assignments = assign_groups()
    if {row.group_id for row in rows} != set(assignments):
        raise ValueError("Observed capture groups differ from the P06 protocol.")
    expected = {(split, label) for split in GROUPS for label in CLASSES}
    if {(row.split, row.label) for row in rows} != expected:
        raise ValueError("One or more splits lack a target class.")
    for row in rows:
        expected_label = TRAFFIC_CLASSES[row.group_id.partition(":")[0]]
        if row.split != assignments[row.group_id] or row.label != expected_label:
            raise ValueError(f"Incorrect group assignment for {row.flow_id}.")
    if len({row.import_id for row in rows}) != 1:
        raise ValueError("Manifest rows refer to different imports.")
    if any(row.seed != SEED or row.packet_count <= 0 for row in rows):
        raise ValueError("Manifest has an unexpected seed or packet count.")


def build_split(client: Client, import_id: str) -> list[SplitRow]:
    assignments = assign_groups()
    rows = []
    excluded = Counter()
    for flow_id, group, label, count, times, sizes, directions in prefix_rows(client, import_id):
        if group not in assignments:
            raise ValueError(f"Unexpected group: {group}")
        try:
            clean_packets(times, sizes, directions)
        except InvalidPackets as error:
            excluded[(label, str(error))] += 1
            continue
        rows.append(SplitRow(import_id, flow_id, group, label, assignments[group], SEED, count))
    if excluded:
        print("Excluded from the manifest:", dict(excluded))
    validate_splits(rows)
    return rows


def save_if_unchanged(path: Path, content: str) -> None:
    if path.exists():
        if path.read_bytes() != content.encode("utf-8"):
            raise ValueError(f"Saved result differs from current result: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8", newline="") as file:
            file.write(content)


def summarize_split(rows: list[SplitRow], digest: str) -> dict:
    counts = Counter((row.split, row.label) for row in rows)
    long_counts = Counter((row.split, row.label) for row in rows if row.packet_count >= 64)
    return {
        "import_id": rows[0].import_id,
        "seed": SEED,
        "manifest_sha256": digest,
        "row_count": len(rows),
        "counts": {
            split: {
                label: {"flows": counts[split, label], "at_least_64": long_counts[split, label]}
                for label in CLASSES
            }
            for split in GROUPS
        },
    }


def save_split(path: Path, rows: list[SplitRow]) -> str:
    buffer = StringIO(newline="")
    writer = csv.DictWriter(
        buffer, fieldnames=list(SplitRow.__dataclass_fields__), lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(asdict(row) for row in rows)
    content = buffer.getvalue()
    digest = sha256(content.encode("utf-8")).hexdigest()
    save_if_unchanged(path, content)
    summary = json.dumps(summarize_split(rows, digest), indent=2, sort_keys=True)
    save_if_unchanged(path.with_name(f"{path.name}.summary.json"), summary + "\n")
    return digest


def read_split(path: Path) -> list[SplitRow]:
    content = path.read_bytes()
    with StringIO(content.decode("utf-8"), newline="") as file:
        rows = [
            SplitRow(
                row["import_id"],
                row["flow_id"],
                row["group_id"],
                row["label"],
                row["split"],
                int(row["seed"]),
                int(row["packet_count"]),
            )
            for row in csv.DictReader(file)
        ]
    validate_splits(rows)
    metadata = json.loads(path.with_name(f"{path.name}.summary.json").read_text(encoding="utf-8"))
    if metadata != summarize_split(rows, sha256(content).hexdigest()):
        raise ValueError(f"Split manifest does not match its saved summary: {path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    with connect() as client:
        rows = build_split(client, completed_import_id(client))
    digest = save_split(args.output, rows)
    counts = Counter((row.split, row.label) for row in rows)
    long_counts = Counter((row.split, row.label) for row in rows if row.packet_count >= 64)
    for (split, label), count in sorted(counts.items()):
        print(f"{split} / {label}: {count}, at least 64 packets: {long_counts[split, label]}")
    print(f"{args.output}: {len(rows)} rows, SHA-256 {digest}")


if __name__ == "__main__":
    main()
