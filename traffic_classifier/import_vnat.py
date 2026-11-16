"""Load the three non-VPN VNAT classes into ClickHouse in batches."""

import argparse
import math
import os
import sys
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import file_digest, sha256
from pathlib import Path
from uuid import UUID, uuid4

from clickhouse_connect.driver.client import Client
from clickhouse_connect.driver.exceptions import ClickHouseError

from traffic_classifier.clickhouse_client import connect
from traffic_classifier.schema import create_schema
from traffic_classifier.vnat import (
    CAPTURE_COLUMN,
    PACKET_COLUMNS,
    TRAFFIC_CLASSES,
    capture_parts,
    read_vnat,
)
from traffic_classifier.vnat_groups import CLASSES, collect_captures, group_nonvpn

RELEASE_SHA256 = "5d0c3d76cd292f19e25b5229719264bc1ddd71920a20bb27a7dec6c7138914de"
FLOW_COLUMNS = [
    "import_id",
    "flow_id",
    "source_row",
    "capture_name",
    "capture_group",
    "traffic_class",
    "first_packet_at",
    "packet_count",
    "total_bytes",
    "arrival_offsets_us",
    "packet_sizes_bytes",
    "directions",
]
EVENT_COLUMNS = [
    "import_id",
    "attempt_id",
    "source_sha256",
    "source_name",
    "selection",
    "status",
    "flow_count",
    "packet_count",
    "total_bytes",
]
ROWS_PER_BATCH = 500
PACKETS_PER_BATCH = 250_000
LOCK_DIR = Path(__file__).resolve().parent.parent / "data"


class BadFlow(ValueError):
    """A malformed row that can be excluded from an otherwise valid import."""


@dataclass
class Totals:
    flows: int = 0
    packets: int = 0
    bytes: int = 0

    def as_tuple(self) -> tuple[int, int, int]:
        return self.flows, self.packets, self.bytes


@dataclass(frozen=True)
class Import:
    import_id: str
    attempt_id: UUID
    digest: str
    source_name: str
    selection: str


def source_import(path: Path, limit: int | None) -> Import:
    with path.open("rb") as dataset:
        digest = file_digest(dataset, "sha256").hexdigest()
    if digest != RELEASE_SHA256:
        raise ValueError("SHA-256 does not match the official VNAT sequence release.")
    selection = f"nonvpn-three-classes-v1:limit={limit if limit is not None else 'none'}"
    import_id = sha256(f"{digest}:{selection}".encode("ascii")).hexdigest()
    return Import(import_id, uuid4(), digest, path.name, selection)


def convert_packets(times: list, sizes: list, directions: list) -> tuple[int, list[int], int]:
    if not all(isinstance(values, list) for values in (times, sizes, directions)):
        raise BadFlow("packet columns must contain lists")
    if not times or len(times) != len(sizes) or len(times) != len(directions):
        raise BadFlow("empty or mismatched packet arrays")
    if any(not isinstance(time, (int, float)) or not math.isfinite(time) for time in times):
        raise BadFlow("invalid timestamp")
    if any(not isinstance(size, int) or not 0 <= size <= 2**32 - 1 for size in sizes):
        raise BadFlow("invalid packet size")
    if any(not isinstance(direction, int) or direction not in (0, 1) for direction in directions):
        raise BadFlow("invalid direction")
    first_us = round(times[0] * 1_000_000)
    offsets_us = [round(time * 1_000_000) - first_us for time in times]
    return first_us, offsets_us, sum(sizes)


def record_event(client: Client, run: Import, status: str, totals: Totals) -> None:
    event = [
        run.import_id,
        run.attempt_id,
        run.digest,
        run.source_name,
        run.selection,
        status,
        *totals.as_tuple(),
    ]
    client.insert("import_runs", [event], column_names=EVENT_COLUMNS)


def database_totals(client: Client, import_id: str) -> tuple[int, int, int]:
    return client.query(
        "SELECT count(), sum(length(packet_sizes_bytes)), sum(arraySum(packet_sizes_bytes)) "
        "FROM flows WHERE import_id = {import_id:String}",
        parameters={"import_id": import_id},
    ).first_row


@contextmanager
def import_lock(path: Path):
    """Serialize imports of the same selection on this host."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(
                "Another local process holds the import lock for this selection."
            ) from error
        yield


def verify_loaded(client: Client, run: Import, expected: tuple[int, int, int]) -> None:
    actual = database_totals(client, run.import_id)
    if actual != expected:
        raise RuntimeError(f"Import verification mismatch: expected {expected}, got {actual}")
    distinct = client.query(
        "SELECT uniqExact(flow_id) FROM flows WHERE import_id = {import_id:String}",
        parameters={"import_id": run.import_id},
    ).first_row[0]
    if distinct != actual[0]:
        raise RuntimeError(f"Duplicate flow_id in import {run.import_id}.")


def recover_or_skip(client: Client, run: Import) -> bool:
    parameters = {"import_id": run.import_id}
    completed = client.query(
        "SELECT flow_count, packet_count, total_bytes FROM import_runs "
        "WHERE import_id = {import_id:String} AND status = 'completed' "
        "ORDER BY recorded_at DESC, attempt_id DESC LIMIT 1",
        parameters=parameters,
    ).result_rows
    if completed:
        verify_loaded(client, run, completed[0])
        print(f"Import {run.import_id} is already complete. Verified totals and unique flow IDs.")
        return True

    if database_totals(client, run.import_id)[0]:
        started = client.query(
            "SELECT count() FROM import_runs "
            "WHERE import_id = {import_id:String} AND status = 'started'",
            parameters=parameters,
        ).first_row[0]
        if not started:
            raise RuntimeError(
                f"Existing rows for import {run.import_id} lack a matching start event."
            )
        client.command("ALTER TABLE flows DROP PARTITION {import_id:String}", parameters=parameters)
        if database_totals(client, run.import_id)[0]:
            raise RuntimeError(f"Incomplete partition {run.import_id} was not cleared.")
        print(
            f"Cleared the incomplete partition for {run.import_id}. Restarting the import.",
            flush=True,
        )
    return False


def load_flows(
    client: Client, path: Path, run: Import, limit: int | None, totals: Totals
) -> Counter[str]:
    frame = read_vnat(path)
    groups = group_nonvpn(collect_captures(frame, allow_invalid=True))
    expected_columns = [CAPTURE_COLUMN, *PACKET_COLUMNS]
    counts: Counter[str] = Counter()
    batch = []
    batch_packets = 0
    rows = frame[expected_columns].itertuples(index=False, name=None)
    for index, (name, times, sizes, directions) in enumerate(rows):
        key = capture_parts(name) if isinstance(name, str) else None
        if key is None or key[1] not in TRAFFIC_CLASSES:
            counts["invalid: unknown capture or application"] += 1
            continue
        mode, app, _ = key
        label = TRAFFIC_CLASSES[app]
        if mode != "nonvpn" or label not in CLASSES:
            counts["excluded by selection"] += 1
            continue
        try:
            start, offsets, total_bytes = convert_packets(times, sizes, directions)
        except BadFlow as error:
            counts[f"invalid: {error}"] += 1
            continue
        if batch and (
            len(batch) >= ROWS_PER_BATCH or batch_packets + len(sizes) > PACKETS_PER_BATCH
        ):
            client.insert("flows", batch, column_names=FLOW_COLUMNS)
            print(f"Inserted {totals.flows:,} flows, {totals.packets:,} packets.", flush=True)
            batch.clear()
            batch_packets = 0
        flow_id = sha256(f"{run.digest}:{index}".encode("ascii")).hexdigest()
        capture_name = name.replace("\\", "/").rsplit("/", 1)[-1]
        batch.append(
            [
                run.import_id,
                flow_id,
                index,
                capture_name,
                groups[key],
                label,
                start,
                len(sizes),
                total_bytes,
                offsets,
                sizes,
                directions,
            ]
        )
        batch_packets += len(sizes)
        totals.flows += 1
        totals.packets += len(sizes)
        totals.bytes += total_bytes
        if totals.flows == limit:
            break
    if batch:
        client.insert("flows", batch, column_names=FLOW_COLUMNS)
    return counts


def run_import(client: Client, path: Path, limit: int | None) -> None:
    run = source_import(path, limit)
    create_schema(client)
    with import_lock(LOCK_DIR / f"{run.import_id}.lock"):
        if not recover_or_skip(client, run):
            insert_new_import(client, path, run, limit)


def insert_new_import(client: Client, path: Path, run: Import, limit: int | None) -> None:
    totals = Totals()
    record_event(client, run, "started", totals)
    try:
        exclusions = load_flows(client, path, run, limit, totals)
        if not totals.flows:
            raise ValueError("No eligible flows were found.")
        verify_loaded(client, run, totals.as_tuple())
        record_event(client, run, "completed", totals)
    except (Exception, KeyboardInterrupt):
        try:
            record_event(client, run, "failed", totals)
        except (ClickHouseError, OSError):
            print("Could not record the failed attempt in ClickHouse.", file=sys.stderr)
        raise
    print(
        f"Import {run.import_id}: {totals.flows:,} flows, {totals.packets:,} packets, {totals.bytes:,} bytes."
    )
    print("Exclusions:", dict(sorted(exclusions.items())), "(scanned rows only)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--limit", type=int, help="stop after this number of eligible flows")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if not args.path.is_file():
        parser.error(f"VNAT file does not exist: {args.path}")
    try:
        with connect() as client:
            run_import(client, args.path, args.limit)
    except (ClickHouseError, OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"Could not import VNAT: {error}\n")


if __name__ == "__main__":
    main()
