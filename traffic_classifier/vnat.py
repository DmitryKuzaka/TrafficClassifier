"""Inspect the official VNAT packet dataframe without exposing flow endpoints."""

import argparse
import math
import re
from collections import Counter
from pathlib import Path

import pandas as pd

PACKET_COLUMNS = ("timestamps", "sizes", "directions")
CAPTURE_COLUMN = "file_names"
CAPTURE_NAME = re.compile(
    r"(?P<mode>nonvpn|vpn)_(?P<app>[a-z0-9-]+)_"
    r"(?P<variant>long_|new)?capture_?(?P<number>[0-9]+)\.pcap",
    re.IGNORECASE,
)
TRAFFIC_CLASSES = {
    "netflix": "video",
    "vimeo": "video",
    "youtube": "video",
    "voip": "voice",
    "sftp": "file_transfer",
    "rsync": "file_transfer",
    "scp": "file_transfer",
    "skype-chat": "chat",
    "ssh": "remote_access",
    "rdp": "remote_access",
}


def capture_parts(value: str) -> tuple[str, str, str] | None:
    """Get capture metadata from a publisher-supplied name."""
    name = value.replace("\\", "/").rsplit("/", 1)[-1]
    match = CAPTURE_NAME.fullmatch(name)
    if match is None:
        return None
    variant = (match["variant"] or "").lower()
    capture = f"{variant}capture{int(match['number'])}"
    return match["mode"].lower(), match["app"].lower(), capture


def read_vnat(path: Path) -> pd.DataFrame:
    """Read the sequence release from MIT Lincoln Laboratory."""
    # The fixed HDF5 layout stores object columns with pickle. Use the official file.
    return pd.read_hdf(path, key="/data", mode="r")


def inspect_vnat(frame: pd.DataFrame, examples: int) -> None:
    print(f"Rows: {len(frame)}")
    print("Columns and missing cells:")
    for name in frame.columns:
        column = frame[name]
        print(f"  {name!r}: dtype={column.dtype}, missing={column.isna().sum()}")

    required = {*PACKET_COLUMNS, CAPTURE_COLUMN}
    if missing := required.difference(frame.columns):
        raise ValueError(f"Unexpected VNAT schema. Missing columns: {sorted(missing)}")

    labels: Counter[str] = Counter()
    for name in frame[CAPTURE_COLUMN]:
        parts = capture_parts(name) if isinstance(name, str) else None
        label = TRAFFIC_CLASSES.get(parts[1], "unknown_app") if parts else "unknown_capture"
        labels[label] += 1
    print("Rows by class:", dict(sorted(labels.items())))

    lengths = []
    empty = 0
    mismatched = 0
    missing_arrays = 0
    missing_elements = Counter(dict.fromkeys(PACKET_COLUMNS, 0))
    sequence_types = {name: Counter() for name in PACKET_COLUMNS}
    direction_values = set()
    first_directions = Counter()
    printed_examples = 0
    for arrays in frame.loc[:, list(PACKET_COLUMNS)].itertuples(index=False, name=None):
        if any(not isinstance(values, list) for values in arrays):
            missing_arrays += 1
            continue
        times, sizes, directions = arrays
        lengths.append(len(times))
        empty += len(times) == 0
        mismatched += len(times) != len(sizes) or len(times) != len(directions)
        for name, values in zip(PACKET_COLUMNS, arrays):
            missing_elements[name] += int(pd.isna(values).sum())
            if len(values):
                sequence_types[name][f"{type(values).__name__}[{type(values[0]).__name__}]"] += 1
        direction_values.update(directions)
        if len(directions):
            first_directions[directions[0]] += 1
        if (
            not times
            or printed_examples >= examples
            or len(times) != len(sizes)
            or len(times) != len(directions)
        ):
            continue
        if any(not isinstance(time, (int, float)) or not math.isfinite(time) for time in times[:8]):
            continue
        printed_examples += 1
        print(f"Anonymized example {printed_examples} (up to 8 entries per channel):")
        print("  relative times (ms):", [round((time - times[0]) * 1000, 3) for time in times[:8]])
        print("  sizes (bytes):", sizes[:8])
        print("  directions:", directions[:8])

    if lengths:
        summary = pd.Series(lengths, dtype="int64").describe(percentiles=[0.5, 0.9, 0.99])
        print("Lengths from timestamps:", summary.to_dict())
        print("Rows below N:", {n: sum(length < n for length in lengths) for n in (8, 16, 32, 64)})
    print(f"Empty timestamp sequences: {empty}")
    print(f"Rows with mismatched sequence lengths: {mismatched}")
    print(f"Rows without all sequences: {missing_arrays}")
    print("Missing sequence elements:", dict(missing_elements))
    print(
        "Non-empty sequence types:", {name: dict(counts) for name, counts in sequence_types.items()}
    )
    print("Direction values:", sorted(direction_values, key=repr))
    print(
        "First packet directions:",
        dict(sorted(first_directions.items(), key=lambda item: repr(item[0]))),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="path to the official VNAT sequence release")
    parser.add_argument("--examples", type=int, default=3, choices=range(6))
    args = parser.parse_args()
    inspect_vnat(read_vnat(args.path), args.examples)


if __name__ == "__main__":
    main()
