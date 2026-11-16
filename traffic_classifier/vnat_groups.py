"""Count VNAT sessions by capture and look for related recording windows."""

import argparse
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from traffic_classifier.vnat import CAPTURE_COLUMN, TRAFFIC_CLASSES, capture_parts, read_vnat

CLASSES = ("video", "voice", "file_transfer")
PREFIX_LENGTHS = (8, 16, 32, 64)


@dataclass
class Capture:
    flows: int = 0
    first: float = math.inf
    last: float = -math.inf
    reaches: Counter[int] = field(default_factory=Counter)

    def add(self, times: list[float]) -> None:
        if not isinstance(times, list) or not times:
            raise ValueError("Empty timestamp sequence in the capture audit.")
        if not all(isinstance(time, (int, float)) and math.isfinite(time) for time in times):
            raise ValueError("Non-finite capture time.")
        first, last = min(times), max(times)
        self.flows += 1
        self.first = min(self.first, first)
        self.last = max(self.last, last)
        for n in PREFIX_LENGTHS:
            self.reaches[n] += len(times) >= n


def collect_captures(
    frame: pd.DataFrame, *, allow_invalid: bool = False
) -> dict[tuple[str, str, str], Capture]:
    captures: dict[tuple[str, str, str], Capture] = {}
    for name, times in frame[[CAPTURE_COLUMN, "timestamps"]].itertuples(index=False, name=None):
        key = capture_parts(name) if isinstance(name, str) else None
        if key is None or key[1] not in TRAFFIC_CLASSES:
            if allow_invalid:
                continue
            raise ValueError("Unrecognized capture name or application.")
        if TRAFFIC_CLASSES[key[1]] not in CLASSES:
            continue
        capture = captures.get(key)
        if capture is None:
            capture = Capture()
        try:
            capture.add(times)
        except ValueError:
            if allow_invalid:
                continue
            raise
        captures[key] = capture
    return captures


def group_nonvpn(captures: dict[tuple[str, str, str], Capture]) -> dict[tuple[str, str, str], str]:
    """Group same-app non-VPN captures whose time windows touch or overlap."""
    by_app: dict[str, list[tuple[tuple[str, str, str], Capture]]] = defaultdict(list)
    for key, capture in captures.items():
        mode, app, _ = key
        if mode == "nonvpn":
            by_app[app].append((key, capture))

    assignments = {}
    for app, records in sorted(by_app.items()):
        group_end = -math.inf
        for key, capture in sorted(records, key=lambda pair: (pair[1].first, pair[0])):
            if capture.first > group_end:
                group_id = f"{app}:{key[2]}"
                group_end = capture.last
            else:
                group_end = max(group_end, capture.last)
            assignments[key] = group_id
    return assignments


def report(captures: dict[tuple[str, str, str], Capture]) -> None:
    groups = group_nonvpn(captures)
    print(
        "class | all flows | all captures | nonvpn flows | nonvpn captures | nonvpn groups | nonvpn reaches 8/16/32/64"
    )
    for label in CLASSES:
        selected = {
            key: stats for key, stats in captures.items() if TRAFFIC_CLASSES[key[1]] == label
        }
        nonvpn = {key: stats for key, stats in selected.items() if key in groups}
        group_ids = {groups[key] for key in nonvpn}
        reaches = [sum(stats.reaches[n] for stats in nonvpn.values()) for n in PREFIX_LENGTHS]
        print(
            label,
            sum(stats.flows for stats in selected.values()),
            len(selected),
            sum(stats.flows for stats in nonvpn.values()),
            len(nonvpn),
            len(group_ids),
            reaches,
            sep=" | ",
        )

    print("\nNon-VPN group membership (flows and flows reaching 64):")
    for key, group_id in sorted(groups.items()):
        capture = captures[key]
        print("/".join(key[1:]), group_id, capture.flows, capture.reaches[64])

    print("\nSame-name VPN/non-VPN pairs and time overlap:")
    for (mode, app, name), vpn in sorted(captures.items()):
        peer = captures.get(("nonvpn", app, name))
        if mode == "vpn" and peer is not None:
            overlap = max(vpn.first, peer.first) <= min(vpn.last, peer.last)
            print(f"{app}/{name}: overlap={overlap}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    report(collect_captures(read_vnat(args.path)))


if __name__ == "__main__":
    main()
