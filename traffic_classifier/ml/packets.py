"""Validate a sequence of packet features and repair out-of-order timestamps."""

from collections.abc import Sequence
from dataclasses import dataclass


class InvalidPackets(ValueError):
    """The supplied packet sequence cannot be used."""


@dataclass
class CleanPackets:
    offsets_us: list[int]
    sizes: list[int]
    directions: list[int]
    repaired_timestamps: int


def clean_packets(
    offsets_us: Sequence[int],
    sizes: Sequence[int],
    directions: Sequence[int],
) -> CleanPackets:
    if len(offsets_us) == 0 or len(offsets_us) != len(sizes) or len(sizes) != len(directions):
        raise InvalidPackets("empty or mismatched packet arrays")
    if any(type(size) is not int or not 0 <= size <= 2**32 - 1 for size in sizes):
        raise InvalidPackets("invalid packet size")
    if any(type(direction) is not int or direction not in (0, 1) for direction in directions):
        raise InvalidPackets("invalid direction")
    if offsets_us[0] != 0 or any(type(offset) is not int for offset in offsets_us):
        raise InvalidPackets("invalid or nonrelative packet time")

    repaired = 0
    monotonic_offsets = []
    previous = 0
    for offset in offsets_us:
        if offset < previous:
            repaired += 1
            offset = previous
        monotonic_offsets.append(offset)
        previous = offset
    first_direction = directions[0]
    return CleanPackets(
        offsets_us=monotonic_offsets,
        sizes=list(sizes),
        directions=[int(direction == first_direction) for direction in directions],
        repaired_timestamps=repaired,
    )


def main() -> None:
    from collections import Counter, defaultdict

    from traffic_classifier.clickhouse_client import connect
    from traffic_classifier.dataset_report import completed_import_id

    counters = defaultdict(Counter)
    with connect() as client:
        import_id = completed_import_id(client)
        query = """
            SELECT traffic_class, arraySlice(arrival_offsets_us, 1, 64),
                   arraySlice(packet_sizes_bytes, 1, 64), arraySlice(directions, 1, 64),
                   arrayExists(delta -> delta < 0, arrayDifference(arrival_offsets_us))
            FROM flows WHERE import_id = {import_id:String}
        """
        with client.query_row_block_stream(query, parameters={"import_id": import_id}) as blocks:
            for block in blocks:
                for label, times, sizes, directions, original_decreases in block:
                    counts = counters[label]
                    counts["total"] += 1
                    counts["full_flow_with_time_decrease"] += original_decreases
                    try:
                        packets = clean_packets(times, sizes, directions)
                    except InvalidPackets as error:
                        counts[f"excluded: {error}"] += 1
                        continue
                    counts["kept"] += 1
                    counts["shorter_than_8"] += len(sizes) < 8
                    counts["prefix_with_repaired_time"] += packets.repaired_timestamps > 0
                    counts["repaired_timestamps_in_prefix"] += packets.repaired_timestamps

    print(f"Cleaning audit for import {import_id} (up to 64 packets)")
    for label, counts in sorted(counters.items()):
        print(label, dict(sorted(counts.items())))


if __name__ == "__main__":
    main()
