import hashlib

import pandas as pd
import pytest

from traffic_classifier import import_vnat
from traffic_classifier.import_vnat import BadFlow, convert_packets
from traffic_classifier.vnat import capture_parts
from traffic_classifier.vnat_groups import Capture, group_nonvpn


def test_conversion_preserves_packet_order():
    actual = convert_packets([1.25, 1.2501, 1.2499], [120, 63, 0], [1, 0, 1])
    assert actual == (1_250_000, [0, 100, -100], 183)


@pytest.mark.parametrize(
    ("times", "sizes", "directions", "reason"),
    (
        ([], [], [], "empty or mismatched"),
        ([1.0], [10, 20], [1], "empty or mismatched"),
        ([1.0, float("nan")], [10, 20], [1, 0], "invalid timestamp"),
        ([1.0, 2.0], [10, -1], [1, 0], "invalid packet size"),
        ([1.0, 2.0], [10, 20], [1, 2], "invalid direction"),
    ),
)
def test_conversion_reports_bad_input(times, sizes, directions, reason):
    with pytest.raises(BadFlow, match=reason):
        convert_packets(times, sizes, directions)


def test_capture_variants_and_transitive_overlap():
    first = ("nonvpn", "scp", "capture1")
    second = ("nonvpn", "scp", "long_capture1")
    third = ("nonvpn", "scp", "newcapture1")
    separate = ("nonvpn", "scp", "capture2")
    assert capture_parts(r"folder\nonvpn_scp_long_capture1.pcap") == second
    assert capture_parts("nonvpn_scp_newcapture1.pcap") == third
    assert capture_parts("nonvpn_scp_capture_2.pcap") == separate

    groups = group_nonvpn(
        {
            separate: Capture(first=20, last=22),
            second: Capture(first=4, last=7),
            first: Capture(first=0, last=5),
            third: Capture(first=7, last=12),
        }
    )
    assert set(groups.values()) == {"scp:capture1", "scp:capture2"}
    assert groups[first] == groups[second] == groups[third]


def test_lock_recovers_after_release(tmp_path):
    path = tmp_path / "same-import.lock"
    with (
        import_vnat.import_lock(path),
        pytest.raises(RuntimeError, match="Another local process"),
        import_vnat.import_lock(path),
    ):
        pytest.fail("Concurrent holder unexpectedly acquired the lock.")
    with import_vnat.import_lock(path):
        assert path.exists()


@pytest.mark.clickhouse
def test_import_metadata_and_repeat_on_temporary_database(tmp_path, monkeypatch):
    from uuid import uuid4

    from traffic_classifier.clickhouse_client import connect

    source = tmp_path / "synthetic.h5"
    source.write_bytes(b"small test fixture")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    frame = pd.DataFrame(
        {
            "file_names": [
                "nonvpn_youtube_capture1.pcap",
                "nonvpn_voip_capture1.pcap",
                "nonvpn_scp_capture1.pcap",
            ],
            "timestamps": [[1.0, 1.1], [2.0], [3.0]],
            "sizes": [[11, 12], [75], [100, 200]],
            "directions": [[1, 0], [1], [1]],
        }
    )
    monkeypatch.setattr(import_vnat, "RELEASE_SHA256", digest)
    monkeypatch.setattr(import_vnat, "LOCK_DIR", tmp_path)
    monkeypatch.setattr(import_vnat, "read_vnat", lambda _: frame)

    database = "tc_test_" + uuid4().hex
    with connect() as admin:
        admin.command(f"CREATE DATABASE {database}")
        try:
            monkeypatch.setenv("CLICKHOUSE_DATABASE", database)
            with connect() as client:
                run = import_vnat.source_import(source, None)
                import_vnat.run_import(client, source, None)
                assert import_vnat.database_totals(client, run.import_id) == (2, 3, 98)
                import_vnat.run_import(client, source, None)
                assert import_vnat.database_totals(client, run.import_id) == (2, 3, 98)
                assert client.query(
                    "SELECT count() FROM import_runs WHERE status = 'completed'"
                ).first_row == (1,)
        finally:
            admin.command(f"DROP DATABASE {database}")
