import hashlib
from dataclasses import replace
from uuid import uuid4

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from traffic_classifier import batch_predictions, import_vnat
from traffic_classifier.api import app as api
from traffic_classifier.clickhouse_client import connect
from traffic_classifier.inference import Predictor
from traffic_classifier.ml.cnn import PacketCNN
from traffic_classifier.ml.model_bundle import ModelBundle
from traffic_classifier.ml.normalization import Normalizer
from traffic_classifier.prediction_statistics import RunNotFound, prediction_statistics


def test_statistics_reports_unavailable_database(monkeypatch):
    def unavailable():
        raise OSError("temporary database outage")

    monkeypatch.setattr(api, "connect", unavailable)
    with TestClient(api.create_app(lambda: None)) as client:
        response = client.get("/api/v1/statistics")
        assert response.status_code == 503
        assert response.json()["detail"] == "ClickHouse statistics are unavailable."
        assert client.get("/api/v1/statistics", params={"n": 9}).status_code == 422


def make_predictor(n):
    return Predictor(
        ModelBundle(
            PacketCNN().eval(), Normalizer(n, (0.0, 0.0), (1.0, 1.0), "i", "s"), f"model-{n}"
        )
    )


@pytest.mark.clickhouse
def test_import_predict_recover_and_report(tmp_path, monkeypatch):
    source = tmp_path / "small.h5"
    source.write_bytes(b"test vnats")
    monkeypatch.setattr(
        import_vnat, "RELEASE_SHA256", hashlib.sha256(source.read_bytes()).hexdigest()
    )
    frame = pd.DataFrame(
        {
            "file_names": [
                "nonvpn_youtube_capture1.pcap",
                "nonvpn_voip_capture1.pcap",
                "nonvpn_scp_capture1.pcap",
            ],
            "timestamps": [[1.0, 1.01], [2.0], [3601.0, 3601.01, 3601.02]],
            "sizes": [[100, 200], [300], [400, 500, 600]],
            "directions": [[1, 0], [1], [1, 0, 1]],
        }
    )
    monkeypatch.setattr(import_vnat, "read_vnat", lambda _: frame)
    monkeypatch.setattr(import_vnat, "LOCK_DIR", tmp_path)
    monkeypatch.setattr(batch_predictions, "LOCK_DIR", tmp_path)
    database = "tc_test_" + uuid4().hex
    with connect() as admin:
        admin.command(f"CREATE DATABASE {database}")
        try:
            monkeypatch.setenv("CLICKHOUSE_DATABASE", database)
            with connect() as reader, connect() as writer:
                import_vnat.run_import(writer, source, None)
                predictor = make_predictor(8)
                predict_rows = batch_predictions.predict_rows

                def interrupted(reader, writer, predictor, run):
                    predict_rows(reader, writer, predictor, replace(run, limit=1))
                    raise RuntimeError("interrupted before completion")

                with monkeypatch.context() as scoped:
                    scoped.setattr(batch_predictions, "predict_rows", interrupted)
                    with pytest.raises(RuntimeError, match="interrupted before completion"):
                        batch_predictions.run_batch(reader, writer, predictor)
                with pytest.raises(RunNotFound):
                    prediction_statistics(reader)

                run = batch_predictions.run_batch(reader, writer, predictor)
                expected = prediction_statistics(reader, run_id=run.run_id)
                assert batch_predictions.run_batch(reader, writer, predictor) == run
                another = batch_predictions.run_batch(reader, writer, make_predictor(16))
                assert another.run_id != run.run_id
                assert expected["totals"] == {
                    "flows": 3,
                    "packets": 6,
                    "observed_packets": 6,
                    "bytes": 2100,
                }
                assert len(expected["bytes_by_start_hour_utc"]) == 2
                assert prediction_statistics(reader, n=8) == expected
                assert (
                    prediction_statistics(reader, model_id="model-16")["totals"]
                    == expected["totals"]
                )

            with TestClient(api.create_app(lambda: predictor)) as client:
                assert (
                    client.get("/api/v1/statistics", params={"run_id": run.run_id}).json()
                    == expected
                )
                assert (
                    client.get("/api/v1/statistics", params={"run_id": "0" * 64}).status_code == 404
                )
                assert (
                    client.get("/api/v1/statistics", params={"run_id": "malformed"}).status_code
                    == 422
                )
        finally:
            admin.command(f"DROP DATABASE {database}")
