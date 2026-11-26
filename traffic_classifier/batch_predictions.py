"""Apply the saved CNN to historical ClickHouse flows in restartable batches."""

import argparse
import json
from dataclasses import dataclass
from hashlib import sha256
from itertools import batched

import torch
from clickhouse_connect.driver.client import Client
from clickhouse_connect.driver.exceptions import ClickHouseError

from traffic_classifier.clickhouse_client import connect
from traffic_classifier.dataset_report import completed_import_id
from traffic_classifier.import_vnat import LOCK_DIR, import_lock
from traffic_classifier.inference import PacketInput, Predictor, load_predictor
from traffic_classifier.schema import SQL_DIR
from traffic_classifier.vnat_groups import CLASSES

BATCH_SIZE = 256
RUN_COLUMNS = ("run_id", "import_id", "model_id", "n", "max_flows", "flow_count", "status")
PREDICTION_COLUMNS = (
    "run_id",
    "import_id",
    "flow_id",
    "model_id",
    "n",
    "observed_packets",
    "predicted_class",
    "score_video",
    "score_voice",
    "score_file_transfer",
)
PREFIX_QUERY = """
    SELECT toString(flow_id), arraySlice(arrival_offsets_us, 1, {n:UInt8}),
           arraySlice(packet_sizes_bytes, 1, {n:UInt8}), arraySlice(directions, 1, {n:UInt8})
    FROM flows WHERE import_id = {import_id:String} ORDER BY flow_id
"""


@dataclass(frozen=True)
class PredictionRun:
    run_id: str
    import_id: str
    model_id: str
    n: int
    limit: int | None
    expected: int

    def event(self, status: str) -> tuple:
        return (
            self.run_id,
            self.import_id,
            self.model_id,
            self.n,
            self.limit,
            self.expected,
            status,
        )


def create_prediction_schema(client: Client) -> None:
    for name in ("predictions.sql", "prediction_runs.sql"):
        client.command((SQL_DIR / name).read_text(encoding="utf-8"))


def identify_run(
    client: Client, predictor: Predictor, import_id: str | None, limit: int | None
) -> PredictionRun:
    if limit is not None and not 1 <= limit < 2**64:
        raise ValueError("Limit must be a positive UInt64.")
    import_id = completed_import_id(client, import_id)
    source_count, unique = client.query(
        "SELECT count(), uniqExact(flow_id) FROM flows WHERE import_id = {import_id:String}",
        parameters={"import_id": import_id},
    ).first_row
    if not source_count or unique != source_count:
        raise ValueError("The source import is empty or contains duplicate flows.")
    expected = min(limit, source_count) if limit is not None else source_count
    key = json.dumps(["prediction-v1", import_id, predictor.model_id, predictor.n, limit])
    run_id = sha256(key.encode("utf-8")).hexdigest()
    return PredictionRun(run_id, import_id, predictor.model_id, predictor.n, limit, expected)


def verify_run(client: Client, run: PredictionRun) -> None:
    rows, unique, mismatched = client.query(
        """
        SELECT count(), uniqExact(flow_id),
               countIf(import_id != {import_id:String} OR model_id != {model_id:String} OR n != {n:UInt8})
        FROM predictions WHERE run_id = {run_id:String}
        """,
        parameters={
            "run_id": run.run_id,
            "import_id": run.import_id,
            "model_id": run.model_id,
            "n": run.n,
        },
    ).first_row
    if (rows, unique, mismatched) != (run.expected, run.expected, 0):
        raise RuntimeError(f"Prediction run {run.run_id} has inconsistent rows or metadata.")


def recovered_or_complete(client: Client, run: PredictionRun) -> bool:
    params = {"run_id": run.run_id}
    events = client.query(
        "SELECT status FROM prediction_runs WHERE run_id = {run_id:String}",
        parameters=params,
    ).result_rows
    if ("completed",) in events:
        verify_run(client, run)
        return True
    (retained,) = client.query(
        "SELECT count() FROM predictions WHERE run_id = {run_id:String}",
        parameters=params,
    ).first_row
    if retained:
        if ("started",) not in events:
            raise RuntimeError(f"Prediction rows for {run.run_id} have no start event.")
        client.command("ALTER TABLE predictions DROP PARTITION {run_id:String}", parameters=params)
        (remaining,) = client.query(
            "SELECT count() FROM predictions WHERE run_id = {run_id:String}",
            parameters=params,
        ).first_row
        if remaining:
            raise RuntimeError(f"Could not clear incomplete prediction run {run.run_id}.")
    return False


def predict_rows(reader: Client, writer: Client, predictor: Predictor, run: PredictionRun) -> None:
    query = PREFIX_QUERY
    params = {"import_id": run.import_id, "n": run.n}
    if run.limit is not None:
        query += " LIMIT {limit:UInt64}"
        params["limit"] = run.limit
    with reader.query_row_block_stream(query, parameters=params) as blocks:
        for block in blocks:
            for rows in batched(block, BATCH_SIZE):
                inputs = [
                    PacketInput(offsets, sizes, directions)
                    for _, offsets, sizes, directions in rows
                ]
                answers = predictor.predict_many(inputs)
                writer.insert(
                    "predictions",
                    [
                        (
                            run.run_id,
                            run.import_id,
                            row[0],
                            answer.model_id,
                            answer.n,
                            answer.observed_packets,
                            answer.category,
                            *(answer.scores[label] for label in CLASSES),
                        )
                        for row, answer in zip(rows, answers, strict=True)
                    ],
                    column_names=PREDICTION_COLUMNS,
                )


def run_batch(
    reader: Client,
    writer: Client,
    predictor: Predictor,
    import_id: str | None = None,
    limit: int | None = None,
) -> PredictionRun:
    run = identify_run(reader, predictor, import_id, limit)
    with import_lock(LOCK_DIR / f"prediction-{run.run_id}.lock"):
        create_prediction_schema(writer)
        if recovered_or_complete(writer, run):
            print(f"Run {run.run_id} already complete ({run.expected} flows).")
            return run
        writer.insert("prediction_runs", [run.event("started")], column_names=RUN_COLUMNS)
        predict_rows(reader, writer, predictor, run)
        verify_run(writer, run)
        writer.insert("prediction_runs", [run.event("completed")], column_names=RUN_COLUMNS)
    print(f"Completed run {run.run_id} ({run.expected} flows).")
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--import-id", help="use a specific completed import")
    parser.add_argument(
        "--limit", type=int, help="process the first given number of flows by flow ID"
    )
    args = parser.parse_args()
    torch.set_num_threads(2)
    try:
        predictor = load_predictor()
        with connect() as reader, connect() as writer:
            run_batch(reader, writer, predictor, args.import_id, args.limit)
    except (ClickHouseError, OSError, RuntimeError, ValueError, KeyError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
