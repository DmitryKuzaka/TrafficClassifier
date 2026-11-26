"""Read aggregates for a single completed historical prediction run."""

from datetime import UTC

from clickhouse_connect.driver.client import Client

from traffic_classifier.schema import SQL_DIR
from traffic_classifier.vnat_groups import CLASSES, PREFIX_LENGTHS


class RunNotFound(ValueError):
    """There is no completed run for the filters."""


def prediction_statistics(
    client: Client, *, run_id: str | None = None, model_id: str | None = None, n: int | None = None
) -> dict:
    if n is not None and n not in PREFIX_LENGTHS:
        raise ValueError(f"Expected packet count in {PREFIX_LENGTHS}.")
    if not client.command("EXISTS TABLE prediction_runs"):
        raise RunNotFound("No completed prediction runs. Run the batch prediction command first.")
    conditions = ["status = 'completed'"]
    params = {}
    for column, kind, value in (
        ("run_id", "String", run_id),
        ("model_id", "String", model_id),
        ("n", "UInt8", n),
    ):
        if value is not None:
            conditions.append(f"{column} = {{{column}:{kind}}}")
            params[column] = value

    candidates = client.query(
        "SELECT toString(run_id), toString(import_id), model_id, n, flow_count "
        "FROM prediction_runs WHERE "
        + " AND ".join(conditions)
        + " ORDER BY recorded_at DESC, run_id DESC LIMIT 1",
        parameters=params,
    ).result_rows
    if not candidates:
        raise RunNotFound("No completed prediction run matches the filters.")
    selected, source, model, prefix, expected = candidates[0]
    query = (SQL_DIR / "prediction_statistics.sql").read_text(encoding="utf-8")
    aggregate_rows = client.query(query, parameters={"run_id": selected}).result_rows

    by_category = {
        label: {"flows": 0, "packets": 0, "observed_packets": 0, "bytes": 0} for label in CLASSES
    }
    by_hour = {}
    index = {label: idx for idx, label in enumerate(CLASSES)}
    confusion = [[0] * len(CLASSES) for _ in CLASSES]
    for hour, predicted, truth, count, packets, observed, size in aggregate_rows:
        by_category[predicted]["flows"] += count
        by_category[predicted]["packets"] += packets
        by_category[predicted]["observed_packets"] += observed
        by_category[predicted]["bytes"] += size
        stamp = hour.replace(tzinfo=UTC).isoformat()
        by_hour.setdefault(stamp, dict.fromkeys(CLASSES, 0))[predicted] += size
        confusion[index[truth]][index[predicted]] += count

    totals = {
        field: sum(category[field] for category in by_category.values())
        for field in ("flows", "packets", "observed_packets", "bytes")
    }
    if not expected or totals["flows"] != expected:
        raise RuntimeError(f"Joined flows do not match completed prediction run {selected}.")
    correct = sum(confusion[i][i] for i in range(len(CLASSES)))
    return {
        "run_id": selected,
        "import_id": source,
        "model_id": model,
        "n": prefix,
        "source": "historical_vnat",
        "bytes_basis": "full_flow",
        "period_first_hour_utc": next(iter(by_hour)),
        "period_last_hour_utc": next(reversed(by_hour)),
        "totals": totals,
        "by_predicted_category": by_category,
        "bytes_by_start_hour_utc": by_hour,
        "known_labels": {
            "correct": correct,
            "accuracy": correct / expected,
            "class_order": list(CLASSES),
            "confusion_matrix": confusion,
        },
    }
