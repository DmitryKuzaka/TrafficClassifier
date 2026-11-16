"""Print analytical reports for a completed VNAT import."""

import argparse

from clickhouse_connect.driver.client import Client

from traffic_classifier.clickhouse_client import connect
from traffic_classifier.schema import SQL_DIR

REPORTS = ("by_class", "lengths", "by_capture", "by_start_hour_utc")


def completed_import_id(client: Client, requested: str | None = None) -> str:
    condition = (
        "import_id = {requested:String}"
        if requested
        else "selection = 'nonvpn-three-classes-v1:limit=none'"
    )
    parameters = {"requested": requested} if requested else None
    rows = client.query(
        "SELECT toString(import_id) FROM import_runs "
        f"WHERE status = 'completed' AND {condition} "
        "ORDER BY recorded_at DESC, import_id DESC LIMIT 1",
        parameters=parameters,
    ).result_rows
    if not rows:
        raise ValueError("No completed import matches the request. Run the VNAT importer first.")
    return rows[0][0]


def display_report(client: Client, import_id: str, report: str) -> None:
    query = (SQL_DIR / "dataset" / f"{report}.sql").read_text(encoding="utf-8")
    result = client.query(query, parameters={"import_id": import_id})
    print(f"\n{report}")
    print("\t".join(result.column_names))
    for row in result.result_rows:
        print(
            "\t".join(f"{value:.6f}" if isinstance(value, float) else str(value) for value in row)
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--import-id", help="use a specific completed import")
    parser.add_argument("--section", choices=REPORTS, help="print only one report")
    args = parser.parse_args()
    with connect() as client:
        try:
            import_id = completed_import_id(client, args.import_id)
        except ValueError as error:
            parser.exit(1, f"{error}\n")
        print(f"VNAT true labels, import {import_id}")
        for report in (args.section,) if args.section else REPORTS:
            display_report(client, import_id, report)


if __name__ == "__main__":
    main()
