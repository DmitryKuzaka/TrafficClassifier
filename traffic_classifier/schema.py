"""Create the ClickHouse tables used for raw flows and import status."""

from pathlib import Path

from clickhouse_connect.driver.client import Client

from traffic_classifier.clickhouse_client import connect

SQL_DIR = Path(__file__).resolve().parent.parent / "sql"


def create_schema(client: Client) -> None:
    for filename in ("flows.sql", "import_runs.sql"):
        client.command((SQL_DIR / filename).read_text(encoding="utf-8"))


def main() -> None:
    with connect() as client:
        create_schema(client)
        print(f"Applied table creation commands in {client.database}.")


if __name__ == "__main__":
    main()
