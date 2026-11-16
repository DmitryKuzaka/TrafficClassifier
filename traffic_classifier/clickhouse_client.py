"""Connect to the project's ClickHouse server and check readiness."""

import os
from pathlib import Path

from clickhouse_connect import get_client
from clickhouse_connect.driver.client import Client
from dotenv import load_dotenv


def connect() -> Client:
    project_dir = Path(__file__).resolve().parent.parent
    load_dotenv(project_dir / ".env", interpolate=False)
    password = os.getenv("CLICKHOUSE_PASSWORD")
    if not password:
        raise RuntimeError("Set CLICKHOUSE_PASSWORD in .env or in the process environment.")

    port = int(os.getenv("CLICKHOUSE_PORT", "8123"))
    if not 1 <= port <= 65535:
        raise ValueError("CLICKHOUSE_PORT must be in the range 1..65535.")

    return get_client(
        host=os.getenv("CLICKHOUSE_HOST", "127.0.0.1"),
        port=port,
        database=os.getenv("CLICKHOUSE_DATABASE", "traffic_classifier"),
        username=os.getenv("CLICKHOUSE_USER", "traffic_classifier"),
        password=password,
        connect_timeout=5,
        send_receive_timeout=30,
    )


def main() -> None:
    with connect() as client:
        if client.query("SELECT 1").first_row != (1,):
            raise RuntimeError("Unexpected SELECT 1 result.")
        print(f"ClickHouse {client.server_version}: SELECT 1 = 1")


if __name__ == "__main__":
    main()
