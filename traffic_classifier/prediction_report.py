"""Export a simple self-contained report from ClickHouse aggregates."""

import argparse
from base64 import b64encode
from datetime import datetime
from html import escape
from io import BytesIO
from pathlib import Path

from clickhouse_connect.driver.exceptions import ClickHouseError
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.dates import DateFormatter
from matplotlib.figure import Figure

from traffic_classifier.clickhouse_client import connect
from traffic_classifier.prediction_statistics import prediction_statistics

TITLES = {"video": "Видео", "voice": "Звонки", "file_transfer": "Файлы"}
COLORS = ("#1959bd", "#6f489b", "#a74620")


def charts(report: dict) -> str:
    figure = Figure(figsize=(10, 3.9), layout="constrained")
    FigureCanvasAgg(figure)
    categories, timeline = figure.subplots(1, 2)
    labels = list(TITLES)
    category_bytes = report["by_predicted_category"]
    categories.bar(
        list(TITLES.values()),
        [category_bytes[label]["bytes"] / 1_000_000 for label in labels],
        color=COLORS,
    )
    categories.set(title="Объём по предсказанной категории", ylabel="МБ")
    hourly = report["bytes_by_start_hour_utc"]
    for label, color in zip(labels, COLORS, strict=True):
        active_hours = [
            (datetime.fromisoformat(hour), row[label]) for hour, row in hourly.items() if row[label]
        ]
        timeline.scatter(
            [hour for hour, _ in active_hours],
            [size / 1_000_000 for _, size in active_hours],
            s=18,
            color=color,
            label=TITLES[label],
        )
    timeline.set(title="Объём по часу начала потока", ylabel="МБ", xlabel="UTC")
    timeline.xaxis.set_major_formatter(DateFormatter("%d.%m %H:%M"))
    timeline.tick_params(axis="x", labelrotation=25)
    timeline.legend()
    figure.savefig(image := BytesIO(), format="png", dpi=145)
    return b64encode(image.getvalue()).decode("ascii")


def write_html(report: dict, output: Path) -> None:
    totals = report["totals"]
    quality = report["known_labels"]
    preview = charts(report)
    heading = "".join(f"<th>{title}</th>" for title in TITLES.values())
    matrix = "".join(
        f"<tr><th>{TITLES[label]}</th>" + "".join(f"<td>{n}</td>" for n in row) + "</tr>"
        for label, row in zip(quality["class_order"], quality["confusion_matrix"], strict=True)
    )
    lines = "".join(
        f"<tr><th>{TITLES[label]}</th><td>{values['flows']:,}</td><td>{values['bytes']:,}</td></tr>"
        for label, values in report["by_predicted_category"].items()
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        f"""<!doctype html><html lang="ru"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TrafficClassifier · отчёт</title>
<style>
body{{font:16px system-ui;color:#1a263b;background:#f2f5fa;margin:0}}
main{{max-width:1050px;margin:28px auto;padding:24px;background:white;border-radius:12px}}
dt{{font-weight:650}}dd{{margin:0 0 8px;overflow-wrap:anywhere}}img{{width:100%;height:auto}}
table{{border-collapse:collapse;width:100%}}th,td{{border-bottom:1px solid #e0e7f0;padding:9px;text-align:left}}
</style>
<main><h1>TrafficClassifier · отчёт</h1>
<p>Исторические записи VNAT. Байты относятся к полным потокам и сгруппированы по времени первого пакета.</p>
<dl><dt>Запуск</dt><dd>{escape(report["run_id"])}</dd><dt>Импорт</dt><dd>{escape(report["import_id"])}</dd>
<dt>Модель</dt><dd>{escape(report["model_id"])}</dd><dt>Первые пакеты</dt><dd>{report["n"]}</dd>
<dt>Период UTC</dt><dd>{escape(report["period_first_hour_utc"])} — {escape(report["period_last_hour_utc"])}</dd></dl>
<p><strong>{totals["flows"]:,}</strong> потоков · <strong>{totals["bytes"]:,}</strong> байт ·
<strong>{totals["packets"]:,}</strong> пакетов в записях, из них <strong>{totals["observed_packets"]:,}</strong> увидела CNN</p>
<img alt="Графики объёмов по классу и по времени начала потока" src="data:image/png;base64,{preview}">
<h2>Предсказанные категории</h2><table><tr><th>Категория</th><th>Потоки</th><th>Байты</th></tr>{lines}</table>
<h2>Сверка с известными метками</h2><p>Совпали {quality["correct"]:,} из {totals["flows"]:,} ответов.
В матрице строки показывают истинную категорию, столбцы — предсказанную.</p>
<table><tr><th>Истинная категория</th>{heading}</tr>{matrix}</table></main></html>""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    parser.add_argument("--model-id")
    parser.add_argument("--n", type=int)
    parser.add_argument("--output", type=Path, default=Path("artifacts/prediction-report.html"))
    args = parser.parse_args()
    try:
        with connect() as client:
            report = prediction_statistics(
                client, run_id=args.run_id, model_id=args.model_id, n=args.n
            )
        write_html(report, args.output)
    except (ClickHouseError, OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"{error}\n")
    print(f"{args.output} ({report['run_id']})")


if __name__ == "__main__":
    main()
