"""Plot the locked test results by packet budget and evaluation cohort."""

import argparse
import json
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from traffic_classifier.ml.prefix_study import DEFAULT_OUTPUT
from traffic_classifier.vnat_groups import PREFIX_LENGTHS


def plot_quality(output: Path) -> Path:
    test = json.loads((output / "test.json").read_text(encoding="utf-8"))
    if test["split"] != "test":
        raise ValueError("The quality plot expects the final test report.")
    results = test["lengths"]
    selected = test["selected_cnn_n"]
    figure = Figure(figsize=(10, 4.1), layout="constrained", facecolor="#f6f8fb")
    FigureCanvasAgg(figure)
    left, right = figure.subplots(1, 2, sharey=True)
    for axis, key, label in (
        (left, "all", "Все потоки"),
        (right, "at_least_64", "Одни и те же потоки от 64 пакетов"),
    ):
        for method, name, color, marker in (
            ("cnn", "1D-CNN", "#1e4ead", "o"),
            ("random_forest", "Random Forest", "#ac461c", "s"),
        ):
            scores = [(results[str(n)][method][key] or {}).get("macro_f1") for n in PREFIX_LENGTHS]
            axis.plot(PREFIX_LENGTHS, scores, label=name, color=color, marker=marker)
        axis.axvline(selected, color="#64748b", linestyle=":", alpha=0.8)
        axis.set(title=label, xlabel="Первые N пакетов", xticks=PREFIX_LENGTHS, ylim=(0, 1.03))
        axis.grid(axis="y", alpha=0.25)
    left.set_ylabel("Macro-F1")
    right.legend(loc="lower right")
    figure.suptitle("VNAT · качество на отложенных захватах")
    path = output / "quality.png"
    figure.savefig(path, dpi=170)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(plot_quality(args.output))


if __name__ == "__main__":
    main()
