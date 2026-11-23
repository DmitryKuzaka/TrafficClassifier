"""Open a local window for trying the selected packet CNN."""

import json
import tkinter as tk
from functools import partial
from tkinter import ttk

import numpy as np
import torch

from traffic_classifier.ml.dataset import DEFAULT_BUNDLE, load_bundle
from traffic_classifier.ml.final_evaluation import DEFAULT_PROTOCOL
from traffic_classifier.ml.model_bundle import ModelBundle, load_model_bundle
from traffic_classifier.ml.prefix_study import DEFAULT_OUTPUT
from traffic_classifier.vnat_groups import CLASSES

CLASS_NAMES = {"video": "Видео", "voice": "Звонок", "file_transfer": "Передача файла"}
DISPLAY_NAMES = tuple(CLASS_NAMES[label] for label in CLASSES)


def parse_packets(text: str, n: int) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    rows = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            offset, size, direction = map(int, line.split())
        except ValueError:
            raise ValueError(f"Строка {number} должна содержать три целых числа.") from None
        if not 0 <= offset <= 2**63 - 1 or not 0 <= size <= 2**32 - 1:
            raise ValueError(f"Проверь время и размер в строке {number}.")
        if direction not in (0, 1):
            raise ValueError(f"В строке {number} направление должно быть 0 или 1.")
        rows.append((offset, size, direction))
    if not 1 <= len(rows) <= n:
        raise ValueError(f"Укажи от 1 до {n} пакетов.")
    if rows[0][0] != 0:
        raise ValueError("Время первого пакета должно быть 0.")
    return tuple(zip(*rows, strict=True))


def load_examples(n: int) -> dict[str, str]:
    validation = load_bundle(DEFAULT_BUNDLE, "validation")
    examples = {}
    for index, title in enumerate(DISPLAY_NAMES):
        candidates = np.flatnonzero((validation.labels == index) & validation.mask[:, n - 1])
        if not len(candidates):
            raise ValueError(f"В validation нет примера «{title}» с {n} пакетами.")
        size, direction, gap_ms = validation.features[candidates[0], :, :n]
        offsets_us = np.rint(gap_ms.astype(np.float64) * 1000).astype(np.int64).cumsum()
        examples[title] = "\n".join(
            f"{offset}  {int(length)}  {int(side)}"
            for offset, length, side in zip(offsets_us, size, direction, strict=True)
        )
    return examples


def build_window(root: tk.Tk, model: ModelBundle, examples: dict[str, str]) -> None:
    n = model.normalizer.n
    root.title("TrafficClassifier — пробная версия")
    root.minsize(620, 500)
    style = ttk.Style(root)
    style.configure("Title.TLabel", font=("Segoe UI", 22, "bold"))
    style.configure("Result.TLabel", font=("Segoe UI", 13, "bold"))
    frame = ttk.Frame(root, padding=22)
    frame.pack(fill="both", expand=True)

    ttk.Label(frame, text="Распознавание сетевого потока", style="Title.TLabel").pack(anchor="w")
    ttk.Label(frame, text=f"Выбери пример из validation или укажи от 1 до {n} пакетов.").pack(
        anchor="w", pady=(3, 13)
    )
    picks = ttk.Frame(frame)
    picks.pack(fill="x")
    ttk.Label(frame, text="Время от первого пакета (мкс)   Размер (байт)   Направление (0/1)").pack(
        anchor="w", pady=(13, 4)
    )
    editor = tk.Text(frame, name="packets", height=min(n + 1, 10), font=("Consolas", 12), undo=True)
    editor.pack(fill="both", expand=True)

    def show_example(name: str) -> None:
        editor.delete("1.0", "end")
        editor.insert("1.0", examples[name])
        editor.edit_modified(False)
        status.set(f"Метка примера — {name.lower()}.")
        clear_scores()

    for title in examples:
        ttk.Button(picks, text=title, command=partial(show_example, title)).pack(
            side="left", padx=(0, 6)
        )

    status = tk.StringVar(root)
    bars = []
    values = []

    def clear_scores() -> None:
        for bar, value in zip(bars, values, strict=True):
            bar.configure(value=0)
            value.set("—")

    def edited(_event: tk.Event) -> None:
        if editor.edit_modified():
            editor.edit_modified(False)
            clear_scores()
            status.set("Вход изменён. Нажми «Распознать».")

    editor.bind("<<Modified>>", edited)

    def predict() -> None:
        editor.edit_modified(False)
        clear_scores()
        try:
            offset, size, direction = parse_packets(editor.get("1.0", "end-1c"), n)
            logits = model.packet_logits(offset, size, direction)
        except ValueError as error:
            status.set(str(error))
            return
        scores = torch.softmax(logits, dim=0)
        category = DISPLAY_NAMES[int(scores.argmax())]
        status.set(f"Результат — {category.lower()} · пакетов {len(size)} из {n}")
        for bar, value, score in zip(bars, values, scores.tolist(), strict=True):
            bar.configure(value=100 * score)
            value.set(f"{100 * score:.1f}%".replace(".", ","))

    action = ttk.Frame(frame)
    action.pack(fill="x", pady=12)
    ttk.Button(action, name="predict", text="Распознать", command=predict).pack(side="left")
    ttk.Label(action, textvariable=status, style="Result.TLabel", wraplength=430).pack(
        side="left", padx=10
    )
    summary = ttk.Frame(frame)
    summary.pack(fill="x")
    summary.columnconfigure(1, weight=1)
    for index, title in enumerate(DISPLAY_NAMES):
        value = tk.StringVar(root, value="—")
        bar = ttk.Progressbar(summary, maximum=100)
        ttk.Label(summary, text=title, width=18).grid(row=index, column=0, sticky="w", pady=3)
        bar.grid(row=index, column=1, sticky="ew")
        ttk.Label(summary, textvariable=value, width=8).grid(row=index, column=2, padx=(8, 0))
        bars.append(bar)
        values.append(value)
    ttk.Label(frame, text="Полосы показывают относительные оценки CNN.").pack(
        anchor="w", pady=(8, 0)
    )
    show_example(DISPLAY_NAMES[0])


def main() -> None:
    try:
        protocol = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
        n = protocol["selected_cnn_n"]
        model = load_model_bundle(DEFAULT_OUTPUT / "cnn" / f"n{n}" / "model.pt", expected_n=n)
        if model.model_id != protocol["lengths"][str(n)]["cnn_model_id"]:
            raise ValueError("Комплект CNN отличается от версии в протоколе.")
        examples = load_examples(n)
    except (OSError, ValueError) as error:
        raise SystemExit(f"Не удалось открыть пробную версию. {error}") from None

    torch.set_num_threads(1)
    root = tk.Tk()
    build_window(root, model, examples)
    root.mainloop()


if __name__ == "__main__":
    main()
