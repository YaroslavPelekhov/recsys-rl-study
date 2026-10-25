"""Vector architecture figures for the two recommendation studies; no model changes."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK, TEAL, RUST = "#243747", "#176C79", "#B45D42"
GRAY, LIGHT = "#71818C", "#F2F5F7"


def box(ax, x, y, w, h, title, detail="", kind="plain"):
    color = TEAL if kind == "model" else RUST if kind == "reward" else GRAY
    fill = "#EDF7F7" if kind == "model" else "#FBF0EB" if kind == "reward" else LIGHT
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.006,rounding_size=0.012",
            linewidth=1.1,
            edgecolor=color,
            facecolor=fill,
        )
    )
    ax.text(
        x + w / 2,
        y + h * 0.60,
        title,
        ha="center",
        va="center",
        fontsize=10.2,
        color=INK,
        weight="bold",
        linespacing=1.4,
    )
    if detail:
        ax.text(
            x + w / 2,
            y + h * 0.18,
            detail,
            ha="center",
            va="center",
            fontsize=8,
            color=GRAY,
            linespacing=1.3,
        )


def arrow(ax, start, end, color=GRAY, bend=None, dashed=False):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            lw=1.15,
            color=color,
            connectionstyle=bend or "arc3",
            linestyle="--" if dashed else "-",
        )
    )


def task1():
    fig, ax = plt.subplots(figsize=(12.8, 4.0))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    box(ax, 0.02, 0.62, 0.18, 0.24, "События и атрибуты\nKION / Yambda", "все строки выбранных UID")
    box(
        ax,
        0.27,
        0.62,
        0.20,
        0.24,
        "Временной срез\nи общий retrieval",
        "прошлое → кандидаты C(u,t)",
    )
    box(ax, 0.55, 0.62, 0.19, 0.24, "Ручные признаки x\nuser · item · связь", "71 KION / 69 Yambda")
    box(
        ax,
        0.80,
        0.62,
        0.18,
        0.24,
        "CatBoost Ranker\nx или [x, φ] → top-K",
        "одни и те же кандидаты",
        kind="model",
    )
    box(
        ax,
        0.27,
        0.17,
        0.20,
        0.25,
        "Обучаемые связи\nEASE / NCE",
        "альтернативы для KION / Yambda",
        kind="model",
    )
    box(
        ax,
        0.55,
        0.17,
        0.19,
        0.25,
        "История → score φ\nrecency + pooling",
        "признаки пары user–item",
        kind="model",
    )
    box(
        ax,
        0.80,
        0.17,
        0.18,
        0.25,
        "Standalone-контроль\nG0 · G1 · A0 · A1",
        "те же векторы и кандидаты",
    )
    for a, b in (
        ((0.20, 0.74), (0.27, 0.74)),
        ((0.47, 0.74), (0.55, 0.74)),
        ((0.74, 0.74), (0.80, 0.74)),
        ((0.37, 0.62), (0.37, 0.42)),
        ((0.47, 0.30), (0.55, 0.30)),
        ((0.74, 0.30), (0.80, 0.30)),
    ):
        arrow(ax, a, b)
    arrow(ax, (0.74, 0.40), (0.84, 0.62), TEAL)
    ax.text(0.32, 0.51, "fit на train", ha="center", fontsize=8, color=GRAY)
    ax.text(0.84, 0.49, "добавить φ", ha="center", fontsize=8, color=TEAL)
    ax.text(0.02, 0.965, "ОТ ИСТОРИИ К РАНЖИРОВАНИЮ", color=INK, fontsize=10, weight="bold")
    ax.text(
        0.50,
        0.035,
        "Manual сохраняет общий retrieval; проверяется вклад дополнительных признаков, а не новый отбор кандидатов.",
        ha="center",
        fontsize=8.2,
        color=GRAY,
    )
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.025, top=0.99)
    return fig


def task2():
    fig, ax = plt.subplots(figsize=(12.8, 5.5))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    xs = [0.015, 0.218, 0.421, 0.624, 0.827]
    width = 0.158
    upper = [
        ("Реальный префикс\nпоследние 64 ID", "строго до запроса", "plain"),
        ("GPT-2 decoder\n3 блока · d=128", "2,70 млн параметров", "model"),
        ("Распределение πθ\nи sampled item", "softmax по 16 384 ID", "model"),
        ("Нейронная RM ×3\ncompletion / Activity", "reward model frozen", "reward"),
        ("REINFORCE + KL\n5 шагов · γ=0.9", "reference SFT frozen", "model"),
    ]
    lower = [
        ("Реальный префикс\nID + прошлый отклик", "история не выдумывается", "plain"),
        ("Frozen GPT-2\n+ copy истории", "редкие ID через copy", "model"),
        ("Обучаемый gate\np = (1−g)p₀ + gpс", "16 параметров", "model"),
        ("CatBoost RM ×3\nсостояние + кандидат", "13 + 7 causal-признаков", "reward"),
        ("CE + reward + KL\njoint / gated", "контроль: CE continuation", "model"),
    ]
    for y, rows in ((0.66, upper), (0.10, lower)):
        for x, (title, detail, kind) in zip(xs, rows, strict=True):
            box(ax, x, y, width, 0.19, title, detail, kind)
        for x, next_x in zip(xs[:-1], xs[1:], strict=True):
            arrow(ax, (x + width, y + 0.095), (next_x, y + 0.095))
    ax.text(0.015, 0.985, "ЦИКЛ A · ОБНОВЛЯЮТСЯ ВЕСА GPT-2", fontsize=10, weight="bold", color=INK)
    ax.text(
        0.015,
        0.425,
        "ЦИКЛ B · ЗАМОРОЖЕННЫЙ DECODER, ОБУЧАЕМЫЙ COPY-АДАПТЕР",
        fontsize=10,
        weight="bold",
        color=INK,
    )
    for a, b, y, label in (
        (0.906, 0.297, 0.905, "градиент policy · RM и reference не обучаются"),
        (0.906, 0.500, 0.348, "градиент адаптера · контекст реальный · H=1"),
    ):
        ax.plot([a, a, b, b], [y - 0.03, y, y, y - 0.03], color=TEAL, lw=1.1)
        arrow(ax, (b, y - 0.025), (b, y - 0.055), TEAL)
        ax.text((a + b) / 2, y + 0.012, label, ha="center", fontsize=8, color=TEAL)
    arrow(ax, (0.50, 0.66), (0.09, 0.66), bend="arc3,rad=-.35", dashed=True)
    ax.text(
        0.30,
        0.57,
        "добавить generated ID в контекст следующего шага",
        ha="center",
        fontsize=8,
        color=GRAY,
    )
    ax.text(
        0.50,
        0.025,
        "Рекомендованный ID не означает наблюдённое прослушивание; прогноз reward проверяется отдельно от ranking quality.",
        ha="center",
        fontsize=8.2,
        color=GRAY,
    )
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.015, top=0.99)
    return fig


def draw(task, output):
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig = task1() if task == 1 else task2()
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(
            output / f"architecture.{suffix}", dpi=190, bbox_inches="tight", facecolor="white"
        )
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=int, choices=(1, 2), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    draw(args.task, args.output)
