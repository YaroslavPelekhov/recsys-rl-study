"""Numerical blocks and publication figures; narrative remains editable LaTeX."""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rec.io import file_hash, write_json
from rec.task2_data import read_json

matplotlib.use("Agg")
ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    "sft": "GPT-2",
    "repeat_head": "Repeat / head",
    "repeat": "Repeat / full",
    "popularity": "Popularity",
    "copy_fixed": "Copy 50/50",
    "copy_ce": "Copy CE",
    "continue_ce": "CE continuation",
    "reward_only": "Reward only",
    "joint": "Joint",
    "gated": "Gated joint",
}
ORDER = (
    "popularity",
    "repeat_head",
    "repeat",
    "sft",
    "copy_fixed",
    "copy_ce",
    "continue_ce",
    "reward_only",
    "joint",
    "gated",
)
TEAL, INK, RED, GRAY = "#167889", "#243747", "#BD6248", "#85949E"


def save(fig, directory, name):
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(
            directory / f"{name}.{suffix}",
            dpi=200,
            bbox_inches="tight",
            metadata={"Creator": "Reproducible Task2b evidence"} if suffix == "pdf" else None,
        )
    plt.close(fig)


def publish():
    from verify_task2b import verify

    verify(ROOT / "reports/task2b")
    e = ROOT / "reports/task2b/evidence"
    result, diagnostic = read_json(e / "results.json"), read_json(e / "diagnostics.json")
    paper = ROOT / "paper/task2b"
    figures = paper / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": INK,
            "text.color": INK,
            "axes.titleweight": "bold",
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.05), gridspec_kw={"width_ratios": [1, 1.15]})
    for ax, names, title, labels in (
        (
            axes[0],
            ("copy_ce-minus-sft", "continue_ce-minus-sft"),
            "A  Copy adapter versus GPT-2",
            ("Copy CE", "CE continuation"),
        ),
        (
            axes[1],
            ("gated-minus-continue_ce", "joint-minus-continue_ce"),
            "B  Reward versus matched CE",
            ("Gated joint (primary)", "Uniform joint"),
        ),
    ):
        for j, name in enumerate(names):
            row = result["comparisons"][name]
            lo, hi = np.asarray(row["ci95"]) * 1000
            delta = row["delta"] * 1000
            ax.errorbar(
                delta,
                1 - j,
                xerr=[[delta - lo], [hi - delta]],
                fmt="o",
                color=TEAL if ax is axes[0] else RED,
                capsize=3,
                markersize=5,
            )
        ax.axvline(0, color=GRAY, lw=0.8)
        ax.set_yticks([1, 0], labels)
        ax.set_ylim(-0.5, 1.5)
        ax.set_title(title, fontsize=8, loc="left", pad=10)
        ax.set_xlabel(r"$\Delta$ NDCG@10 $\times 10^3$ (95% user CI)")
        ax.grid(axis="x", alpha=0.15)
    fig.tight_layout(w_pad=2.2)
    save(fig, figures, "effects")

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.1), gridspec_kw={"width_ratios": [1.65, 1]})
    rm = result["rm"]
    names = ["Old neural", "Constant", "Item only", "Last20 mean", "State only", "Full ensemble"]
    scores = [
        rm["controls"]["old_neural"],
        rm["constant_mse"],
        rm["controls"]["item_only"],
        rm["controls"]["last20"],
        rm["controls"]["state_only"],
        rm["mse"],
    ]
    axes[0].barh(np.arange(6), scores, color=[GRAY] * 4 + [INK, TEAL], height=0.65)
    axes[0].set_yticks(np.arange(6), names)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 0.218)
    axes[0].set_xlabel("Logged-action completion MSE (lower is better)")
    for j, val in enumerate(scores):
        axes[0].text(val + 0.004, j, f"{val:.3f}", va="center", fontsize=7.5)
    axes[0].set_title("A  Most prediction gain needs no candidate", fontsize=8, loc="left")
    corr = np.asarray(diagnostic["rm_error_correlation"])
    axes[1].imshow(corr, vmin=0.99, vmax=1, cmap="Blues")
    for a in range(3):
        for b in range(3):
            axes[1].text(
                b,
                a,
                f"{corr[a, b]:.3f}",
                ha="center",
                va="center",
                color="white" if corr[a, b] > 0.996 else INK,
                fontsize=8,
            )
    axes[1].set_xticks(range(3), ["RM17", "RM42", "RM73"])
    axes[1].set_yticks(range(3), ["RM17", "RM42", "RM73"])
    axes[1].set_title("B  Correlated prediction errors", fontsize=8, loc="left")
    fig.tight_layout(w_pad=2)
    save(fig, figures, "reward")

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.05))
    for name, offset in {
        "popularity": (-35, 8),
        "repeat": (5, -12),
        "sft": (4, -10),
        "continue_ce": (5, 5),
    }.items():
        row = result["methods"][name]
        axes[0].scatter(
            row["reward"], row["ndcg"] * 1000, color=TEAL if name == "continue_ce" else GRAY, s=26
        )
        axes[0].annotate(
            LABELS[name],
            (row["reward"], row["ndcg"] * 1000),
            xytext=offset,
            textcoords="offset points",
            fontsize=7,
        )
    axes[0].set_xlabel("Mean predicted top1 completion")
    axes[0].set_ylabel(r"NDCG@10 $\times 10^3$")
    axes[0].set_xlim(0.603, 0.653)
    axes[0].set_ylim(2, 14.4)
    axes[0].set_title("A  A high proxy does not rank relevant items", loc="left", fontsize=8)
    for name, offset in {
        "continue_ce": (-58, 5),
        "gated": (-45, -14),
        "joint": (5, 5),
        "reward_only": (-48, -12),
    }.items():
        row = result["methods"][name]
        axes[1].scatter(
            row["reward"], row["ndcg"] * 1000, color=TEAL if name == "continue_ce" else RED, s=27
        )
        axes[1].annotate(
            LABELS[name],
            (row["reward"], row["ndcg"] * 1000),
            xytext=offset,
            textcoords="offset points",
            fontsize=7,
        )
    axes[1].set_xlabel("Mean predicted top1 completion")
    axes[1].set_ylabel(r"NDCG@10 $\times 10^3$")
    axes[1].set_title("B  Reward improves slightly, NDCG declines", loc="left", fontsize=8)
    axes[1].ticklabel_format(useOffset=False, axis="x")
    from matplotlib.ticker import MaxNLocator

    axes[1].xaxis.set_major_locator(MaxNLocator(3))
    for ax in axes:
        ax.grid(alpha=0.15)
    fig.tight_layout(w_pad=2)
    save(fig, figures, "tradeoff")

    commands = {
        "EvalUsers": result["users"],
        "EvalQueries": result["queries"],
        "SftNdcg": f"{result['methods']['sft']['ndcg']:.6f}",
        "CeNdcg": f"{result['methods']['continue_ce']['ndcg']:.6f}",
        "GatedNdcg": f"{result['methods']['gated']['ndcg']:.6f}",
        "GainPercent": f"{diagnostic['ce_relative_gain_percent']:.1f}",
        "RmMse": f"{rm['mse']:.6f}",
        "OldRmMse": f"{rm['controls']['old_neural']:.6f}",
    }
    for prefix, key in (("Primary", "gated-minus-continue_ce"), ("Copy", "continue_ce-minus-sft")):
        row = result["comparisons"][key]
        commands[prefix + "Delta"], commands[prefix + "Lo"], commands[prefix + "Hi"] = (
            f"{row['delta']:+.6f}",
            f"{row['ci95'][0]:+.6f}",
            f"{row['ci95'][1]:+.6f}",
        )
    (paper / "numbers.tex").write_text(
        "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in commands.items()) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    table = []
    for name in ORDER:
        row = result["methods"][name]
        table.append(
            f"{LABELS[name]} & {row['ndcg']:.6f} & {row['recall']:.6f} & {row['reward']:.4f} \\\\"
        )
    (paper / "ranking.tex").write_text(
        "\\begin{tabular}{lrrr}\\toprule\nМетод & NDCG@10 & Recall@10 & Proxy\\\\\\midrule\n"
        + "\n".join(table)
        + "\n\\bottomrule\\end{tabular}\n",
        encoding="utf-8",
        newline="\n",
    )
    report = [
        "# Полный журнал второго цикла Task2",
        "",
        "## Внешняя проверка",
        "",
        "| Метод | NDCG@10 | Recall@10 | MAP@10 | Proxy | Repeat top1 | Duration270 (audit) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ORDER:
        r = result["methods"][name]
        duration = read_json(e / "descriptive.json")["methods"][name]["duration270"]
        report.append(
            f"| {LABELS[name]} | {r['ndcg']:.6f} | {r['recall']:.6f} | {r['map']:.6f} | {r['reward']:.6f} | {r['repeat']:.3f} | {duration:.1f} |"
        )
    report += [
        "",
        "Duration270 — описательный post-test аудит по train-only metadata; исходный кэш duration200 с unknown=0 сохранён в evidence. Первичные метрики/политики не менялись.",
        "",
        "## Внешние метрики каждого model seed",
        "",
        "| Метод | Seed | NDCG | Recall | Proxy |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    raw = pd.read_parquet(e / "query_metrics.parquet")
    for (name, seed), frame in raw.loc[raw.seed != 0].groupby(["method", "seed"]):
        mean = frame.groupby("uid")[["ndcg", "recall", "reward"]].mean().mean()
        report.append(
            f"| {LABELS[name]} | {seed} | {mean['ndcg']:.6f} | {mean['recall']:.6f} | {mean['reward']:.6f} |"
        )
    report += [
        "",
        "## Парные user-cluster интервалы",
        "",
        "| Контраст | Delta NDCG | 95% CI |",
        "| --- | ---: | --- |",
    ]
    for name, r in result["comparisons"].items():
        report.append(
            f"| {name} | {r['delta']:+.8f} | [{r['ci95'][0]:+.8f}; {r['ci95'][1]:+.8f}] |"
        )
    report += [
        "",
        "## Все adapter fits и checkpoints",
        "",
        "Колонки search: только development. Seeds42/73 не меняют выбранный рецепт.",
        "",
        "| Seed | Метод | Lambda/update | NDCG | Proxy | Admitted | Fit seconds |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for seed in (17, 42, 73):
        for r in read_json(e / f"adapter_search_{seed}.json"):
            if "trajectory" in r:
                for p in r["trajectory"]:
                    report.append(
                        f"| {seed} | copy_ce checkpoint | {p['update']} | {p['ndcg']:.6f} | {p['predicted_ratio']:.6f} | - | {r['seconds']:.3f} (whole fit) |"
                    )
            else:
                report.append(
                    f"| {seed} | {r['method']} | {r['lambda']} | {r['ndcg']:.6f} | {r['predicted_ratio']:.6f} | {r['admitted']:.3f} | {r['seconds']:.3f} |"
                )
    report += [
        "",
        "## RM calibration search",
        "",
        "| Seed | Depth | Trees | Calibration MSE |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for r in read_json(e / "rm_search.json")["trials"]:
        report.append(
            f"| {r.get('seed', 17)} | {r['depth']} | {r['trees']} | {r['calibration_mse']:.6f} |"
        )
    report += [
        "",
        "RM external MSE: "
        + "; ".join(f"{name}={value:.6f}" for name, value in zip(names, scores, strict=True)),
        "",
        "Технические сбои и пределы измерения затрат: evidence/diagnostics.json; runtime_fix.json. Веса frozen, ни один рецепт не изменён после external labels.",
    ]
    (ROOT / "reports/task2b/RESULTS.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_json(
        paper / "figure_inputs.json",
        {name: file_hash(e / name) for name in ("results.json", "diagnostics.json")},
    )


if __name__ == "__main__":
    publish()
