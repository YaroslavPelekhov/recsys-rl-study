"""Collect immutable experiment evidence and draw publication figures.

The narrative is edited in paper/task2/article.tex. This script refreshes only
tables, numerical commands, plots, and a compact readable results ledger.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rec.io import file_hash, write_json
from rec.task2_data import SPECIAL, read_json

ROOT = Path(__file__).resolve().parents[1]
NAMES = {
    "sft": "GPT-2 / CE",
    "ce": "+ CE",
    "ratio_b0.1_h5": "Ratio",
    "seconds_b0.5_h5": "Seconds",
    "return_b0.5_h5": "Activity",
    "ensemble_b0.1_h5": "Ensemble",
    "anchored_b0.1_h5": "Anchored",
    "ensemble_b0.1_h1": "Ensemble H=1",
    "repeat": "Repeat",
    "popularity": "Popularity",
    "markov": "Markov",
}
ORDER = (
    "popularity",
    "markov",
    "repeat",
    "sft",
    "ce",
    "ratio_b0.1_h5",
    "seconds_b0.5_h5",
    "return_b0.5_h5",
    "ensemble_b0.1_h1",
    "ensemble_b0.1_h5",
    "anchored_b0.1_h5",
)
COLORS = {
    "ratio": "#3876A5",
    "seconds": "#B86436",
    "return": "#8073A6",
    "ensemble": "#245365",
    "anchored": "#19816A",
    "sft": "#8D9398",
}


def collect(run, evidence):
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / "predictions").mkdir(exist_ok=True)
    for source in (run / "test_predictions").glob("*.npz"):
        shutil.copyfile(source, evidence / "predictions" / source.name)
    mapping = {"test_queries.parquet": "queries.parquet", "test_labels.parquet": "labels.parquet"}
    for name in (
        "sample.json",
        "protocol.json",
        "data_audit.json",
        "environment.json",
        "sft_recipe.json",
        "rl_recipe.json",
        "search_trials.json",
        "search_controls.json",
        "frozen.json",
        "prediction_seal.json",
        "test_opened.json",
    ):
        mapping[name] = name
    for source, destination in mapping.items():
        shutil.copyfile(run / source, evidence / destination)
    np.savez_compressed(evidence / "contexts.npz", x=np.load(run / "test_x.npy"))
    for directory, source_root in (("logs", run), ("pilot_logs", Path("runs/task2"))):
        target = evidence / directory
        target.mkdir(exist_ok=True)
        folder = source_root / "logs"
        if directory == "pilot_logs" and not folder.exists():
            folder = Path("reports/task2/evidence/pilot_logs")
        for source in folder.glob("*.json"):
            if source.resolve() != (target / source.name).resolve():
                shutil.copyfile(source, target / source.name)
    for source_name, target_name in (
        ("search_trials.json", "pilot_trials.json"),
        ("sft_recipe.json", "pilot_sft_recipe.json"),
    ):
        source = Path("runs/task2") / source_name
        if not source.exists():
            source = Path("reports/task2/evidence") / target_name
        if source.resolve() != (evidence / target_name).resolve():
            shutil.copyfile(source, evidence / target_name)


def statistics(evidence):
    results = read_json(evidence / "results.json")
    rm = read_json(evidence / "reward_metrics.json")
    meta, labels = (
        pd.read_parquet(evidence / "queries.parquet"),
        pd.read_parquet(evidence / "labels.parquet"),
    )
    predictions = np.load(evidence / "reward_predictions.npz")
    rm_contrasts = {}
    truth = predictions["test_truth"][:, 0]
    baseline = (predictions["test_global_mean"][:, 0] - truth) ** 2
    for method in ("item_mean", "ensemble", "audit_101", "ensemble_shuffled_history"):
        values = (predictions[f"test_{method}"][:, 0] - truth) ** 2
        delta = (
            pd.DataFrame({"uid": meta.uid, "difference": values - baseline})
            .groupby("uid")
            .difference.agg(["sum", "count"])
        )
        rng = np.random.default_rng(718283)
        estimates = []
        for _ in range(3000):
            drawn = rng.integers(len(delta), size=len(delta))
            estimates.append(
                delta["sum"].to_numpy()[drawn].sum() / delta["count"].to_numpy()[drawn].sum()
            )
        rm_contrasts[method] = {
            "delta": float((values - baseline).mean()),
            "ci95": np.quantile(estimates, [0.025, 0.975]).tolist(),
            "users": len(delta),
        }
    macro_ceiling = (
        pd.DataFrame({"uid": meta.uid, "covered": labels.token >= SPECIAL})
        .groupby("uid")
        .covered.mean()
        .mean()
    )
    resources = []
    for directory in ("logs", "pilot_logs"):
        resources.extend(
            {"source": directory, "fit": p.stem, **read_json(p)["resources"]}
            for p in (evidence / directory).glob("*.json")
        )
    # Four reused RMs are counted only once, in the final logs directory.
    resources = [
        r for r in resources if not (r["source"] == "pilot_logs" and r["fit"].startswith("rm_"))
    ]
    return {
        "results": results,
        "rm": rm,
        "rm_contrasts": rm_contrasts,
        "macro_catalog_ceiling": float(macro_ceiling),
        "resources": resources,
        "completed_fits": len(resources),
        "sum_recorded_fit_seconds": sum(r["seconds"] for r in resources),
        "max_cuda_peak_mib": max(r["cuda_peak_allocated_mib"] for r in resources),
        "max_rss_at_end_mib": max(r["rss_mib_at_end"] for r in resources),
        "search": read_json(evidence / "search_trials.json"),
        "rollouts": pd.read_csv(evidence / "rollout_summary.csv").to_dict("records"),
    }


def setup_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#B2BBC2",
            "text.color": "#243747",
            "axes.labelcolor": "#243747",
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )


def save_figure(fig, directory, name):
    directory.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "svg", "png"):
        fig.savefig(
            directory / f"{name}.{extension}", dpi=180, bbox_inches="tight", pad_inches=0.07
        )
    plt.close(fig)


def figures(evidence, directory, data):
    setup_style()
    result, search = data["results"], data["search"]
    fig, axes = plt.subplots(
        1, 2, figsize=(7.15, 2.85), gridspec_kw={"width_ratios": [1.05, 1]}, layout="constrained"
    )
    ax = axes[0]
    for kind in COLORS:
        if kind == "sft":
            continue
        points = sorted(
            [r for r in search["trials"] if r["kind"] == kind],
            key=lambda r: r["beta"],
            reverse=True,
        )
        xs = [r["metrics"]["ndcg"] * 100 for r in points]
        ys = [r["metrics"]["predicted_ratio"] for r in points]
        ax.plot(xs, ys, color=COLORS[kind], lw=0.9, alpha=0.65)
        ax.scatter(xs, ys, s=[15, 30, 50], color=COLORS[kind], label=kind.title(), zorder=3)
    ax.scatter(
        [search["sft"]["ndcg"] * 100],
        [search["sft"]["predicted_ratio"]],
        color="#243747",
        marker="*",
        s=80,
        label="GPT-2",
        zorder=5,
    )
    ax.axvline(0.95 * search["sft"]["ndcg"] * 100, color="#7E8C95", lw=0.7, ls=":")
    ax.set(
        xlabel="NDCG@10 × 100",
        ylabel="Предсказанная доля прослушивания",
        title="a   Development: все 15 проб, seed17",
    )
    ax.grid(alpha=0.15)
    ax.legend(frameon=False, fontsize=6.7, ncol=2, loc="best")
    ax = axes[1]
    methods = [
        "repeat",
        "ce",
        "seconds_b0.5_h5",
        "return_b0.5_h5",
        "ratio_b0.1_h5",
        "ensemble_b0.1_h1",
        "ensemble_b0.1_h5",
        "anchored_b0.1_h5",
    ]
    for i, method in enumerate(methods):
        contrast = result["methods"][method]["vs_sft"]
        lo, hi = np.asarray(contrast["ci95"]) * 1000
        value = contrast["delta"] * 1000
        color = COLORS.get(method.split("_")[0], "#657785")
        ax.plot([lo, hi], [i, i], color=color, lw=1.3)
        ax.scatter(value, i, s=24, color=color, zorder=3)
    ax.axvline(0, color="#73828D", lw=0.8)
    ax.set_yticks(np.arange(len(methods)), [NAMES[m] for m in methods])
    ax.invert_yaxis()
    ax.set(
        xlabel="Δ NDCG@10 к GPT-2, ×1000",
        title=f"b   Внешняя оценка: {result['users']} пользователей",
    )
    ax.grid(axis="x", alpha=0.15)
    save_figure(fig, directory, "tradeoff")

    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.45), layout="constrained")
    arrays = np.load(evidence / "reward_predictions.npz")
    pred, truth = arrays["test_ensemble"][:, 0], arrays["test_truth"][:, 0]
    bins = np.minimum((pred * 10).astype(int), 9)
    xs, ys, sizes = [], [], []
    for b in range(10):
        mask = bins == b
        if mask.sum() >= 10:
            xs.append(pred[mask].mean())
            ys.append(truth[mask].mean())
            sizes.append(mask.sum())
    axes[0].plot([0, 1], [0, 1], color="#A0ABB2", ls="--", lw=0.8)
    axes[0].scatter(xs, ys, s=np.sqrt(sizes) * 3, color=COLORS["ensemble"], alpha=0.85)
    axes[0].set(
        xlim=(0, 1.02),
        ylim=(0, 1.02),
        xlabel="Прогноз RM",
        ylabel="Средний реальный отклик",
        title="a   Калибровка, logged actions",
    )
    for i, (method, label) in enumerate(
        (
            ("item_mean", "Item mean"),
            ("ensemble", "Ensemble"),
            ("audit_101", "Audit RM"),
            ("ensemble_shuffled_history", "Чужая история"),
        )
    ):
        stat = data["rm_contrasts"][method]
        lo, hi = np.asarray(stat["ci95"]) * 1000
        axes[1].plot([lo, hi], [i, i], color=COLORS["ensemble"], lw=1.3)
        axes[1].scatter(stat["delta"] * 1000, i, s=22, color=COLORS["ensemble"])
    axes[1].set_yticks(range(4), ["Item mean", "Ensemble", "Audit RM", "Чужая история"])
    axes[1].invert_yaxis()
    axes[1].axvline(0, lw=0.7, color="#8B989F")
    axes[1].set(xlabel="Δ MSE к global mean, ×1000", title="b   Ошибка RM: парные 95% CI")
    methods = [
        "sft",
        "ratio_b0.1_h5",
        "seconds_b0.5_h5",
        "return_b0.5_h5",
        "ensemble_b0.1_h5",
        "anchored_b0.1_h5",
    ]
    for i, method in enumerate(methods):
        v = result["methods"][method]["behavior"]["duration"]
        axes[2].scatter(v, i, s=24, color=COLORS.get(method.split("_")[0], "#657785"))
    axes[2].set_yticks(range(len(methods)), [NAMES[m] for m in methods])
    axes[2].invert_yaxis()
    axes[2].set(xlabel="Длительность top1, секунд", title="c   Что рекомендует политика")
    for ax in axes:
        ax.grid(alpha=0.12)
    save_figure(fig, directory, "reward_audit")

    frame = pd.DataFrame(data["rollouts"])
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.4), layout="constrained")
    for method, key in (
        ("sft", "sft"),
        ("ensemble_b0.1_h5", "ensemble"),
        ("anchored_b0.1_h5", "anchored"),
    ):
        for mode, style in (("evolving", "-"), ("frozen", "--")):
            selected = frame[(frame.method == method) & (frame["mode"] == mode)].sort_values("step")
            axes[0].plot(
                selected.step,
                selected.predicted_ratio,
                style,
                color=COLORS[key],
                lw=1.4,
                label=NAMES[method] if mode == "evolving" else None,
            )
            if mode == "evolving":
                axes[1].plot(
                    selected.step,
                    selected.context_shift,
                    marker="o",
                    markersize=3,
                    color=COLORS[key],
                    lw=1.2,
                )
                axes[2].plot(
                    selected.step,
                    selected.duplicate * 100,
                    marker="o",
                    markersize=3,
                    color=COLORS[key],
                    lw=1.2,
                )
    axes[0].set(ylabel="Предсказанная доля", title="a   Награда при генерации")
    axes[0].legend(frameon=False, fontsize=6.7)
    axes[1].set(ylabel="|RM(hₜ,a) − RM(h₀,a)|", title="b   Сдвиг оценки из-за контекста")
    axes[2].set(ylabel="Повторы, %", title="c   Повторы внутри списка")
    for ax in axes:
        ax.set(xlabel="Шаг генерации", xticks=range(1, 6))
        ax.grid(alpha=0.15)
    save_figure(fig, directory, "rollouts")

    trajectory = read_json(evidence / "logs/sft_17.json")["trajectory"]
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.25), layout="constrained")
    axes[0].plot(
        [r["update"] for r in trajectory],
        [r["search_ndcg"] for r in trajectory],
        "o-",
        color=COLORS["ensemble"],
        lw=1.1,
        markersize=3,
    )
    axes[1].plot(
        [r["update"] for r in trajectory],
        [r["train_ce_last_batch"] for r in trajectory],
        "o-",
        color=COLORS["seconds"],
        lw=1.1,
        markersize=3,
    )
    axes[0].set(ylabel="Development NDCG@10", title="Выбор checkpoint по development")
    axes[1].set(ylabel="CE последнего train batch", title="Снижение loss не гарантирует точность")
    for ax in axes:
        ax.set(xlabel="Updates")
        ax.grid(alpha=0.15)
    save_figure(fig, directory, "sft_learning")


def numerical_blocks(evidence, paper, data):
    result = data["results"]
    primary = result["primary_anchored_minus_ensemble"]
    methods = result["methods"]
    commands = {
        "EvalUsers": str(result["users"]),
        "EvalQueries": str(result["queries"]),
        "CatalogCoverage": f"{result['catalog_coverage'] * 100:.1f}",
        "MacroCeiling": f"{data['macro_catalog_ceiling']:.4f}",
        "PrimaryDelta": f"{primary['delta']:+.6f}",
        "PrimaryLo": f"{primary['ci95'][0]:+.6f}",
        "PrimaryHi": f"{primary['ci95'][1]:+.6f}",
        "SftNdcg": f"{methods['sft']['ndcg']:.5f}",
        "RepeatNdcg": f"{methods['repeat']['ndcg']:.5f}",
        "ProxyBefore": f"{methods['sft']['behavior']['predicted_ratio']:.4f}",
        "ProxyAfter": f"{methods['ensemble_b0.1_h5']['behavior']['predicted_ratio']:.4f}",
        "FitCount": str(data["completed_fits"]),
        "FitMinutes": f"{data['sum_recorded_fit_seconds'] / 60:.1f}",
    }
    text = "% Generated from reports/task2/evidence; edit narrative in article.tex.\n"
    text += (
        "\n".join(r"\newcommand{\%s}{%s}" % (name, value) for name, value in commands.items())
        + "\n"
    )
    rows = []
    for method in ORDER:
        values = methods[method]
        rows.append(
            f"{NAMES[method]} & {values['ndcg']:.5f} & {values['recall']:.5f} & {values['map']:.5f} \\\\"
        )
    text += "\\newcommand{\\MainRows}{%\n" + "\n".join(rows) + "\n}\n"
    rows = []
    for name in ("global_mean", "item_mean", "ensemble", "audit_101", "ensemble_shuffled_history"):
        values = next(
            r for r in data["rm"]["metrics"] if r["split"] == "test" and r["model"] == name
        )
        label = {
            "global_mean": "Global mean",
            "item_mean": "Item mean",
            "ensemble": "Ensemble RM",
            "audit_101": "Audit RM",
            "ensemble_shuffled_history": "Чужая история",
        }[name]
        rows.append(
            f"{label} & {values['ratio_mse']:.4f} & {values['ratio_mae']:.4f} & {values['return24_auc']:.3f} & {values['return24_mse']:.4f} \\\\"
        )
    text += "\\newcommand{\\RewardRows}{%\n" + "\n".join(rows) + "\n}\n"
    paper.mkdir(parents=True, exist_ok=True)
    (paper / "numbers.tex").write_text(text, "utf-8", newline="\n")
    return commands


def results_ledger(evidence, output, data):
    result, search = data["results"], data["search"]
    lines = [
        "# Полные результаты второй задачи",
        "",
        "Числа сформированы из сохранённых прогнозов. Интервалы — user-cluster bootstrap, 3000 resamples; primary один, остальные описательные.",
        "",
        "## Внешняя оценка",
        "",
        "| Метод | NDCG@10 | Recall@10 | MAP@10 | ΔNDCG к SFT, 95% CI |",
        "|---|---:|---:|---:|---|",
    ]
    for method in ORDER:
        r = result["methods"][method]
        d = r["vs_sft"]
        lines.append(
            f"| {NAMES[method]} | {r['ndcg']:.6f} | {r['recall']:.6f} | {r['map']:.6f} | {d['delta']:+.6f} [{d['ci95'][0]:+.6f}; {d['ci95'][1]:+.6f}] |"
        )
    lines += [
        "",
        "## Поведение выбранных политик",
        "",
        "Top1 share — доля одного самого частого трека среди всех запросов; остальные средние — user macro. Audit RM имеет другую инициализацию, те же training data; это не истинная награда.",
        "",
        "| Метод | RM ratio | Audit ratio | KL | Entropy | Recent repeat | Seconds | Top1 share | Coverage@10 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in ORDER:
        if "behavior" not in result["methods"][method]:
            continue
        b = result["methods"][method]["behavior"]
        lines.append(
            f"| {NAMES[method]} | {b['predicted_ratio']:.4f} | {b['audit_ratio']:.4f} | {b['kl']:.4f} | {b['entropy']:.3f} | {b['recent_repeat']:.3f} | {b['duration']:.1f} | {b['top1_item_share']:.3%} | {b['catalog_coverage10']:.3%} |"
        )
    lines += [
        "",
        "## Полный development search",
        "",
        "Все15 trials, seed17. Здесь выбирались beta; эти числа не являются внешним подтверждением. Размер маркера на рисунке растёт при уменьшении beta.",
        "",
        "| Reward | beta | NDCG | Predicted ratio | KL | Seconds | Top1 share |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in search["trials"]:
        m = row["metrics"]
        lines.append(
            f"| {row['kind']} | {row['beta']} | {m['ndcg']:.6f} | {m['predicted_ratio']:.4f} | {m['kl']:.4f} | {m['duration']:.1f} | {m['top1_item_share']:.3%} |"
        )
    lines += [
        "",
        "## Reward model на внешних logged actions",
        "",
        "| Модель | Ratio MSE | MAE | Bias | ECE10 | Activity AUC | Brier |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in data["rm"]["metrics"]:
        if row["split"] != "test":
            continue
        lines.append(
            f"| {row['model']} | {row['ratio_mse']:.6f} | {row['ratio_mae']:.6f} | {row['ratio_bias']:+.5f} | {row['ratio_ece10']:.5f} | {row['return24_auc']:.4f} | {row['return24_mse']:.6f} |"
        )
    lines += [
        "",
        "## Полные model seeds",
        "",
        "Это отдельные SFT/RL fits, не повтор одного прогноза. Общий ансамбль RM фиксирован.",
        "",
        "| Модель | NDCG | Recall | MAP |",
        "|---|---:|---:|---:|",
    ]
    for row in result["seeds"]:
        lines.append(
            f"| {row['model']} | {row['ndcg']:.6f} | {row['recall']:.6f} | {row['map']:.6f} |"
        )
    lines += [
        "",
        "## Пилот и бюджет",
        "",
        f"Всего {data['completed_fits']} завершённых fits, включая пилот; повторно используемые4 RM посчитаны один раз. Сумма записанных durations {data['sum_recorded_fit_seconds']:.1f}s. Это сумма стадий, не полный elapsed исследования. CUDA peak allocated {data['max_cuda_peak_mib']:.1f}MiB; максимальный end-of-stage RSS {data['max_rss_at_end_mib']:.1f}MiB, не peak RSS.",
        "",
        "Сначала SFT имела бюджет1500 updates и NDCG .009576 на search. Сохранены9 завершённых RL trials; ensemble trial прерван. До открытия внешних меток зарегистрировано увеличение SFT budget до12000. Выбран checkpoint3000. Старые пробы находятся в evidence/pilot_trials.json и evidence/pilot_logs/. Неуспешные технические запуски (Arrow/Windows host commit) не выдаются за завершённые fits.",
        "",
        "## Данные графиков",
        "",
        "[Полные численные данные](evidence/figure_data.json), [rollouts по шагам](evidence/rollout_summary.csv), [observed-trajectory RM errors](evidence/logged_rollout_metrics.json), [audit](evidence/audit.json).",
    ]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n", "utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/task2_final"))
    parser.add_argument("--report", type=Path, default=Path("reports/task2"))
    parser.add_argument("--paper-dir", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    evidence = args.report / "evidence"
    paper = args.paper_dir or Path("paper") / args.report.name
    if args.check:
        current = statistics(evidence)
        if current != read_json(evidence / "figure_data.json"):
            raise ValueError("Figure statistics differ from source evidence")
        print("Task 2 figure statistics match source evidence")
        return
    collect(args.run, evidence)
    data = statistics(evidence)
    write_json(evidence / "figure_data.json", data)
    figures(evidence, paper / "figures", data)
    numerical_blocks(evidence, paper, data)
    results_ledger(evidence, args.report, data)
    digests = {
        p.relative_to(evidence).as_posix(): file_hash(p)
        for p in sorted(evidence.rglob("*"))
        if p.is_file() and p.name != "checksums.json"
    }
    write_json(evidence / "checksums.json", digests)
    print(json.dumps({"evidence_files": len(digests), "completed_fits": data["completed_fits"]}))


if __name__ == "__main__":
    main()
