"""Bind the unified Task 2 submission article to the two frozen experiment cycles."""

import shutil
from pathlib import Path

from rec.io import file_hash, write_json
from rec.task2_data import read_json

ROOT = Path(__file__).resolve().parents[1]


def prepare():
    paper = ROOT / "paper/submission_task2"
    figures = paper / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    first = read_json(ROOT / "reports/task2/evidence/results.json")
    second = read_json(ROOT / "reports/task2b/evidence/results.json")
    blocks = {}
    for prefix, values, primary in (
        ("A", first, first["primary_anchored_minus_ensemble"]),
        ("B", second, second["comparisons"]["gated-minus-continue_ce"]),
    ):
        blocks.update(
            {
                prefix + "Users": str(values["users"]),
                prefix + "Queries": str(values["queries"]),
                prefix + "Delta": f"{primary['delta']:+.6f}",
                prefix + "Lo": f"{primary['ci95'][0]:+.6f}",
                prefix + "Hi": f"{primary['ci95'][1]:+.6f}",
            }
        )
    blocks["Gain"] = (
        f"{100 * (second['methods']['continue_ce']['ndcg'] / second['methods']['sft']['ndcg'] - 1):.1f}"
    )
    (paper / "numbers.tex").write_text(
        "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in blocks.items()) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    first_methods = (
        ("popularity", "Popularity"),
        ("markov", "Markov"),
        ("repeat", "Repeat"),
        ("sft", "GPT-2"),
        ("ce", "+ CE"),
        ("ratio_b0.1_h5", "Ratio"),
        ("seconds_b0.5_h5", "Seconds"),
        ("return_b0.5_h5", "Activity"),
        ("ensemble_b0.1_h1", "Ensemble H1"),
        ("ensemble_b0.1_h5", "Ensemble H5"),
        ("anchored_b0.1_h5", "Anchored H5"),
    )
    second_methods = (
        ("popularity", "Popularity"),
        ("repeat_head", "Repeat / head"),
        ("repeat", "Repeat / full"),
        ("sft", "GPT-2"),
        ("copy_fixed", "Copy 50/50"),
        ("copy_ce", "Copy CE"),
        ("continue_ce", "CE continuation"),
        ("reward_only", "Reward-only"),
        ("joint", "Joint"),
        ("gated", "Gated joint"),
    )
    for name, values, methods in (
        ("cycle_a", first, first_methods),
        ("cycle_b", second, second_methods),
    ):
        rows = []
        for key, label in methods:
            row = values["methods"][key]
            rows.append(
                f"{label} & {row['ndcg']:.6f} & {row['recall']:.6f} & {row['map']:.6f} \\\\"
            )
        table = (
            "\\begin{tabular}{lrrr}\\toprule\nМетод & NDCG@10 & Recall@10 & MAP@10\\\\\\midrule\n"
            + "\n".join(rows)
            + "\n\\bottomrule\\end{tabular}\n"
        )
        (paper / f"{name}.tex").write_text(table, encoding="utf-8", newline="\n")
    for source, name in (
        ("paper/task2/figures/tradeoff.pdf", "cycle_a_search.pdf"),
        ("paper/task2/figures/rollouts.pdf", "cycle_a_rollouts.pdf"),
        ("paper/task2b/figures/effects.pdf", "cycle_b_effects.pdf"),
        ("paper/task2b/figures/reward.pdf", "cycle_b_reward.pdf"),
    ):
        shutil.copy2(ROOT / source, figures / name)
    inputs = [
        "reports/task2/evidence/results.json",
        "reports/task2/evidence/figure_data.json",
        "reports/task2b/evidence/results.json",
        "reports/task2b/evidence/diagnostics.json",
        "reports/task2b/evidence/descriptive.json",
    ]
    write_json(paper / "evidence_inputs.json", {p: file_hash(ROOT / p) for p in inputs})


if __name__ == "__main__":
    prepare()
