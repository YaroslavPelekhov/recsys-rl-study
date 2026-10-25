"""Frozen forecasts, one-time labels, rollout audit and compact evidence."""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse

import research_task2 as study
from rec.io import file_hash, write_json
from rec.task2_data import (
    CONTEXT,
    DAY,
    SPECIAL,
    histories,
    make_queries,
    prefix,
    read_json,
    return_labels,
)
from rec.task2_metrics import (
    behavior_summary,
    paired_interval,
    ranking_rows,
    reward_metrics,
    user_macro,
)
from rec.task2_models import feedback, load_feedback, load_policy, seed_everything


def validate_freeze(root):
    frozen = read_json(root / "frozen.json")
    for name, expected in frozen["files"].items():
        if file_hash(Path(name)) != expected:
            raise ValueError(f"Frozen input changed: {name}")
    return frozen


def simple_forecasts(root, split, x, catalog):
    arrays = np.load(root / "train.npz")
    tx, ty = arrays["x"], arrays["y"]
    valid = ty >= SPECIAL
    rows, cols = tx[valid].astype(np.int64), ty[valid]
    n = len(catalog["items"]) + SPECIAL
    transition = sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(n, n))
    pop = np.r_[np.zeros(SPECIAL), catalog["counts"]]
    pop /= pop.sum()
    predictions = {m: [] for m in ("popularity", "repeat", "markov")}
    for context in x:
        repeat = (
            np.bincount(context, weights=0.97 ** np.arange(CONTEXT)[::-1], minlength=n) + pop * 1e-5
        )
        markov = transition.getrow(int(context[-1])).toarray().ravel() + 20 * pop
        for name, score in (("popularity", pop.copy()), ("repeat", repeat), ("markov", markov)):
            score[:SPECIAL] = -1
            # Stable lower-token tie break; policy scores almost never tie.
            chosen = np.lexsort((np.arange(n), -score))[:10]
            predictions[name].append(chosen)
    for method, values in predictions.items():
        np.savez_compressed(
            root / f"{split}_predictions/{method}.npz", top=np.stack(values).astype(np.int32)
        )


def forecast(root):
    frozen = validate_freeze(root)
    if (root / "test_labels.parquet").exists():
        raise ValueError("Test already opened; no new forecasts may be added")
    start = time.perf_counter()
    catalog = np.load(root / "catalog.npz")
    frame = pd.read_parquet(root / "sampled_events.parquet")
    history = histories(frame, catalog["items"])
    sample = read_json(root / "sample.json")
    meta, x, _ = make_queries(history, sample["external_users"], range(293, 299), False)
    del frame, history
    gc.collect()
    meta.to_parquet(root / "test_queries.parquet", index=False)
    np.save(root / "test_x.npy", x)
    paths = []
    for seed in (17, 42, 73):
        paths.extend([root / f"models/sft_{seed}.pt", root / f"models/ce_{seed}.pt"])
        paths.extend(
            root / f"models/{kind}_b{beta:g}_h{length}_{seed}.pt"
            for kind, beta, length in frozen["configs"]
        )
    for path in paths:
        study.score_policy(root, path, "test", int(path.stem.rsplit("_", 1)[1]))
        print(f"Saved forecast {path.stem}", flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    simple_forecasts(root, "test", x, catalog)
    files = [
        root / "test_queries.parquet",
        root / "test_x.npy",
        *sorted((root / "test_predictions").glob("*.npz")),
    ]
    write_json(
        root / "prediction_seal.json",
        {
            "files": {p.as_posix(): file_hash(p) for p in files},
            "frozen_sha256": file_hash(root / "frozen.json"),
            "test_labels_constructed": False,
            "queries": len(x),
            "users": meta.uid.nunique(),
            "resources": study.resource(start),
        },
    )


def materialize_labels(root):
    seal = read_json(root / "prediction_seal.json")
    for name, expected in seal["files"].items():
        if file_hash(Path(name)) != expected:
            raise ValueError(f"Sealed forecasts changed: {name}")
    catalog = np.load(root / "catalog.npz")
    h = histories(pd.read_parquet(root / "sampled_events.parquet"), catalog["items"])
    meta = pd.read_parquet(root / "test_queries.parquet")
    rows = []
    returns = {u: return_labels(g["t"]) for u, g in h.items()}
    for row in meta.itertuples(index=False):
        g, i = h[row.uid], row.position
        before = np.searchsorted(g["t"], row.timestamp, side="left")
        rows.append(
            (
                int(g["tokens"][i]),
                float(g["ratio"][i]),
                float(returns[row.uid][i]),
                bool(g["item"][i] in g["item"][:before]),
                int(g["item"][i]),
            )
        )
    labels = pd.DataFrame(rows, columns=["token", "ratio", "return24", "repeat_full", "item_id"])
    labels.to_parquet(root / "test_labels.parquet", index=False)
    write_json(
        root / "test_opened.json",
        {
            "labels_sha256": file_hash(root / "test_labels.parquet"),
            "prediction_seal_sha256": file_hash(root / "prediction_seal.json"),
            "rows": len(labels),
            "catalog_coverage": float((labels.token >= SPECIAL).mean()),
        },
    )


def aggregate(root, output):
    output.mkdir(parents=True, exist_ok=True)
    meta = pd.read_parquet(root / "test_queries.parquet")
    labels = pd.read_parquet(root / "test_labels.parquet")
    grouped, seed_rows, behaviors, all_metrics = {}, [], {}, []
    for path in sorted((root / "test_predictions").glob("*.npz")):
        name = path.stem
        parts = name.rsplit("_", 1)
        method = parts[0] if parts[-1].isdigit() else name
        arrays = np.load(path)
        values = ranking_rows(arrays["top"], labels.token)
        seed_rows.append({"model": name, **user_macro(values, meta.uid)})
        values["query"] = np.arange(len(meta))
        values["model"] = name
        values["method"] = method
        all_metrics.append(values)
        grouped.setdefault(method, []).append(values[["ndcg", "recall", "map"]].to_numpy())
        if "diagnostics" in arrays:
            behaviors.setdefault(method, []).append(
                behavior_summary(
                    arrays["top"],
                    pd.DataFrame(arrays["diagnostics"], columns=study.DIAGNOSTICS),
                    meta.uid,
                    16384,
                )
            )
    metric_frames = pd.concat(all_metrics, ignore_index=True)
    metric_frames.to_parquet(output / "query_metrics.parquet", index=False)
    means = {k: np.mean(v, 0) for k, v in grouped.items()}
    results = {}
    for method, values in means.items():
        result = user_macro(pd.DataFrame(values, columns=["ndcg", "recall", "map"]), meta.uid)
        result["vs_sft"] = paired_interval(values[:, 0], means["sft"][:, 0], meta.uid)
        for name, mask in (
            ("positive", labels.ratio >= 0.5),
            ("novel", ~labels.repeat_full),
            ("repeat", labels.repeat_full),
            ("oov", labels.token < SPECIAL),
        ):
            result[name] = {
                "queries": int(mask.sum()),
                **user_macro(
                    pd.DataFrame(values[mask], columns=["ndcg", "recall", "map"]), meta.uid[mask]
                ),
            }
        if method in behaviors:
            result["behavior"] = {
                k: float(np.mean([b[k] for b in behaviors[method]])) for k in behaviors[method][0]
            }
        results[method] = result
    primary = paired_interval(
        means["anchored_b0.1_h5"][:, 0], means["ensemble_b0.1_h5"][:, 0], meta.uid
    )
    write_json(
        output / "results.json",
        {
            "methods": results,
            "primary_anchored_minus_ensemble": primary,
            "seeds": seed_rows,
            "queries": len(meta),
            "users": meta.uid.nunique(),
            "catalog_coverage": float((labels.token >= SPECIAL).mean()),
            "mean_logged_ratio": float(labels.ratio.mean()),
            "mean_logged_return24": float(labels.return24.mean()),
        },
    )
    return results


def audit_rm(root, output):
    start = time.perf_counter()
    train = np.load(root / "train.npz")
    a, y = train["rm_a"], train["rm_y"]
    mean = y.mean(0)
    n = len(np.load(root / "catalog.npz")["items"]) + SPECIAL
    counts = np.bincount(a, minlength=n)
    sums = np.stack([np.bincount(a, weights=y[:, j], minlength=n) for j in range(2)], -1)
    shrunk = (sums + 20 * mean) / (counts[:, None] + 20)
    records = []
    predictions = {}
    for split in ("search", "test"):
        x = np.load(root / f"{split}_x.npy")
        labels = pd.read_parquet(root / f"{split}_labels.parquet")
        truth, actions = labels[["ratio", "return24"]].to_numpy(), labels.token.to_numpy()
        models = [
            load_feedback(root / f"models/rm_{seed}.pt", study.DEVICE) for seed in (17, 42, 73, 101)
        ]
        pred = study.evaluate_feedback(models, x, actions)
        split_meta = pd.read_parquet(root / f"{split}_queries.parquet")
        order = split_meta.sort_values("timestamp", kind="stable").index.to_numpy()
        sources = np.arange(len(x))
        for j, current in enumerate(order):
            for earlier in order[max(0, j - 50) : j][::-1]:
                if split_meta.uid.iloc[earlier] != split_meta.uid.iloc[current]:
                    sources[current] = earlier
                    break
        shuffled = study.evaluate_feedback(models[:3], x[sources], actions).mean(0)
        options = {
            "global_mean": np.broadcast_to(mean, truth.shape),
            "item_mean": shrunk[actions],
            "ensemble": pred[:3].mean(0),
            "audit_101": pred[3],
            "ensemble_shuffled_history": shuffled,
        }
        options.update({f"rm_{seed}": pred[j] for j, seed in enumerate((17, 42, 73))})
        for name, p in options.items():
            records.append({"split": split, "model": name, **reward_metrics(truth, p)})
            predictions[f"{split}_{name}"] = p
        predictions[f"{split}_truth"] = truth
        predictions[f"{split}_actions"] = actions
        for bin_name, mask in (("catalog", actions >= SPECIAL), ("oov", actions < SPECIAL)):
            records.append(
                {
                    "split": split,
                    "model": f"ensemble_{bin_name}",
                    **reward_metrics(truth[mask], pred[:3, mask].mean(0)),
                }
            )
    np.savez_compressed(output / "reward_predictions.npz", **predictions)
    write_json(
        output / "reward_metrics.json", {"metrics": records, "resources": study.resource(start)}
    )


@torch.no_grad()
def rollout_audit(root, output):
    seed_everything(17)
    start = time.perf_counter()
    meta = pd.read_parquet(root / "test_queries.parquet")
    candidates = meta.drop_duplicates("uid").index.to_numpy()
    indices = np.sort(
        np.random.default_rng(718284).choice(candidates, min(256, len(candidates)), replace=False)
    )
    x = np.load(root / "test_x.npy")[indices]
    rms = [load_feedback(root / f"models/rm_{s}.pt", study.DEVICE) for s in (17, 42, 73)]
    auditor = load_feedback(root / "models/rm_101.pt", study.DEVICE)
    frozen = read_json(root / "frozen.json")
    configs = [("sft", None), *[(f"{k}_b{b:g}_h{h}", (k, b, h)) for k, b, h in frozen["configs"]]]
    all_rows = []
    for seed in (17, 42, 73):
        for name, _ in configs:
            model = load_policy(root / f"models/{name}_{seed}.pt", study.DEVICE)
            for mode in ("evolving", "frozen"):
                torch.manual_seed(9000 + seed)
                initial = study.tensor(x)
                context, previous = initial, []
                for step in range(5):
                    with study.amp():
                        logits = model(context, last_only=True).float()
                        action = torch.distributions.Categorical(logits=logits).sample()
                        p = feedback(rms, context, action).float()
                        original = feedback(rms, initial, action).float()
                        audit = auditor(context, action).sigmoid().float()
                    for j, uid in enumerate(meta.iloc[indices].uid):
                        all_rows.append(
                            {
                                "method": name,
                                "seed": seed,
                                "mode": mode,
                                "step": step + 1,
                                "uid": int(uid),
                                "action": int(action[j]),
                                "predicted_ratio": float(p[:, j, 0].mean()),
                                "anchored_ratio": float(original[:, j, 0].mean()),
                                "context_shift": float(
                                    (p[:, j, 0].mean() - original[:, j, 0].mean()).abs()
                                ),
                                "rm_std": float(p[:, j, 0].std(unbiased=False)),
                                "audit_ratio": float(audit[j, 0]),
                                "predicted_return24": float(p[:, j, 1].mean()),
                                "duplicate": float(any(action[j] == prev[j] for prev in previous)),
                            }
                        )
                    previous.append(action)
                    if mode == "evolving":
                        context = torch.cat((context[:, 1:], action[:, None]), 1)
            del model
            torch.cuda.empty_cache()
    frame = pd.DataFrame(all_rows)
    frame.to_parquet(output / "rollouts.parquet", index=False)
    summary = (
        frame.drop(columns=["action", "uid", "seed"])
        .groupby(["method", "mode", "step"])
        .mean()
        .reset_index()
    )
    summary.to_csv(output / "rollout_summary.csv", index=False)
    # Observed trajectories are calibration data, not generated-policy outcomes.
    catalog = np.load(root / "catalog.npz")
    real = histories(pd.read_parquet(root / "sampled_events.parquet"), catalog["items"])
    logged_x, logged_a, logged_y, logged_meta = [], [], [], []
    for row in meta.iloc[indices].itertuples():
        g = real[row.uid]
        times = g["t"]
        unique = np.r_[True, np.diff(times) != 0] & np.r_[np.diff(times) != 0, True]
        positions = np.flatnonzero(
            (times >= row.timestamp) & (times < 299 * DAY) & unique & (g["organic"] == 0)
        )[:5]
        ret = return_labels(times)
        for step, pos in enumerate(positions, 1):
            end = np.searchsorted(times, times[pos], side="left")
            logged_x.append(prefix(g["tokens"], end))
            logged_a.append(g["tokens"][pos])
            logged_y.append((g["ratio"][pos], ret[pos]))
            logged_meta.append((row.uid, step))
    logged_p = study.evaluate_feedback(rms, np.stack(logged_x), np.asarray(logged_a)).mean(0)
    np.savez_compressed(
        output / "logged_rollout_reward.npz",
        predictions=logged_p,
        truth=np.asarray(logged_y),
        meta=np.asarray(logged_meta),
    )
    logged_stats = [
        {
            "step": step,
            **reward_metrics(
                np.asarray(logged_y)[np.asarray(logged_meta)[:, 1] == step],
                logged_p[np.asarray(logged_meta)[:, 1] == step],
            ),
        }
        for step in range(1, 6)
    ]
    write_json(output / "logged_rollout_metrics.json", {"steps": logged_stats})
    write_json(
        output / "rollout_resources.json",
        {"users": len(indices), "resources": study.resource(start)},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase", choices=("forecast", "open", "aggregate", "reward", "rollouts", "all")
    )
    parser.add_argument("--root", type=Path, default=Path("runs/task2_final"))
    parser.add_argument("--output", type=Path, default=Path("reports/task2/evidence"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    seed_everything(17)
    actions = {
        "forecast": lambda: forecast(args.root),
        "open": lambda: materialize_labels(args.root),
        "aggregate": lambda: aggregate(args.root, args.output),
        "reward": lambda: audit_rm(args.root, args.output),
        "rollouts": lambda: rollout_audit(args.root, args.output),
    }
    if args.phase == "all":
        for phase in actions:
            actions[phase]()
    else:
        actions[args.phase]()


if __name__ == "__main__":
    main()
