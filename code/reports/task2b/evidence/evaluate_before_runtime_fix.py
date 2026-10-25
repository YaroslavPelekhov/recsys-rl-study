"""Seal all new-cohort forecasts before labels; CPU metric replay evidence."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import research_task2b as study
from rec.io import file_hash, write_json
from rec.task2_data import histories, make_queries, read_json
from rec.task2_metrics import paired_interval, ranking_rows, user_macro
from rec.task2b import item_indices, make_records

ROOT = Path("runs/task2b")


def validate():
    frozen = read_json(ROOT / "frozen.json")
    for name, digest in frozen["files"].items():
        if file_hash(Path(name)) != digest:
            raise ValueError(f"Frozen source/model changed: {name}")
    return frozen


def methods(seed):
    selected = read_json(ROOT / "selection.json")
    return {
        "copy_ce": ROOT / f"ce_{seed}.npy",
        "continue_ce": ROOT / f"continue_ce_0_{seed}.npy",
        **{k: ROOT / f"{k}_{v['lambda']:g}_{seed}.npy" for k, v in selected.items()},
    }


def forecast():
    validate()
    if (ROOT / "test_labels.parquet").exists() or (ROOT / "prediction_seal.json").exists():
        raise ValueError("Test forecasts already sealed/opened")
    cat = dict(np.load(ROOT / "catalog.npz"))
    frame = pd.read_parquet(ROOT / "sampled_events.parquet")
    h = histories(frame, cat["head"])
    users = read_json(ROOT / "sample.json")["external_users"]
    meta, x, _ = make_queries(h, users, range(293, 299), False)
    arrays = make_records(h, zip(meta.uid, meta.position, strict=True), cat)
    # Dummy target never reads labels and only permits reuse of training-cache
    # mechanics. All probability/rank/reward calculations depend on prefix only.
    np.savez_compressed(
        ROOT / "test.npz", x=x, target=np.full(len(x), -5, np.int64), users=meta.uid, **arrays
    )
    meta.to_parquet(ROOT / "test_queries.parquet", index=False)
    del h, frame
    data = study.load("test")
    del data["target"]
    predictions = ROOT / "test_predictions"
    predictions.mkdir(exist_ok=True)
    for seed in (17, 42, 73):
        study.cache(seed, "test")
        with np.load(ROOT / f"cache_test_{seed}.npz") as f:
            saved = dict(f)
        for method, path in methods(seed).items():
            _, out = study.evaluate(np.load(path), data, saved, cat)
            np.savez_compressed(predictions / f"{method}_{seed}.npz", **out)
        # Frozen GPT-2 and fixed copy mixture are both same-catalog controls.
        parameter = np.zeros(16, np.float32)
        parameter[14] = -3.5
        for method, bias in (("sft", -40.0), ("copy_fixed", 0.0)):
            parameter[13] = bias
            _, out = study.evaluate(parameter, data, saved, cat)
            np.savez_compressed(predictions / f"{method}_{seed}.npz", **out)
        print(f"Saved new cohort forecasts seed{seed}", flush=True)
    head = set(item_indices(cat["head"], cat["items"]).tolist())
    pop_order = np.lexsort((np.arange(len(cat["items"])), -cat["counts"]))
    for name in ("repeat", "repeat_head", "popularity"):
        top = []
        for context in data["history"]:
            score = {}
            for position, action in enumerate(context):
                if action >= 0 and (name != "repeat_head" or action in head):
                    score[int(action)] = score.get(int(action), 0.0) + 0.97 ** (63 - position)
            if name == "popularity":
                chosen = pop_order[:10].tolist()
            else:
                chosen = sorted(score, key=lambda a: (-score[a], a))[:10]
                chosen += [int(a) for a in pop_order if a not in chosen][: 10 - len(chosen)]
            top.append(chosen)
        top = np.asarray(top, np.int32)
        reward = study.predict_reward(data, top[:, :1], cat, study.models()).mean(0)[:, 0]
        np.savez_compressed(
            predictions / f"{name}.npz",
            top=top,
            reward=reward,
            repeat=(top[:, 0, None] == data["history"]).any(1),
            duration=cat["lengths"][top[:, 0]],
        )
    files = [ROOT / "test.npz", ROOT / "test_queries.parquet"] + sorted(predictions.glob("*.npz"))
    write_json(
        ROOT / "prediction_seal.json",
        dict(
            frozen_sha256=file_hash(ROOT / "frozen.json"),
            files={str(p): file_hash(p) for p in files},
            labels_materialized=False,
        ),
    )


def labels():
    validate()
    seal = read_json(ROOT / "prediction_seal.json")
    for name, digest in seal["files"].items():
        assert file_hash(Path(name)) == digest
    if (ROOT / "test_labels.parquet").exists():
        raise FileExistsError("Already opened")
    cat = dict(np.load(ROOT / "catalog.npz"))
    frame = pd.read_parquet(ROOT / "sampled_events.parquet")
    h = histories(frame, cat["head"])
    meta = pd.read_parquet(ROOT / "test_queries.parquet")
    target = np.asarray(
        [h[int(u)]["item"][int(p)] for u, p in zip(meta.uid, meta.position, strict=True)]
    )
    ratio = np.asarray(
        [h[int(u)]["ratio"][int(p)] for u, p in zip(meta.uid, meta.position, strict=True)]
    )
    del h, frame
    pd.DataFrame(
        dict(item_id=target, item_index=item_indices(target, cat["items"]), ratio=ratio)
    ).to_parquet(ROOT / "test_labels.parquet", index=False)
    data = study.load("test")
    rms = study.models()
    predicted = study.predict_reward(data, item_indices(target, cat["items"]), cat, rms)[:, :, 0]
    from catboost import CatBoostRegressor
    from rec.task2b import reward_features

    features = reward_features(data, item_indices(target, cat["items"]), cat)
    controls = {}
    for name, cols in (("state_only", np.arange(13)), ("item_only", np.arange(13, 16))):
        model = CatBoostRegressor()
        model.load_model(str(ROOT / f"rm_{name}.cbm"))
        controls[name] = np.clip(model.predict(features[:, cols], thread_count=4), 0, 1)
    controls["last20"] = data["state"][:, 2]
    import torch
    from rec.task2_data import encode
    from rec.task2_models import load_feedback, seed_everything

    seed_everything(17)
    neural = []
    actions = encode(target, cat["head"])
    for seed in (17, 42, 73):
        model = load_feedback(Path(f"runs/task2_final/models/rm_{seed}.pt"))
        values = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for start in range(0, len(target), 256):
                x = torch.tensor(data["x"][start : start + 256], dtype=torch.long, device="cuda")
                a = torch.tensor(actions[start : start + 256], dtype=torch.long, device="cuda")
                values.append(model(x, a).sigmoid()[:, 0].float().cpu().numpy())
        neural.append(np.concatenate(values))
        del model
    controls["old_neural"] = np.mean(neural, axis=0)
    np.savez_compressed(
        ROOT / "rm_test.npz",
        truth=ratio,
        predictions=predicted,
        users=meta.uid,
        constant=np.full(len(ratio), read_json(ROOT / "rm_constant.json")["mean"]),
        **controls,
    )
    write_json(
        ROOT / "opened.json",
        dict(
            seal_sha256=file_hash(ROOT / "prediction_seal.json"),
            labels_sha256=file_hash(ROOT / "test_labels.parquet"),
            queries=len(meta),
            users=meta.uid.nunique(),
        ),
    )


def results():
    meta = pd.read_parquet(ROOT / "test_queries.parquet")
    truth = pd.read_parquet(ROOT / "test_labels.parquet")
    frames, summaries = [], {}
    for path in sorted((ROOT / "test_predictions").glob("*.npz")):
        with np.load(path) as f:
            pred = dict(f)
        frame = ranking_rows(pred["top"], truth.item_index)
        frame["uid"] = meta.uid
        for key in ("reward", "repeat", "duration"):
            frame[key] = pred[key]
        pieces = path.stem.rsplit("_", 1)
        frame["method"] = pieces[0] if pieces[-1] in ("17", "42", "73") else path.stem
        frame["seed"] = int(pieces[-1]) if pieces[-1] in ("17", "42", "73") else 0
        frame["query"] = np.arange(len(frame))
        frames.append(frame)
    rows = pd.concat(frames, ignore_index=True)
    mean = rows.groupby(["method", "query", "uid"], as_index=False).mean(numeric_only=True)
    for method, frame in mean.groupby("method"):
        summaries[method] = user_macro(
            frame[["ndcg", "recall", "map", "reward", "repeat", "duration"]], frame.uid
        )
    intervals = {}

    def values(method):
        return mean.loc[mean.method == method].sort_values("query").ndcg.to_numpy()

    for a, b in (
        ("gated", "continue_ce"),
        ("joint", "continue_ce"),
        ("copy_ce", "sft"),
        ("continue_ce", "sft"),
        ("gated", "copy_ce"),
        ("continue_ce", "repeat"),
        ("copy_ce", "repeat"),
    ):
        intervals[f"{a}-minus-{b}"] = paired_interval(values(a), values(b), meta.uid, seed=161804)
    with np.load(ROOT / "rm_test.npz") as rm:
        actual, predicted, constant = rm["truth"], rm["predictions"].mean(0), rm["constant"]
        reward = dict(
            mse=float(np.mean((predicted - actual) ** 2)),
            constant_mse=float(np.mean((constant - actual) ** 2)),
            mae=float(np.abs(predicted - actual).mean()),
        )
        # Event-weighted MSE difference with user-cluster bootstrap, not macro MSE.
        errors = (
            pd.DataFrame(
                dict(uid=meta.uid, error=(predicted - actual) ** 2 - (constant - actual) ** 2)
            )
            .groupby("uid")
            .error.agg(["sum", "size"])
        )
        rng = np.random.default_rng(161804)
        draw = rng.integers(len(errors), size=(3000, len(errors)))
        delta = errors["sum"].to_numpy()[draw].sum(1) / errors["size"].to_numpy()[draw].sum(1)
        reward["mse_difference_ci95"] = np.quantile(delta, [0.025, 0.975]).tolist()
        reward["controls"] = {
            name: float(np.mean((rm[name] - actual) ** 2))
            for name in ("state_only", "item_only", "last20", "old_neural")
        }
        errors = (
            pd.DataFrame(
                dict(
                    uid=meta.uid, error=(predicted - actual) ** 2 - (rm["state_only"] - actual) ** 2
                )
            )
            .groupby("uid")
            .error.agg(["sum", "size"])
        )
        delta = errors["sum"].to_numpy()[draw].sum(1) / errors["size"].to_numpy()[draw].sum(1)
        reward["minus_state_only_ci95"] = np.quantile(delta, [0.025, 0.975]).tolist()
    cat = dict(np.load(ROOT / "catalog.npz"))
    data = study.load("test")
    support = (item_indices(truth.item_id.to_numpy(), cat["head"]) >= 0) | (
        (truth.item_index.to_numpy() >= 0)
        & (data["history"] == truth.item_index.to_numpy()[:, None]).any(1)
    )
    write_json(
        ROOT / "results.json",
        dict(
            methods=summaries,
            comparisons=intervals,
            rm=reward,
            users=meta.uid.nunique(),
            queries=len(meta),
            full_catalog=len(cat["items"]),
            head_coverage=float((item_indices(truth.item_id.to_numpy(), cat["head"]) >= 0).mean()),
            support_coverage=float(support.mean()),
            full_catalog_coverage=float((truth.item_index >= 0).mean()),
        ),
    )
    rows.to_parquet(ROOT / "query_metrics.parquet", index=False)
    print(read_json(ROOT / "results.json"), flush=True)


def publish():
    destination = Path("reports/task2b/evidence")
    destination.mkdir(parents=True, exist_ok=True)
    names = (
        "sample.json",
        "frozen.json",
        "prediction_seal.json",
        "opened.json",
        "results.json",
        "selection.json",
        "ce_recipe.json",
        "rm_search.json",
        "rm_constant.json",
        "rm_ablation.json",
        "development_controls.json",
        "audit.json",
        "prepared.json",
        "test_queries.parquet",
        "test_labels.parquet",
        "query_metrics.parquet",
        "rm_test.npz",
        "adapter_search_17.json",
        "adapter_search_42.json",
        "adapter_search_73.json",
    )
    for name in names:
        shutil.copy2(ROOT / name, destination / name)
    shutil.copytree(ROOT / "test_predictions", destination / "test_predictions", dirs_exist_ok=True)
    shutil.copy2(ROOT / "test.npz", destination / "test.npz")
    with np.load(ROOT / "test.npz") as data:
        np.savez_compressed(
            destination / "test_support.npz", history=data["history"], users=data["users"]
        )
    cat = dict(np.load(ROOT / "catalog.npz"))
    np.savez_compressed(destination / "catalog.npz", items=cat["items"], head=cat["head"])
    files = [p for p in destination.rglob("*") if p.is_file() and p.name != "checksums.json"]
    write_json(
        destination / "checksums.json",
        {p.relative_to(destination).as_posix(): file_hash(p) for p in files},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("forecast", "labels", "results", "publish"))
    args = parser.parse_args()
    globals()[args.stage]()
