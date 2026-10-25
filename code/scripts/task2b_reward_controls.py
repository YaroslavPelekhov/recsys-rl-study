"""Independent reward ablations on calibration/development, never test tuning."""

from pathlib import Path

import numpy as np
from catboost import CatBoostRegressor

from rec.io import write_json
from rec.task2b import item_indices, reward_features


def run():
    root = Path("runs/task2b")
    cat = dict(np.load(root / "catalog.npz"))
    train = dict(np.load(root / "rm_train.npz"))
    val = dict(np.load(root / "rm_calibration.npz"))
    dev = dict(np.load(root / "development.npz"))
    tx = reward_features(train, item_indices(train["target"], cat["items"]), cat)
    vx = reward_features(val, item_indices(val["target"], cat["items"]), cat)
    dx = reward_features(dev, item_indices(dev["target"], cat["items"]), cat)
    result = {}
    for name, cols in (("state_only", np.arange(13)), ("item_only", np.arange(13, 16))):
        model = CatBoostRegressor(
            iterations=400,
            depth=6,
            learning_rate=0.05,
            loss_function="RMSE",
            random_seed=17,
            thread_count=6,
            allow_writing_files=False,
            verbose=False,
        )
        model.fit(
            tx[:, cols],
            train["ratio"],
            eval_set=(vx[:, cols], val["ratio"]),
            early_stopping_rounds=40,
        )
        model.save_model(str(root / f"rm_{name}.cbm"))
        pred = np.clip(model.predict(dx[:, cols]), 0, 1)
        result[name] = dict(
            trees=model.tree_count_, development_mse=float(np.mean((pred - dev["ratio"]) ** 2))
        )
    predictions = []
    for seed in (17, 42, 73):
        model = CatBoostRegressor()
        model.load_model(str(root / f"rm_{seed}.cbm"))
        predictions.append(np.clip(model.predict(dx), 0, 1))
    for name, pred in (
        ("ensemble", np.mean(predictions, axis=0)),
        ("constant", np.full(len(dx), train["ratio"].mean())),
        ("last20", dev["state"][:, 2]),
    ):
        result[name] = dict(development_mse=float(np.mean((pred - dev["ratio"]) ** 2)))
    write_json(root / "rm_ablation.json", result)


def policy_controls():
    import research_task2b as study
    from rec.task2_metrics import ranking_rows, user_macro

    root = Path("runs/task2b")
    cat, data = dict(np.load(root / "catalog.npz")), study.load("development")
    saved = dict(np.load(root / "cache_development_17.npz"))
    result = {}
    for name, bias in (("sft", -40.0), ("copy_fixed", 0.0)):
        p = np.zeros(16, np.float32)
        p[14], p[13] = -3.5, bias
        result[name] = study.evaluate(p, data, saved, cat)[0]
    popularity = np.lexsort((np.arange(len(cat["items"])), -cat["counts"]))
    head = set(item_indices(cat["head"], cat["items"]).tolist())
    for name in ("repeat", "repeat_head", "popularity"):
        top = []
        for context in data["history"]:
            score = {}
            for pos, action in enumerate(context):
                if action >= 0 and (name != "repeat_head" or action in head):
                    score[int(action)] = score.get(int(action), 0.0) + 0.97 ** (63 - pos)
            chosen = (
                popularity[:10].tolist()
                if name == "popularity"
                else sorted(score, key=lambda a: (-score[a], a))[:10]
            )
            if len(chosen) < 10:
                chosen += [int(a) for a in popularity[:30] if a not in chosen][: 10 - len(chosen)]
            top.append(chosen)
        result[name] = user_macro(
            ranking_rows(np.asarray(top), item_indices(data["target"], cat["items"])), data["users"]
        )
    write_json(root / "development_controls.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    import sys

    policy_controls() if "--policy" in sys.argv else run()
