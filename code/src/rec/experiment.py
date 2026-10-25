from __future__ import annotations

import json
import logging
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from catboost import CatBoostRanker, Pool

from rec.data import download, load_sample
from rec.features import CAT_FEATURES, FeatureBuilder, search_kernels
from rec.io import environment, file_hash, write_json
from rec.metrics import paired_bootstrap, summarize, user_metrics
from rec.protocol import Config, Fold, future_truth, history_before, temporal_folds
from rec.retrieval import retrieve

LOG = logging.getLogger(__name__)


@contextmanager
def measurement(records: list[dict], stage: str):
    start = time.perf_counter()
    process = psutil.Process()
    peak = [process.memory_info().rss]
    stop = threading.Event()

    def sample() -> None:
        while not stop.wait(0.2):
            peak[0] = max(peak[0], process.memory_info().rss)

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()
        records.append(
            {
                "stage": stage,
                "seconds": time.perf_counter() - start,
                "peak_process_rss_mb": peak[0] / 1024**2,
            }
        )
        LOG.info(
            "%s: %.1f s; peak RSS %.0f MB",
            stage,
            records[-1]["seconds"],
            records[-1]["peak_process_rss_mb"],
        )


def prepare_fold(
    events: pd.DataFrame,
    items: pd.DataFrame,
    users: pd.DataFrame,
    fold: Fold,
    config: Config,
    output: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    history = history_before(events, fold.cutoff)
    retrieval = retrieve(history, items, fold.cutoff, config)
    truth_all = future_truth(events, fold, history)
    truth = truth_all[truth_all.user_id.isin(retrieval.eligible_users)].copy()
    # Save computation on users without a measurable ranking target. Their existence and
    # target items never affect retrieval fitting, scores, candidate membership or features.
    pairs = retrieval.candidates[retrieval.candidates.user_id.isin(truth.user_id)].reset_index(
        drop=True
    )
    frame, groups = FeatureBuilder(history, items, users, fold.cutoff).build(pairs, config)
    labels = truth.assign(label=np.int8(1))
    frame = frame.merge(labels, on=["user_id", "item_id"], how="left", validate="one_to_one")
    frame["label"] = frame.label.fillna(0).astype(np.int8)
    frame["query_id"] = fold.name + ":" + frame.user_id.astype(str)
    diagnostics = {
        **retrieval.diagnostics,
        "fold": fold.name,
        "cutoff": str(fold.cutoff),
        "end_exclusive": str(fold.end),
        "history_rows": len(history),
        "future_rows_all": len(truth_all),
        "future_users_all": truth_all.user_id.nunique(),
        "warm_active_users": truth.user_id.nunique(),
        "warm_future_rows": len(truth),
        "cold_item_targets": int((~truth.item_id.isin(retrieval.catalog)).sum()),
        "labeled_candidates": len(frame),
        "positive_candidates": int(frame.label.sum()),
    }
    output.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output / f"{fold.name}.parquet", index=False)
    truth.to_parquet(output / f"{fold.name}_truth.parquet", index=False)
    write_json(output / f"{fold.name}_audit.json", diagnostics)
    return frame, truth, groups


def make_pool(frame: pd.DataFrame, columns: list[str], training: bool = False) -> Pool:
    if training:
        # Constant-label queries have no ranking pairs and contribute no ranking gradient.
        mixed = frame.groupby("query_id").label.transform("nunique") > 1
        frame = frame.loc[mixed]
    if frame.empty:
        raise ValueError("No trainable ranking queries; increase sample size or candidate coverage")
    cats = [c for c in columns if c in CAT_FEATURES]
    return Pool(frame[columns], label=frame.label, group_id=frame.query_id, cat_features=cats)


def fit_model(
    train: pd.DataFrame,
    columns: list[str],
    config: Config,
    depth: int,
    seed: int,
    iterations: int | None = None,
) -> CatBoostRanker:
    model = CatBoostRanker(
        iterations=iterations or config.iterations,
        depth=depth,
        learning_rate=config.learning_rate,
        loss_function=f"YetiRank:mode=NDCG;top={config.k}",
        random_seed=seed,
        thread_count=config.threads,
        l2_leaf_reg=5,
        allow_writing_files=False,
        verbose=False,
    )
    model.fit(make_pool(train, columns, training=True))
    return model


def score_model(model: CatBoostRanker, frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return model.predict(prediction_pool(frame, columns))


def prediction_pool(frame: pd.DataFrame, columns: list[str]) -> Pool:
    return Pool(frame[columns], cat_features=[c for c in columns if c in CAT_FEATURES])


def tune_trial(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    truth: pd.DataFrame,
    columns: list[str],
    config: Config,
    depth: int,
) -> dict:
    """Choose iteration using end-to-end NDCG, including missed retrieval targets."""
    start = time.perf_counter()
    model = fit_model(train, columns, config, depth, config.model_seeds[0])
    pool = prediction_pool(validation, columns)
    best = {"ndcg": -1.0}
    checkpoint = min(25, config.iterations)
    for index, scores in enumerate(model.staged_predict(pool, eval_period=checkpoint)):
        iteration = min((index + 1) * checkpoint, model.tree_count_)
        per_user = user_metrics(validation, scores, truth, truth.user_id.unique(), config.k)
        value = float(per_user.ndcg.mean())
        if value > best["ndcg"]:
            best = {"ndcg": value, "iterations": iteration}
    return {
        **best,
        "depth": depth,
        "features": len(columns),
        "seconds": time.perf_counter() - start,
    }


def select_recipe(
    train: pd.DataFrame,
    search: pd.DataFrame,
    truth: pd.DataFrame,
    groups: dict,
    config: Config,
    output: Path,
    resources: list,
) -> dict:
    trials = []

    def trial(name: str, columns: list[str], depth: int) -> dict:
        with measurement(resources, f"search/{name}"):
            result = {"trial": name, **tune_trial(train, search, truth, columns, config, depth)}
        trials.append(result)
        write_json(output / "search_trials.json", trials)
        return result

    manual = groups["manual"]
    baselines = [trial(f"manual_depth{depth}", manual, depth) for depth in config.depths]
    best_manual = max(baselines, key=lambda r: r["ndcg"])
    depth = best_manual["depth"]
    chosen = []
    columns = list(manual)
    incumbent = best_manual
    remaining = [kernel.name for kernel in search_kernels(config)]
    for step in range(config.search_steps):
        candidates = []
        for name in remaining:
            result = trial(f"step{step + 1}/{name}", columns + groups[name], depth)
            candidates.append((name, result))
        if not candidates:
            break
        name, result = max(candidates, key=lambda pair: pair[1]["ndcg"])
        if result["ndcg"] < incumbent["ndcg"] + config.min_gain:
            break
        chosen.append(name)
        remaining.remove(name)
        columns += groups[name]
        incumbent = result
    all_columns = list(
        dict.fromkeys(
            manual + [c for name in [k.name for k in search_kernels(config)] for c in groups[name]]
        )
    )
    expanded = trial("all_features", all_columns, depth)
    no_relations = [c for c in manual if not c.startswith("ui_")]
    ablated = trial("manual_without_relational_aggregates", no_relations, depth)
    # All methods are predeclared. Confirmation and test never select the winner or recipe.
    recipe = {
        "manual": {"columns": manual, **best_manual},
        "auto": {"columns": columns, "selected_groups": chosen, **incumbent},
        "all_features": {"columns": all_columns, **expanded},
        "no_relations": {"columns": no_relations, **ablated},
        "selection_fold": "search",
        "confirmation_used_for_selection": False,
        "test_used_for_selection": False,
        "search_trials": len(trials),
    }
    write_json(output / "frozen_recipe.json", recipe)
    return recipe


def evaluate_methods(
    train: pd.DataFrame,
    frame: pd.DataFrame,
    truth: pd.DataFrame,
    recipe: dict,
    config: Config,
    output: Path,
    fold_name: str,
    resources: list,
) -> tuple[list[dict], dict[str, pd.DataFrame]]:
    output.mkdir(parents=True, exist_ok=True)
    summaries, per_method = [], {}
    population = np.sort(truth.user_id.unique())
    for name, column in (
        ("popularity", "retrieval_pop"),
        ("item_knn", "retrieval_cf"),
        ("rrf", "retrieval_rrf"),
    ):
        scores = frame[column].to_numpy()
        per_user = user_metrics(frame, scores, truth, population, config.k)
        per_method[name] = per_user
        summaries.append(
            {
                "fold": fold_name,
                "method": name,
                "seed": None,
                **summarize(per_user, frame, scores, config.k),
            }
        )
        per_user.to_parquet(output / f"{name}_users.parquet", index=False)
    for name in ("manual", "auto", "all_features", "no_relations"):
        specification = recipe[name]
        columns = specification["columns"]
        user_runs = []
        for seed in config.model_seeds:
            with measurement(resources, f"{fold_name}/{name}/seed{seed}"):
                model = fit_model(
                    train,
                    columns,
                    config,
                    specification["depth"],
                    seed,
                    specification["iterations"],
                )
                scores = score_model(model, frame, columns)
                per_user = user_metrics(frame, scores, truth, population, config.k)
                summaries.append(
                    {
                        "fold": fold_name,
                        "method": name,
                        "seed": seed,
                        "features": len(columns),
                        "iterations": specification["iterations"],
                        **summarize(per_user, frame, scores, config.k),
                    }
                )
                per_user.to_parquet(output / f"{name}_seed{seed}_users.parquet", index=False)
                predictions = frame[["user_id", "item_id", "label"]].assign(score=scores)
                predictions.to_parquet(
                    output / f"{name}_seed{seed}_predictions.parquet", index=False
                )
                model.save_model(str(output / f"{name}_seed{seed}.cbm"))
                importance = pd.DataFrame(
                    {
                        "feature": columns,
                        "importance": model.get_feature_importance(
                            make_pool(train, columns, training=True), type="PredictionValuesChange"
                        ),
                    }
                )
                importance.sort_values("importance", ascending=False).to_csv(
                    output / f"{name}_seed{seed}_importance.csv", index=False
                )
            user_runs.append(per_user)
        # Average per-seed metrics, not predictions: these are separate fits, not an ensemble.
        averaged = pd.concat(user_runs).groupby("user_id", as_index=False).mean(numeric_only=True)
        per_method[name] = averaged
        averaged.to_parquet(output / f"{name}_users.parquet", index=False)
    write_json(output / "metrics.json", summaries)
    return summaries, per_method


def run(config: Config, raw: Path, output: Path) -> dict:
    if (output / "config.json").exists():
        raise FileExistsError(f"Run directory already exists: {output}; use a new --output")
    output.mkdir(parents=True, exist_ok=True)
    resources = []
    write_json(output / "config.json", config.to_dict())
    write_json(output / "environment.json", environment())
    with measurement(resources, "load_data"):
        download(raw)
        events, items, users, sample = load_sample(raw, config.n_users, config.sample_seed)
    write_json(output / "sample.json", sample)
    write_json(
        output / "source_manifest.json", json.loads((raw / "manifest.json").read_text("utf-8"))
    )
    folds = temporal_folds(pd.Timestamp(sample["source_end_exclusive"]), config)
    write_json(output / "protocol.json", [vars(fold) for fold in folds])
    prepared = {}
    groups = None
    for fold in folds[:-1]:
        with measurement(resources, f"prepare/{fold.name}"):
            frame, truth, groups = prepare_fold(
                events, items, users, fold, config, output / "cache"
            )
        prepared[fold.name] = (frame, truth)
        write_json(output / "resources.json", resources)
    train = pd.concat(
        [prepared[f.name][0] for f in folds if f.name.startswith("train_")], ignore_index=True
    )
    assert groups is not None
    write_json(output / "feature_groups.json", groups)
    recipe = select_recipe(train, *prepared["search"], groups, config, output, resources)
    frozen_hash = file_hash(output / "frozen_recipe.json")
    confirmation_train = pd.concat([train, prepared["search"][0]], ignore_index=True)
    confirm_results, _ = evaluate_methods(
        confirmation_train,
        *prepared["confirm"],
        recipe,
        config,
        output / "confirm",
        "confirm",
        resources,
    )
    final_train = pd.concat([confirmation_train, prepared["confirm"][0]], ignore_index=True)
    # No test labels/candidates/metrics have been materialized before freezing every method.
    test_fold = folds[-1]
    with measurement(resources, "prepare/test"):
        test, test_truth, _ = prepare_fold(
            events, items, users, test_fold, config, output / "cache"
        )
    test_results, per_method = evaluate_methods(
        final_train, test, test_truth, recipe, config, output / "test", "test", resources
    )
    comparisons = {}
    for name in ("auto", "all_features", "no_relations"):
        comparisons[name] = paired_bootstrap(
            per_method["manual"], per_method[name], config.sample_seed, config.bootstrap_samples
        )
    if file_hash(output / "frozen_recipe.json") != frozen_hash:
        raise RuntimeError("Recipe changed after freeze")
    result = {
        "confirmation": confirm_results,
        "test": test_results,
        "paired_comparisons_vs_manual": comparisons,
        "frozen_recipe_sha256": frozen_hash,
    }
    write_json(output / "results.json", result)
    write_json(output / "resources.json", resources)
    write_json(
        output / "complete.json",
        {
            "status": "complete",
            "recipe_sha256": frozen_hash,
            "results_sha256": file_hash(output / "results.json"),
        },
    )
    return result
