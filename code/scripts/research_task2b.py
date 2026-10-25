"""Bounded development-only copy / reward adapter experiments."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from catboost import CatBoostRegressor

from rec.io import file_hash, write_json
from rec.task2_data import read_json
from rec.task2_metrics import ranking_rows, user_macro
from rec.task2b import item_indices, reward_features

ROOT = Path("runs/task2b")


def subset(data, ids):
    return {k: v[ids] for k, v in data.items()}


def load(name):
    with np.load(ROOT / f"{name}.npz") as archive:
        return dict(archive)


def train_rm():
    cat = dict(np.load(ROOT / "catalog.npz"))
    train, val = load("rm_train"), load("rm_calibration")
    tx = reward_features(train, item_indices(train["target"], cat["items"]), cat)
    vx = reward_features(val, item_indices(val["target"], cat["items"]), cat)
    # Constant uses the same logged nonorganic training distribution as RM.
    write_json(ROOT / "rm_constant.json", {"mean": float(train["ratio"].mean())})
    log, best, depth = [], float("inf"), None
    start = time.perf_counter()
    for d in (4, 6):
        model = CatBoostRegressor(
            iterations=400,
            depth=d,
            learning_rate=0.05,
            loss_function="RMSE",
            random_seed=17,
            thread_count=6,
            allow_writing_files=False,
            verbose=False,
        )
        model.fit(tx, train["ratio"], eval_set=(vx, val["ratio"]), early_stopping_rounds=40)
        mse = float(np.mean((model.predict(vx) - val["ratio"]) ** 2))
        log.append(dict(depth=d, trees=model.tree_count_, calibration_mse=mse))
        if mse < best:
            depth, best = d, mse
            model.save_model(str(ROOT / "rm_17.cbm"))
    for seed in (42, 73):
        model = CatBoostRegressor(
            iterations=400,
            depth=depth,
            learning_rate=0.05,
            loss_function="RMSE",
            random_seed=seed,
            thread_count=6,
            allow_writing_files=False,
            verbose=False,
        )
        model.fit(tx, train["ratio"], eval_set=(vx, val["ratio"]), early_stopping_rounds=40)
        model.save_model(str(ROOT / f"rm_{seed}.cbm"))
        log.append(
            dict(
                seed=seed,
                depth=depth,
                trees=model.tree_count_,
                calibration_mse=float(np.mean((model.predict(vx) - val["ratio"]) ** 2)),
            )
        )
    write_json(
        ROOT / "rm_search.json",
        dict(
            trials=log,
            selected_depth=depth,
            seconds=time.perf_counter() - start,
            train=len(tx),
            calibration=len(vx),
        ),
    )
    print(read_json(ROOT / "rm_search.json"), flush=True)


def models():
    result = []
    for seed in (17, 42, 73):
        model = CatBoostRegressor()
        model.load_model(str(ROOT / f"rm_{seed}.cbm"))
        result.append(model)
    return result


def predict_reward(data, actions, cat, rms):
    out = []
    if actions.ndim == 1:
        actions = actions[:, None]
    for i in range(0, len(actions), 256):
        part = reward_features(subset(data, slice(i, i + 256)), actions[i : i + 256], cat)
        out.append(
            np.stack([np.clip(m.predict(part, thread_count=4), 0, 1) for m in rms]).reshape(
                3, min(256, len(actions) - i), -1
            )
        )
    return np.concatenate(out, axis=1).astype(np.float32)


def cache(seed, split="policy"):
    """Frozen decoder probabilities and independent MC actions; no test labels."""
    import torch
    from rec.task2_models import load_policy, seed_everything

    destination = ROOT / f"cache_{split}_{seed}.npz"
    if destination.exists():
        return
    seed_everything(seed)
    data, cat = load(split), dict(np.load(ROOT / "catalog.npz"))
    policy = load_policy(Path(f"runs/task2_final/models/sft_{seed}.pt"))
    head_idx = item_indices(cat["head"], cat["items"])
    target_head = np.searchsorted(cat["head"], data["target"])
    valid = target_head < len(cat["head"])
    valid[valid] &= cat["head"][target_head[valid]] == data["target"][valid]
    target_head = np.where(valid, target_head + 3, 0)
    values = {
        k: []
        for k in ("target_probability", "sample", "top", "top_probability", "history_probability")
    }
    rng = np.random.default_rng(seed + 900)
    start = time.perf_counter()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(data["x"]), 128):
            x = torch.as_tensor(data["x"][i : i + 128], dtype=torch.long, device="cuda")
            probs = policy(x, last_only=True).float().softmax(-1)
            n = len(x)
            values["target_probability"].append(
                probs[
                    torch.arange(n, device="cuda"),
                    torch.as_tensor(target_head[i : i + n], device="cuda"),
                ]
                .cpu()
                .numpy()
            )
            # 8 IID MC samples from the frozen head policy.
            cumulative = probs[:, 3:].cpu().numpy().cumsum(-1)
            u = rng.random((n, 8))
            sample = np.stack([np.searchsorted(cumulative[j], u[j]) for j in range(n)])
            sample = np.minimum(sample, len(head_idx) - 1)
            values["sample"].append(head_idx[sample])
            p, top = probs[:, 3:].topk(128, -1)
            values["top"].append(head_idx[top.cpu().numpy()])
            values["top_probability"].append(p.cpu().numpy())
            hidx = data["history"][i : i + n]
            hraw = cat["items"][np.maximum(hidx, 0)]
            from rec.task2_data import encode

            ht = encode(hraw, cat["head"])
            hp = probs.gather(1, torch.as_tensor(ht, dtype=torch.long, device="cuda")).cpu().numpy()
            hp[(ht < 3) | (hidx < 0)] = 0
            values["history_probability"].append(hp)
    out = {k: np.concatenate(v) for k, v in values.items()}
    out["target_probability"][~valid] = 0
    rms = models()
    actions = np.concatenate([out["sample"], np.maximum(data["history"], 0)], axis=1)
    out["reward"] = predict_reward(data, actions, cat, rms)
    np.savez_compressed(destination, **out)
    write_json(
        ROOT / f"cache_{split}_{seed}.json",
        dict(
            seconds=time.perf_counter() - start,
            queries=len(data["x"]),
            file_sha256=file_hash(destination),
        ),
    )
    print(f"Cached {split} seed{seed}: {len(data['x']):,}", flush=True)


def distribution(parameters, state, outcome, valid):
    import torch

    gate = torch.sigmoid(state @ parameters[:13] + parameters[13])
    # Positive decay, initialized at .03 per position; outcome weight starts0.
    age = torch.arange(63, -1, -1, dtype=state.dtype, device=state.device)
    logits = -torch.nn.functional.softplus(parameters[14]) * age + parameters[15] * outcome
    logits = logits.masked_fill(~valid, -1e4)
    copy = logits.softmax(-1) * valid
    copy = copy / copy.sum(-1, keepdim=True).clamp_min(1e-12)
    gate = gate * valid.any(-1)
    return gate, copy


def evaluate(parameters, data, saved, cat):
    import torch

    with torch.no_grad():
        gate, copy = distribution(
            torch.as_tensor(parameters),
            torch.as_tensor(data["state"]),
            torch.as_tensor(data["outcome"]),
            torch.as_tensor(data["history"] >= 0),
        )
    g, c = gate.numpy(), copy.numpy()
    tops, predicted, durations, repeats = [], [], [], []
    rms = models()
    # Pool contains frozen head top128 and ALL64 copy positions. Head-only items
    # below top128 cannot enter final top10: at least128 larger unmodified head
    # scores remain. Duplicate copy contributions are aggregated exactly.
    for j in range(len(g)):
        score = {
            int(a): (1 - g[j]) * float(p)
            for a, p in zip(saved["top"][j], saved["top_probability"][j], strict=True)
        }
        for pos, a in enumerate(data["history"][j]):
            if a < 0:
                continue
            score.setdefault(int(a), (1 - g[j]) * float(saved["history_probability"][j, pos]))
            score[int(a)] += g[j] * c[j, pos]
        top = sorted(score, key=lambda a: (-score[a], a))[:10]
        tops.append(top)
        repeats.append(top[0] in data["history"][j])
        durations.append(float(cat["lengths"][top[0]]))
    top = np.asarray(tops, np.int32)
    predicted = predict_reward(data, top[:, :1], cat, rms).mean(0)[:, 0]
    target = item_indices(data["target"], cat["items"]) if "target" in data else None
    metrics = user_macro(ranking_rows(top, target), data["users"]) if target is not None else {}
    metrics.update(
        dict(
            predicted_ratio=float(predicted.mean()),
            gate=float(g.mean()),
            repeat=float(np.mean(repeats)),
            duration=float(np.mean(durations)),
            top1_share=float(np.unique(top[:, 0], return_counts=True)[1].max() / len(top)),
        )
    )
    return metrics, dict(
        top=top,
        reward=predicted,
        gate=g,
        repeat=np.asarray(repeats),
        duration=np.asarray(durations),
    )


def train_adapter(seed=17):
    import torch
    from rec.task2_models import seed_everything

    seed_everything(seed)
    torch.set_num_threads(4)
    data, dev, cat = load("policy"), load("development"), dict(np.load(ROOT / "catalog.npz"))
    with np.load(ROOT / f"cache_policy_{seed}.npz") as f:
        saved = dict(f)
    with np.load(ROOT / f"cache_development_{seed}.npz") as f:
        search = dict(f)
    state, outcomes = torch.tensor(data["state"]), torch.tensor(data["outcome"])
    valid = torch.tensor(data["history"] >= 0)
    match = (
        torch.tensor(data["history"] == item_indices(data["target"], cat["items"])[:, None]) & valid
    )
    py = torch.tensor(saved["target_probability"])
    rewards = torch.tensor(saved["reward"])
    head_reward = rewards[:, :, :8].mean(-1).T
    copy_reward = rewards[:, :, 8:].permute(1, 0, 2)
    rng = np.random.default_rng(seed)
    initial = torch.zeros(16)
    initial[14] = -3.5
    parameter = torch.nn.Parameter(initial)
    optimizer = torch.optim.Adam([parameter], lr=0.01)
    log, best, best_parameters, best_step = [], -1, None, None
    budget = 1200 if seed == 17 else read_json(ROOT / "ce_recipe.json")["updates"]
    start = time.perf_counter()
    for step in range(1, budget + 1):
        ids = rng.integers(len(state), size=256)
        gate, copy = distribution(parameter, state[ids], outcomes[ids], valid[ids])
        probability = (1 - gate) * py[ids] + gate * (copy * match[ids]).sum(-1)
        loss = -probability.clamp_min(1e-12).log().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step in (300, 600, 1200) or step == budget:
            metrics, _ = evaluate(parameter.detach().numpy(), dev, search, cat)
            log.append(dict(update=step, loss=float(loss.detach()), **metrics))
            if seed != 17 or metrics["ndcg"] > best:
                best, best_parameters, best_step = metrics["ndcg"], parameter.detach().clone(), step
    assert best_parameters is not None
    np.save(ROOT / f"ce_{seed}.npy", best_parameters.numpy())
    if seed == 17:
        write_json(ROOT / "ce_recipe.json", dict(updates=best_step, search_ndcg=best))
    records = [
        dict(method="copy_ce", seed=seed, trajectory=log, seconds=time.perf_counter() - start)
    ]
    with torch.no_grad():
        reference_gate, reference_copy = distribution(best_parameters, state, outcomes, valid)
    configs = [("continue_ce", 0.0)] + [
        (kind, lam) for kind in ("reward_only", "joint", "gated") for lam in (0.1, 0.5, 2.0, 8.0)
    ]
    if seed != 17:
        configs = [("continue_ce", 0.0)] + [
            (k, v["lambda"]) for k, v in read_json(ROOT / "selection.json").items()
        ]
    for kind, lam in configs:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed + 100)
        parameter = torch.nn.Parameter(best_parameters.clone())
        optimizer = torch.optim.Adam([parameter], lr=0.003)
        start = time.perf_counter()
        admitted = []
        for _ in range(300):
            ids = rng.integers(len(state), size=256)
            gate, copy = distribution(parameter, state[ids], outcomes[ids], valid[ids])
            p = (1 - gate) * py[ids] + gate * (copy * match[ids]).sum(-1)
            ce = -p.clamp_min(1e-12).log().mean()
            expected_copy = (copy[:, None] * copy_reward[ids]).sum(-1)
            improvement = expected_copy.detach() - head_reward[ids]
            admit = (
                ((improvement.min(-1).values > 0) | (improvement.max(-1).values < 0)).float()
                if kind == "gated"
                else torch.ones_like(gate)
            )
            value = ((1 - gate[:, None]) * head_reward[ids] + gate[:, None] * expected_copy).mean(
                -1
            )
            rg = reference_gate[ids].clamp(0.000001, 0.999999)
            gg = gate.clamp(0.000001, 0.999999)
            # KL over latent branch+position bounds KL of marginal item policy.
            kl = (
                gg * (gg / rg).log()
                + (1 - gg) * ((1 - gg) / (1 - rg)).log()
                + gg
                * (copy * (copy.clamp_min(1e-12) / reference_copy[ids].clamp_min(1e-12)).log()).sum(
                    -1
                )
            )
            loss = (
                (0 if kind == "reward_only" else ce)
                - lam * (admit * value).mean()
                + 0.1 * kl.mean()
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            admitted.append(float(admit.mean()))
        parameters = parameter.detach().numpy()
        name = f"{kind}_{lam:g}_{seed}"
        np.save(ROOT / f"{name}.npy", parameters)
        metrics, _ = evaluate(parameters, dev, search, cat)
        records.append(
            dict(
                method=kind,
                name=name,
                seed=seed,
                **{"lambda": lam},
                admitted=float(np.mean(admitted)),
                **metrics,
                seconds=time.perf_counter() - start,
            )
        )
        print(records[-1], flush=True)
    write_json(ROOT / f"adapter_search_{seed}.json", records)
    if seed == 17:
        base = log[max(range(len(log)), key=lambda j: log[j]["ndcg"])]["predicted_ratio"]
        selection = {}
        for kind in ("reward_only", "joint", "gated"):
            trials = [r for r in records if r["method"] == kind]
            eligible = [r for r in trials if r["predicted_ratio"] >= base]
            chosen = max(eligible or trials, key=lambda r: r["ndcg"])
            selection[kind] = {
                "lambda": chosen["lambda"],
                "reward_guard_passed": bool(eligible),
                "search": chosen,
            }
        write_json(ROOT / "selection.json", selection)


def freeze():
    sources = [
        "docs/TASK2B_PROTOCOL.md",
        "src/rec/task2b.py",
        "scripts/research_task2b.py",
        "scripts/evaluate_task2b.py",
        "scripts/audit_task2b.py",
        "scripts/task2b_reward_controls.py",
        "src/rec/task2_data.py",
        "src/rec/task2_models.py",
        "src/rec/task2_metrics.py",
    ]
    files = [Path(x) for x in sources] + [
        ROOT / x
        for x in (
            "sample.json",
            "catalog.npz",
            "selection.json",
            "ce_recipe.json",
            "rm_constant.json",
            "rm_ablation.json",
            "development_controls.json",
            "audit.json",
        )
    ]
    files += list(ROOT.glob("*.npy")) + list(ROOT.glob("rm_*.cbm"))
    files += [Path(f"runs/task2_final/models/sft_{s}.pt") for s in (17, 42, 73)]
    files += [Path(f"runs/task2_final/models/rm_{s}.pt") for s in (17, 42, 73)]
    write_json(
        ROOT / "frozen.json",
        dict(
            files={str(p): file_hash(p) for p in files},
            primary="gated-minus-continue_ce",
            opened=False,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("rm", "cache", "adapter", "freeze"))
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    if args.stage == "rm":
        train_rm()
    elif args.stage == "cache":
        for split in ("policy", "development"):
            cache(args.seed, split)
    elif args.stage == "adapter":
        train_adapter(args.seed)
    else:
        freeze()
