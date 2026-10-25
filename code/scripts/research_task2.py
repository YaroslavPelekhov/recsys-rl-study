"""Reproducible stages of GPT-2 / reward-model / REINFORCE study.

No test labels are constructed until every frozen model's forecasts are saved.
Use --help; the default project lock remains unchanged.
"""

from __future__ import annotations

import argparse
import gc
import math
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch
from torch.nn import functional as F

from rec.io import file_hash, write_json
from rec.task2_data import (
    SPECIAL,
    context_memmap,
    prepare,
    read_json,
    reserve,
)
from rec.task2_metrics import (
    DIAGNOSTICS,
    behavior_summary,
    ranking_rows,
    reward_metrics,
    user_macro,
)
from rec.task2_models import (
    FeedbackNet,
    ItemGPT2,
    feedback,
    load_feedback,
    load_policy,
    rollout_train,
    save_model,
    seed_everything,
)

KINDS = ("ratio", "seconds", "return", "ensemble", "anchored")
BETAS = (0.02, 0.1, 0.5)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def amp():
    return torch.autocast("cuda", dtype=torch.bfloat16) if DEVICE == "cuda" else nullcontext()


def tensor(array, dtype=torch.long):
    if isinstance(array, np.ndarray) and not array.flags.writeable:
        array = array.copy()
    return torch.as_tensor(array, dtype=dtype, device=DEVICE)


def resource(start):
    return {
        "seconds": time.perf_counter() - start,
        "rss_mib_at_end": psutil.Process().memory_info().rss / 2**20,
        "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20
        if DEVICE == "cuda"
        else 0.0,
    }


@torch.no_grad()
def recommend(model, x, batch=192):
    model.eval()
    return np.concatenate(
        [
            model(tensor(x[i : i + batch]), last_only=True)
            .float()
            .topk(10, -1)
            .indices.cpu()
            .numpy()
            for i in range(0, len(x), batch)
        ]
    )


def search_metric(model, root):
    x = np.load(root / "search_x.npy", mmap_mode="r")
    meta = pd.read_parquet(root / "search_queries.parquet")
    y = pd.read_parquet(root / "search_labels.parquet").token.to_numpy()
    with amp():
        top = recommend(model, x)
    return user_macro(ranking_rows(top, y), meta.uid)["ndcg"]


def train_sft(root, seed=17, updates=None, budget=12000):
    destination = root / f"models/sft_{seed}.pt"
    if (
        destination.exists()
        and (root / f"logs/sft_{seed}.json").exists()
        and (root / "sft_recipe.json").exists()
    ):
        return
    seed_everything(seed)
    if updates is None:
        updates = budget if seed == 17 else read_json(root / "sft_recipe.json")["updates"]
    data = np.load(root / "train.npz")
    x, y = data["x"], data["y"]
    catalog = np.load(root / "catalog.npz")
    model = ItemGPT2(len(catalog["items"]) + SPECIAL).to(DEVICE)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    rng, trajectory, best, best_step = np.random.default_rng(seed), [], -1.0, 0
    start = time.perf_counter()
    for step in range(1, updates + 1):
        ids = rng.integers(len(x), size=64)
        warmup = 100 if budget == 1500 else 300
        rate = min(1.0, step / warmup) * (0.1 + 0.9 * (1 + math.cos(math.pi * step / budget)) / 2)
        for group in optimizer.param_groups:
            group["lr"] = 3e-4 * rate
        optimizer.zero_grad(set_to_none=True)
        # Same effective batch64, four microbatches: WDDM also needs host commit
        # for CUDA allocations on a nearly full desktop drive.
        denominator = int((y[ids] >= SPECIAL).sum())
        loss_value = 0.0
        for micro in np.array_split(ids, 4):
            with amp():
                logits = model(tensor(x[micro]))
                loss = (
                    F.cross_entropy(
                        logits.flatten(0, 1).float(), tensor(y[micro]).flatten(), reduction="sum"
                    )
                    / denominator
                )
            loss.backward()
            loss_value += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 500 == 0:
            print(f"SFT seed={seed} update={step} CE={loss_value:.4f}", flush=True)
        interval = 300 if budget == 1500 else 1500
        if step % interval == 0 or step == updates:
            value = search_metric(model, root) if seed == 17 else None
            trajectory.append(
                {"update": step, "train_ce_last_batch": loss_value, "search_ndcg": value}
            )
            if seed != 17 or value > best:
                best, best_step = value if value is not None else 0.0, step
                save_model(destination, model)
            model.train()
    write_json(
        root / f"logs/sft_{seed}.json",
        {
            "trajectory": trajectory,
            "parameters": sum(p.numel() for p in model.parameters()),
            "resources": resource(start),
        },
    )
    if seed == 17:
        write_json(
            root / "sft_recipe.json",
            {"updates": best_step, "search_ndcg": best, "model_sha256": file_hash(destination)},
        )


@torch.no_grad()
def evaluate_feedback(models, x, actions):
    out = []
    for start in range(0, len(x), 512):
        with amp():
            predicted = feedback(
                models, tensor(x[start : start + 512]), tensor(actions[start : start + 512])
            )
        out.append(predicted.float().cpu().numpy())
    return np.concatenate(out, axis=1)


def train_rm(root, seed):
    destination = root / f"models/rm_{seed}.pt"
    if destination.exists():
        return
    seed_everything(seed)
    data = np.load(root / "train.npz")
    x, a, y = context_memmap(root), data["rm_a"], data["rm_y"]
    catalog = np.load(root / "catalog.npz")
    model = FeedbackNet(len(catalog["items"]) + SPECIAL, catalog["lengths"]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rng, trajectory, start = np.random.default_rng(seed), [], time.perf_counter()
    model.train()
    for step in range(1, 801):
        ids = rng.integers(len(x), size=512)
        optimizer.zero_grad(set_to_none=True)
        with amp():
            loss = F.binary_cross_entropy_with_logits(
                model(tensor(x[ids]), tensor(a[ids])).float(), tensor(y[ids], torch.float32)
            )
        loss.backward()
        optimizer.step()
        if step % 200 == 0:
            trajectory.append({"update": step, "train_bce_last_batch": loss.item()})
            print(f"RM seed={seed} update={step} BCE={loss.item():.4f}", flush=True)
    save_model(destination, model)
    sx = np.load(root / "search_x.npy")
    sy = pd.read_parquet(root / "search_labels.parquet")
    model.eval()
    pred = evaluate_feedback([model], sx, sy.token.to_numpy())[0]
    write_json(
        root / f"logs/rm_{seed}.json",
        {
            "trajectory": trajectory,
            "search_metrics": reward_metrics(sy[["ratio", "return24"]], pred),
            "resources": resource(start),
        },
    )


def train_rl(root, kind, beta, seed=17, length=5):
    name = f"{kind}_b{beta:g}_h{length}_{seed}"
    destination = root / f"models/{name}.pt"
    if destination.exists():
        return destination
    seed_everything(seed)
    policy = load_policy(root / f"models/sft_{seed}.pt", DEVICE)
    reference = load_policy(root / f"models/sft_{seed}.pt", DEVICE)
    reference.requires_grad_(False)
    models = [
        load_feedback(root / f"models/rm_{s}.pt", DEVICE).requires_grad_(False)
        for s in (17, 42, 73)
    ]
    policy.eval()  # policy probabilities must match sampling; dropout disabled
    optimizer = torch.optim.AdamW(policy.parameters(), lr=5e-5, weight_decay=0.01)
    data = np.load(root / "train.npz")
    x, users = context_memmap(root), data["rm_u"]
    choices = [np.flatnonzero(users == u) for u in np.unique(users)]
    rng, trajectory, start = np.random.default_rng(seed + 4000), [], time.perf_counter()
    for step in range(1, 121):
        selected = rng.integers(len(choices), size=96)
        ids = np.array([rng.choice(choices[u]) for u in selected])
        optimizer.zero_grad(set_to_none=True)
        with amp():
            loss, diagnostic = rollout_train(
                policy, reference, models, tensor(x[ids]), kind, beta, length
            )
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        if step % 20 == 0:
            row = {
                "update": step,
                "policy_loss": loss.item(),
                "gradient_norm": float(norm),
                "reward": diagnostic[0].item(),
                "ratio": diagnostic[1].item(),
                "exact_kl": diagnostic[2].item(),
            }
            trajectory.append(row)
            print(
                f"RL {name} {step}/120 reward={row['reward']:.4f} KL={row['exact_kl']:.4f}",
                flush=True,
            )
    save_model(destination, policy)
    write_json(
        root / f"logs/{name}.json",
        {
            "kind": kind,
            "beta": beta,
            "length": length,
            "seed": seed,
            "trajectory": trajectory,
            "resources": resource(start),
        },
    )
    return destination


def train_ce(root, seed):
    destination = root / f"models/ce_{seed}.pt"
    if destination.exists():
        return
    seed_everything(seed)
    policy = load_policy(root / f"models/sft_{seed}.pt", DEVICE).train()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=5e-5, weight_decay=0.01)
    arrays = np.load(root / "train.npz")
    x, y = arrays["x"], arrays["y"]
    rng, start = np.random.default_rng(seed + 4000), time.perf_counter()
    for _ in range(120):
        ids = rng.integers(len(x), size=64)
        optimizer.zero_grad(set_to_none=True)
        denominator = int((y[ids] >= SPECIAL).sum())
        for micro in np.array_split(ids, 4):
            with amp():
                loss = (
                    F.cross_entropy(
                        policy(tensor(x[micro])).flatten(0, 1).float(),
                        tensor(y[micro]).flatten(),
                        reduction="sum",
                    )
                    / denominator
                )
            loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
    save_model(destination, policy)
    write_json(
        root / f"logs/ce_{seed}.json",
        {"updates": 120, "last_ce": loss.item(), "resources": resource(start)},
    )


@torch.no_grad()
def score_policy(root, path, split="search", seed=17):
    directory = root / f"{split}_predictions"
    directory.mkdir(exist_ok=True)
    destination = directory / (path.stem + ".npz")
    if destination.exists():
        return destination
    model = load_policy(path, DEVICE)
    reference = load_policy(root / f"models/sft_{seed}.pt", DEVICE)
    rms = [load_feedback(root / f"models/rm_{s}.pt", DEVICE) for s in (17, 42, 73)]
    audit = load_feedback(root / "models/rm_101.pt", DEVICE)
    x = np.load(root / f"{split}_x.npy", mmap_mode="r")
    top, rows = [], []
    for start in range(0, len(x), 192):
        context = tensor(x[start : start + 192])
        with amp():
            lp = model(context, last_only=True).float().log_softmax(-1)
            ref = reference(context, last_only=True).float().log_softmax(-1)
            chosen = lp.topk(10, -1).indices
            p = feedback(rms, context, chosen[:, 0]).float()
            auditor = audit(context, chosen[:, 0]).sigmoid().float()
        chosen_np = chosen.cpu().numpy()
        top.append(chosen_np)
        rows.append(
            np.stack(
                (
                    p[..., 0].mean(0).cpu().numpy(),
                    p[..., 0].std(0, unbiased=False).cpu().numpy(),
                    auditor[:, 0].cpu().numpy(),
                    p[..., 1].mean(0).cpu().numpy(),
                    (lp.exp() * (lp - ref)).sum(-1).cpu().numpy(),
                    -(lp.exp() * lp).sum(-1).cpu().numpy(),
                    (chosen[:, :1] == context).any(1).float().cpu().numpy(),
                    np.expm1(rms[0].lengths[chosen[:, 0]].float().cpu().numpy() * 7),
                ),
                -1,
            )
        )
    np.savez_compressed(destination, top=np.concatenate(top), diagnostics=np.concatenate(rows))
    return destination


def search_summary(root, predictions):
    p = np.load(predictions)
    meta = pd.read_parquet(root / "search_queries.parquet")
    labels = pd.read_parquet(root / "search_labels.parquet")
    result = user_macro(ranking_rows(p["top"], labels.token), meta.uid)
    result.update(
        behavior_summary(
            p["top"], pd.DataFrame(p["diagnostics"], columns=DIAGNOSTICS), meta.uid, 16384
        )
    )
    return result


def development(root):
    (root / "models").mkdir(exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    train_sft(root)
    for seed in (17, 42, 73, 101):
        train_rm(root, seed)
        gc.collect()
        torch.cuda.empty_cache()
    trials = []
    baseline = search_summary(root, score_policy(root, root / "models/sft_17.pt"))
    for kind in KINDS:
        for beta in BETAS:
            path = train_rl(root, kind, beta)
            result = search_summary(root, score_policy(root, path))
            trials.append({"kind": kind, "beta": beta, "metrics": result})
            write_json(root / "search_trials.json", {"sft": baseline, "trials": trials})
            gc.collect()
            torch.cuda.empty_cache()
    selected = {}
    for kind in KINDS:
        options = [r for r in trials if r["kind"] == kind]
        feasible = [r for r in options if r["metrics"]["ndcg"] >= 0.95 * baseline["ndcg"]]
        selected[kind] = max(
            feasible or options,
            key=lambda r: (
                r["metrics"]["predicted_ratio"] if feasible else r["metrics"]["ndcg"],
                r["beta"],
            ),
        )["beta"]
    write_json(
        root / "rl_recipe.json",
        {
            "selected_beta": selected,
            "sft_updates": read_json(root / "sft_recipe.json")["updates"],
            "primary": ["anchored_b0.1_h5", "ensemble_b0.1_h5"],
            "search_trials": len(trials),
            "test_opened": False,
        },
    )
    final_fit(root)


def final_fit(root):
    recipe = read_json(root / "rl_recipe.json")
    configs = {(kind, beta, 5) for kind, beta in recipe["selected_beta"].items()}
    configs.update({("anchored", 0.1, 5), ("ensemble", 0.1, 5), ("ensemble", 0.1, 1)})
    for seed in (17, 42, 73):
        train_sft(root, seed)
        train_ce(root, seed)
        for kind, beta, length in sorted(configs):
            train_rl(root, kind, beta, seed, length)
            gc.collect()
            torch.cuda.empty_cache()
    files = [
        Path("docs/TASK2_PROTOCOL.md"),
        Path("docs/TASK2_AMENDMENT_01.md"),
        *Path("src/rec").glob("task2*.py"),
        Path("scripts/research_task2.py"),
        Path("scripts/evaluate_task2.py"),
        Path("scripts/audit_task2.py"),
        root / "rl_recipe.json",
        root / "sample.json",
        root / "catalog.npz",
        root / "protocol.json",
        root / "train.npz",
        root / "rm_contexts.npy",
        root / "search_trials.json",
        *sorted((root / "models").glob("*.pt")),
    ]
    write_json(
        root / "frozen.json",
        {
            "files": {p.as_posix(): file_hash(p) for p in files},
            "test_labels_constructed": False,
            "configs": [list(c) for c in sorted(configs)],
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase", choices=("reserve", "prepare", "development", "final-fit", "sft", "rm", "rl")
    )
    parser.add_argument("--root", type=Path, default=Path("runs/task2_final"))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--kind", choices=KINDS, default="ratio")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--budget", type=int, choices=(1500, 12000), default=12000)
    args = parser.parse_args()
    (args.root / "models").mkdir(parents=True, exist_ok=True)
    (args.root / "logs").mkdir(exist_ok=True)
    {
        "reserve": lambda: reserve(args.root),
        "prepare": lambda: prepare(args.root),
        "development": lambda: development(args.root),
        "final-fit": lambda: final_fit(args.root),
        "sft": lambda: train_sft(args.root, args.seed, budget=args.budget),
        "rm": lambda: train_rm(args.root, args.seed),
        "rl": lambda: train_rl(args.root, args.kind, args.beta, args.seed),
    }[args.phase]()


if __name__ == "__main__":
    main()
