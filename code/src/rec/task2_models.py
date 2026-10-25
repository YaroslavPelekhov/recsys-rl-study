"""Small GPT-2 architecture, independent feedback predictors and policy gradient.

PyTorch is optional for the original feature-search project. This module is
imported only by Task 2 training; report verification remains CPU/NumPy only.
"""

from __future__ import annotations

import math
import os

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from rec.task2_data import CONTEXT, SPECIAL


def seed_everything(seed):
    os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = True


class CausalBlock(nn.Module):
    def __init__(self, width, heads, layers, dropout):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv = nn.Linear(width, width * 3)
        self.proj = nn.Linear(width, width)
        self.mlp = nn.Sequential(
            nn.Linear(width, width * 4), nn.GELU(approximate="tanh"), nn.Linear(width * 4, width)
        )
        self.dropout, self.heads = dropout, heads
        for layer in (self.qkv, self.proj, self.mlp[0], self.mlp[-1]):
            nn.init.normal_(layer.weight, std=0.02)
            nn.init.zeros_(layer.bias)
        nn.init.normal_(self.proj.weight, std=0.02 / math.sqrt(2 * layers))
        nn.init.normal_(self.mlp[-1].weight, std=0.02 / math.sqrt(2 * layers))

    def forward(self, x, mask):
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)

        def reshape(a):
            return a.reshape(b, t, self.heads, d // self.heads).transpose(1, 2)

        a = F.scaled_dot_product_attention(
            reshape(q),
            reshape(k),
            reshape(v),
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        a = a.transpose(1, 2).reshape(b, t, d)
        x = x + F.dropout(self.proj(a), p=self.dropout, training=self.training)
        return x + F.dropout(self.mlp(self.ln2(x)), p=self.dropout, training=self.training)


class ItemGPT2(nn.Module):
    def __init__(self, vocab_size, width=128, layers=3, heads=4, dropout=0.1):
        super().__init__()
        self.config = dict(
            vocab_size=vocab_size, width=width, layers=layers, heads=heads, dropout=dropout
        )
        self.token = nn.Embedding(vocab_size, width, padding_idx=0)
        self.position = nn.Embedding(CONTEXT, width)
        nn.init.normal_(self.token.weight, std=0.02)
        nn.init.normal_(self.position.weight, std=0.02)
        self.blocks = nn.ModuleList(
            [CausalBlock(width, heads, layers, dropout) for _ in range(layers)]
        )
        self.ln = nn.LayerNorm(width)

    def forward(self, ids, last_only=False):
        _, t = ids.shape
        occupied = ids != 0
        positions = (occupied.long().cumsum(-1) - 1).clamp_min(0)
        x = self.token(ids) + self.position(positions)
        causal = torch.ones(t, t, device=ids.device, dtype=torch.bool).tril()
        # Diagonal for padding queries avoids an all-masked SDPA row; their
        # hidden states never enter a loss or a real-token attention context.
        mask = (
            (causal[None, :, :] & occupied[:, None, :])
            | torch.eye(t, dtype=torch.bool, device=ids.device)[None]
        )[:, None]
        for block in self.blocks:
            x = block(x, mask)
        x = self.ln(x[:, -1:] if last_only else x)
        logits = F.linear(x, self.token.weight)
        logits[..., :SPECIAL] = -1e4
        return logits[:, 0] if last_only else logits


class FeedbackNet(nn.Module):
    def __init__(self, vocab_size, lengths, width=48):
        super().__init__()
        self.config = dict(vocab_size=vocab_size, width=width)
        self.item = nn.Embedding(vocab_size, width, padding_idx=0)
        self.item_bias = nn.Embedding(vocab_size, 2)
        self.register_buffer(
            "lengths",
            torch.tensor(np.r_[np.zeros(SPECIAL), np.log1p(lengths) / 7], dtype=torch.float32),
        )
        self.mlp = nn.Sequential(
            nn.Linear(width * 4 + 2, 96), nn.SiLU(), nn.Linear(96, 32), nn.SiLU(), nn.Linear(32, 2)
        )
        nn.init.normal_(self.item.weight, std=0.05)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, context, actions):
        h = self.item(context)
        valid = (context >= SPECIAL).float()
        mean = (h * valid[:, :, None]).sum(1) / valid.sum(1).clamp_min(1)[:, None]
        last = h[:, -1]
        candidate = self.item(actions)
        numeric = torch.stack((self.lengths[actions], valid.mean(1)), -1)
        features = torch.cat((mean, last, candidate, mean * candidate, numeric), -1)
        return self.mlp(features) + self.item_bias(actions)


def discounted_returns(rewards, gamma=0.9):
    out = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[:, 0])
    for step in reversed(range(rewards.shape[1])):
        running = rewards[:, step] + gamma * running
        out[:, step] = running
    return out


def reinforce_loss(log_prob, regularized_rewards, gamma=0.9):
    returns = discounted_returns(regularized_rewards.detach(), gamma)
    # Other independent trajectories give an action-independent baseline.
    baseline = (returns.sum(0, keepdim=True) - returns) / max(1, len(returns) - 1)
    weights = gamma ** torch.arange(returns.shape[1], device=returns.device)
    return -((returns - baseline).detach() * log_prob * weights).sum(1).mean()


@torch.no_grad()
def feedback(models, context, actions):
    return torch.stack([m(context, actions).sigmoid() for m in models])


def reward_value(kind, predictions, anchored_predictions, duration):
    ratio = predictions[..., 0]
    if kind == "ratio":
        return ratio[0]
    if kind == "seconds":
        return ratio[0] * duration.clamp(max=600) / 600
    if kind == "return":
        return 0.5 * ratio[0] + 0.5 * predictions[0, ..., 1]
    value = ratio.mean(0) - 0.5 * ratio.std(0, unbiased=False)
    if kind == "anchored":
        value = value - (ratio.mean(0) - anchored_predictions[..., 0].mean(0)).abs()
    elif kind != "ensemble":
        raise ValueError(kind)
    return value


def rollout_train(policy, reference, rewards, context, kind, beta, length=5):
    initial = context
    logps, adjusted, diagnostics = [], [], []
    for _ in range(length):
        logits = policy(context, last_only=True).float()
        log_policy = logits.log_softmax(-1)
        distribution = torch.distributions.Categorical(logits=logits)
        action = distribution.sample()
        logp = log_policy.gather(1, action[:, None])[:, 0]
        with torch.no_grad():
            log_reference = reference(context, last_only=True).float().log_softmax(-1)
            ref = log_reference.gather(1, action[:, None])[:, 0]
            predicted = feedback(rewards, context, action)
            anchored = feedback(rewards, initial, action) if kind == "anchored" else predicted
            duration = torch.expm1(rewards[0].lengths[action] * 7)
            reward = reward_value(kind, predicted, anchored, duration)
            exact_kl = (log_policy.exp() * (log_policy - log_reference)).sum(-1)
        logps.append(logp)
        adjusted.append(reward - beta * (logp.detach() - ref))
        diagnostics.append(torch.stack((reward, predicted[..., 0].mean(0), exact_kl), -1).mean(0))
        context = torch.cat((context[:, 1:], action[:, None]), 1)
    return reinforce_loss(torch.stack(logps, 1), torch.stack(adjusted, 1)), torch.stack(
        diagnostics
    ).mean(0)


def save_model(path, model):
    # Weights only, no optimizer states: bounded disk use and safe torch load.
    torch.save(
        {"config": model.config, "state_dict": {k: v.cpu() for k, v in model.state_dict().items()}},
        path,
    )


def load_policy(path, device="cuda"):
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = ItemGPT2(**state["config"])
    model.load_state_dict(state["state_dict"])
    return model.to(device).eval()


def load_feedback(path, device="cuda"):
    state = torch.load(path, map_location="cpu", weights_only=True)
    lengths = np.expm1(state["state_dict"]["lengths"].numpy()[SPECIAL:] * 7)
    model = FeedbackNet(lengths=lengths, **state["config"])
    model.load_state_dict(state["state_dict"])
    return model.to(device).eval()
