"""
HalfCheetah-v5 experiments for the paper's CCMoE-PPO:

- Single-Gaussian PPO baseline (Section 3, standard actor–critic).
- MoE baseline: multiple Gaussian heads with global gate (no chart structure).
- PPO + Graph-Laplacian baseline: hard Voronoi partition + parameter Laplacian.
- CCMoE-PPO (proposed): chart-consistent mixture-of-experts policy class:
    * Shared encoder φ_θ(o) ∈ R^d
    * Euclidean chart cover in feature space
    * Masked gate restricted to active charts 𝓘(z)
    * Convex interpolation in log-variance coordinates
    * Overlap-conditioned consistency on 𝓤_i ∩ 𝓤_j
    * Optional gradient balancing for the overlap loss

This script reproduces the HalfCheetah-v5 part of the methodology + experiments.
"""

import os, json, math, time, argparse, random
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict

import numpy as np
import gymnasium as gym

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent

from scipy.spatial.distance import cdist

# --------------------------
# Globals / Utilities
# --------------------------

MASTER_DIR = os.path.join(".", "CCMoE_HalfCheetah")

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0
EPS = 1e-6


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def timestamp():
    return time.strftime("%Y%m%d-%H%M%S")


def trapz_auc(xs, ys):
    """Area under learning curve via trapezoidal rule (AUC vs. env steps)."""
    if len(xs) < 2:
        return 0.0
    return float(np.trapz(ys, xs))


# --------------------------
# Loggers
# --------------------------

class CSVLogger:
    def __init__(self, path, headers: List[str]):
        self.path = path
        self.headers = headers
        ensure_dir(os.path.dirname(path))
        with open(self.path, "w") as f:
            f.write(",".join(headers) + "\n")

    def write(self, row: Dict):
        with open(self.path, "a") as f:
            f.write(",".join(str(row.get(h, "")) for h in self.headers) + "\n")


class RunLogger:
    """
    Per-run logger:
      - episode.csv: episodic returns vs. environment steps
      - train.csv: PPO diagnostics + CCMoE regularizer stats
      - config.json: TrainConfig
      - summary.json: final metrics (AUC, final return, time-to-thresholds)
    """
    def __init__(self, base_dir, algo_tag, run_tag):
        self.root = os.path.join(base_dir, algo_tag, run_tag)
        ensure_dir(self.root)

        self.episode_log = CSVLogger(
            os.path.join(self.root, "episode.csv"),
            ["episode", "global_step", "ep_return", "len",
             "reward_forward", "reward_ctrl", "wall_time_s"]
        )

        self.train_log = CSVLogger(
            os.path.join(self.root, "train.csv"),
            [
                "update", "global_step",
                "policy_loss", "value_loss", "entropy",
                "approx_kl", "clipfrac",
                "penalty", "gate_entropy", "lambda_t",
                "lr", "vf_coef", "ent_coef"
            ],
        )

        self.config_path = os.path.join(self.root, "config.json")
        self.summary_path = os.path.join(self.root, "summary.json")
        self.ckpt_dir = os.path.join(self.root, "checkpoints")
        ensure_dir(self.ckpt_dir)

    def save_config(self, cfg: dict):
        with open(self.config_path, "w") as f:
            json.dump(cfg, f, indent=2)

    def save_summary(self, summary: dict):
        with open(self.summary_path, "w") as f:
            json.dump(summary, f, indent=2)


# --------------------------
# PPO Buffer (GAE-λ)
# --------------------------

class RolloutBuffer:
    """
    On-policy buffer with Generalized Advantage Estimation (GAE).
    `dones` must be 1.0 ONLY for true terminals (terminated=True), not for time-limit truncations.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95, device="cpu"):
        self.obs = torch.zeros((size, obs_dim), dtype=torch.float32, device=device)
        self.acts = torch.zeros((size, act_dim), dtype=torch.float32, device=device)
        self.rews = torch.zeros((size,), dtype=torch.float32, device=device)
        self.dones = torch.zeros((size,), dtype=torch.float32, device=device)
        self.vals = torch.zeros((size,), dtype=torch.float32, device=device)
        self.logps = torch.zeros((size,), dtype=torch.float32, device=device)
        self.ptr, self.max_size = 0, size
        self.gamma, self.lam = gamma, lam
        self.device = device

    def store(self, o, a, r, d, v, logp):
        """
        Store one transition.
        d = 1.0 only if the episode truly terminated (terminated=True), not for truncations.
        """
        assert self.ptr < self.max_size
        self.obs[self.ptr] = torch.as_tensor(o, dtype=torch.float32, device=self.device)
        self.acts[self.ptr] = torch.as_tensor(a, dtype=torch.float32, device=self.device)
        self.rews[self.ptr] = r
        self.dones[self.ptr] = d
        self.vals[self.ptr] = v
        self.logps[self.ptr] = logp
        self.ptr += 1

    def finish(self, last_val: float):
        """
        Compute GAE(λ) advantages and returns for the filled portion of the buffer.
        Handles partially filled last batch.
        """
        n = self.ptr
        if n == 0:
            return {
                "obs": self.obs[:0], "acts": self.acts[:0],
                "rets": self.rews[:0], "advs": self.rews[:0],
                "logps": self.logps[:0]
            }

        adv = torch.zeros(n, dtype=torch.float32, device=self.device)
        last_gae = 0.0
        for t in reversed(range(n)):
            # nonterminal=1 if not true terminal, so we bootstrap through time-limit truncations
            nonterminal = 1.0 - self.dones[t]
            delta = self.rews[t] + self.gamma * last_val * nonterminal - self.vals[t]
            last_gae = delta + self.gamma * self.lam * nonterminal * last_gae
            adv[t] = last_gae
            last_val = self.vals[t]

        ret = adv + self.vals[:n]

        data = {
            "obs": self.obs[:n],
            "acts": self.acts[:n],
            "rets": ret,
            "advs": adv,
            "logps": self.logps[:n],
        }
        self.ptr = 0
        return data


# --------------------------
# Squashed Diagonal Gaussian Policy
# --------------------------

class SquashedDiagGaussian:
    """
    Tanh-squashed diagonal Gaussian:
      - Unsquashed distribution N(μ, Σ)
      - Squashed actions a = tanh(z)
      - Log prob and entropy use the standard change-of-variables formula.

    This is the basic building block; CCMoE composes μ and log σ before squashing.
    """

    def __init__(self, mu, log_std):
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        self.mu = mu
        self.std = torch.exp(log_std)
        self.normal = Independent(Normal(self.mu, self.std), 1)

    def sample(self):
        z = self.mu + self.std * torch.randn_like(self.mu)
        a = torch.tanh(z)
        logp = self.normal.log_prob(z) - torch.sum(torch.log(1 - a.pow(2) + EPS), dim=-1)
        return a, logp

    def log_prob(self, a):
        # atanh(a) = 0.5 * (log(1+a) - log(1-a))
        atanh = 0.5 * (torch.log1p(a + EPS) - torch.log1p(-a + EPS))
        logp = self.normal.log_prob(atanh) - torch.sum(torch.log(1 - a.pow(2) + EPS), dim=-1)
        return logp

    def entropy(self):
        # Monte Carlo entropy estimate for the squashed distribution
        a, logp = self.sample()
        return -logp.mean()


# --------------------------
# Networks
# --------------------------

class MLP(nn.Module):
    """
    Shared encoder φ_θ(o):
      - Two hidden layers, tanh nonlinearity.
      - Returns (out, feat), where feat is the latent representation.
      - In the paper: φ_θ defines the feature space with chart cover.
    """
    def __init__(self, in_dim, out_dim, hid=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.Tanh(),
            nn.Linear(hid, hid), nn.Tanh(),
        )
        self.out = nn.Linear(hid, out_dim)

    def forward(self, x):
        h = self.net(x)
        return self.out(h), h  # out (unused here), features h


class ValueHead(nn.Module):
    """
    Value head V_ψ(z):
      - Operates on encoder features z ∈ ℝ^d (shared with the policy).
      - Matches the paper: value and policy share φ_θ.
    """
    def __init__(self, feat_dim, hid=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hid), nn.Tanh(),
            nn.Linear(hid, hid), nn.Tanh(),
            nn.Linear(hid, 1),
        )

    def forward(self, z):
        return self.net(z).squeeze(-1)


class Gate(nn.Module):
    """
    Gating network:
      - Produces *logits* over m experts.
      - MoE baseline: uses a global softmax over these logits.
      - CCMoE: applies chart mask + locality prior before softmax.
    """
    def __init__(self, feat_dim, m):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.Tanh(),
            nn.Linear(128, m),
        )

    def forward(self, z):
        return self.net(z)  # [B, m] logits


class LocalGaussianHead(nn.Module):
    """
    Local expert: maps features to μ_i(z) and log σ_i (diagonal covariance).
    """
    def __init__(self, feat_dim, act_dim):
        super().__init__()
        self.mu = nn.Linear(feat_dim, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, z):
        mu = self.mu(z)
        log_std = self.log_std.expand_as(mu)
        return mu, log_std


class RestrictionMap(nn.Module):
    """
    Restriction map R_i in the paper:
      - Affine map on actions: R_i(a) = W_i a + b_i.
      - Here implemented as a full linear map for simplicity
        (paper uses diagonal for closed-form KL; diagonal is a subset of this).
    """
    def __init__(self, act_dim):
        super().__init__()
        self.W = nn.Linear(act_dim, act_dim, bias=True)
        nn.init.eye_(self.W.weight)
        nn.init.zeros_(self.W.bias)

    def forward(self, mu):
        return self.W(mu)


# --------------------------
# Chart Cover: feature-space balls
# --------------------------

class ChartCover(nn.Module):
    """
    Chart cover in encoder feature space (Section 3.2 in the paper):

      - m centers c_i ∈ ℝ^d
      - Each chart 𝓤_i = { z : ‖z - c_i‖₂ ≤ r }
      - We maintain centers in a whitened coordinate space to stabilize the metric
        (here, whitening statistics are estimated from a warmup batch).

    This module is used by:
      - GraphLaplacianPPO: hard Voronoi partition (no overlaps).
      - CCMoE: overlapping ball cover, masked gating, and overlap detection.
    """
    def __init__(self, m: int, feat_dim: int, r: float,
                 tau_center: float = 0.01, device="cpu"):
        super().__init__()
        self.m = m
        self.feat_dim = feat_dim
        self.r = r
        self.tau = tau_center
        self.device = device

        # Centers live in (approximately) whitened feature space.
        self.centers = nn.Parameter(
            torch.zeros(m, feat_dim, device=device),
            requires_grad=False
        )

        # Whitening statistics (fixed after warmup for simplicity).
        self.stats_mean = nn.Parameter(
            torch.zeros(feat_dim, device=device),
            requires_grad=False
        )
        self.stats_var = nn.Parameter(
            torch.ones(feat_dim, device=device),
            requires_grad=False
        )

        self.initialized = False

    def whiten(self, z: torch.Tensor):
        """
        Whiten encoder features to stabilize distances:

            z̃ = (z - μ) / sqrt(σ² + ε)

        This matches the paper's notion of working in a feature space with
        approximately isotropic covariance. For simplicity we keep μ, σ²
        fixed after initialization (they could also be updated slowly).
        """
        return (z - self.stats_mean) / torch.sqrt(self.stats_var + 1e-6)

    def init_with_batch(self, z: torch.Tensor):
        """
        Initialize centers with a k-means++-style seeding on a warmup batch
        of features z. This matches the paper's initialization on a warmup buffer.
        """
        with torch.no_grad():
            # Estimate whitening statistics from warmup batch
            mean = z.mean(dim=0)
            var = z.var(dim=0, unbiased=False) + 1e-6
            self.stats_mean.copy_(mean)
            self.stats_var.copy_(var)

            z_w = self.whiten(z)  # whitened features

            B = z_w.shape[0]
            idx0 = np.random.randint(B)
            centers = [z_w[idx0]]

            for _ in range(1, self.m):
                # k-means++ style: sample proportional to squared distance to closest center
                d2 = torch.stack(
                    [torch.cdist(z_w, c[None, :]).squeeze(-1).pow(2) for c in centers],
                    dim=1
                ).min(dim=1)[0]
                probs = (d2 + 1e-8) / torch.sum(d2 + 1e-8)
                idx = torch.multinomial(probs, 1).item()
                centers.append(z_w[idx])

            self.centers.data = torch.stack(centers, dim=0)
            self.initialized = True

    def memberships(self, z: torch.Tensor):
        """
        Compute distances and chart memberships:

          - dists[b, i] = ‖z̃_b - c_i‖
          - hard_idx[b] = argmin_i dists[b, i]
          - masks[i][b] = 1 if z̃_b ∈ 𝓤_i

        Where z̃ is the whitened feature.
        """
        if not self.initialized:
            raise RuntimeError("ChartCover must be initialized with init_with_batch() before use.")

        z_w = self.whiten(z)
        dists = torch.cdist(z_w, self.centers)  # [B, m]
        hard_idx = torch.argmin(dists, dim=1)
        masks = [(dists[:, i] <= self.r) for i in range(self.m)]
        return dists, hard_idx, masks

    def soft_update_centers(self, z: torch.Tensor):
        """
        Two-timescale center adaptation (end of each PPO update cycle):

          c_i ← (1 - τ) c_i + τ * mean{ z̃ | argmin_j ‖z̃ - c_j‖ = i }.

        This matches the paper's off-gradient center update that tracks
        the evolving feature distribution without direct gradient feedback.
        """
        if not self.initialized:
            return

        with torch.no_grad():
            z_w = self.whiten(z)
            dists = torch.cdist(z_w, self.centers)
            hard_idx = torch.argmin(dists, dim=1)

            for i in range(self.m):
                sel = (hard_idx == i)
                if sel.any():
                    mean_i = z_w[sel].mean(dim=0)
                    self.centers.data[i].mul_(1.0 - self.tau).add_(mean_i, alpha=self.tau)


# --------------------------
# Agents (4 variants)
# --------------------------

class SingleGaussianPPO(nn.Module):
    """
    Baseline: single diagonal Gaussian policy (standard PPO).

    - Shared encoder φ_θ(o) for policy and value.
    - Policy: μ, log σ from encoder features.
    - Value: V_ψ(z) from encoder features.
    """
    def __init__(self, obs_dim, act_dim, device="cpu"):
        super().__init__()
        self.device = device
        self.encoder = MLP(obs_dim, 256)          # φ_θ
        self.policy_head = LocalGaussianHead(256, act_dim)
        self.value_head = ValueHead(256)

    def forward(self, obs):
        _, feat = self.encoder(obs)
        mu, log_std = self.policy_head(feat)
        pi = SquashedDiagGaussian(mu, log_std)
        v = self.value_head(feat)
        return pi, v


class MoEPPO(nn.Module):
    """
    Mixture-of-experts PPO baseline:

      - Shared encoder φ_θ(o) ∈ ℝ^d
      - m local Gaussian heads (μ_i, log σ_i)
      - Global softmax gate over experts (no chart structure)
      - Convex interpolation in log-variance coordinates
      - Single squashed Gaussian policy

    This matches the MoE baseline described in the paper (no locality nor overlap regularization).
    """
    def __init__(self, obs_dim, act_dim, m=4, device="cpu"):
        super().__init__()
        self.device = device
        self.encoder = MLP(obs_dim, 256)
        self.heads = nn.ModuleList([LocalGaussianHead(256, act_dim) for _ in range(m)])
        self.gate = Gate(256, m)
        self.value_head = ValueHead(256)
        self.m = m

    def forward(self, obs):
        _, feat = self.encoder(obs)

        # Local heads
        mus, log_stds = [], []
        for h in self.heads:
            mu_i, ls_i = h(feat)
            mus.append(mu_i)
            log_stds.append(ls_i)
        mus = torch.stack(mus, dim=1)       # [B, m, A]
        log_stds = torch.stack(log_stds, 1) # [B, m, A]

        # Global gate: softmax over all experts (no masking)
        gate_logits = self.gate(feat)       # [B, m]
        g_probs = F.softmax(gate_logits, dim=-1)  # [B, m]
        g = g_probs.unsqueeze(-1)           # [B, m, 1]

        # Convex interpolation in log-variance coordinates
        mu = torch.sum(g * mus, dim=1)
        log_std = torch.sum(g * log_stds, dim=1)

        pi = SquashedDiagGaussian(mu, log_std)
        v = self.value_head(feat)
        gate_entropy = -(g_probs * torch.log(g_probs + 1e-8)).sum(dim=1).mean()

        return pi, v, gate_entropy, (mus, log_stds, g_probs, feat)


class GraphLaplacianPPO(nn.Module):
    """
    PPO + Graph-Laplacian baseline:

      - Shared encoder φ_θ(o) ∈ ℝ^d
      - m local Gaussian heads
      - Hard partition: each state uses the nearest head (Voronoi diagram)
      - Parameter-space graph Laplacian penalty over head parameters

    This tests whether a generic smoothness prior over head parameters
    can match CCMoE's structured overlap consistency.
    """
    def __init__(self, obs_dim, act_dim, m=4, r=2.0, device="cpu"):
        super().__init__()
        self.device = device
        self.encoder = MLP(obs_dim, 256)
        self.heads = nn.ModuleList([LocalGaussianHead(256, act_dim) for _ in range(m)])
        self.value_head = ValueHead(256)
        self.cover = ChartCover(m, 256, r=r, device=device)
        self.m = m

    def forward(self, obs):
        _, feat = self.encoder(obs)

        # Lazy chart initialization on first batch
        if not self.cover.initialized:
            self.cover.init_with_batch(feat.detach())

        dists, hard_idx, _ = self.cover.memberships(feat.detach())

        # Evaluate all local heads
        mu_all, ls_all = [], []
        for h in self.heads:
            mu_i, ls_i = h(feat)
            mu_all.append(mu_i)
            ls_all.append(ls_i)
        mu_all = torch.stack(mu_all, dim=1)  # [B, m, A]
        ls_all = torch.stack(ls_all, dim=1)  # [B, m, A]

        # Select nearest head per state
        sel = F.one_hot(hard_idx, num_classes=self.m).unsqueeze(-1)  # [B, m, 1]
        mu = torch.sum(sel * mu_all, dim=1)
        log_std = torch.sum(sel * ls_all, dim=1)

        pi = SquashedDiagGaussian(mu, log_std)
        v = self.value_head(feat)

        return pi, v, (mu_all, ls_all, feat)

    def graph_laplacian_loss(self, k=2, w_scale=1e-4):
        """
        Parameter-space Laplacian penalty:
          - Build k-NN graph over chart centers.
          - Penalize squared differences between neighboring head parameters.

        This matches the Graph-Laplacian baseline described in the paper.
        """
        C = self.cover.centers.detach().cpu().numpy()
        if len(C) < 2:
            return torch.tensor(0.0, device=self.device)

        D = cdist(C, C)
        adj = np.zeros((self.m, self.m), dtype=np.float32)
        for i in range(self.m):
            nbrs = np.argsort(D[i])[1:k + 1]
            for j in nbrs:
                adj[i, j] = adj[j, i] = 1.0

        penalty = torch.tensor(0.0, device=self.device)
        for i in range(self.m):
            for j in range(i + 1, self.m):
                if adj[i, j] > 0:
                    Wi, bi = self.heads[i].mu.weight, self.heads[i].mu.bias
                    Wj, bj = self.heads[j].mu.weight, self.heads[j].mu.bias
                    penalty = penalty + w_scale * (
                        (Wi - Wj).pow(2).sum() + (bi - bj).pow(2).sum()
                    )
        return penalty


class CCMoE(nn.Module):
    """
    CCMoE-PPO actor-critic (proposed method in the paper) for HalfCheetah-v5.

    Key ingredients (matching Section 3):

      - Shared encoder φ_θ(o) ∈ ℝ^d.
      - Chart cover {𝓤_i} defined by centers c_i and radius r in feature space.
      - Masked gate:
          * gate logits from G(z)
          * locality prior −β ‖z − c_i‖² / r²
          * softmax restricted to active charts 𝓘(z) = { i : z ∈ 𝓤_i }.
      - Convex interpolation in log-variance coordinates:
          * μ̄(z) = Σ_i g_i(z) μ_i(z)
          * s̄(z) = Σ_i g_i(z) s_i(z), s_i = log σ_i
          * Single diagonal Gaussian before tanh.
      - Overlap-conditioned consistency:
          * Only states in overlaps 𝓤_i ∩ 𝓤_j are regularized.
          * Symmetric KL between local Gaussians (optional restrictions R_i).
    """
    def __init__(self, obs_dim, act_dim,
                 m=4, r=2.0,
                 restrictions: str = "identity",  # "identity" | "learned"
                 locality_beta: float = 1.0,
                 device: str = "cpu"):
        super().__init__()
        self.device = device
        self.encoder = MLP(obs_dim, 256)
        self.heads = nn.ModuleList([LocalGaussianHead(256, act_dim) for _ in range(m)])
        self.gate = Gate(256, m)
        self.value_head = ValueHead(256)
        self.cover = ChartCover(m, 256, r=r, device=device)
        self.m = m
        self.restrictions = restrictions
        self.locality_beta = locality_beta

        if restrictions == "learned":
            self.R = nn.ModuleList([RestrictionMap(act_dim) for _ in range(m)])
        else:
            self.R = None

    def forward(self, obs):
        """
        Forward pass:

          - Compute encoder features z = φ_θ(o).
          - Lazily initialize chart centers on first batch.
          - Evaluate local heads (μ_i, log σ_i).
          - Compute chart memberships, masked gate with locality prior.
          - Convex interpolation in log-variance coordinates → global Gaussian.
        """
        _, feat = self.encoder(obs)

        # Lazy chart initialization using a warmup batch of features.
        if not self.cover.initialized:
            self.cover.init_with_batch(feat.detach())

        # Local experts
        mus, log_stds = [], []
        for h in self.heads:
            mu_i, ls_i = h(feat)
            mus.append(mu_i)
            log_stds.append(ls_i)
        mus = torch.stack(mus, dim=1)        # [B, m, A]
        log_stds = torch.stack(log_stds, 1)  # [B, m, A]

        # Chart memberships and distances (whitened space)
        dists, hard_idx, masks_list = self.cover.memberships(feat.detach())
        mask_tensor = torch.stack(masks_list, dim=1)  # [B, m] boolean mask for active charts 𝓘(z)

        # Gate logits
        gate_logits = self.gate(feat)  # [B, m]

        # Locality prior: decay contributions away from chart centers.
        if self.locality_beta > 0.0:
            gate_logits = gate_logits - self.locality_beta * (dists / (self.cover.r + 1e-8)).pow(2)

        # Masked gating: restrict support to active charts 𝓘(z).
        very_neg = torch.finfo(gate_logits.dtype).min
        logits_masked = gate_logits.masked_fill(~mask_tensor, very_neg)

        # Some states may fall outside all balls; fallback to global softmax for them.
        no_active = (~mask_tensor).all(dim=1)  # [B]
        g_probs = torch.zeros_like(gate_logits)

        if no_active.any():
            g_probs[no_active] = F.softmax(gate_logits[no_active], dim=-1)
        if (~no_active).any():
            g_probs[~no_active] = F.softmax(logits_masked[~no_active], dim=-1)

        g = g_probs.unsqueeze(-1)  # [B, m, 1]

        # Global parameters via convex interpolation in log-variance coordinates.
        mu = torch.sum(g * mus, dim=1)
        log_std = torch.sum(g * log_stds, dim=1)

        pi = SquashedDiagGaussian(mu, log_std)
        v = self.value_head(feat)

        # Gate entropy encourages non-degenerate mixing early in training.
        gate_entropy = -(g_probs * torch.log(g_probs + 1e-8)).sum(dim=1).mean()

        # pack carries all tensors needed for the overlap consistency loss.
        pack = (mus, log_stds, g_probs, feat, mask_tensor)
        return pi, v, gate_entropy, pack

    def overlap_consistency_loss(self, pack):
        """
        Overlap-conditioned consistency loss (Section 3.3):

          - For each feature z in the batch, identify overlapping chart pairs
            (i, j) such that z ∈ 𝓤_i ∩ 𝓤_j.
          - Compute symmetric KL between local Gaussians (with optional
            restrictions R_i, R_j) on those overlaps.
          - Average over all overlapping pairs and states.

        This aligns experts ONLY where they actually co-activate under the
        current state visitation distribution.
        """
        mus, log_stds, g_probs, feat, mask_tensor = pack
        B, m, A = mus.shape
        assert m == self.m

        pairs = []
        for i in range(self.m):
            for j in range(i + 1, self.m):
                overlap = mask_tensor[:, i] & mask_tensor[:, j]
                if overlap.any():
                    pairs.append((i, j))

        if len(pairs) == 0:
            return torch.tensor(0.0, device=self.device)

        loss = 0.0
        count = 0

        for (i, j) in pairs:
            overlap = mask_tensor[:, i] & mask_tensor[:, j]
            if not overlap.any():
                continue

            mu_i = mus[:, i, :][overlap]
            ls_i = log_stds[:, i, :][overlap]
            mu_j = mus[:, j, :][overlap]
            ls_j = log_stds[:, j, :][overlap]

            # Optional linear restrictions R_i:
            if self.R is not None:
                mu_i = self.R[i](mu_i)
                mu_j = self.R[j](mu_j)

            var_i = torch.exp(2 * ls_i)
            var_j = torch.exp(2 * ls_j)

            # KL(N_i || N_j) for diagonal Gaussians
            term_ij = 0.5 * torch.sum(
                var_i / (var_j + EPS)
                + (mu_j - mu_i).pow(2) / (var_j + EPS)
                + 2 * (ls_j - ls_i) - 1,
                dim=-1,
            )
            term_ji = 0.5 * torch.sum(
                var_j / (var_i + EPS)
                + (mu_i - mu_j).pow(2) / (var_i + EPS)
                + 2 * (ls_i - ls_j) - 1,
                dim=-1,
            )

            loss = loss + (term_ij + term_ji).mean()
            count += 1

        if count == 0:
            return torch.tensor(0.0, device=self.device)
        return loss / count


# --------------------------
# Training configuration
# --------------------------

@dataclass
class TrainConfig:
    # Algorithm: "ccmoe" | "ppo" | "ppo_glap" | "moe"
    algo: str = "ccmoe"

    # CCMoE / MoE / Graph parameters
    m: int = 4
    r: float = 2.0        # chart radius in whitened feature space
    lam_pen: float = 0.01 # λ_max for CCMoE overlap consistency (when grad balancing is on)
    restrictions: str = "identity"  # "identity" | "learned"

    # General training
    seed: int = 0
    steps: int = 400_000
    update_freq: int = 4096
    epochs: int = 10
    mb_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr: float = 3e-4
    vf_coef: float = 0.5
    ent_coef: float = 0.0  # policy entropy weight
    clip_ratio: float = 0.2
    max_grad_norm: float = 1.0
    device: str = "cpu"
    eval_every: int = 20_000

    # Graph-Laplacian baseline
    k_lap: int = 2
    lap_scale: float = 1e-4

    # Thresholds used for "time to X return" metrics
    thresholds: Tuple[float, float] = (3000.0, 5000.0)

    # Checkpointing
    save_every: int = 0
    save_tag: str = "final"

    # CCMoE-specific extras (matching the paper)
    gate_ent_coef: float = 0.0      # η: gate entropy weight in the total loss
    lam_min: float = 0.0            # λ_min: lower bound for overlap weight
    grad_balance_alpha: float = 0.0 # α: gradient balancing factor (0 = disable)


def make_env(seed):
    env = gym.make("HalfCheetah-v5")
    env.reset(seed=seed)
    return env


def save_checkpoint(model, cfg: TrainConfig, obs_dim, act_dim, path: str):
    ensure_dir(os.path.dirname(path))
    ckpt = {
        "algo": cfg.algo,
        "model_class": model.__class__.__name__,
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "config": asdict(cfg),
        "state_dict": model.state_dict(),
    }
    if isinstance(model, (CCMoE, GraphLaplacianPPO)):
        ckpt["cover_centers"] = model.cover.centers.detach().cpu().numpy()
    torch.save(ckpt, path)


def encoder_grad_norm(model: nn.Module) -> float:
    """
    Compute L2 norm of encoder gradients ‖∇_θ encoder‖.

    Used in CCMoE's gradient-balancing scheme to adapt λ_t based on the
    relative strength of PPO and overlap-consistency gradients.
    """
    total = 0.0
    for p in model.encoder.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().pow(2).sum().item())
    return math.sqrt(total + 1e-12)


def ppo_update(model, optimizer, data, cfg: TrainConfig):
    obs, acts, rets, advs, logps_old = data["obs"], data["acts"], data["rets"], data["advs"], data["logps"]

    # Advantage normalization
    advs = (advs - advs.mean()) / (advs.std() + 1e-8)

    n = obs.shape[0]
    idx = np.arange(n)

    policy_loss_epoch, value_loss_epoch = [], []
    entropy_epoch, approx_kl_epoch, clipfrac_epoch = [], [], []
    last_penalty = 0.0
    last_gate_ent = 0.0
    last_lambda_t = 0.0

    for _ in range(cfg.epochs):
        np.random.shuffle(idx)
        for start in range(0, n, cfg.mb_size):
            mb_idx = idx[start:start + cfg.mb_size]
            b_obs, b_acts = obs[mb_idx], acts[mb_idx]
            b_advs, b_rets = advs[mb_idx], rets[mb_idx]
            b_logps_old = logps_old[mb_idx]

            # --------------------------
            # CCMoE with gradient balancing (Section 3.3)
            # --------------------------
            if isinstance(model, CCMoE) and cfg.grad_balance_alpha > 0.0 and cfg.lam_pen > 0.0:
                # 1) PPO encoder gradient norm
                optimizer.zero_grad()
                pi, v, gate_entropy, pack = model(b_obs)
                logp = pi.log_prob(b_acts)
                ratio = torch.exp(logp - b_logps_old)
                clip_adv = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * b_advs
                policy_loss = -(torch.min(ratio * b_advs, clip_adv)).mean()
                value_loss = F.mse_loss(v, b_rets)
                entropy = pi.entropy() if hasattr(pi, "entropy") else torch.tensor(0.0, device=cfg.device)

                L_ppo = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy
                if cfg.gate_ent_coef != 0.0:
                    L_ppo = L_ppo - cfg.gate_ent_coef * gate_entropy

                L_ppo.backward()
                norm_ppo = encoder_grad_norm(model)
                optimizer.zero_grad()

                # 2) Overlap-only encoder gradient norm
                pi2, v2, gate_entropy2, pack2 = model(b_obs)
                overlap_loss = model.overlap_consistency_loss(pack2)
                overlap_loss.backward()
                norm_ov = encoder_grad_norm(model)
                optimizer.zero_grad()

                # 3) Adapt λ_t based on gradient norms
                if norm_ov < 1e-8:
                    lambda_t = cfg.lam_min
                else:
                    raw = cfg.grad_balance_alpha * (norm_ppo / norm_ov)
                    lambda_t = max(cfg.lam_min, min(cfg.lam_pen, raw))

                # 4) Final joint loss with λ_t
                pi, v, gate_entropy, pack = model(b_obs)
                logp = pi.log_prob(b_acts)
                ratio = torch.exp(logp - b_logps_old)
                clip_adv = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * b_advs
                policy_loss = -(torch.min(ratio * b_advs, clip_adv)).mean()
                value_loss = F.mse_loss(v, b_rets)
                entropy = pi.entropy() if hasattr(pi, "entropy") else torch.tensor(0.0, device=cfg.device)
                overlap_loss = model.overlap_consistency_loss(pack)

                L_ppo = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy
                if cfg.gate_ent_coef != 0.0:
                    L_ppo = L_ppo - cfg.gate_ent_coef * gate_entropy

                loss = L_ppo + lambda_t * overlap_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()

                with torch.no_grad():
                    approx_kl = (b_logps_old - logp).mean().abs()
                    clipfrac = (torch.abs(ratio - 1.0) > cfg.clip_ratio).float().mean()

                policy_loss_epoch.append(float(policy_loss.item()))
                value_loss_epoch.append(float(value_loss.item()))
                entropy_epoch.append(float(entropy.item()))
                approx_kl_epoch.append(float(approx_kl.item()))
                clipfrac_epoch.append(float(clipfrac.item()))
                last_penalty = float(overlap_loss.item())
                last_gate_ent = float(gate_entropy.item())
                last_lambda_t = float(lambda_t)
                continue

            # --------------------------
            # Generic PPO step (no gradient balancing)
            # --------------------------
            optimizer.zero_grad()
            penalty = torch.tensor(0.0, device=cfg.device)
            gate_entropy = torch.tensor(0.0, device=cfg.device)
            lambda_t = 0.0

            if isinstance(model, CCMoE):
                pi, v, gate_entropy, pack = model(b_obs)
                if cfg.lam_pen > 0.0:
                    penalty = model.overlap_consistency_loss(pack)
                    lambda_t = cfg.lam_pen
            elif isinstance(model, MoEPPO):
                pi, v, gate_entropy, _ = model(b_obs)
            elif isinstance(model, GraphLaplacianPPO):
                pi, v, _ = model(b_obs)
                if cfg.lap_scale > 0.0:
                    penalty = model.graph_laplacian_loss(k=cfg.k_lap, w_scale=cfg.lap_scale)
            else:
                # SingleGaussianPPO
                pi, v = model(b_obs)

            logp = pi.log_prob(b_acts)
            ratio = torch.exp(logp - b_logps_old)
            clip_adv = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * b_advs

            policy_loss = -(torch.min(ratio * b_advs, clip_adv)).mean()
            value_loss = F.mse_loss(v, b_rets)
            entropy = pi.entropy() if hasattr(pi, "entropy") else torch.tensor(0.0, device=cfg.device)

            # Base PPO loss
            loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy

            # Gate entropy (for CCMoE and MoE; for others gate_entropy=0)
            if cfg.gate_ent_coef != 0.0:
                loss = loss - cfg.gate_ent_coef * gate_entropy

            # Overlap / Laplacian penalty
            if isinstance(model, CCMoE) and cfg.lam_pen > 0.0:
                loss = loss + cfg.lam_pen * penalty
            elif isinstance(model, GraphLaplacianPPO) and cfg.lap_scale > 0.0:
                loss = loss + penalty

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approx_kl = (b_logps_old - logp).mean().abs()
                clipfrac = (torch.abs(ratio - 1.0) > cfg.clip_ratio).float().mean()

            policy_loss_epoch.append(float(policy_loss.item()))
            value_loss_epoch.append(float(value_loss.item()))
            entropy_epoch.append(float(entropy.item()))
            approx_kl_epoch.append(float(approx_kl.item()))
            clipfrac_epoch.append(float(clipfrac.item()))
            last_penalty = float(penalty.item()) if torch.is_tensor(penalty) else 0.0
            last_gate_ent = float(gate_entropy.item()) if torch.is_tensor(gate_entropy) else 0.0
            last_lambda_t = float(lambda_t)

    stats = {
        "policy_loss": float(np.mean(policy_loss_epoch)),
        "value_loss": float(np.mean(value_loss_epoch)),
        "entropy": float(np.mean(entropy_epoch)),
        "approx_kl": float(np.mean(approx_kl_epoch)),
        "clipfrac": float(np.mean(clipfrac_epoch)),
        "penalty": last_penalty,
        "gate_entropy": last_gate_ent,
        "lambda_t": last_lambda_t,
    }
    return stats


def run_one(cfg: TrainConfig, base_dir: str, algo_tag: str, run_tag: str):
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    env = make_env(cfg.seed)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Build model according to algorithm
    if cfg.algo == "ppo":
        model = SingleGaussianPPO(obs_dim, act_dim, device=device).to(device)
    elif cfg.algo == "moe":
        model = MoEPPO(obs_dim, act_dim, m=cfg.m, device=device).to(device)
    elif cfg.algo == "ppo_glap":
        model = GraphLaplacianPPO(obs_dim, act_dim, m=cfg.m, r=cfg.r, device=device).to(device)
    elif cfg.algo == "ccmoe":
        model = CCMoE(obs_dim, act_dim, m=cfg.m, r=cfg.r,
                      restrictions=cfg.restrictions,
                      device=device).to(device)
    else:
        raise ValueError(f"Unknown algo {cfg.algo}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    buf = RolloutBuffer(obs_dim, act_dim, size=cfg.update_freq,
                        gamma=cfg.gamma, lam=cfg.gae_lambda,
                        device=device)

    logger = RunLogger(base_dir, algo_tag, run_tag)
    logger.save_config(asdict(cfg))

    ep = 0
    global_step = 0
    obs, info = env.reset(seed=cfg.seed)
    ep_return, ep_len = 0.0, 0
    returns_x, returns_y = [], []
    t_start = time.time()
    next_save = cfg.save_every if cfg.save_every > 0 else None

    while global_step < cfg.steps:
        # Collect one rollout batch
        for _ in range(cfg.update_freq):
            o_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            out = model(o_t)

            if isinstance(model, (CCMoE, MoEPPO, GraphLaplacianPPO)):
                pi, v = out[0], out[1]
            else:
                pi, v = out

            with torch.no_grad():
                a_t, logp_t = pi.sample()
            a_np = a_t.squeeze(0).detach().cpu().numpy()

            next_obs, reward, terminated, truncated, info = env.step(a_np)

            done_env = terminated or truncated      # for resetting environment
            done_bootstrap = float(terminated)      # true terminal only for GAE cut

            buf.store(obs, a_np, reward, done_bootstrap, v.item(), float(logp_t.item()))
            ep_return += reward
            ep_len += 1
            global_step += 1
            obs = next_obs

            # Optional periodic checkpoints
            if cfg.save_every and next_save is not None and global_step >= next_save:
                ck = os.path.join(logger.ckpt_dir, f"step_{global_step}.pt")
                save_checkpoint(model, cfg, obs_dim, act_dim, ck)
                next_save += cfg.save_every

            # Episode boundary
            if done_env:
                r_forward = info.get("reward_forward", np.nan)
                r_ctrl = info.get("reward_ctrl", np.nan)
                logger.episode_log.write({
                    "episode": ep,
                    "global_step": global_step,
                    "ep_return": ep_return,
                    "len": ep_len,
                    "reward_forward": r_forward,
                    "reward_ctrl": r_ctrl,
                    "wall_time_s": time.time() - t_start,
                })
                returns_x.append(global_step)
                returns_y.append(ep_return)
                ep += 1
                ep_return, ep_len = 0.0, 0
                obs, info = env.reset()

            if global_step >= cfg.steps:
                break

        # PPO update at end of rollout batch
        with torch.no_grad():
            o_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            v_end = model(o_t)[1]
            last_val = v_end.item()

        data = buf.finish(last_val)
        stats = ppo_update(model, optimizer, data, cfg)
        logger.train_log.write({
            "update": global_step // cfg.update_freq,
            "global_step": global_step,
            "policy_loss": stats["policy_loss"],
            "value_loss": stats["value_loss"],
            "entropy": stats["entropy"],
            "approx_kl": stats["approx_kl"],
            "clipfrac": stats["clipfrac"],
            "penalty": stats["penalty"],
            "gate_entropy": stats["gate_entropy"],
            "lambda_t": stats["lambda_t"],
            "lr": cfg.lr,
            "vf_coef": cfg.vf_coef,
            "ent_coef": cfg.ent_coef,
        })

        # Two-timescale center adaptation (Section 3.4)
        if isinstance(model, (CCMoE, GraphLaplacianPPO)):
            with torch.no_grad():
                obs_batch = data["obs"]
                _, feats = model.encoder(obs_batch)
                model.cover.soft_update_centers(feats)

    # Summaries
    auc = trapz_auc(np.array(returns_x, dtype=np.float64),
                    np.array(returns_y, dtype=np.float64))
    last_k = 10 if len(returns_y) >= 10 else len(returns_y)
    final_avg = float(np.mean(returns_y[-last_k:])) if last_k > 0 else float("nan")
    best = float(np.max(returns_y)) if len(returns_y) else float("nan")

    def time_to(th):
        for x, y in zip(returns_x, returns_y):
            if y >= th:
                return int(x)
        return -1

    t3k = time_to(cfg.thresholds[0])
    t5k = time_to(cfg.thresholds[1])

    logger.save_summary({
        "seed": cfg.seed,
        "steps": cfg.steps,
        "episodes": ep,
        "final_avg_return_last10": final_avg,
        "best_return": best,
        "auc_return_vs_steps": auc,
        "time_to_3k": t3k,
        "time_to_5k": t5k,
    })

    # Final checkpoint
    final_ckpt_path = os.path.join(logger.ckpt_dir, f"{cfg.save_tag}.pt")
    save_checkpoint(model, cfg, obs_dim, act_dim, final_ckpt_path)
    env.close()

    return {
        "algo": cfg.algo,
        "seed": cfg.seed,
        "m": cfg.m,
        "r": cfg.r,
        "lambda": cfg.lam_pen,
        "restrictions": cfg.restrictions,
        "steps": cfg.steps,
        "final_avg_last10": final_avg,
        "best": best,
        "auc": auc,
        "t3k": t3k,
        "t5k": t5k,
    }


# --------------------------
# Orchestration & Aggregation (no plotting)
# --------------------------

def write_master_index(rows: List[Dict]):
    path = os.path.join(MASTER_DIR, "master_index.csv")
    headers = ["algo", "seed", "m", "r", "lambda", "restrictions",
               "steps", "final_avg_last10", "best", "auc", "t3k", "t5k"]
    if not os.path.exists(path):
        CSVLogger(path, headers)
    logger = CSVLogger(path, headers)
    for r in rows:
        logger.write(r)


def aggregate_summary():
    agg = {}
    for algo in ["ccmoe", "ppo", "ppo_glap", "moe"]:
        a_dir = os.path.join(MASTER_DIR, algo)
        if not os.path.isdir(a_dir):
            continue
        for run in os.listdir(a_dir):
            s_path = os.path.join(a_dir, run, "summary.json")
            c_path = os.path.join(a_dir, run, "config.json")
            if not (os.path.isfile(s_path) and os.path.isfile(c_path)):
                continue
            with open(s_path) as f:
                summ = json.load(f)
            with open(c_path) as f:
                cfg = json.load(f)
            key = (
                algo,
                cfg.get("m", 1),
                cfg.get("r", 0.0),
                cfg.get("lam_pen", 0.0),
                cfg.get("restrictions", "-"),
                cfg.get("steps"),
            )
            val = {
                "final": summ.get("final_avg_return_last10", float("nan")),
                "best": summ.get("best_return", float("nan")),
                "auc": summ.get("auc_return_vs_steps", float("nan")),
                "t3k": summ.get("time_to_3k", -1),
                "t5k": summ.get("time_to_5k", -1),
            }
            agg.setdefault(key, []).append(val)

    rows = []
    for key, vals in agg.items():
        algo, m, r, lam, rest, steps = key
        arrf = np.array([v["final"] for v in vals], dtype=np.float64)
        arrb = np.array([v["best"] for v in vals], dtype=np.float64)
        arra = np.array([v["auc"] for v in vals], dtype=np.float64)
        arrt3 = np.array([v["t3k"] for v in vals], dtype=np.float64)
        arrt5 = np.array([v["t5k"] for v in vals], dtype=np.float64)
        rows.append({
            "algo": algo,
            "m": m,
            "r": r,
            "lambda": lam,
            "restrictions": rest,
            "steps": steps,
            "n_seeds": len(vals),
            "final_mean": float(np.nanmean(arrf)),
            "final_std": float(np.nanstd(arrf)),
            "best_mean": float(np.nanmean(arrb)),
            "best_std": float(np.nanstd(arrb)),
            "auc_mean": float(np.nanmean(arra)),
            "auc_std": float(np.nanstd(arra)),
            "t3k_mean": float(np.nanmean(arrt3)),
            "t5k_mean": float(np.nanmean(arrt5)),
        })

    csv_path = os.path.join(MASTER_DIR, "aggregate_summary.csv")
    headers = [
        "algo", "m", "r", "lambda", "restrictions", "steps", "n_seeds",
        "final_mean", "final_std", "best_mean", "best_std",
        "auc_mean", "auc_std", "t3k_mean", "t5k_mean",
    ]
    CSVLogger(csv_path, headers)
    csv = CSVLogger(csv_path, headers)
    for r in rows:
        csv.write(r)

    with open(os.path.join(MASTER_DIR, "aggregate_summary.json"), "w") as f:
        json.dump(rows, f, indent=2)


# --------------------------
# Suite Runner
# --------------------------

def run_suite(args):
    ensure_dir(MASTER_DIR)
    all_rows = []

    def run_block(algo, m, r, lam, restrictions, seeds, steps):
        for seed in seeds:
            cfg = TrainConfig(
                algo=algo,
                m=m,
                r=r,
                lam_pen=lam,
                restrictions=restrictions,
                seed=seed,
                steps=steps,
                device="cpu",
                update_freq=args.update_freq,
                epochs=args.epochs,
                mb_size=args.mb_size,
                lr=args.lr,
                vf_coef=args.vf_coef,
                ent_coef=args.ent_coef,
                clip_ratio=args.clip_ratio,
                eval_every=args.eval_every,
                k_lap=args.k_lap,
                lap_scale=args.lap_scale,
                save_every=args.save_every,
                save_tag=args.save_tag,
                gate_ent_coef=args.gate_ent_coef,
                lam_min=args.lam_min,
                grad_balance_alpha=args.grad_balance_alpha,
            )

            if algo == "ccmoe":
                run_tag = f"m{m}_r{r}_lam{lam}_{restrictions}_seed{seed}_{timestamp()}"
            elif algo == "ppo_glap":
                run_tag = f"m{m}_r{r}_seed{seed}_{timestamp()}"
            elif algo == "moe":
                run_tag = f"m{m}_seed{seed}_{timestamp()}"
            else:
                run_tag = f"seed{seed}_{timestamp()}"

            res = run_one(cfg, MASTER_DIR, algo, run_tag)
            all_rows.append(res)

    seeds = [int(s) for s in args.seeds.split(",")]
    m_list = [int(x) for x in args.m_list.split(",")]
    r_list = [float(x) for x in args.r_list.split(",")]
    lam_list = [float(x) for x in args.lambda_list.split(",")]
    rest_list = [x.strip() for x in args.restrictions.split(",")]

    if args.run_all:
        # Proposed CCMoE-PPO (grid over {m, r, λ_max, restrictions})
        for m in m_list:
            for r in r_list:
                for lam in lam_list:
                    for rest in rest_list:
                        run_block("ccmoe", m, r, lam, rest, seeds, args.steps)

        # Single-Gaussian PPO baseline
        run_block("ppo", 1, 0.0, 0.0, "-", seeds, args.steps)

        # PPO + Graph-Laplacian baseline
        for m in m_list:
            for r in r_list:
                run_block("ppo_glap", m, r, 0.0, "-", seeds, args.steps)

        # MoE baseline
        for m in m_list:
            run_block("moe", m, 0.0, 0.0, "-", seeds, args.steps)

    else:
        # Single configuration (use first elements of lists)
        run_block(args.algo,
                  int(m_list[0]),
                  float(r_list[0]),
                  float(lam_list[0]),
                  rest_list[0],
                  seeds,
                  args.steps)

    write_master_index(all_rows)
    aggregate_summary()


# --------------------------
# CLI
# --------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="HalfCheetah-v5 CCMoE-PPO (Proposed) + Baselines"
    )
    p.add_argument("--run_all", action="store_true",
                   help="Run CCMoE + baselines + ablations")
    p.add_argument("--algo", type=str, default="ccmoe",
                   choices=["ccmoe", "ppo", "ppo_glap", "moe"],
                   help="When not using --run_all")
    p.add_argument("--seeds", type=str, default="0,1,2")
    p.add_argument("--steps", type=int, default=400000,
                   help="Env steps per run")
    p.add_argument("--update_freq", type=int, default=4096)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--mb_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--vf_coef", type=float, default=0.5)
    p.add_argument("--ent_coef", type=float, default=0.0)
    p.add_argument("--clip_ratio", type=float, default=0.2)
    p.add_argument("--eval_every", type=int, default=20000)
    p.add_argument("--m_list", type=str, default="2,4,8")
    p.add_argument("--r_list", type=str, default="1.5,2.0,2.5")
    p.add_argument("--lambda_list", type=str, default="0.0,0.01,0.05")
    p.add_argument("--restrictions", type=str, default="identity,learned")
    p.add_argument("--k_lap", type=int, default=2,
                   help="kNN for Graph-Laplacian baseline")
    p.add_argument("--lap_scale", type=float, default=1e-4,
                   help="Graph-Laplacian weight")
    p.add_argument("--save_every", type=int, default=0,
                   help="Save checkpoint every N env steps (0=off)")
    p.add_argument("--save_tag", type=str, default="final",
                   help="Filename tag for final checkpoint")

    # CCMoE-specific CLI knobs (paper parameters)
    p.add_argument("--gate_ent_coef", type=float, default=0.0,
                   help="η: gate entropy weight in CCMoE/MoE.")
    p.add_argument("--lam_min", type=float, default=0.0,
                   help="λ_min: lower bound for overlap weight λ_t.")
    p.add_argument("--grad_balance_alpha", type=float, default=0.0,
                   help="α: gradient balancing coefficient for CCMoE (0 = disable).")

    args = p.parse_args()
    run_suite(args)
