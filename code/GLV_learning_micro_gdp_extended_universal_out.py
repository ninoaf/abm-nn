#!/usr/bin/env python3
"""
Extended GLV model for GDP + macro latent variables with an out-of-sample split.

Training only uses years strictly before --train-cut-year, while evaluation / plots
cover the entire time span so we can assess post-2015 performance.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple
import shlex
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import matplotlib.pyplot as plt


DEFAULT_MACRO_CSV = Path("/Users/nino-aisot/CodingProjects/abm-nn/data_cache/macro_dataset_kalman_processed.csv")
FEATURE_COLUMNS = [
    "Unemployment",
    "Interest_Rate",
    "Debt",
    "Working_Age_Pop",
    "Inflation",
    "Current_Account",
]


@dataclass
class ExtendedDataset:
    years: np.ndarray
    times: torch.Tensor           # (T,)
    gdp_values: torch.Tensor      # (T, N)
    y_values: torch.Tensor        # (T, N, F)
    names: List[str]
    baselines: np.ndarray         # (N,)
    y_means: np.ndarray           # (N, F)
    y_stds: np.ndarray            # (N, F)


def select_top_countries(df: pd.DataFrame, end_year: int, top_n: int) -> List[str]:
    latest = df[df["year"] == end_year].copy()
    latest = latest.sort_values("GDP", ascending=False)
    countries = latest["country"].tolist()
    if len(countries) < top_n:
        return countries
    return countries[:top_n]


def build_extended_dataset(args) -> ExtendedDataset:
    csv_path = Path(args.macro_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Macro dataset not found at {csv_path}")
    df = pd.read_csv(csv_path)
    mask = (df["year"] >= args.start_year) & (df["year"] <= args.end_year)
    df = df.loc[mask].copy()
    years = np.arange(args.start_year, args.end_year + 1, dtype=np.int32)
    if df.empty:
        raise RuntimeError("Filtered macro dataset is empty for the requested years.")

    top_countries = select_top_countries(df, args.end_year, args.top_n)
    country_series = []
    y_series_list = []
    names = []
    baselines = []
    y_means = []
    y_stds = []

    for country in top_countries:
        sub = df[df["country"] == country].set_index("year")
        if sub.empty:
            continue
        sub = sub.reindex(years)
        if sub.isnull().any().any():
            continue
        gdp = sub["GDP"].values.astype(np.float64)
        gdp = gdp / args.gdp_scale
        baseline = gdp[0] if gdp[0] > 0 else 1.0
        baselines.append(baseline)
        gdp = gdp / baseline
        country_series.append(gdp)

        features = sub[FEATURE_COLUMNS].values.astype(np.float64)
        mean = features.mean(axis=0)
        std = features.std(axis=0)
        std[std < 1e-6] = 1.0
        normalized = (features - mean) / std
        y_series_list.append(normalized)
        y_means.append(mean)
        y_stds.append(std)
        names.append(country)

    if not country_series:
        raise RuntimeError("No countries with complete macro data for the requested interval.")

    gdp_matrix = np.stack(country_series, axis=1)  # (T, N)
    y_tensor = np.stack(y_series_list, axis=1)     # (T, N, F)
    baselines_np = np.asarray(baselines, dtype=np.float32)
    y_means_np = np.asarray(y_means, dtype=np.float32)
    y_stds_np = np.asarray(y_stds, dtype=np.float32)

    times = years - years[0]
    dataset = ExtendedDataset(
        years=years,
        times=torch.from_numpy(times.astype(np.float32)),
        gdp_values=torch.from_numpy(gdp_matrix.astype(np.float32)),
        y_values=torch.from_numpy(y_tensor.astype(np.float32)),
        names=names,
        baselines=baselines_np,
        y_means=y_means_np,
        y_stds=y_stds_np,
    )
    return dataset


def rk4_step(rhs, t, x, dt):
    k1 = rhs(t, x)
    k2 = rhs(t + 0.5 * dt, x + 0.5 * dt * k1)
    k3 = rhs(t + 0.5 * dt, x + 0.5 * dt * k2)
    k4 = rhs(t + dt, x + dt * k3)
    return x + (k1 + 2 * k2 + 2 * k3 + k4) * (dt / 6.0)


class PhiSelfCoupled(nn.Module):
    """Shared local dynamics with FiLM adapters for country-specific behavior."""

    def __init__(
        self,
        num_countries: int,
        feature_dim: int,
        hidden_dim: int = 64,
        num_hidden: int = 2,
        normalize_input: bool = False,
        country_embed_dim: int = 16,
        use_country_embeddings: bool = True,
    ):
        super().__init__()
        self.normalize_input = normalize_input
        self.feature_dim = feature_dim
        self.num_countries = num_countries
        self.use_country_embeddings = use_country_embeddings
        self.country_embed_dim = country_embed_dim if use_country_embeddings else 0

        if use_country_embeddings:
            self.country_embedding = nn.Embedding(num_countries, country_embed_dim)
        else:
            self.register_buffer("country_embedding", None, persistent=False)

        input_dim = 1 + feature_dim + self.country_embed_dim
        hidden_dim = max(4, hidden_dim)
        num_hidden = max(1, num_hidden)

        layers: List[nn.Linear] = []
        self.film_scale = nn.ModuleList()
        self.film_shift = nn.ModuleList()

        prev_dim = input_dim
        for _ in range(num_hidden):
            layer = nn.Linear(prev_dim, hidden_dim)
            nn.init.xavier_uniform_(layer.weight, gain=0.1)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
            layers.append(layer)
            if use_country_embeddings:
                self.film_scale.append(nn.Linear(country_embed_dim, hidden_dim))
                self.film_shift.append(nn.Linear(country_embed_dim, hidden_dim))
            else:
                self.film_scale.append(None)
                self.film_shift.append(None)
            prev_dim = hidden_dim

        self.hidden_layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(hidden_dim, 2)
        nn.init.xavier_uniform_(self.output_layer.weight, gain=0.1)
        nn.init.zeros_(self.output_layer.bias)

    def _normalize_x(self, xi: torch.Tensor) -> torch.Tensor:
        if self.normalize_input:
            return xi / 100.0
        return xi

    def forward(self, x: torch.Tensor, y: torch.Tensor, country_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert y.shape[0] == x.shape[0]
        if country_indices is None:
            country_indices = torch.arange(x.shape[0], device=x.device)

        xi = self._normalize_x(x).unsqueeze(-1)
        base_inp = torch.cat([xi, y], dim=-1)

        if self.use_country_embeddings:
            embeds = self.country_embedding(country_indices)
            inp = torch.cat([base_inp, embeds], dim=-1)
        else:
            embeds = None
            inp = base_inp

        h = inp
        for idx, layer in enumerate(self.hidden_layers):
            h = layer(h)
            if self.use_country_embeddings and embeds is not None:
                gamma = torch.tanh(self.film_scale[idx](embeds)) + 1.0
                beta = self.film_shift[idx](embeds)
                h = gamma * h + beta
            h = torch.relu(h)

        out = self.output_layer(h)
        growth_raw = out[:, 0]
        drift_raw = out[:, 1]
        return x * growth_raw + drift_raw


class PhiYIndividual(nn.Module):
    """Shared latent dynamics psi(Y_i, X_i) -> dY_i/dt with FiLM adapters."""

    def __init__(
        self,
        num_countries: int,
        feature_dim: int,
        hidden_dim: int = 64,
        num_hidden: int = 2,
        normalize_input: bool = False,
        country_embed_dim: int = 16,
        use_country_embeddings: bool = True,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.normalize_input = normalize_input
        self.num_countries = num_countries
        self.use_country_embeddings = use_country_embeddings
        self.country_embed_dim = country_embed_dim if use_country_embeddings else 0

        if use_country_embeddings:
            self.country_embedding = nn.Embedding(num_countries, country_embed_dim)
        else:
            self.register_buffer("country_embedding", None, persistent=False)

        input_dim = feature_dim + 1 + self.country_embed_dim
        hidden_dim = max(4, hidden_dim)
        num_hidden = max(1, num_hidden)

        self.hidden_layers = nn.ModuleList()
        self.film_scale = nn.ModuleList()
        self.film_shift = nn.ModuleList()

        prev_dim = input_dim
        for _ in range(num_hidden):
            layer = nn.Linear(prev_dim, hidden_dim)
            nn.init.xavier_uniform_(layer.weight, gain=0.1)
            nn.init.zeros_(layer.bias)
            self.hidden_layers.append(layer)
            if use_country_embeddings:
                self.film_scale.append(nn.Linear(country_embed_dim, hidden_dim))
                self.film_shift.append(nn.Linear(country_embed_dim, hidden_dim))
            else:
                self.film_scale.append(None)
                self.film_shift.append(None)
            prev_dim = hidden_dim

        self.output_layer = nn.Linear(hidden_dim, feature_dim)
        nn.init.xavier_uniform_(self.output_layer.weight, gain=0.1)
        nn.init.zeros_(self.output_layer.bias)

    def _normalize_x(self, xi: torch.Tensor) -> torch.Tensor:
        if self.normalize_input:
            return xi / 100.0
        return xi

    def forward(self, x: torch.Tensor, y: torch.Tensor, country_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        if country_indices is None:
            country_indices = torch.arange(x.shape[0], device=x.device)

        xi = self._normalize_x(x).unsqueeze(-1)
        base_inp = torch.cat([y, xi], dim=-1)

        if self.use_country_embeddings:
            embeds = self.country_embedding(country_indices)
            inp = torch.cat([base_inp, embeds], dim=-1)
        else:
            embeds = None
            inp = base_inp

        h = inp
        for idx, layer in enumerate(self.hidden_layers):
            h = layer(h)
            if self.use_country_embeddings and embeds is not None:
                gamma = torch.tanh(self.film_scale[idx](embeds)) + 1.0
                beta = self.film_shift[idx](embeds)
                h = gamma * h + beta
            h = torch.relu(h)

        return self.output_layer(h)


class GDPGLVExtendedModel(nn.Module):
    def __init__(
        self,
        num_countries: int,
        feature_dim: int,
        hidden_dim: int,
        num_hidden: int,
        a_init: float,
        a_diag_init: float,
        normalize_input: bool = False,
        country_embed_dim: int = 16,
        use_country_embeddings: bool = True,
        use_interactions: bool = True,
        interaction_gain: float = 1.0,
        learn_interaction_scales: bool = False,
        interaction_scale_max: float = 1.0,
        interaction_beta_init: float = 0.2,
    ):
        super().__init__()
        self.num_countries = num_countries
        self.feature_dim = feature_dim
        self.state_dim = num_countries + num_countries * feature_dim
        self.use_interactions = use_interactions
        self.interaction_gain = interaction_gain
        self.learn_interaction_scales = learn_interaction_scales
        self.interaction_scale_max = interaction_scale_max
        self.interaction_factor = 1.0
        self.register_buffer("country_indices", torch.arange(num_countries, dtype=torch.long), persistent=False)

        self.phi_self = PhiSelfCoupled(
            num_countries,
            feature_dim,
            hidden_dim,
            num_hidden,
            normalize_input=normalize_input,
            country_embed_dim=country_embed_dim,
            use_country_embeddings=use_country_embeddings,
        )
        self.phi_y = PhiYIndividual(
            num_countries,
            feature_dim,
            hidden_dim,
            num_hidden,
            normalize_input=normalize_input,
            country_embed_dim=country_embed_dim,
            use_country_embeddings=use_country_embeddings,
        )

        off_diag_value = float(a_init) if a_init is not None else float(1.0 / max(1, num_countries))
        diag_value = float(a_diag_init) if a_diag_init is not None else 0.0
        init = torch.full((num_countries, num_countries), off_diag_value)
        init.fill_diagonal_(diag_value)
        self.A = nn.Parameter(init)

        if learn_interaction_scales and use_interactions:
            self.interaction_logits = nn.Parameter(torch.zeros(num_countries))
        else:
            self.register_buffer("interaction_logits", None, persistent=False)

        beta_clamped = float(min(max(interaction_beta_init, 1e-4), 1.0 - 1e-4))
        beta_raw = math.log(beta_clamped / (1.0 - beta_clamped))
        self.interaction_beta_raw = nn.Parameter(torch.tensor(beta_raw, dtype=torch.float32))

    @property
    def interaction_beta(self) -> torch.Tensor:
        return torch.sigmoid(self.interaction_beta_raw)

    def _split_state(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        flat = state.reshape(-1)
        x = flat[: self.num_countries]
        y = flat[self.num_countries :].view(self.num_countries, self.feature_dim)
        return x, y

    def rhs(self, state: torch.Tensor) -> torch.Tensor:
        x, y = self._split_state(state)
        phi = self.phi_self(x, y, self.country_indices)
        if self.use_interactions and self.interaction_factor > 0:
            beta = self.interaction_beta
            positive_states = torch.clamp(x, min=0.0)
            powered_states = torch.pow(positive_states + 1e-9, beta)
            interaction = positive_states * (self.A @ powered_states)
            if self.learn_interaction_scales and self.interaction_logits is not None:
                scales = torch.tanh(self.interaction_logits) * self.interaction_scale_max
                interaction = interaction * scales
            interaction = interaction * (self.interaction_gain * self.interaction_factor)
            dx = phi + interaction
        else:
            dx = phi
        dy = self.phi_y(x, y, self.country_indices)
        return torch.cat([dx, dy.reshape(-1)], dim=0)

    def rhs_batch(self, states: torch.Tensor) -> torch.Tensor:
        if states.dim() == 1:
            return self.rhs(states)
        outputs = []
        for sample in states:
            outputs.append(self.rhs(sample))
        return torch.stack(outputs, dim=0)

    def interaction_parameters(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        if self.use_interactions:
            params.append(self.A)
            if self.learn_interaction_scales and isinstance(self.interaction_logits, nn.Parameter):
                params.append(self.interaction_logits)
            params.append(self.interaction_beta_raw)
        return params

    def set_interaction_factor(self, value: float):
        self.interaction_factor = float(value)

    def state_dict_to_save(self):
        return {"model": self.state_dict()}

    def load_from_state(self, state):
        self.load_state_dict(state["model"])


def finite_difference(traj: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
    dt = times[1:] - times[:-1]
    shape = [dt.shape[0]] + [1] * (traj.dim() - 1)
    dt = dt.view(*shape)
    diffs = traj[1:] - traj[:-1]
    return diffs / (dt + 1e-6)


def pretrain_coupled_model(
    model: GDPGLVExtendedModel,
    states: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    args,
):
    if args.pretrain_epochs <= 0:
        return
    print(f"[INFO] Teacher-forcing pretrain for {args.pretrain_epochs} epochs …")
    params = list(model.phi_self.parameters()) + list(model.phi_y.parameters())
    interaction_params = model.interaction_parameters()
    if interaction_params:
        params += interaction_params
    optimizer = optim.Adam(params, lr=args.pretrain_lr)
    prev_factor = model.interaction_factor
    model.set_interaction_factor(1.0)
    batch_size = min(args.pretrain_batch_size, states.shape[0])
    progress = tqdm(range(1, args.pretrain_epochs + 1), desc="Pretrain", ncols=120)
    for epoch in progress:
        optimizer.zero_grad()
        idx = torch.randint(0, states.shape[0], (batch_size,), device=device)
        batch_states = states[idx]
        batch_targets = targets[idx]
        preds = model.rhs_batch(batch_states)
        loss = (preds - batch_targets).abs().mean()
        loss.backward()
        optimizer.step()
        progress.set_postfix({"loss": f"{loss.item():.4f}"})
    model.set_interaction_factor(prev_factor)
    print("[INFO] Pretraining finished.")


def rollout_extended_model(
    model: GDPGLVExtendedModel,
    state0: torch.Tensor,
    times: torch.Tensor,
    substeps: int,
    clamp_max: Optional[float],
) -> torch.Tensor:
    states = [state0]
    current = state0
    for i in range(len(times) - 1):
        total_dt = times[i + 1] - times[i]
        n_steps = max(1, int(substeps))
        dt = total_dt / n_steps
        for _ in range(n_steps):
            current = rk4_step(lambda t, vec: model.rhs(vec), times[i], current, dt)
            x = current[: model.num_countries]
            y = current[model.num_countries :]
            if clamp_max is not None and clamp_max > 0:
                x = torch.clamp(x, min=0.0, max=clamp_max)
            else:
                x = torch.clamp(x, min=0.0)
            current = torch.cat([x, y], dim=0)
        states.append(current)
    return torch.stack(states, dim=0)


def rollout_train_and_holdout(
    model: GDPGLVExtendedModel,
    init_train: torch.Tensor,
    times_train: torch.Tensor,
    init_hold: Optional[torch.Tensor],
    times_hold: Optional[torch.Tensor],
    steps: int,
    clamp_max: Optional[float],
):
    train_traj = rollout_extended_model(model, init_train, times_train, steps, clamp_max)
    hold_traj = None
    if init_hold is not None and times_hold is not None and times_hold.shape[0] > 0:
        hold_traj = rollout_extended_model(model, init_hold, times_hold, steps, clamp_max)
    return train_traj, hold_traj


def plot_trajectories(years, true_traj, pred_traj, names, output_path, cutoff_year: Optional[int] = None):
    plt.figure(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))
    for i, (name, color) in enumerate(zip(names, colors)):
        plt.plot(years, true_traj[:, i], color=color, label=f"{name} (true)")
        plt.plot(
            years,
            pred_traj[:, i],
            color=color,
            linestyle="--",
            marker="D",
            markevery=max(1, len(years) // 20),
            markersize=4,
            label=f"{name} (fit)",
        )
    if cutoff_year is not None:
        plt.axvline(cutoff_year, color="black", linestyle="--", linewidth=1.5, label="Train/Test split")
    plt.xlabel("Year")
    plt.ylabel("GDP (billions USD, log scale)")
    plt.yscale("log")
    plt.title("Empirical vs GLV fit (Extended)")
    handles, labels = plt.gca().get_legend_handles_labels()
    target_rows = 4
    ncol = max(1, math.ceil(len(labels) / target_rows)) if labels else 1
    legend = plt.legend(
        handles,
        labels,
        ncol=ncol,
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_A_matrix(model: GDPGLVExtendedModel, names: List[str], output_path: Path):
    A = model.A.detach().cpu().numpy()
    abs_max = np.abs(A).max()
    vmin = -abs_max if abs_max > 0 else -1.0
    vmax = abs_max if abs_max > 0 else 1.0
    plt.figure(figsize=(6, 5))
    im = plt.imshow(A, cmap="RdBu_r", vmin=vmin, vmax=vmax)
    plt.colorbar(im, label="A_ij")
    plt.xticks(np.arange(len(names)), names, rotation=90, fontsize=8)
    plt.yticks(np.arange(len(names)), names, fontsize=8)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_beta_history(history: List[dict], output_path: Path):
    epochs = [h["epoch"] for h in history]
    betas = [h.get("beta", 0.0) for h in history]
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, betas, linewidth=2, color="blue")
    plt.xlabel("Epoch")
    plt.ylabel("Beta (interaction exponent)")
    plt.title("Beta Parameter Evolution During Training")
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Extended GLV model with coupled GDP and latent macro dynamics.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--macro-csv", type=str, default=str(DEFAULT_MACRO_CSV))
    parser.add_argument("--start-year", type=int, default=1995)
    parser.add_argument("--end-year", type=int, default=2023)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--gdp-scale", type=float, default=1e9)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--train-cut-year",
        type=int,
        default=2015,
        help="First year reserved for holdout evaluation (training uses years strictly before this).",
    )
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=4, help="RK4 substeps per observed interval.")
    parser.add_argument("--state-clamp", type=float, default=100.0)
    parser.add_argument("--a-init", type=float, default=None)
    parser.add_argument("--a-diag-init", type=float, default=None)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--cyclical-max-lr", type=float, default=None)
    parser.add_argument("--cyclical-step", type=int, default=200)
    parser.add_argument("--curriculum-start", type=float, default=5.0)
    parser.add_argument("--curriculum-increment", type=float, default=1.0)
    parser.add_argument("--curriculum-period", type=int, default=10)
    parser.add_argument("--curriculum-auto", action="store_true")
    parser.add_argument("--curriculum-threshold", type=float, default=0.5)
    parser.add_argument("--stagnation-epochs", type=int, default=50)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-hidden", type=int, default=2)
    parser.add_argument("--phi-normalize-input", action="store_true")
    parser.add_argument("--country-embed-dim", type=int, default=16)
    parser.add_argument("--no-country-embeds", action="store_true")
    parser.add_argument("--pretrain-epochs", type=int, default=0)
    parser.add_argument("--pretrain-batch-size", type=int, default=128)
    parser.add_argument("--pretrain-lr", type=float, default=1e-3)
    parser.add_argument("--relative-eps", type=float, default=1e-3)
    parser.add_argument("--lambda-a", type=float, default=1e-4)
    parser.add_argument("--lambda-a-warmup-mult", type=float, default=1.0)
    parser.add_argument("--lambda-a-cooldown-epochs", type=int, default=0)
    parser.add_argument("--interaction-warmup-epochs", type=int, default=0)
    parser.add_argument("--interaction-ramp-epochs", type=int, default=0)
    parser.add_argument("--interaction-gain", type=float, default=1.0)
    parser.add_argument("--learn-interaction-scales", action="store_true")
    parser.add_argument("--interaction-scale-max", type=float, default=1.0)
    parser.add_argument("--interaction-beta-init", type=float, default=1.0)
    parser.add_argument("--a-lr-mult", type=float, default=0.5)
    parser.add_argument("--a-max-abs", type=float, default=0.0)
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--y-loss-weight", type=float, default=1.0)
    parser.add_argument("--exp-root", type=str, default="experiments")
    parser.add_argument("--exp-prefix", type=str, default="GLV_learning_micro_gdp_extended")
    parser.add_argument("--preview-interval", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset = build_extended_dataset(args)
    times_full = dataset.times.to(device)
    true_gdp_full = dataset.gdp_values.to(device)
    true_y_full = dataset.y_values.to(device)
    num_countries = true_gdp_full.shape[1]
    feature_dim = true_y_full.shape[2]
    baseline_tensor = torch.from_numpy(dataset.baselines).to(device).view(1, -1)

    cut_year = args.train_cut_year
    years_np = dataset.years
    if not (years_np[0] < cut_year <= years_np[-1]):
        raise ValueError(f"--train-cut-year ({cut_year}) must be within ({years_np[0]+1} .. {years_np[-1]})")
    split_idx = int(np.searchsorted(years_np, cut_year))
    if split_idx < 2:
        raise ValueError("Training split must contain at least two timesteps.")

    train_years = years_np[:split_idx]
    times = times_full[:split_idx]
    true_gdp = true_gdp_full[:split_idx]
    true_y = true_y_full[:split_idx]

    holdout_len = times_full.shape[0] - split_idx
    if holdout_len > 0:
        times_hold = (times_full[split_idx:] - times_full[split_idx]).to(device)
        true_gdp_hold = true_gdp_full[split_idx:]
        true_y_hold = true_y_full[split_idx:]
        holdout_init_state = torch.cat(
            [true_gdp_full[split_idx], true_y_full[split_idx].reshape(-1)], dim=0
        ).to(device)
    else:
        times_hold = None
        true_gdp_hold = None
        true_y_hold = None
        holdout_init_state = None

    deriv_gdp = finite_difference(true_gdp, times)
    deriv_y = finite_difference(true_y, times)
    state_samples = torch.cat(
        [
            true_gdp[:-1],
            true_y[:-1].reshape(true_y.shape[0] - 1, -1),
        ],
        dim=1,
    )
    target_samples = torch.cat(
        [
            deriv_gdp,
            deriv_y.reshape(deriv_y.shape[0], -1),
        ],
        dim=1,
    )

    exp_root = Path(args.exp_root)
    exp_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    exp_dir = exp_root / f"{args.exp_prefix}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=False)

    command_tokens = [sys.executable] + sys.argv
    command_str = shlex.join(command_tokens)
    (exp_dir / "command.txt").write_text(command_str + "\n")

    traj_plot_path = exp_dir / "trajectory.png"
    a_plot_path = exp_dir / "A_matrix.png"
    beta_plot_path = exp_dir / "beta_history.png"

    model = GDPGLVExtendedModel(
        num_countries,
        feature_dim,
        args.hidden_dim,
        args.num_hidden,
        args.a_init,
        args.a_diag_init,
        normalize_input=args.phi_normalize_input,
        country_embed_dim=args.country_embed_dim,
        use_country_embeddings=not args.no_country_embeds,
        use_interactions=not args.local_only,
        interaction_gain=args.interaction_gain,
        learn_interaction_scales=args.learn_interaction_scales,
        interaction_scale_max=args.interaction_scale_max,
        interaction_beta_init=args.interaction_beta_init,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model learnable parameters: {total_params:,}")

    if args.pretrain_epochs > 0:
        pretrain_states = state_samples.to(device)
        pretrain_targets = target_samples.to(device)
        pretrain_coupled_model(model, pretrain_states, pretrain_targets, device, args)
        with torch.no_grad():
            init_state = torch.cat([true_gdp[0], true_y[0].reshape(-1)], dim=0)
            prev_factor = model.interaction_factor
            model.set_interaction_factor(1.0)
            train_pre, hold_pre = rollout_train_and_holdout(
                model,
                init_state,
                times,
                holdout_init_state,
                times_hold,
                args.steps,
                args.state_clamp,
            )
            model.set_interaction_factor(prev_factor)
        preds_to_concat = [train_pre]
        if hold_pre is not None:
            preds_to_concat.append(hold_pre)
        pred_pretrain = torch.cat(preds_to_concat, dim=0)
        pred_x = pred_pretrain[:, :num_countries].detach().cpu().numpy() * dataset.baselines[None, :]
        plot_trajectories(
            dataset.years,
            true_gdp_full.cpu().numpy() * dataset.baselines[None, :],
            pred_x,
            dataset.names,
            exp_dir / "trajectory_epoch_0000.png",
            cutoff_year=cut_year,
        )

    phi_params = list(model.phi_self.parameters()) + list(model.phi_y.parameters())
    interaction_params = model.interaction_parameters()
    optim_groups = [
        {"params": phi_params, "lr": args.lr, "weight_decay": args.weight_decay},
    ]
    if interaction_params:
        optim_groups.append(
            {"params": interaction_params, "lr": args.lr * args.a_lr_mult, "weight_decay": args.weight_decay}
        )
    optimizer = optim.AdamW(optim_groups)
    trainable_params = phi_params + interaction_params
    scheduler = None
    if args.cyclical_max_lr is not None:
        if interaction_params:
            base_lrs = [args.lr, args.lr * args.a_lr_mult]
            max_lrs = [args.cyclical_max_lr, args.cyclical_max_lr * args.a_lr_mult]
        else:
            base_lrs = args.lr
            max_lrs = args.cyclical_max_lr
        scheduler = optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=base_lrs,
            max_lr=max_lrs,
            step_size_up=max(1, args.cyclical_step),
            mode="triangular2",
            cycle_momentum=False,
        )

    total_steps = times.shape[0]
    curriculum_T = float(min(total_steps, max(2.0, args.curriculum_start)))
    best_loss = float("inf")
    last_improve_epoch = 0
    history = []

    def compute_interaction_factor(epoch: int) -> float:
        if not model.use_interactions:
            return 0.0
        warm = max(0, args.interaction_warmup_epochs)
        if epoch <= warm:
            return 0.0
        ramp = max(0, args.interaction_ramp_epochs)
        if ramp <= 0:
            return 1.0
        progress = min(1.0, (epoch - warm) / float(ramp))
        return progress

    def compute_lambda_a(epoch: int) -> float:
        base = args.lambda_a
        if not model.use_interactions or args.lambda_a_warmup_mult <= 1.0:
            return base
        warm = max(0, args.interaction_warmup_epochs)
        if epoch <= warm:
            return base * args.lambda_a_warmup_mult
        cooldown = args.lambda_a_cooldown_epochs
        if cooldown <= 0:
            cooldown = max(1, args.interaction_ramp_epochs)
        progress = min(1.0, (epoch - warm) / float(max(1, cooldown)))
        factor = args.lambda_a_warmup_mult - (args.lambda_a_warmup_mult - 1.0) * progress
        return base * factor

    init_state = torch.cat([true_gdp[0], true_y[0].reshape(-1)], dim=0)
    progress = tqdm(range(1, args.epochs + 1), desc="Training", ncols=140)
    for epoch in progress:
        optimizer.zero_grad()
        interaction_factor = compute_interaction_factor(epoch)
        model.set_interaction_factor(interaction_factor)
        lambda_a_curr = compute_lambda_a(epoch)

        step_limit = int(max(2, min(total_steps, round(curriculum_T))))
        curr_year = int(train_years[min(step_limit - 1, len(train_years) - 1)])
        times_slice = times[:step_limit]
        true_x_slice = true_gdp[:step_limit]
        true_y_slice = true_y[:step_limit]

        pred_slice = rollout_extended_model(model, init_state, times_slice, args.steps, args.state_clamp)
        pred_x_slice = pred_slice[:, :num_countries]
        pred_y_slice = pred_slice[:, num_countries:].view(step_limit, num_countries, feature_dim)

        true_real = true_x_slice * baseline_tensor
        pred_real = pred_x_slice * baseline_tensor
        rel_error = (pred_real - true_real).abs() / (true_real.abs() + args.relative_eps)
        loss_gdp = rel_error.mean()

        loss_y = torch.mean((pred_y_slice - true_y_slice) ** 2)

        reg_A = lambda_a_curr * torch.mean(model.A ** 2)
        loss = loss_gdp + args.y_loss_weight * loss_y + reg_A

        loss.backward()
        if args.grad_clip > 0:
            clip_grad_norm_(trainable_params, args.grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if args.a_max_abs > 0:
            with torch.no_grad():
                model.A.clamp_(-abs(args.a_max_abs), abs(args.a_max_abs))

        current_lr = optimizer.param_groups[0]["lr"]
        progress.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "loss_x": f"{loss_gdp.item():.4f}",
                "loss_y": f"{loss_y.item():.4f}",
                "steps": f"{step_limit}/{total_steps}",
                "year": curr_year,
                "lr": f"{current_lr:.4e}",
                "int": f"{interaction_factor:.2f}",
            }
        )

        beta_value = float(model.interaction_beta.item()) if model.use_interactions else 0.0
        history.append(
            {
                "epoch": epoch,
                "loss": float(loss.item()),
                "loss_gdp": float(loss_gdp.item()),
                "loss_y": float(loss_y.item()),
                "reg_A": float(reg_A.item()),
                "lr": float(current_lr),
                "curriculum_T": float(curriculum_T),
                "curriculum_steps": int(step_limit),
                "lambda_a": float(lambda_a_curr),
                "interaction_factor": float(interaction_factor),
                "beta": beta_value,
            }
        )

        preview_due = args.preview_interval > 0 and epoch % args.preview_interval == 0
        auto_due = epoch % 100 == 0
        if preview_due or auto_due:
            with torch.no_grad():
                train_prev, hold_prev = rollout_train_and_holdout(
                    model,
                    init_state,
                    times,
                    holdout_init_state,
                    times_hold,
                    args.steps,
                    args.state_clamp,
                )
            preds_prev = [train_prev]
            if hold_prev is not None:
                preds_prev.append(hold_prev)
            pred_preview = torch.cat(preds_prev, dim=0).detach().cpu().numpy()
            pred_preview_x = pred_preview[:, :num_countries] * dataset.baselines[None, :]
            plot_path = exp_dir / f"trajectory_epoch_{epoch:04d}.png"
            plot_trajectories(
                dataset.years,
                true_gdp_full.cpu().numpy() * dataset.baselines[None, :],
                pred_preview_x,
                dataset.names,
                plot_path,
                cutoff_year=cut_year,
            )

        if loss_gdp.item() + 1e-6 < best_loss:
            best_loss = loss_gdp.item()
            last_improve_epoch = epoch

        should_increase = False
        if args.curriculum_auto:
            if loss_gdp.item() <= args.curriculum_threshold:
                should_increase = True
        else:
            if epoch % max(1, args.curriculum_period) == 0:
                should_increase = True
        if (epoch - last_improve_epoch) >= args.stagnation_epochs:
            should_increase = True
            last_improve_epoch = epoch
        if should_increase and curriculum_T < total_steps:
            curriculum_T = min(total_steps, curriculum_T + args.curriculum_increment)

    model.set_interaction_factor(1.0)
    train_traj, hold_traj = rollout_train_and_holdout(
        model,
        init_state,
        times,
        holdout_init_state,
        times_hold,
        args.steps,
        args.state_clamp,
    )
    pred_parts = [train_traj.detach().cpu()]
    if hold_traj is not None:
        pred_parts.append(hold_traj.detach().cpu())
    full_traj = torch.cat(pred_parts, dim=0).numpy()

    pred_x = full_traj[:, :num_countries]
    pred_y = full_traj[:, num_countries:].reshape(len(times_full), num_countries, feature_dim)
    true_x_np = true_gdp_full.cpu().numpy()
    true_y_np = true_y_full.cpu().numpy()

    true_real_full = true_x_np * dataset.baselines[None, :]
    pred_real_full = pred_x * dataset.baselines[None, :]
    plot_trajectories(dataset.years, true_real_full, pred_real_full, dataset.names, traj_plot_path, cutoff_year=cut_year)
    plot_A_matrix(model, dataset.names, a_plot_path)
    if model.use_interactions:
        plot_beta_history(history, beta_plot_path)

    np.save(exp_dir / "times.npy", dataset.times.cpu().numpy())
    np.save(exp_dir / "true_gdp.npy", true_real_full)
    np.save(exp_dir / "pred_gdp.npy", pred_real_full)
    np.save(exp_dir / "true_y.npy", true_y_np)
    np.save(exp_dir / "pred_y.npy", pred_y)
    torch.save(model.state_dict_to_save(), exp_dir / "model.pt")

    with (exp_dir / "history.json").open("w") as f_hist:
        json.dump(history, f_hist, indent=2)

    holdout_mask = years_np >= cut_year
    holdout_years = years_np[holdout_mask].tolist() if holdout_mask.any() else []
    holdout_loss = None
    if holdout_mask.any():
        diff = pred_real_full[holdout_mask] - true_real_full[holdout_mask]
        holdout_rel = np.abs(diff) / (np.abs(true_real_full[holdout_mask]) + args.relative_eps)
        holdout_loss = float(holdout_rel.mean())
        print(f"[INFO] Holdout relative loss ({cut_year}-{years_np[-1]}): {holdout_loss:.4f}")

    config = {
        "timestamp": timestamp,
        "args": vars(args),
        "names": dataset.names,
        "years": dataset.years.tolist(),
        "feature_columns": FEATURE_COLUMNS,
        "best_traj_loss": best_loss,
        "train_years": years_np[:split_idx].tolist(),
        "holdout_years": holdout_years,
        "holdout_rel_loss": holdout_loss,
    }
    with (exp_dir / "config.json").open("w") as f_cfg:
        json.dump(config, f_cfg, indent=2)

    print(f"[INFO] Experiment saved to {exp_dir}")


if __name__ == "__main__":
    main()

