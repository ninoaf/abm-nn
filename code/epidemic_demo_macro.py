"""
Macro-level learning of SIR-on-graph RHS using only aggregate curves s(t), i(t), r(t).

Procedure:
- Generate one Erdős–Rényi graph (n1=100, p=0.05)
- Simulate true ODE with (beta=0.4, gamma=0.2) to obtain s(t), i(t), r(t)
- Train GraphRHSNet (phi1, phi2) by minimizing L1 loss over aggregate curves
  L = sum_t |s(t)-s~(t)| + |i(t)-i~(t)| + |r(t)-r~(t)|

This script reuses simulator and GraphRHSNet from epidemic_demo_micro.py,
and provides a differentiable torch-based simulator for training.
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from typing import Tuple, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import time
import shutil
import datetime
import json

from epidemic_demo_micro import (
    generate_erdos_renyi_adjacency_networkx,
    simulate_sir_on_graph,
)


class PairwiseMLPReLU(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, num_hidden: int = 2, out_dim: int = 1):
        super().__init__()
        layers = []
        last = in_dim
        for _ in range(num_hidden):
            layers.append(nn.Linear(last, hidden_dim, bias=True))
            layers.append(nn.LeakyReLU())
            last = hidden_dim
        layers.append(nn.Linear(last, out_dim, bias=True))
        layers.append(nn.ReLU())  # nonnegative, zero at zero
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ScalarMLPReLU(nn.Module):
    def __init__(self, hidden_dim: int = 64, num_hidden: int = 2, out_dim: int = 1):
        super().__init__()
        layers = []
        last = 1
        for _ in range(num_hidden):
            layers.append(nn.Linear(last, hidden_dim, bias=True))
            layers.append(nn.LeakyReLU())
            last = hidden_dim
        layers.append(nn.Linear(last, out_dim, bias=True))
        layers.append(nn.ReLU())  # nonnegative, zero at zero
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GraphRHSNet(nn.Module):
    def __init__(self, adjacency: np.ndarray, hidden_dim: int = 64, num_hidden: int = 2, gamma: float = 0.2):
        super().__init__()
        adj = np.asarray(adjacency)
        if adj.shape[0] != adj.shape[1]:
            raise ValueError("adjacency must be square")
        self.num_nodes = adj.shape[0]
        ii, jj = np.where(adj > 0.0)
        self.register_buffer("edge_i", torch.from_numpy(ii.astype(np.int64)))
        self.register_buffer("edge_j", torch.from_numpy(jj.astype(np.int64)))
        self.phi1 = PairwiseMLPReLU(in_dim=2, hidden_dim=hidden_dim, num_hidden=num_hidden, out_dim=1)
        self.phi2 = ScalarMLPReLU(hidden_dim=hidden_dim, num_hidden=num_hidden, out_dim=1)
        self.gamma = float(gamma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N = self.num_nodes
        S = x[:, :N]
        I = x[:, N:2 * N]
        # R not needed in RHS

        dS_list = []
        dI_list = []
        dR_list = []
        for b in range(x.shape[0]):
            Si = S[b, self.edge_i]
            Ij = I[b, self.edge_j]
            vals_s = self.phi1(torch.stack([Si, Ij], dim=1)).squeeze(-1)
            acc_s = torch.zeros(N, device=x.device, dtype=x.dtype)
            acc_s.index_add_(0, self.edge_i, vals_s)
            rec = self.phi2(I[b].unsqueeze(-1)).squeeze(-1)
            dS_b = -acc_s
            dI_b = acc_s - rec
            dR_b = rec
            dS_list.append(dS_b)
            dI_list.append(dI_b)
            dR_list.append(dR_b)

        dS = torch.stack(dS_list, dim=0)
        dI = torch.stack(dI_list, dim=0)
        dR = torch.stack(dR_list, dim=0)
        return torch.cat([dS, dI, dR], dim=1)


def torch_rk4_step_rhsnet(model: GraphRHSNet, x: torch.Tensor, dt: float) -> torch.Tensor:
    """Single RK4 step for state x with derivatives given by model(x)."""
    k1 = model(x)
    k2 = model(x + 0.5 * dt * k1)
    k3 = model(x + 0.5 * dt * k2)
    k4 = model(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def simulate_with_model_torch(
    model: GraphRHSNet,
    S0: np.ndarray,
    I0: np.ndarray,
    R0: np.ndarray,
    t_max: float,
    dt: float,
    device: torch.device,
) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Differentiable simulation using GraphRHSNet and RK4 in torch.
    Returns times (np), and S/I/R histories as numpy arrays but keeps computation graph on model params.
    """
    model.train()
    N = S0.shape[0]
    num_steps = int(np.ceil(t_max / dt))
    times = np.linspace(0.0, num_steps * dt, num_steps + 1, dtype=np.float32)

    # Initialize state tensor
    x = torch.from_numpy(np.concatenate([S0, I0, R0]).astype(np.float32)).to(device)
    x = x.unsqueeze(0)  # (1, 3N)

    S_hist, I_hist, R_hist = [], [], []
    # store initial
    S_hist.append(x[0, :N].clone())
    I_hist.append(x[0, N:2 * N].clone())
    R_hist.append(x[0, 2 * N:3 * N].clone())

    for _ in range(num_steps):
        x = torch_rk4_step_rhsnet(model, x, dt)
        # Clamp and renormalize per node (non-differentiable at bounds, acceptable for stability)
        S = x[:, :N]
        I = x[:, N:2 * N]
        R = x[:, 2 * N:3 * N]
        S = torch.clamp(S, 0.0, 1.0)
        I = torch.clamp(I, 0.0, 1.0)
        R = torch.clamp(R, 0.0, 1.0)
        total = S + I + R
        S = S / total
        I = I / total
        R = R / total
        x = torch.cat([S, I, R], dim=1)

        S_hist.append(S[0].clone())
        I_hist.append(I[0].clone())
        R_hist.append(R[0].clone())

    return times, torch.stack(S_hist, dim=0), torch.stack(I_hist, dim=0), torch.stack(R_hist, dim=0)


def compute_regularizers(model: GraphRHSNet, device: torch.device, reg_n: int = 256):
    s_samples = torch.rand(reg_n, 1, device=device)
    zeros = torch.zeros_like(s_samples)
    reg_axis_val = model.phi1(torch.cat([s_samples, zeros], dim=1)).abs().mean()
    reg_phi2_origin_val = model.phi2(torch.zeros(reg_n, 1, device=device)).abs().mean()

    # Raw outputs for diagnostics
    phi1_axis_out = model.phi1(torch.cat([s_samples, zeros], dim=1)).squeeze()
    phi2_zero_out = model.phi2(torch.zeros(reg_n, 1, device=device)).squeeze()
    return reg_axis_val, reg_phi2_origin_val, phi1_axis_out, phi2_zero_out


def compute_grad_norm(model: nn.Module) -> float:
    grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            g = p.grad.detach()
            grad_norm += float(g.norm(2).item() ** 2)
    return grad_norm ** 0.5


def print_init_metrics(t_curr0: float, base_loss0: torch.Tensor, reg_axis0: torch.Tensor,
                       reg_phi2_origin0: torch.Tensor, loss_total0: torch.Tensor,
                       phi1_axis_out0: torch.Tensor, phi2_zero_out0: torch.Tensor) -> None:
    with torch.no_grad():
        frac_zero_phi1 = float((phi1_axis_out0 == 0).float().mean().detach().cpu().item())
        frac_zero_phi2 = float((phi2_zero_out0 == 0).float().mean().detach().cpu().item())
        min_phi1 = float(phi1_axis_out0.min().detach().cpu().item())
        max_phi1 = float(phi1_axis_out0.max().detach().cpu().item())
        min_phi2 = float(phi2_zero_out0.min().detach().cpu().item())
        max_phi2 = float(phi2_zero_out0.max().detach().cpu().item())
    print(
        f"Init (pre-train) - t_curr={t_curr0:.1f} - loss_base: {float(base_loss0):.3e} "
        f"- reg_axis_phi1: {float(reg_axis0):.3e} - reg_phi2_origin: {float(reg_phi2_origin0):.3e} "
        f"- loss_total: {float(loss_total0):.3e}"
    )
    print(
        f"  phi1(S,0): min={min_phi1:.3e}, max={max_phi1:.3e}, frac==0: {frac_zero_phi1:.3f}; "
        f"phi2(0): min={min_phi2:.3e}, max={max_phi2:.3e}, frac==0: {frac_zero_phi2:.3f}"
    )


def print_epoch_metrics(epoch: int, epochs: int, t_curr: float, base_loss: float, reg_axis: float,
                        reg_phi2: float, final_loss: float, lr: float, grad_norm: float) -> None:
    print(
        f"Epoch {epoch}/{epochs} - t_curr={t_curr:.1f} - loss_base: {base_loss:.6f} "
        f"- reg_axis_phi1: {reg_axis:.6f} - reg_phi2_origin: {reg_phi2:.6f} "
        f"- loss_total: {final_loss:.6f} - lr: {lr:.2e} - grad_norm: {grad_norm:.3e}"
    )


def train_macro(
    adjacency: np.ndarray,
    S0: np.ndarray,
    I0: np.ndarray,
    R0: np.ndarray,
    target_times: np.ndarray,
    target_S_sum: np.ndarray,
    target_I_sum: np.ndarray,
    target_R_sum: np.ndarray,
    epochs: int = 200,
    dt: float = 0.1,
    lr: float = 1e-4,
    hidden_dim: int = 32,
    num_hidden: int = 1,
    gamma: float = 0.2,
    device: str | None = None,
    horizon_increment: float = 5.0,
    horizon_step_epochs: int = 10,
    horizon_base: float = 5.0,
    lambda_phi1_axis: float = 1e-3,
    lambda_phi2_zero: float = 1e-5,
    clip_grad: float | None = 1.0,
    lr_decay_gamma: float = 1.0,
    triangular_max_lr: float | None = None,
    triangular_step_epochs: int = 50,
    t_train_max: float | None = None,
    pretrained_state_dict: dict | None = None,
    reg_warmup_epochs: int = 0,
) -> Tuple[GraphRHSNet, dict]:
    """Train GraphRHSNet via macroscopic L1 loss on aggregate curves."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)

    model = GraphRHSNet(adjacency=adjacency, hidden_dim=hidden_dim, num_hidden=num_hidden, gamma=gamma).to(device_t)
    if pretrained_state_dict is not None:
        try:
            # Accept either nested or flat dicts
            if "phi1" in pretrained_state_dict and "phi2" in pretrained_state_dict:
                model.phi1.load_state_dict(pretrained_state_dict["phi1"])  # type: ignore[index]
                model.phi2.load_state_dict(pretrained_state_dict["phi2"])  # type: ignore[index]
            else:
                model.load_state_dict(pretrained_state_dict, strict=False)
            print("Loaded pretrained weights into model for resume training.")
        except Exception as e:
            print(f"Warning: failed to load pretrained weights (continuing from scratch): {e}")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = None
    use_triangular = (
        triangular_max_lr is not None and float(triangular_max_lr) > lr and triangular_step_epochs > 0
    )
    if use_triangular:
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=lr,
            max_lr=float(triangular_max_lr),
            step_size_up=int(triangular_step_epochs),
            step_size_down=int(triangular_step_epochs),
            mode="triangular",
            cycle_momentum=False,
        )
    elif lr_decay_gamma is not None and lr_decay_gamma < 1.0:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=float(lr_decay_gamma))

    # Convert targets to torch
    s_target = torch.from_numpy(target_S_sum.astype(np.float32)).to(device_t)
    i_target = torch.from_numpy(target_I_sum.astype(np.float32)).to(device_t)
    r_target = torch.from_numpy(target_R_sum.astype(np.float32)).to(device_t)

    history = {"loss": [], "loss_base": [], "reg_axis_phi1": [], "reg_phi2_origin": [], "loss_total": []}
    t_max_total = float(target_times[-1]) if t_train_max is None else float(t_train_max)
    train_start_ts = time.time()
    printed_100 = False

    # Record initial (pre-training) metrics to inspect NN initialization
    try:
        stage0 = 0
        t_curr0 = min(t_max_total, float(horizon_base) + float(horizon_increment) * stage0)
        L_curr0 = int(np.ceil(t_curr0 / dt)) + 1
        with torch.no_grad():
            _times0, _S0, _I0, _R0 = simulate_with_model_torch(
                model, S0=S0, I0=I0, R0=R0, t_max=t_curr0, dt=dt, device=device_t
            )
            s_pred0 = _S0.sum(dim=1)
            i_pred0 = _I0.sum(dim=1)
            r_pred0 = _R0.sum(dim=1)
            Nloc0 = adjacency.shape[0]
            denom0 = float(max(1, L_curr0) * max(1, Nloc0))
            base_loss0 = (
                (s_pred0 - s_target[:L_curr0]).abs().sum()
                + (i_pred0 - i_target[:L_curr0]).abs().sum()
                + (r_pred0 - r_target[:L_curr0]).abs().sum()
            ) / denom0
            reg_axis0, reg_phi2_origin0, phi1_axis_out0, phi2_zero_out0 = compute_regularizers(model, device_t, reg_n=256)
            loss_total0 = base_loss0
            if int(reg_warmup_epochs) <= 0:
                if lambda_phi1_axis > 0.0:
                    loss_total0 = loss_total0 + lambda_phi1_axis * reg_axis0
                if lambda_phi2_zero > 0.0:
                    loss_total0 = loss_total0 + lambda_phi2_zero * reg_phi2_origin0

        history["loss"].append(float(loss_total0.detach().cpu().item()))
        history["loss_base"].append(float(base_loss0.detach().cpu().item()))
        history["reg_axis_phi1"].append(float(reg_axis0.detach().cpu().item()))
        history["reg_phi2_origin"].append(float(reg_phi2_origin0.detach().cpu().item()))
        history["loss_total"].append(float(loss_total0.detach().cpu().item()))

        print_init_metrics(t_curr0, base_loss0, reg_axis0, reg_phi2_origin0, loss_total0, phi1_axis_out0, phi2_zero_out0)
    except Exception:
        pass

    for epoch in range(epochs):
        # Curriculum on simulation horizon: first 10 epochs t=5, then +5 every 10 epochs
        stage = epoch // max(1, horizon_step_epochs)
        t_curr = min(t_max_total, float(horizon_base) + float(horizon_increment) * stage)
        # Corresponding target length
        L_curr = int(np.ceil(t_curr / dt)) + 1
        optimizer.zero_grad()
        times_pred, S_hist, I_hist, R_hist = simulate_with_model_torch(
            model, S0=S0, I0=I0, R0=R0, t_max=t_curr, dt=dt, device=device_t
        )
        # Aggregate sums
        s_pred = S_hist.sum(dim=1)
        i_pred = I_hist.sum(dim=1)
        r_pred = R_hist.sum(dim=1)
        # L1 macro loss normalized by time length and node count to stabilize scale
        Nloc = adjacency.shape[0]
        denom = float(max(1, L_curr) * max(1, Nloc))
        base_loss = (
            (s_pred - s_target[:L_curr]).abs().sum()
            + (i_pred - i_target[:L_curr]).abs().sum()
            + (r_pred - r_target[:L_curr]).abs().sum()
        ) / denom
        # Regularizers (raw, unweighted values) for logging
        reg_axis_val, reg_phi2_origin_val, _phi1_axis_out, _phi2_zero_out = compute_regularizers(model, device_t, reg_n=256)
        # Compose final loss with warmup
        loss = base_loss
        if epoch >= int(reg_warmup_epochs):
            if lambda_phi1_axis > 0.0:
                loss = loss + lambda_phi1_axis * reg_axis_val
            if lambda_phi2_zero > 0.0:
                loss = loss + lambda_phi2_zero * reg_phi2_origin_val
        loss.backward()
        # Compute grad L2 norm before clipping
        grad_norm = compute_grad_norm(model)
        if clip_grad and float(clip_grad) > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(clip_grad))
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        base_v = float(base_loss.detach().cpu().item())
        reg_axis_v = float(reg_axis_val.detach().cpu().item())
        reg_phi2_origin_v = float(reg_phi2_origin_val.detach().cpu().item())
        final_v = float(loss.detach().cpu().item())
        history["loss"].append(final_v)
        history["loss_base"].append(base_v)
        history["reg_axis_phi1"].append(reg_axis_v)
        history["reg_phi2_origin"].append(reg_phi2_origin_v)
        history["loss_total"].append(final_v)
        if (epoch + 1) == 100 and not printed_100:
            elapsed_100 = time.time() - train_start_ts
            print(f"Time for first 100 epochs: {elapsed_100:.2f}s (~{elapsed_100/100.0:.3f}s/epoch)")
            printed_100 = True
        if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 1:
            curr_lr = optimizer.param_groups[0]["lr"]
            print_epoch_metrics(epoch + 1, epochs, t_curr, base_v, reg_axis_v, reg_phi2_origin_v, final_v, curr_lr, grad_norm)

    total_elapsed = time.time() - train_start_ts
    print(f"Total training time: {total_elapsed:.2f}s (~{total_elapsed/max(1, epochs):.3f}s/epoch)")
    return model, history


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Macro learning of SIR RHS from aggregate curves")
    parser.add_argument("--nodes", type=int, default=100)
    parser.add_argument("--p", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument("--gamma", type=float, default=0.2)
    parser.add_argument("--t_max", type=float, default=30.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--num_hidden", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--horizon_increment", type=float, default=1.0, help="Time horizon increment every 10 epochs")
    parser.add_argument("--horizon_step_epochs", type=int, default=50, help="Number of epochs between horizon increments")
    parser.add_argument("--horizon_base", type=float, default=1.0, help="Starting horizon for curriculum")
    parser.add_argument("--clip_grad", type=float, default=10.0, help="Max gradient norm for clipping (<=0 to disable)")
    parser.add_argument("--lr_decay_gamma", type=float, default=1.0, help="Exponential LR decay gamma (<1 to enable)")
    parser.add_argument("--t_train_max", type=float, default=30, help="Optional cap on training horizon (<= t_max)")
    parser.add_argument("--save_model", type=str, default=None, help="Path to save trained model (phi1, phi2, adjacency, gamma)")
    parser.add_argument("--load_model", type=str, default=None, help="Path to load a saved model")
    parser.add_argument("--load_only", action="store_true", help="If set, load the model and only run simulation/plotting")
    parser.add_argument("--simulate_only", action="store_true", help="Run simulation/plotting from a saved model; do not train (requires --load_model)")
    parser.add_argument("--lambda_phi1_axis", type=float, default=1, help="Weight for |phi1(S,0)| axis regularizer")
    parser.add_argument("--lambda_phi2_zero", type=float, default=1, help="Weight for |phi2(0)| zero regularizer")
    parser.add_argument("--reg_warmup_epochs", type=int, default=100, help="Epochs to delay adding regularizers")
    parser.add_argument("--exp_name", type=str, default=None, help="Name prefix for experiment folder")
    parser.add_argument("--exp_root", type=str, default="experiments", help="Root folder to store experiments")
    return parser


def setup_experiment(exp_root: str, exp_name: str | None) -> Tuple[str, str]:
    """Create experiment directory and return (exp_dir, timestamp_str)."""
    ts = datetime.datetime.now().strftime("%d_%m_%Y_%H:%M")
    exp_prefix = exp_name if (exp_name is not None and len(exp_name) > 0) else ""
    exp_root_dir = os.path.join(os.path.dirname(__file__), exp_root)
    os.makedirs(exp_root_dir, exist_ok=True)
    exp_dirname = f"{exp_prefix}_{ts}" if len(exp_prefix) > 0 else ts
    exp_dir = os.path.join(exp_root_dir, exp_dirname)
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir, ts


def create_initial_conditions(G, nodes: int, seed: int, k_infected: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create S0, I0, R0 with k infected nodes sampled from the GCC if available."""
    rng = np.random.default_rng(seed)
    I0 = np.zeros(nodes, dtype=np.float32)
    try:
        import networkx as nx
        if G.number_of_nodes() > 0:
            largest_cc_nodes = max(nx.connected_components(G), key=len)
            cc_list = list(largest_cc_nodes)
            k = min(k_infected, len(cc_list))
            seeds = rng.choice(cc_list, size=k, replace=False)
            I0[np.asarray(seeds, dtype=int)] = 1.0
    except Exception:
        pass
    S0 = np.ones(nodes, dtype=np.float32) - I0
    R0 = np.zeros(nodes, dtype=np.float32)
    return S0, I0, R0


def maybe_load_checkpoint(args, adjacency, G):
    """Optionally load checkpoint; if adjacency present in ckpt, override current graph."""
    ckpt = None
    if args.load_model is not None and os.path.exists(args.load_model):
        ckpt = torch.load(args.load_model, map_location="cpu", weights_only=False)
        if "adjacency" in ckpt:
            adjacency = ckpt["adjacency"]
            if isinstance(adjacency, torch.Tensor):
                adjacency = adjacency.cpu().numpy()
            import networkx as nx
            G = nx.from_numpy_array(adjacency)
        elif args.load_only:
            print("Warning: checkpoint missing adjacency; using freshly generated graph.")
    return ckpt, adjacency, G


def save_model_ckpt(model: GraphRHSNet, adjacency: np.ndarray, gamma: float, path: str) -> None:
    torch.save(
        {
            "state_dict": {"phi1": model.phi1.state_dict(), "phi2": model.phi2.state_dict()},
            "adjacency": np.asarray(adjacency, dtype=np.float32),
            "gamma": gamma,
        },
        path,
    )
    print(f"Saved model to: {path}")


def save_experiment_artifacts(
    exp_dir: str,
    history: Dict[str, Any],
    adjacency: np.ndarray,
    args,
    ts: str,
    out_path: str,
    command: str | None = None,
) -> None:
    # Persist history, adjacency, and a copy of this script for reproducibility
    try:
        with open(os.path.join(exp_dir, "history.json"), "w") as f:
            json.dump(history, f)
    except Exception as e:
        print(f"Warning: failed to save history.json: {e}")

    try:
        np.save(os.path.join(exp_dir, "adjacency.npy"), np.asarray(adjacency, dtype=np.float32))
    except Exception as e:
        print(f"Warning: failed to save adjacency.npy: {e}")

    try:
        config_payload = {
            "args": vars(args),
            "timestamp": ts,
            "plot_path": out_path,
        }
        if command:
            config_payload["command"] = command
        with open(os.path.join(exp_dir, "config.json"), "w") as f:
            json.dump(config_payload, f, default=str)
    except Exception as e:
        print(f"Warning: failed to save config.json: {e}")

    try:
        shutil.copy2(__file__, os.path.join(exp_dir, os.path.basename(__file__)))
    except Exception as e:
        print(f"Warning: failed to copy script to experiment folder: {e}")


def build_and_save_plot(
    args,
    G,
    adjacency,
    times,
    S_hist,
    I_hist,
    R_hist,
    model,
    history,
    exp_dir: str,
    S_sum_obs: np.ndarray | None = None,
    I_sum_obs: np.ndarray | None = None,
    R_sum_obs: np.ndarray | None = None,
    plot_obs_only: bool = False,
) -> str:
    """
    Construct a 1x2 figure (loss diagnostics + trajectories). If observational macro curves are provided,
    they can be plotted instead of the clean aggregates, and noise statistics are reported in the title.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_nn, S_nn, I_nn, R_nn = simulate_with_model_torch(
        model,
        S0=S_hist[0].astype(np.float32),
        I0=I_hist[0].astype(np.float32),
        R0=R_hist[0].astype(np.float32),
        t_max=args.t_max,
        dt=args.dt,
        device=device,
    )

    S_sum_true = S_hist.sum(axis=1)
    I_sum_true = I_hist.sum(axis=1)
    R_sum_true = R_hist.sum(axis=1)
    S_sum_nn = S_nn.sum(dim=1).detach().cpu().numpy()
    I_sum_nn = I_nn.sum(dim=1).detach().cpu().numpy()
    R_sum_nn = R_nn.sum(dim=1).detach().cpu().numpy()

    fig = plt.figure(figsize=(12, 5))
    gs = fig.add_gridspec(1, 2, wspace=0.28, hspace=0.15)

    # Row 1, Col 1: training losses and regularizers
    ax_loss = fig.add_subplot(gs[0, 0])
    epochs_axis = np.arange(0, len(history.get("loss_base", [])))
    if len(epochs_axis) > 0:
        ax_loss.plot(epochs_axis, history.get("loss_base", []), label="loss_base", color="#1f77b4")
        ax_loss.plot(epochs_axis, history.get("reg_axis_phi1", []), label="reg_axis_phi1", linestyle=":", color="#ff7f0e")
        ax_loss.plot(epochs_axis, history.get("reg_phi2_origin", []), label="reg_phi2_origin", linestyle="-.", color="#d62728")
        ax_loss.plot(epochs_axis, history.get("loss_total", []), label="loss_total", linestyle="-", color="#9467bd")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss / Reg terms")
        ax_loss.set_yscale('log')
        ax_loss.set_title("Training losses and regularizers")
        ax_loss.legend(ncol=1, fontsize=8, loc="upper right")

    # Row 1, Col 2: trajectories (true vs NN dashed, optional observational overlays)
    ax_traj = fig.add_subplot(gs[0, 1])
    obs_available = (
        S_sum_obs is not None
        and I_sum_obs is not None
        and R_sum_obs is not None
        and len(S_sum_obs) == len(times)
    )
    if plot_obs_only and obs_available:
        base_curves = [
            (S_sum_obs, "#1f77b4", "s_obs(t)"),
            (I_sum_obs, "#ff7f0e", "i_obs(t)"),
            (R_sum_obs, "#2ca02c", "r_obs(t)"),
        ]
    else:
        base_curves = [
            (S_sum_true, "#1f77b4", "s(t)"),
            (I_sum_true, "#ff7f0e", "i(t)"),
            (R_sum_true, "#2ca02c", "r(t)"),
        ]
    for curve, color, label in base_curves:
        ax_traj.plot(times, curve, label=label, color=color)

    if obs_available and not plot_obs_only:
        ax_traj.plot(times, S_sum_obs, label="s_obs(t)", linestyle=":", color="#9ecae1", alpha=0.8)
        ax_traj.plot(times, I_sum_obs, label="i_obs(t)", linestyle=":", color="#ffb347", alpha=0.8)
        ax_traj.plot(times, R_sum_obs, label="r_obs(t)", linestyle=":", color="#a1d99b", alpha=0.8)

    nn_kw = {
        "marker": "D",
        "markersize": 4,
        "markevery": max(1, len(t_nn) // 30),
    }
    # Use distinct colors for dashed NN curves to distinguish overlaps
    ax_traj.plot(t_nn, S_sum_nn, label="ŝ(t)", linestyle="--", color="#6baed6", **nn_kw)
    ax_traj.plot(t_nn, I_sum_nn, label="î(t)", linestyle="--", color="#ffbb78", **nn_kw)
    ax_traj.plot(t_nn, R_sum_nn, label="r̂(t)", linestyle="--", color="#98df8a", **nn_kw)
    ax_traj.set_xlabel("Time")
    ax_traj.set_ylabel("Sum across nodes")
    if obs_available and hasattr(args, "obs_noise_sigma"):
        sigma_noise = float(getattr(args, "obs_noise_sigma", 0.0))

        def compute_snr(signal: np.ndarray, sigma: float) -> float:
            var_signal = float(np.var(signal))
            var_noise = float(sigma**2)
            if var_noise <= 0:
                return float("inf")
            return var_signal / var_noise if var_signal > 0 else 0.0

        snr_s = compute_snr(S_sum_true, sigma_noise)
        snr_i = compute_snr(I_sum_true, sigma_noise)
        snr_r = compute_snr(R_sum_true, sigma_noise)
        snr_mean = (snr_s + snr_i + snr_r) / 3.0
        ax_traj.set_title(f"σ_noise={sigma_noise:.3f}, SNR={snr_mean:.2f}")
    else:
        ax_traj.set_title("Ground-truth vs ABM-NN trajectory")
    ax_traj.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    out_path = args.out or os.path.join(exp_dir, "sir_on_graph_macro.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved macro plot to: {out_path}")
    return out_path


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Seed PyTorch RNGs for deterministic initialization and sampling
    try:
        torch.manual_seed(int(args.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.seed))
    except Exception:
        pass

    # Prepare experiment directory
    exp_dir, ts = setup_experiment(args.exp_root, args.exp_name)

    # Build graph and initial condition
    G, adjacency = generate_erdos_renyi_adjacency_networkx(args.nodes, args.p, args.seed)
    S0, I0, R0 = create_initial_conditions(G, args.nodes, args.seed, k_infected=1)

    # If loading-only, prefer adjacency from checkpoint for consistent simulation
    ckpt, adjacency, G = maybe_load_checkpoint(args, adjacency, G)

    times, S_hist, I_hist, R_hist = simulate_sir_on_graph(
        adjacency=adjacency, beta=args.beta, gamma=args.gamma, S0=S0, I0=I0, R0=R0, t_max=args.t_max, dt=args.dt
    )
    S_sum = S_hist.sum(axis=1)
    I_sum = I_hist.sum(axis=1)
    R_sum = R_hist.sum(axis=1)

    # Train GraphRHSNet via macro loss
    model = None
    history = {}
    if (args.load_only or args.simulate_only) and ckpt is not None:
        # Construct and load model
        model = GraphRHSNet(adjacency=adjacency, gamma=ckpt.get("gamma", args.gamma))
        state_dict = ckpt.get("state_dict", ckpt)
        # Backward-compatible: accept either separate or flat dicts
        try:
            model.phi1.load_state_dict(state_dict["phi1"])  # type: ignore[index]
            model.phi2.load_state_dict(state_dict["phi2"])  # type: ignore[index]
        except Exception:
            model.load_state_dict(state_dict)
    elif not args.simulate_only:
        pretrained_sd = None
        if ckpt is not None and not args.load_only:
            pretrained_sd = ckpt.get("state_dict", ckpt)
        model, history = train_macro(
            adjacency=adjacency,
            S0=S0,
            I0=I0,
            R0=R0,
            target_times=times.astype(np.float32),
            target_S_sum=S_sum.astype(np.float32),
            target_I_sum=I_sum.astype(np.float32),
            target_R_sum=R_sum.astype(np.float32),
            epochs=args.epochs,
            dt=args.dt,
            lr=args.lr,
            hidden_dim=args.hidden_dim,
            num_hidden=args.num_hidden,
            gamma=args.gamma,
            horizon_increment=args.horizon_increment,
            horizon_step_epochs=args.horizon_step_epochs,
            horizon_base=args.horizon_base,
            lambda_phi1_axis=args.lambda_phi1_axis,
            lambda_phi2_zero=args.lambda_phi2_zero,
            clip_grad=args.clip_grad,
            lr_decay_gamma=args.lr_decay_gamma,
            t_train_max=args.t_train_max,
            pretrained_state_dict=pretrained_sd,
            reg_warmup_epochs=args.reg_warmup_epochs,
        )
        # Save if requested
        save_model_path = args.save_model if args.save_model else os.path.join(exp_dir, "macro_rhs.ckpt")
        save_model_ckpt(model, adjacency, args.gamma, save_model_path)

    # Build and save figure
    out_path = build_and_save_plot(args, G, adjacency, times, S_hist, I_hist, R_hist, model, history, exp_dir)
    # Save metadata
    try:
        cmd = " ".join(shlex.quote(part) for part in sys.argv)
    except Exception:
        cmd = None
    save_experiment_artifacts(exp_dir, history, adjacency, args, ts, out_path, command=cmd)


if __name__ == "__main__":
    main()


