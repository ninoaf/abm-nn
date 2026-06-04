import os
import math
import random
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from tqdm import tqdm


# ============================
# Configuration and Utilities
# ============================

@dataclass
class TrainingConfig:
    random_seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_train_samples: int = 4096
    batch_size: int = 256
    learning_rate: float = 1e-3
    max_epochs: int = 2000
    print_every: int = 200

    rollout_time: float = 1000.0
    rollout_dt: float = 0.01
    num_plot_trajectories: int = 1


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================
# Mass-Spring Ground Truth
# ============================

# Mass m = 1, spring constant k = 1
# Hamiltonian H(q,p) = 0.5 * (q^2 + p^2)
# dq/dt = + dH/dp = p
# dp/dt = - dH/dq = -q

def true_time_derivative(state: torch.Tensor) -> torch.Tensor:
    """Compute true time derivative for mass-spring system.

    Args:
        state: Tensor of shape (N, 2) where columns are (q, p)
    Returns:
        Tensor of shape (N, 2) containing (dq/dt, dp/dt)
    """
    q = state[:, 0:1]
    p = state[:, 1:2]
    dq_dt = p
    dp_dt = -q
    return torch.cat([dq_dt, dp_dt], dim=1)


def rk4_step(dynamics_fn, state: torch.Tensor, dt: float) -> torch.Tensor:
    """Runge-Kutta 4th order single step.

    Args:
        dynamics_fn: function(state) -> dstate/dt with same shape as state
        state: Tensor of shape (N, D)
        dt: time step
    Returns:
        Next state tensor of shape (N, D)
    """
    k1 = dynamics_fn(state)
    k2 = dynamics_fn(state + 0.5 * dt * k1)
    k3 = dynamics_fn(state + 0.5 * dt * k2)
    k4 = dynamics_fn(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def rollout(dynamics_fn, initial_state: torch.Tensor, t_final: float, dt: float, *, progress_desc: str | None = None) -> torch.Tensor:
    """Rollout trajectory using RK4 with optional tqdm progress bar.

    Args:
        dynamics_fn: function(state) -> dstate/dt
        initial_state: Tensor (1, 2) [q0, p0]
        t_final: final time
        dt: step size
        progress_desc: optional description for tqdm bar
    Returns:
        Trajectory tensor of shape (T, 2)
    """
    num_steps = int(math.ceil(t_final / dt))
    trajectory = [initial_state]
    state = initial_state

    iterator = range(num_steps)
    if progress_desc is not None:
        iterator = tqdm(iterator, desc=progress_desc, leave=False)

    for _ in iterator:
        state = rk4_step(dynamics_fn, state, dt)
        trajectory.append(state)
    return torch.cat(trajectory, dim=0)


# ============================
# HNN Model
# ============================

class HamiltonianNN(nn.Module):
    """Small MLP that outputs scalar Hamiltonian H(q,p)."""

    def __init__(self, hidden_sizes: Tuple[int, int] = (64, 64)) -> None:
        super().__init__()
        layers = []
        input_dim = 2
        last_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.Tanh())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, 1))  # scalar H
        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)  # shape (N, 1)


def hnn_time_derivative(model: HamiltonianNN, state: torch.Tensor) -> torch.Tensor:
    """Compute time derivative via Hamilton's equations using learned H.

    Args:
        model: HamiltonianNN
        state: Tensor (N, 2) with requires_grad=True
    Returns:
        Tensor (N, 2) of predicted time derivatives
    """
    state = state.requires_grad_(True)
    H = model(state)  # (N, 1)
    grad_H = torch.autograd.grad(H.sum(), state, create_graph=True)[0]  # (N, 2)
    dH_dq = grad_H[:, 0:1]
    dH_dp = grad_H[:, 1:2]
    dq_dt = dH_dp
    dp_dt = -dH_dq
    return torch.cat([dq_dt, dp_dt], dim=1)


# ============================
# Dataset
# ============================

def sample_states_and_derivatives(num_samples: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample (q,p) and compute true derivatives.

    Samples radii and angles to cover different energy levels.
    """
    # Sample radius r ~ Uniform(0.2, 2.0), angle theta ~ Uniform(0, 2pi)
    radii = torch.distributions.Uniform(0.2, 2.0).sample((num_samples,))
    angles = torch.distributions.Uniform(0.0, 2 * math.pi).sample((num_samples,))

    q = radii * torch.cos(angles)
    p = radii * torch.sin(angles)
    states = torch.stack([q, p], dim=1).to(device)
    derivatives = true_time_derivative(states)
    return states, derivatives


# ============================
# Training
# ============================

def train_hnn_on_data(states: torch.Tensor, derivatives: torch.Tensor, config: TrainingConfig) -> HamiltonianNN:
    model = HamiltonianNN().to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.MSELoss()

    num_samples = states.shape[0]
    num_batches = math.ceil(num_samples / config.batch_size)

    for epoch in range(1, config.max_epochs + 1):
        permutation = torch.randperm(num_samples, device=config.device)
        epoch_loss = 0.0
        for b in range(num_batches):
            batch_indices = permutation[b * config.batch_size : (b + 1) * config.batch_size]
            batch_states = states[batch_indices]
            batch_derivatives = derivatives[batch_indices]

            pred_derivatives = hnn_time_derivative(model, batch_states)
            loss = loss_fn(pred_derivatives, batch_derivatives)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        if epoch % config.print_every == 0 or epoch == 1:
            avg_loss = epoch_loss / num_batches
            print(f"[HNN ] Epoch {epoch:4d} | Loss: {avg_loss:.6f}")
    return model


class NeuralODERHS(nn.Module):
    """Feedforward NN that predicts time derivatives (dq/dt, dp/dt)."""

    def __init__(self, hidden_sizes: Tuple[int, int] = (64, 64)) -> None:
        super().__init__()
        layers = []
        input_dim = 2
        last_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.Tanh())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, 2))  # outputs (dq_dt, dp_dt)
        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


def neural_ode_time_derivative(model: NeuralODERHS, state: torch.Tensor) -> torch.Tensor:
    return model(state)


def train_neural_ode_on_data(states: torch.Tensor, derivatives: torch.Tensor, config: TrainingConfig) -> NeuralODERHS:
    model = NeuralODERHS().to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.MSELoss()

    num_samples = states.shape[0]
    num_batches = math.ceil(num_samples / config.batch_size)

    for epoch in range(1, config.max_epochs + 1):
        permutation = torch.randperm(num_samples, device=config.device)
        epoch_loss = 0.0
        for b in range(num_batches):
            batch_indices = permutation[b * config.batch_size : (b + 1) * config.batch_size]
            batch_states = states[batch_indices]
            batch_derivatives = derivatives[batch_indices]

            pred_derivatives = neural_ode_time_derivative(model, batch_states)
            loss = loss_fn(pred_derivatives, batch_derivatives)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        if epoch % config.print_every == 0 or epoch == 1:
            avg_loss = epoch_loss / num_batches
            print(f"[NODE] Epoch {epoch:4d} | Loss: {avg_loss:.6f}")
    return model


# ============================
# Plotting
# ============================

def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def plot_phase_space(trajectory_true: np.ndarray, trajectory_hnn: np.ndarray, out_path: str, trajectory_neuralode: np.ndarray | None = None) -> None:
    plt.figure(figsize=(6, 6))

    # Determine plotting bounds from trajectories with a margin
    stacks = [trajectory_true, trajectory_hnn]
    if trajectory_neuralode is not None:
        stacks.append(trajectory_neuralode)
    all_points = np.vstack(stacks)
    q_min, q_max = all_points[:, 0].min(), all_points[:, 0].max()
    p_min, p_max = all_points[:, 1].min(), all_points[:, 1].max()
    q_range = q_max - q_min
    p_range = p_max - p_min
    margin_q = 0.15 * (q_range if q_range > 0 else 1.0)
    margin_p = 0.15 * (p_range if p_range > 0 else 1.0)
    q_lo, q_hi = q_min - margin_q, q_max + margin_q
    p_lo, p_hi = p_min - margin_p, p_max + margin_p

    # Quiver vector field of true dynamics: dq/dt = p, dp/dt = -q
    grid_size = 21
    q_vals = np.linspace(q_lo, q_hi, grid_size)
    p_vals = np.linspace(p_lo, p_hi, grid_size)
    Q, P = np.meshgrid(q_vals, p_vals)
    U = P  # dq/dt
    V = -Q # dp/dt
    plt.quiver(Q, P, U, V, color="0.75", alpha=0.6, pivot="mid", angles="xy", scale_units="xy", scale=15.0, zorder=1)

    # True trajectory (circle)
    plt.plot(trajectory_true[:, 0], trajectory_true[:, 1], label="True", linewidth=2, alpha=0.9, zorder=3)
    # HNN rollout
    plt.plot(trajectory_hnn[:, 0], trajectory_hnn[:, 1], label="HNN", linewidth=2, alpha=0.9, zorder=4)
    # Neural ODE rollout
    if trajectory_neuralode is not None:
        plt.plot(trajectory_neuralode[:, 0], trajectory_neuralode[:, 1], label="NeuralODE", linewidth=2, alpha=0.9, zorder=4)

    plt.xlabel("q (position)")
    plt.ylabel("p (momentum)")
    plt.title("Mass-Spring Phase Space: True vs HNN vs NeuralODE")
    plt.xlim(q_lo, q_hi)
    plt.ylim(p_lo, p_hi)
    plt.axis("equal")
    plt.grid(True, linestyle=":", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def sample_initial_conditions(num: int, radius: float, device: str) -> torch.Tensor:
    """Return tensor of shape (num, 2) with evenly spaced angles on a circle of given radius."""
    angles = torch.linspace(0.0, 2 * math.pi, steps=num + 1, device=device)[:-1]
    q0 = radius * torch.cos(angles)
    p0 = radius * torch.sin(angles)
    return torch.stack([q0, p0], dim=1)


def add_colored_trajectory(ax, trajectory: np.ndarray, *, cmap: str = "winter", linewidth: float = 2.0, alpha: float = 0.95) -> None:
    """Draw a trajectory with a color gradient along time.

    Colors progress from start (t=0) to end (t=T) using the chosen colormap.
    """
    pts = trajectory[:, :2]
    if pts.shape[0] < 2:
        return
    segments = np.concatenate([pts[:-1, None, :], pts[1:, None, :]], axis=1)
    t = np.linspace(0.0, 1.0, pts.shape[0] - 1)  # color by time
    lc = LineCollection(segments, cmap=cmap, norm=Normalize(0.0, 1.0))
    lc.set_array(t)
    lc.set_linewidth(linewidth)
    lc.set_alpha(alpha)
    ax.add_collection(lc)


def plot_side_by_side(
    trajectories_neuralode: list[np.ndarray],
    trajectories_hnn: list[np.ndarray],
    out_path: str,
) -> None:
    """Create side-by-side plots: left NeuralODE, right HNN, both with quiver of true dynamics."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

    # Compute global bounds across all trajectories
    all_pts = []
    for traj in trajectories_neuralode:
        all_pts.append(traj)
    for traj in trajectories_hnn:
        all_pts.append(traj)
    all_points = np.vstack(all_pts)
    q_min, q_max = all_points[:, 0].min(), all_points[:, 0].max()
    p_min, p_max = all_points[:, 1].min(), all_points[:, 1].max()
    q_range = q_max - q_min
    p_range = p_max - p_min
    margin_q = 0.15 * (q_range if q_range > 0 else 1.0)
    margin_p = 0.15 * (p_range if p_range > 0 else 1.0)
    q_lo, q_hi = q_min - margin_q, q_max + margin_q
    p_lo, p_hi = p_min - margin_p, p_max + margin_p

    # Prepare quiver grid
    grid_size = 21
    q_vals = np.linspace(q_lo, q_hi, grid_size)
    p_vals = np.linspace(p_lo, p_hi, grid_size)
    Q, P = np.meshgrid(q_vals, p_vals)
    U = P  # dq/dt
    V = -Q # dp/dt

    # Left: NeuralODE
    ax = axes[0]
    ax.quiver(Q, P, U, V, color="0.8", alpha=0.6, pivot="mid", angles="xy", scale_units="xy", scale=15.0, zorder=1)
    for traj in trajectories_neuralode:
        add_colored_trajectory(ax, traj, cmap="winter", linewidth=2.0, alpha=0.95)
    ax.set_title("NeuralODE")
    ax.set_xlabel("q (position)")
    ax.set_ylabel("p (momentum)")
    ax.set_xlim(q_lo, q_hi)
    ax.set_ylim(p_lo, p_hi)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", alpha=0.4)

    # Right: HNN
    ax = axes[1]
    ax.quiver(Q, P, U, V, color="0.8", alpha=0.6, pivot="mid", angles="xy", scale_units="xy", scale=15.0, zorder=1)
    for traj in trajectories_hnn:
        add_colored_trajectory(ax, traj, cmap="winter", linewidth=2.0, alpha=0.95)
    ax.set_title("HNN")
    ax.set_xlabel("q (position)")
    ax.set_ylabel("p (momentum)")
    ax.set_xlim(q_lo, q_hi)
    ax.set_ylim(p_lo, p_hi)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", alpha=0.4)

    fig.suptitle("Phase Space: NeuralODE (left) vs HNN (right)")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def compute_mse_vs_time(truth: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute MSE over time between two trajectories of shape (T, 2)."""
    T = min(truth.shape[0], pred.shape[0])
    diff = truth[:T] - pred[:T]
    return np.mean(diff * diff, axis=1)


def plot_mse_over_time(mses_node: list[np.ndarray], mses_hnn: list[np.ndarray], dt: float, out_path: str) -> None:
    """Plot MSE vs time for NeuralODE and HNN, with mean and shaded std across trajectories."""
    # Align lengths
    T = min(min(m.shape[0] for m in mses_node), min(m.shape[0] for m in mses_hnn))
    mses_node_arr = np.stack([m[:T] for m in mses_node], axis=0)
    mses_hnn_arr = np.stack([m[:T] for m in mses_hnn], axis=0)

    t = np.arange(T) * dt

    mean_node = mses_node_arr.mean(axis=0)
    std_node = mses_node_arr.std(axis=0)
    mean_hnn = mses_hnn_arr.mean(axis=0)
    std_hnn = mses_hnn_arr.std(axis=0)

    plt.figure(figsize=(10, 5))
    plt.plot(t, mean_node, label="NeuralODE MSE", color="C0")
    plt.fill_between(t, mean_node - std_node, mean_node + std_node, color="C0", alpha=0.2)
    plt.plot(t, mean_hnn, label="HNN MSE", color="C1")
    plt.fill_between(t, mean_hnn - std_hnn, mean_hnn + std_hnn, color="C1", alpha=0.2)
    plt.yscale("log")
    plt.xlabel("time")
    plt.ylabel("MSE(q,p)")
    plt.title("Rollout Error vs Time")
    plt.grid(True, linestyle=":", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


# ============================
# Main
# ============================

def main() -> None:
    config = TrainingConfig()
    print(f"Using device: {config.device}")

    set_random_seed(config.random_seed)

    # Shared training data for fair comparison
    states, derivatives = sample_states_and_derivatives(config.num_train_samples, config.device)

    # Train HNN and Neural ODE on the same data
    hnn_model = train_hnn_on_data(states, derivatives, config)
    neuralode_model = train_neural_ode_on_data(states, derivatives, config)

    # Multiple initial conditions on a circle of radius 1.2
    initial_states = sample_initial_conditions(config.num_plot_trajectories, radius=1.2, device=config.device)

    trajectories_true: list[np.ndarray] = []
    trajectories_hnn: list[np.ndarray] = []
    trajectories_node: list[np.ndarray] = []

    for i in tqdm(range(initial_states.shape[0]), desc="Trajectories", leave=True):
        s0 = initial_states[i : i + 1]
        traj_true = rollout(true_time_derivative, s0, config.rollout_time, config.rollout_dt, progress_desc=f"True #{i+1}")
        traj_hnn = rollout(lambda s: hnn_time_derivative(hnn_model, s), s0, config.rollout_time, config.rollout_dt, progress_desc=f"HNN #{i+1}")
        traj_node = rollout(lambda s: neural_ode_time_derivative(neuralode_model, s), s0, config.rollout_time, config.rollout_dt, progress_desc=f"NeuralODE #{i+1}")
        trajectories_true.append(traj_true.detach().cpu().numpy())
        trajectories_hnn.append(traj_hnn.detach().cpu().numpy())
        trajectories_node.append(traj_node.detach().cpu().numpy())

    out_dir = os.path.join(os.getcwd(), "plots")
    ensure_dir(out_dir)

    # Side-by-side phase space
    out_path_ps = os.path.join(out_dir, "neuralode_vs_hnn_side_by_side.png")
    plot_side_by_side(trajectories_node, trajectories_hnn, out_path_ps)

    # MSE over time plot
    mses_node = [compute_mse_vs_time(tru, nod) for tru, nod in zip(trajectories_true, trajectories_node)]
    mses_hnn = [compute_mse_vs_time(tru, hnn) for tru, hnn in zip(trajectories_true, trajectories_hnn)]
    out_path_mse = os.path.join(out_dir, "mse_vs_time.png")
    plot_mse_over_time(mses_node, mses_hnn, config.rollout_dt, out_path_mse)

    print(f"Saved side-by-side comparison to: {out_path_ps}")
    print(f"Saved MSE vs time to: {out_path_mse}")


if __name__ == "__main__":
    main()
