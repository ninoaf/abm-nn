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
from typing import Tuple, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
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
        layers.append(nn.LeakyReLU())  # nonnegative, zero at zero
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
        layers.append(nn.LeakyReLU())  # nonnegative, zero at zero
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
        self.functional_coeff_F = nn.Linear(2, 1, bias=False)
        self.functional_coeff_G = nn.Linear(2, 1, bias=False)
        self.functional_coeff_H = nn.Linear(2, 1, bias=False)
        
        # Initialize functional coefficients with specific values
        # F = -1*acc_s + 0*rec  (dS = -infection_rate)
        with torch.no_grad():
            self.functional_coeff_F.weight.data = torch.tensor([[-1.0, 0.0]], dtype=torch.float32)

                
        for param in self.functional_coeff_F.parameters():
            param.requires_grad = False
        
        # # G = +1*acc_s - 1*rec  (dI = infection_rate - recovery_rate)  
        # with torch.no_grad():
        #     self.functional_coeff_G.weight.data = torch.tensor([[1.0, -1.0]], dtype=torch.float32)
        
        # # H = 0*acc_s + 1*rec   (dR = recovery_rate)
        # with torch.no_grad():
        #     self.functional_coeff_H.weight.data = torch.tensor([[0.0, 1.0]], dtype=torch.float32)

        # for param in self.functional_coeff_G.parameters():
        #     param.requires_grad = False
        # for param in self.functional_coeff_H.parameters():
        #     param.requires_grad = False
        
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
            dS_b = self.functional_coeff_F(torch.stack([acc_s, rec], dim=1)).squeeze(-1)
            dI_b = self.functional_coeff_G(torch.stack([acc_s, rec], dim=1)).squeeze(-1)
            dR_b = self.functional_coeff_H(torch.stack([acc_s, rec], dim=1)).squeeze(-1)
            dS_list.append(dS_b)
            dI_list.append(dI_b)
            dR_list.append(dR_b)

        dS = torch.stack(dS_list, dim=0)
        dI = torch.stack(dI_list, dim=0)
        dR = torch.stack(dR_list, dim=0)
        return torch.cat([dS, dI, dR], dim=1)

def functionals_constraint_evaluator(x: torch.Tensor) -> torch.Tensor:
    """Regularizer for conserved mass."""
    N = x.shape[1] // 3
    dS = x[:, :N]
    dI = x[:, N:2*N]
    dR = x[:, 2*N:3*N]
    return (dS + dI + dR).abs().mean()



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
        x= torch_rk4_step_rhsnet(model, x, dt)
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
                        reg_phi2: float, conservation: float, final_loss: float, lr: float, grad_norm: float) -> None:
    print(
        f"Epoch {epoch}/{epochs} - t_curr={t_curr:.1f} - loss_base: {base_loss:.6f} "
        f"- reg_axis_phi1: {reg_axis:.6f} - reg_phi2_origin: {reg_phi2:.6f} "
        f"- reg_conservation: {conservation:.6f} - loss_total: {final_loss:.6f} - lr: {lr:.2e} - grad_norm: {grad_norm:.3e}"
    )


def pretrain_functional_coefficients(
    model: GraphRHSNet,
    device: torch.device,
    num_samples: int = 1000,
    pretrain_epochs: int = 100,
    pretrain_lr: float = 1e-3,
) -> dict:
    """
    Pretrain functional coefficients F, G, H to satisfy conservation constraint.
    Only optimizes functional_coeff_* parameters while keeping phi1, phi2 frozen.
    
    Args:
        model: GraphRHSNet model to pretrain
        device: torch device
        num_samples: number of random state samples to generate
        pretrain_epochs: number of pretraining epochs
        pretrain_lr: learning rate for pretraining
    
    Returns:
        dict with pretraining history
    """
    print(f"\n{'='*60}")
    print("Stage 1: Pretraining Functional Coefficients")
    print(f"{'='*60}")
    print(f"Samples: {num_samples}, Epochs: {pretrain_epochs}, LR: {pretrain_lr}")
    
    # Freeze phi1 and phi2 parameters
    for param in model.phi1.parameters():
        param.requires_grad = False
    for param in model.phi2.parameters():
        param.requires_grad = False
    
    # Create optimizer with only functional coefficient parameters
    functional_params = list(model.functional_coeff_F.parameters()) + \
                       list(model.functional_coeff_G.parameters()) + \
                       list(model.functional_coeff_H.parameters())
    
    pretrain_optimizer = torch.optim.Adam(functional_params, lr=pretrain_lr)
    
    # Generate random dataset once
    N = model.num_nodes
    random_states = torch.rand(num_samples, 3 * N, device=device, dtype=torch.float32)
    
    history = {"pretrain_loss": []}
    best_loss = float('inf')
    
    model.train()
    for epoch in range(pretrain_epochs):
        pretrain_optimizer.zero_grad()
        
        # Forward pass through model with frozen phi1, phi2
        outputs = model(random_states)
        
        # Evaluate conservation constraint: dS + dI + dR should be ~0
        loss = functionals_constraint_evaluator(outputs)
        
        loss.backward()
        pretrain_optimizer.step()
        
        loss_val = float(loss.item())
        history["pretrain_loss"].append(loss_val)
        
        if loss_val < best_loss:
            best_loss = loss_val
        
        # Print progress
        if (epoch + 1) % max(1, pretrain_epochs // 10) == 0 or epoch == 0:
            print(f"  Pretrain Epoch {epoch+1:4d}/{pretrain_epochs} - Loss: {loss_val:.6e} - Best: {best_loss:.6e}")
    
    print(f"Pretraining Complete - Final Loss: {loss_val:.6e}")
    print(f"{'='*60}\n")
    
    # Unfreeze phi1 and phi2 for main training
    for param in model.phi1.parameters():
        param.requires_grad = True
    for param in model.phi2.parameters():
        param.requires_grad = True
    
    return history


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
    lr_coeff_F: float = 1e-3,
    lr_coeff_G: float = 1e-3,
    lr_coeff_H: float = 1e-3,
    hidden_dim: int = 32,
    num_hidden: int = 1,
    gamma: float = 0.2,
    device: str | None = None,
    horizon_increment: float = 5.0,
    horizon_step_epochs: int = 10,
    horizon_base: float = 5.0,
    lambda_phi1_axis: float = 1e-3,
    lambda_phi2_zero: float = 1e-5,
    lambda_conservation: float = 1e-3,
    clip_grad: float | None = 1.0,
    lr_decay_gamma: float = 1.0,
    t_train_max: float | None = None,
    pretrained_state_dict: dict | None = None,
    reg_warmup_epochs: int = 0,
    pretrain_coefficients: bool = False,
    pretrain_epochs: int = 100,
    pretrain_lr: float = 1e-3,
    pretrain_samples: int = 1000,
    pretrain_interval: int = 100,
    use_cosine_scheduler: bool = False,
    cosine_T_0: int = 50,
    cosine_T_mult: int = 2,
    cosine_eta_min: float = 1e-6,
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
    
    # Stage 1: Initial pretrain functional coefficients if requested
    pretrain_history = {}
    if pretrain_coefficients:
        pretrain_history = pretrain_functional_coefficients(
            model=model,
            device=device_t,
            num_samples=pretrain_samples,
            pretrain_epochs=pretrain_epochs,
            pretrain_lr=pretrain_lr,
        )
        print(f"\n{'='*60}")
        print("Stage 2: Training All Parameters")
        print(f"{'='*60}\n")
    
    # During main training, use separate optimizers for different parameter groups
    # Neural networks (phi1, phi2)
    nn_params = list(model.phi1.parameters()) + list(model.phi2.parameters())
    nn_optimizer = torch.optim.AdamW(nn_params, lr=lr, weight_decay=1e-5)
    
    # Functional coefficients with separate learning rates
    optimizers = {"nn": nn_optimizer}
    
    # Add functional coefficient optimizers only if they are trainable
    if model.functional_coeff_F.weight.requires_grad:
        optimizers["coeff_F"] = torch.optim.AdamW(model.functional_coeff_F.parameters(), lr=lr_coeff_F, weight_decay=1e-5)
    if model.functional_coeff_G.weight.requires_grad:
        optimizers["coeff_G"] = torch.optim.AdamW(model.functional_coeff_G.parameters(), lr=lr_coeff_G, weight_decay=1e-5)
    if model.functional_coeff_H.weight.requires_grad:
        optimizers["coeff_H"] = torch.optim.AdamW(model.functional_coeff_H.parameters(), lr=lr_coeff_H, weight_decay=1e-5)
    
    # Create schedulers for each optimizer
    schedulers = {}
    if lr_decay_gamma is not None and lr_decay_gamma < 1.0:
        for name, opt in optimizers.items():
            schedulers[name] = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=float(lr_decay_gamma))
    elif use_cosine_scheduler:
        # CosineAnnealingWarmRestarts scheduler for each optimizer
        for name, opt in optimizers.items():
            schedulers[name] = CosineAnnealingWarmRestarts(
                opt, 
                T_0=cosine_T_0, 
                T_mult=cosine_T_mult, 
                eta_min=cosine_eta_min
            )

    # Convert targets to torch
    s_target = torch.from_numpy(target_S_sum.astype(np.float32)).to(device_t)
    i_target = torch.from_numpy(target_I_sum.astype(np.float32)).to(device_t)
    r_target = torch.from_numpy(target_R_sum.astype(np.float32)).to(device_t)

    history = {
        "loss": [], "loss_base": [], "reg_axis_phi1": [], "reg_phi2_origin": [], 
        "reg_conservation": [], "loss_total": [],
        "coeff_F": [], "coeff_G": [], "coeff_H": [],
        "learning_rate": []
    }
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
            
            # Conservation constraint on initial state
            x_batch0 = torch.cat([_S0, _I0, _R0], dim=1)
            derivatives0 = model(x_batch0)
            conservation0 = functionals_constraint_evaluator(derivatives0)
            
            loss_total0 = base_loss0
            if int(reg_warmup_epochs) <= 0:
                if lambda_phi1_axis > 0.0:
                    loss_total0 = loss_total0 + lambda_phi1_axis * reg_axis0
                if lambda_phi2_zero > 0.0:
                    loss_total0 = loss_total0 + lambda_phi2_zero * reg_phi2_origin0
                if lambda_conservation > 0.0:
                    loss_total0 = loss_total0 + lambda_conservation * conservation0

        history["loss"].append(float(loss_total0.detach().cpu().item()))
        history["loss_base"].append(float(base_loss0.detach().cpu().item()))
        history["reg_axis_phi1"].append(float(reg_axis0.detach().cpu().item()))
        history["reg_phi2_origin"].append(float(reg_phi2_origin0.detach().cpu().item()))
        history["reg_conservation"].append(float(conservation0.detach().cpu().item()))
        history["loss_total"].append(float(loss_total0.detach().cpu().item()))
        
        # Track functional coefficients
        with torch.no_grad():
            coeff_F = model.functional_coeff_F.weight.detach().cpu().numpy().flatten().tolist()
            coeff_G = model.functional_coeff_G.weight.detach().cpu().numpy().flatten().tolist()
            coeff_H = model.functional_coeff_H.weight.detach().cpu().numpy().flatten().tolist()
        history["coeff_F"].append(coeff_F)
        history["coeff_G"].append(coeff_G)
        history["coeff_H"].append(coeff_H)
        
        # Track learning rates for all optimizers
        lr_dict = {}
        for name, opt in optimizers.items():
            lr_dict[name] = opt.param_groups[0]["lr"]
        history["learning_rate"].append(lr_dict)

        print_init_metrics(t_curr0, base_loss0, reg_axis0, reg_phi2_origin0, loss_total0, phi1_axis_out0, phi2_zero_out0)
    except Exception:
        pass

    for epoch in range(epochs):
        # Curriculum on simulation horizon: first 10 epochs t=5, then +5 every 10 epochs
        stage = epoch // max(1, horizon_step_epochs)
        t_curr = min(t_max_total, float(horizon_base) + float(horizon_increment) * stage)
        # Corresponding target length
        L_curr = int(np.ceil(t_curr / dt)) + 1
        # Zero gradients for all optimizers
        for opt in optimizers.values():
            opt.zero_grad()
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
        
        # Conservation constraint: evaluate on predicted derivatives
        # Reconstruct state from histories and compute derivatives
        x_batch = torch.cat([S_hist, I_hist, R_hist], dim=1)  # (T+1, 3N)
        derivatives = model(x_batch)  # (T+1, 3N) with [dS, dI, dR]
        conservation_loss = functionals_constraint_evaluator(derivatives)
        
        # Compose final loss with warmup
        loss = base_loss
        if epoch >= int(reg_warmup_epochs):
            if lambda_phi1_axis > 0.0:
                loss = loss + lambda_phi1_axis * reg_axis_val
            if lambda_phi2_zero > 0.0:
                loss = loss + lambda_phi2_zero * reg_phi2_origin_val
            if lambda_conservation > 0.0:
                loss = loss + lambda_conservation * conservation_loss
        loss.backward()
        # Compute grad L2 norm before clipping
        grad_norm = compute_grad_norm(model)
        if clip_grad and float(clip_grad) > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(clip_grad))
        # Step all optimizers
        for opt in optimizers.values():
            opt.step()
        
        # Step all schedulers
        for sched in schedulers.values():
            sched.step()

        base_v = float(base_loss.detach().cpu().item())
        reg_axis_v = float(reg_axis_val.detach().cpu().item())
        reg_phi2_origin_v = float(reg_phi2_origin_val.detach().cpu().item())
        conservation_v = float(conservation_loss.detach().cpu().item())
        final_v = float(loss.detach().cpu().item())
        history["loss"].append(final_v)
        history["loss_base"].append(base_v)
        history["reg_axis_phi1"].append(reg_axis_v)
        history["reg_phi2_origin"].append(reg_phi2_origin_v)
        history["reg_conservation"].append(conservation_v)
        history["loss_total"].append(final_v)
        
        # Track functional coefficients
        with torch.no_grad():
            coeff_F = model.functional_coeff_F.weight.detach().cpu().numpy().flatten().tolist()
            coeff_G = model.functional_coeff_G.weight.detach().cpu().numpy().flatten().tolist()
            coeff_H = model.functional_coeff_H.weight.detach().cpu().numpy().flatten().tolist()
        history["coeff_F"].append(coeff_F)
        history["coeff_G"].append(coeff_G)
        history["coeff_H"].append(coeff_H)
        
        # Track learning rates for all optimizers
        lr_dict = {}
        for name, opt in optimizers.items():
            lr_dict[name] = opt.param_groups[0]["lr"]
        history["learning_rate"].append(lr_dict)
        
        if (epoch + 1) == 100 and not printed_100:
            elapsed_100 = time.time() - train_start_ts
            print(f"Time for first 100 epochs: {elapsed_100:.2f}s (~{elapsed_100/100.0:.3f}s/epoch)")
            printed_100 = True
        if (epoch + 1) % max(1, epochs // 50) == 0 or epoch == 1:
            # Use neural network learning rate for display
            curr_lr = optimizers["nn"].param_groups[0]["lr"]
            print_epoch_metrics(epoch + 1, epochs, t_curr, base_v, reg_axis_v, reg_phi2_origin_v, conservation_v, final_v, curr_lr, grad_norm)
        
        # Re-run pretraining every pretrain_interval epochs
        if pretrain_coefficients and (epoch + 1) % pretrain_interval == 0 and (epoch + 1) < epochs:
            print(f"\n{'='*60}")
            print(f"Re-running Pretraining at Epoch {epoch + 1}")
            print(f"{'='*60}")
            
            # Run pretraining to refine functional coefficients
            pretrain_hist_interval = pretrain_functional_coefficients(
                model=model,
                device=device_t,
                num_samples=pretrain_samples,
                pretrain_epochs=pretrain_epochs,
                pretrain_lr=pretrain_lr,
            )
            
            # Append interval pretraining losses to history
            if "pretrain_loss" not in history:
                history["pretrain_loss"] = []
            history["pretrain_loss"].extend(pretrain_hist_interval["pretrain_loss"])
            
            print(f"{'='*60}")
            print(f"Resuming Main Training")
            print(f"{'='*60}\n")
            
            # Recreate optimizers to ensure all parameters are trainable
            nn_params = list(model.phi1.parameters()) + list(model.phi2.parameters())
            optimizers["nn"] = torch.optim.AdamW(nn_params, lr=optimizers["nn"].param_groups[0]["lr"], weight_decay=1e-5)
            
            # Recreate functional coefficient optimizers if they are trainable
            if model.functional_coeff_F.weight.requires_grad:
                optimizers["coeff_F"] = torch.optim.AdamW(model.functional_coeff_F.parameters(), lr=optimizers["coeff_F"].param_groups[0]["lr"], weight_decay=1e-5)
            if model.functional_coeff_G.weight.requires_grad:
                optimizers["coeff_G"] = torch.optim.AdamW(model.functional_coeff_G.parameters(), lr=optimizers["coeff_G"].param_groups[0]["lr"], weight_decay=1e-5)
            if model.functional_coeff_H.weight.requires_grad:
                optimizers["coeff_H"] = torch.optim.AdamW(model.functional_coeff_H.parameters(), lr=optimizers["coeff_H"].param_groups[0]["lr"], weight_decay=1e-5)
            
            # Recreate schedulers
            schedulers = {}
            if lr_decay_gamma is not None and lr_decay_gamma < 1.0:
                for name, opt in optimizers.items():
                    schedulers[name] = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=float(lr_decay_gamma))
            elif use_cosine_scheduler:
                for name, opt in optimizers.items():
                    schedulers[name] = CosineAnnealingWarmRestarts(
                        opt, 
                        T_0=cosine_T_0, 
                        T_mult=cosine_T_mult, 
                        eta_min=cosine_eta_min
                    )

    total_elapsed = time.time() - train_start_ts
    print(f"Total training time: {total_elapsed:.2f}s (~{total_elapsed/max(1, epochs):.3f}s/epoch)")
    
    # Merge pretraining and main training histories
    if pretrain_history:
        history.update(pretrain_history)
    
    return model, history


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Macro learning of SIR RHS from aggregate curves")
    parser.add_argument("--nodes", type=int, default=100)
    parser.add_argument("--p", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument("--gamma", type=float, default=0.2)
    parser.add_argument("--t_max", type=float, default=30.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for neural networks (phi1, phi2)")
    parser.add_argument("--lr_coeff_F", type=float, default=0.1, help="Learning rate for functional coefficient F")
    parser.add_argument("--lr_coeff_G", type=float, default=0.1, help="Learning rate for functional coefficient G")
    parser.add_argument("--lr_coeff_H", type=float, default=0.1, help="Learning rate for functional coefficient H")
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--num_hidden", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--horizon_increment", type=float, default=1.0, help="Time horizon increment every 10 epochs")
    parser.add_argument("--horizon_step_epochs", type=int, default=10, help="Number of epochs between horizon increments")
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
    parser.add_argument("--lambda_conservation", type=float, default=10.0, help="Weight for conservation constraint |dS+dI+dR|")
    parser.add_argument("--reg_warmup_epochs", type=int, default=100, help="Epochs to delay adding regularizers")
    parser.add_argument("--pretrain_coefficients", action="store_true", help="Pretrain functional coefficients before main training")
    parser.add_argument("--pretrain_epochs", type=int, default=100, help="Number of pretraining epochs for coefficients")
    parser.add_argument("--pretrain_lr", type=float, default=1e-3, help="Learning rate for coefficient pretraining")
    parser.add_argument("--pretrain_samples", type=int, default=1000, help="Number of random samples for pretraining")
    parser.add_argument("--pretrain_interval", type=int, default=100, help="Re-run pretraining every N epochs during main training")
    parser.add_argument("--use_cosine_scheduler", default=False,action="store_true", help="Use CosineAnnealingWarmRestarts scheduler instead of exponential decay")
    parser.add_argument("--cosine_T_0", type=int, default=50, help="Number of iterations for the first restart in cosine scheduler")
    parser.add_argument("--cosine_T_mult", type=int, default=2, help="A factor increases T_i after a restart in cosine scheduler")
    parser.add_argument("--cosine_eta_min", type=float, default=1e-6, help="Minimum learning rate in cosine scheduler")
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


def save_cli_commands(exp_dir: str, args: argparse.Namespace) -> None:
    """Save CLI commands and arguments to experiment folder."""
    # Save the command line arguments
    cmd_file = os.path.join(exp_dir, "command.txt")
    with open(cmd_file, "w") as f:
        f.write("Command line arguments:\n")
        f.write("=" * 50 + "\n")
        for arg, value in vars(args).items():
            f.write(f"--{arg}: {value}\n")
        f.write("\n" + "=" * 50 + "\n")
        f.write("Full command:\n")
        f.write("python epidemic_demo_macro_functionals.py")
        for arg, value in vars(args).items():
            if isinstance(value, bool):
                if value:
                    f.write(f" --{arg}")
            else:
                f.write(f" --{arg} {value}")
        f.write("\n")
    
    # Copy the script to the experiment folder
    script_src = os.path.join(os.path.dirname(__file__), "epidemic_demo_macro_functionals.py")
    script_dst = os.path.join(exp_dir, "epidemic_demo_macro_functionals.py")
    shutil.copy2(script_src, script_dst)


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


def save_experiment_artifacts(exp_dir: str, history: Dict[str, Any], adjacency: np.ndarray, args, ts: str, out_path: str) -> None:
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
        with open(os.path.join(exp_dir, "config.json"), "w") as f:
            json.dump({"args": vars(args), "timestamp": ts, "plot_path": out_path}, f, default=str)
    except Exception as e:
        print(f"Warning: failed to save config.json: {e}")

    try:
        shutil.copy2(__file__, os.path.join(exp_dir, os.path.basename(__file__)))
    except Exception as e:
        print(f"Warning: failed to copy script to experiment folder: {e}")


def build_and_save_plot(args, G, adjacency, times, S_hist, I_hist, R_hist, model, history, exp_dir: str) -> str:
    """Construct a 4x2 figure and save it. Returns the file path."""
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

    fig = plt.figure(figsize=(14, 20))
    gs = fig.add_gridspec(5, 2, wspace=0.28, hspace=0.36)

    # Row 1, Col 1: training losses and regularizers
    ax_loss = fig.add_subplot(gs[0, 0])
    # Plot pretraining loss if available
    if "pretrain_loss" in history and len(history["pretrain_loss"]) > 0:
        pretrain_epochs = np.arange(len(history["pretrain_loss"]))
        ax_loss.plot(pretrain_epochs, history["pretrain_loss"], 
                    label="pretrain_loss", linestyle="--", color="#17becf", linewidth=2)
    
    # Plot main training losses
    epochs_axis = np.arange(0, len(history.get("loss_base", [])))
    if len(epochs_axis) > 0:
        # Offset by pretraining epochs if applicable
        offset = len(history.get("pretrain_loss", []))
        epochs_axis_shifted = epochs_axis + offset
        
        ax_loss.plot(epochs_axis_shifted, history.get("loss_base", []), label="loss_base", color="#1f77b4")
        ax_loss.plot(epochs_axis_shifted, history.get("reg_axis_phi1", []), label="reg_axis_phi1", linestyle=":", color="#ff7f0e")
        ax_loss.plot(epochs_axis_shifted, history.get("reg_phi2_origin", []), label="reg_phi2_origin", linestyle="-.", color="#d62728")
        ax_loss.plot(epochs_axis_shifted, history.get("reg_conservation", []), label="reg_conservation", linestyle="-.", color="#2ca02c")
        ax_loss.plot(epochs_axis_shifted, history.get("loss_total", []), label="loss_total", linestyle="-", color="#9467bd", linewidth=2)
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss / Reg terms")
        title = "Training losses (with pretraining)" if offset > 0 else "Training losses and regularizers"
        ax_loss.set_title(title)
        ax_loss.legend(ncol=1, fontsize=6, loc="upper right")

    # Row 1, Col 2: trajectories (true vs NN dashed) with inset of (S,I)
    ax_traj = fig.add_subplot(gs[0, 1])
    ax_traj.plot(times, S_sum_true, label="s(t)", color="#1f77b4")
    ax_traj.plot(times, I_sum_true, label="i(t)", color="#ff7f0e")
    ax_traj.plot(times, R_sum_true, label="r(t)", color="#2ca02c")
    # Use distinct colors for dashed NN curves to distinguish overlaps
    ax_traj.plot(t_nn, S_sum_nn, label="ŝ(t)", linestyle="--", color="#6baed6")
    ax_traj.plot(t_nn, I_sum_nn, label="î(t)", linestyle="--", color="#ffbb78")
    ax_traj.plot(t_nn, R_sum_nn, label="r̂(t)", linestyle="--", color="#98df8a")
    ax_traj.set_xlabel("Time")
    ax_traj.set_ylabel("Sum across nodes")
    ax_traj.set_title("Grund-truth vs ABM-NN trajectory")
    ax_traj.legend(loc="lower right", fontsize=8)

    try:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        ax_in = inset_axes(ax_traj, width="28%", height="28%", loc="upper right", borderpad=1.2)
        S_samples = S_hist.reshape(-1)
        I_samples = I_hist.reshape(-1)
        nsamp = min(10000, S_samples.size)
        if S_samples.size > nsamp:
            rng_vis = np.random.default_rng(args.seed)
            idx = rng_vis.choice(S_samples.size, size=nsamp, replace=False)
            S_plot = S_samples[idx]
            I_plot = I_samples[idx]
        else:
            S_plot = S_samples
            I_plot = I_samples
        ax_in.scatter(S_plot, I_plot, s=1.0, alpha=0.3, color="#1f77b4")
        ax_in.set_xlim(0.0, 1.0)
        ax_in.set_ylim(0.0, 1.0)
        ax_in.set_xticks([0.0, 1.0])
        ax_in.set_yticks([0.0, 1.0])
        ax_in.set_xlabel("S", fontsize=7)
        ax_in.set_ylabel("I", fontsize=7)
        ax_in.tick_params(labelsize=6)
    except Exception:
        pass

    # Row 2, Col 1: degree distribution with inset graph visualization
    ax_deg = fig.add_subplot(gs[1, 0])
    try:
        import networkx as nx  # local import for plotting
        if G.number_of_nodes() > 0:
            largest_cc_nodes = max(nx.connected_components(G), key=len)
            G_vis = G.subgraph(largest_cc_nodes).copy()
        else:
            G_vis = G
        degrees = [d for _, d in G_vis.degree()]
        bins = max(5, min(25, int(np.sqrt(max(1, len(degrees))))))
        ax_deg.hist(degrees, bins=bins, color="#888888", edgecolor="black")
        ax_deg.set_xlabel("Degree")
        ax_deg.set_ylabel("Count")
        n_nodes_total = int(getattr(adjacency, "shape", (len(G),)) [0]) if hasattr(adjacency, "shape") else G.number_of_nodes()
        ax_deg.set_title(f"Degree distribution (GCC), n={n_nodes_total}")
        ax_deg.set_ylim(0, 25)

        # Inset with graph visualization
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        ax_gin = inset_axes(ax_deg, width="40%", height="40%", loc="upper right", borderpad=1.0)
        pos = nx.spring_layout(G_vis, seed=getattr(args, "seed", 7))
        nx.draw_networkx(G_vis, pos=pos, ax=ax_gin, node_size=10, width=0.3, with_labels=False, edge_color="#aaaaaa", node_color="#1f77b4")
        ax_gin.set_axis_off()
    except Exception:
        # Fallback: only show histogram if graph plotting fails
        ax_deg.set_title("Degree distribution")

    # Row 2, Col 2: φ1(S=0.5, I) slice
    ax_phi1 = fig.add_subplot(gs[1, 1])
    i_line = np.linspace(0.0, 1.0, 200, dtype=np.float32)
    s_fixed = np.full_like(i_line, 0.5, dtype=np.float32)
    X_slice = np.stack([s_fixed, i_line], axis=1)
    with torch.no_grad():
        yhat_phi1 = model.phi1(torch.from_numpy(X_slice)).cpu().numpy().reshape(-1)
    ytrue_phi1 = args.beta * 0.5 * i_line
    ax_phi1.plot(i_line, ytrue_phi1, label="β·0.5·I", color="#1f77b4")
    ax_phi1.plot(i_line, yhat_phi1, label="φ1(0.5, I)", linestyle=":", color="#ff7f0e")
    ax_phi1.set_xlabel("I")
    ax_phi1.set_ylabel("value")
    ax_phi1.set_title("phi_1(S=0.5, I)")
    ax_phi1.legend()

    # Row 3, Col 1: Functional coefficients F
    ax_coeff_F = fig.add_subplot(gs[2, 0])
    if "coeff_F" in history and len(history["coeff_F"]) > 0:
        coeff_F_array = np.array(history["coeff_F"])  # Shape: (epochs, 2)
        epochs_coeff = np.arange(len(coeff_F_array))
        ax_coeff_F.plot(epochs_coeff, coeff_F_array[:, 0], label="F_coeff[0]", color="#1f77b4", linewidth=1.5)
        ax_coeff_F.plot(epochs_coeff, coeff_F_array[:, 1], label="F_coeff[1]", color="#ff7f0e", linewidth=1.5)
        ax_coeff_F.set_xlabel("Epoch")
        ax_coeff_F.set_ylabel("Coefficient Value")
        ax_coeff_F.set_title("Functional F Coefficients")
        ax_coeff_F.legend(loc="best", fontsize=8)
        ax_coeff_F.grid(True, alpha=0.3)
    
    # Row 3, Col 2: Functional coefficients G
    ax_coeff_G = fig.add_subplot(gs[2, 1])
    if "coeff_G" in history and len(history["coeff_G"]) > 0:
        coeff_G_array = np.array(history["coeff_G"])
        epochs_coeff = np.arange(len(coeff_G_array))
        ax_coeff_G.plot(epochs_coeff, coeff_G_array[:, 0], label="G_coeff[0]", color="#1f77b4", linewidth=1.5)
        ax_coeff_G.plot(epochs_coeff, coeff_G_array[:, 1], label="G_coeff[1]", color="#ff7f0e", linewidth=1.5)
        ax_coeff_G.set_xlabel("Epoch")
        ax_coeff_G.set_ylabel("Coefficient Value")
        ax_coeff_G.set_title("Functional G Coefficients")
        ax_coeff_G.legend(loc="best", fontsize=8)
        ax_coeff_G.grid(True, alpha=0.3)
    
    # Row 4, Col 1: Functional coefficients H
    ax_coeff_H = fig.add_subplot(gs[3, 0])
    if "coeff_H" in history and len(history["coeff_H"]) > 0:
        coeff_H_array = np.array(history["coeff_H"])
        epochs_coeff = np.arange(len(coeff_H_array))
        ax_coeff_H.plot(epochs_coeff, coeff_H_array[:, 0], label="H_coeff[0]", color="#1f77b4", linewidth=1.5)
        ax_coeff_H.plot(epochs_coeff, coeff_H_array[:, 1], label="H_coeff[1]", color="#ff7f0e", linewidth=1.5)
        ax_coeff_H.set_xlabel("Epoch")
        ax_coeff_H.set_ylabel("Coefficient Value")
        ax_coeff_H.set_title("Functional H Coefficients")
        ax_coeff_H.legend(loc="best", fontsize=8)
        ax_coeff_H.grid(True, alpha=0.3)
    
    # Row 4, Col 2: Conservation constraint loss over epochs
    ax_conservation = fig.add_subplot(gs[3, 1])
    if "reg_conservation" in history and len(history["reg_conservation"]) > 0:
        conservation_vals = history["reg_conservation"]
        epochs_conservation = np.arange(len(conservation_vals))
        ax_conservation.plot(epochs_conservation, conservation_vals, color="#2ca02c", linewidth=2)
        ax_conservation.set_xlabel("Epoch")
        ax_conservation.set_ylabel("Conservation Loss")
        ax_conservation.set_title("Conservation Constraint |dS+dI+dR|")
        ax_conservation.set_yscale('log')  # Log scale for better visibility
        ax_conservation.grid(True, alpha=0.3)

    # Row 5, Col 1: Learning Rates
    ax_lr = fig.add_subplot(gs[4, 0])
    if "learning_rate" in history and len(history["learning_rate"]) > 0:
        lr_data = history["learning_rate"]
        epochs_lr = np.arange(len(lr_data))
        
        # Plot each optimizer's learning rate
        colors = ["#d62728", "#2ca02c", "#1f77b4", "#ff7f0e"]
        for i, (name, lr_val) in enumerate(lr_data[0].items()):
            lr_values = [lr_dict[name] for lr_dict in lr_data]
            ax_lr.plot(epochs_lr, lr_values, color=colors[i % len(colors)], linewidth=2, label=f"LR {name}")
        
        ax_lr.set_xlabel("Epoch")
        ax_lr.set_ylabel("Learning Rate")
        ax_lr.set_title("Learning Rate Schedules")
        ax_lr.set_yscale('log')  # Log scale for better visibility
        ax_lr.grid(True, alpha=0.3)
        ax_lr.legend(loc="best", fontsize=8)

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
    
    # Save CLI commands and script to experiment folder
    save_cli_commands(exp_dir, args)

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
            lr_coeff_F=args.lr_coeff_F,
            lr_coeff_G=args.lr_coeff_G,
            lr_coeff_H=args.lr_coeff_H,
            hidden_dim=args.hidden_dim,
            num_hidden=args.num_hidden,
            gamma=args.gamma,
            horizon_increment=args.horizon_increment,
            horizon_step_epochs=args.horizon_step_epochs,
            horizon_base=args.horizon_base,
            lambda_phi1_axis=args.lambda_phi1_axis,
            lambda_phi2_zero=args.lambda_phi2_zero,
            lambda_conservation=args.lambda_conservation,
            clip_grad=args.clip_grad,
            lr_decay_gamma=args.lr_decay_gamma,
            t_train_max=args.t_train_max,
            pretrained_state_dict=pretrained_sd,
            reg_warmup_epochs=args.reg_warmup_epochs,
            pretrain_coefficients=args.pretrain_coefficients,
            pretrain_epochs=args.pretrain_epochs,
            pretrain_lr=args.pretrain_lr,
            pretrain_samples=args.pretrain_samples,
            pretrain_interval=args.pretrain_interval,
            use_cosine_scheduler=args.use_cosine_scheduler,
            cosine_T_0=args.cosine_T_0,
            cosine_T_mult=args.cosine_T_mult,
            cosine_eta_min=args.cosine_eta_min,
        )
        # Save if requested
        save_model_path = args.save_model if args.save_model else os.path.join(exp_dir, "macro_rhs.ckpt")
        save_model_ckpt(model, adjacency, args.gamma, save_model_path)

    # Build and save figure
    out_path = build_and_save_plot(args, G, adjacency, times, S_hist, I_hist, R_hist, model, history, exp_dir)
    # Save metadata
    save_experiment_artifacts(exp_dir, history, adjacency, args, ts, out_path)


if __name__ == "__main__":
    main()


