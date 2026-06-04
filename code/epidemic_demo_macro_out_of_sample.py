"""
Out-of-sample evaluation for macro-learned SIR RHS.

Loads a saved macro model (phi1, phi2), instantiates it on a new ER graph of
potentially different size, and compares NN simulation against the true ODE
on the new graph. Plots s(t), i(t), r(t) for both.
"""

from __future__ import annotations

import argparse
import os
import datetime
from typing import Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

from epidemic_demo_micro import (
    generate_erdos_renyi_adjacency_networkx,
    simulate_sir_on_graph,
)


from epidemic_demo_macro import GraphRHSNet, simulate_with_model_torch


def compute_layout_no_overlap(G, seed: int, base_k_scale: float = 1.5):
    """Spring layout with a light post-process to reduce node overlaps.

    - Uses Fruchterman–Reingold (spring) with k scaled by 1/sqrt(n)
    - Applies a small repulsive nudge for pairs that are too close
    """
    import networkx as nx  # local import

    n = max(1, G.number_of_nodes())
    k = float(base_k_scale) / np.sqrt(float(n))
    pos = nx.spring_layout(G, seed=seed, k=k, iterations=100)

    # Convert to array for adjustments
    nodes = list(G.nodes())
    if len(nodes) <= 1:
        return pos
    coords = np.array([pos[n] for n in nodes], dtype=np.float64)

    min_dist = 0.04  # target minimum separation in normalized coords
    step = 0.002
    for _ in range(30):  # a few smoothing passes
        moved_any = False
        for i in range(len(nodes)):
            delta = coords[i] - coords
            dist = np.linalg.norm(delta, axis=1) + 1e-12
            too_close = (dist > 0) & (dist < min_dist)
            if not np.any(too_close):
                continue
            # Repulsion away from close neighbors
            rep_dirs = delta[too_close] / dist[too_close, None]
            coords[i] += step * rep_dirs.sum(axis=0)
            moved_any = True
        if not moved_any:
            break

    # Normalize to [0,1]^2 for consistent rendering
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    coords = (coords - lo) / span
    return {nodes[i]: coords[i] for i in range(len(nodes))}


def mask_adjacency_per_node(adjacency: np.ndarray, drop_fraction: float, seed: int | None = None) -> np.ndarray:
    """Disable a fraction of outgoing links per node by zeroing A[i, j] entries.

    - drop_fraction in [0,1]: fraction of links to drop per node (e.g., 0.8 drops 80%).
    - Operates row-wise; resulting adjacency may be asymmetric, which is acceptable for the simulator.
    """
    rng = np.random.default_rng(seed)
    A = np.array(adjacency, dtype=float, copy=True)
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("adjacency must be square")
    drop_fraction = float(np.clip(drop_fraction, 0.0, 1.0))
    if drop_fraction >= 1.0:
        # Disable all links
        A[:, :] = 0.0
        np.fill_diagonal(A, 0.0)
        return A
    for i in range(n):
        nbrs = np.where(A[i] > 0.0)[0]
        deg = int(nbrs.size)
        if deg <= 0:
            continue
        keep_count = int(np.round((1.0 - drop_fraction) * deg))
        keep_count = int(np.clip(keep_count, 0, deg))
        if keep_count == deg:
            continue
        if keep_count == 0:
            # drop all
            A[i, nbrs] = 0.0
        else:
            keep_idx = rng.choice(nbrs, size=keep_count, replace=False)
            drop_idx = np.setdiff1d(nbrs, keep_idx, assume_unique=False)
            if drop_idx.size > 0:
                A[i, drop_idx] = 0.0
    # Ensure no self-loops
    np.fill_diagonal(A, 0.0)
    return A


def create_initial_conditions(G, nodes: int, seed: int, k_infected: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Out-of-sample macro NN evaluation on two ER graphs")
    p.add_argument("--load_model", type=str, required=True, help="Path to saved macro model checkpoint")
    p.add_argument("--nodes1", type=int, default=150, help="Nodes in first ER graph")
    p.add_argument("--p1", type=float, default=0.05, help="Edge prob in first ER graph")
    p.add_argument("--nodes2", type=int, default=250, help="Nodes in second ER graph")
    p.add_argument("--p2", type=float, default=0.05, help="Edge prob in second ER graph")
    p.add_argument("--beta", type=float, default=0.4)
    p.add_argument("--gamma", type=float, default=0.2)
    p.add_argument("--t_max", type=float, default=30.0)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--k_infected", type=int, default=1)
    p.add_argument("--hidden_dim", type=int, default=32)
    p.add_argument("--num_hidden", type=int, default=3)
    p.add_argument("--device", type=str, default=None, help="cpu|cuda|mps (auto if None)")
    p.add_argument("--out", type=str, default=None, help="Output plot path; defaults to experiments/<ts>_oos.png")
    # Social restriction options
    p.add_argument("--restrict_time", type=float, default=None, help="Time at which to apply social restriction (disable links). If omitted, no restriction.")
    p.add_argument("--restrict_end_time", type=float, default=None, help="End time until which restriction remains active. If omitted, restriction persists after start.")
    p.add_argument("--restrict_drop_fraction", type=float, default=0.8, help="Fraction of links to disable per node at restriction time (default: 0.8)")
    p.add_argument("--restrict_seed", type=int, default=None, help="Optional RNG seed for restriction masking (defaults to --seed)")
    return p


def main():
    args = build_parser().parse_args()

    device = args.device or ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    device_t = torch.device(device)

    # Load checkpoint
    ckpt = torch.load(args.load_model, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    gamma_ckpt = ckpt.get("gamma", args.gamma)

    # Helper: run one out-of-sample graph
    def run_one(n_nodes: int, p_edge: float, seed_offset: int = 0, apply_restriction: bool = True):
        Gx, Ax = generate_erdos_renyi_adjacency_networkx(n_nodes, p_edge, args.seed + seed_offset)
        S0x, I0x, R0x = create_initial_conditions(Gx, n_nodes, args.seed + seed_offset, k_infected=args.k_infected)

        # Determine restriction setup
        restrict_time = args.restrict_time if apply_restriction else None
        restrict_end_time = args.restrict_end_time if apply_restriction else None
        drop_frac = float(args.restrict_drop_fraction)
        rseed = args.restrict_seed if args.restrict_seed is not None else (args.seed + seed_offset)

        # Determine if we have a valid restriction window [t_start, t_end)
        has_window = (
            restrict_time is not None
            and restrict_end_time is not None
            and float(restrict_end_time) > float(restrict_time)
        )

        if (restrict_time is None and restrict_end_time is None) or (not has_window):
            # No restriction or immediate zero duration pre-phase
            times_x, S_hist_x, I_hist_x, R_hist_x = simulate_sir_on_graph(
                adjacency=Ax, beta=args.beta, gamma=args.gamma, S0=S0x, I0=I0x, R0=R0x, t_max=args.t_max, dt=args.dt
            )
            model_x = GraphRHSNet(adjacency=Ax, hidden_dim=args.hidden_dim, num_hidden=args.num_hidden, gamma=gamma_ckpt).to(device_t)
            try:
                model_x.phi1.load_state_dict(state_dict["phi1"])  # type: ignore[index]
                model_x.phi2.load_state_dict(state_dict["phi2"])  # type: ignore[index]
            except Exception:
                model_x.load_state_dict(state_dict, strict=False)
            t_nn_x, S_nn_x, I_nn_x, R_nn_x = simulate_with_model_torch(model_x, S0=S0x, I0=I0x, R0=R0x, t_max=args.t_max, dt=args.dt, device=device_t)
            return Gx, Ax, times_x, S_hist_x, I_hist_x, R_hist_x, t_nn_x, S_nn_x, I_nn_x, R_nn_x
        else:
            # Three-phase simulation: pre [0, t_start], restriction [t_start, t_end], post [t_end, t_max]
            t_start = float(restrict_time)
            t_end = float(restrict_end_time)
            # Clamp to [0, t_max]
            t_pre = max(0.0, min(args.t_max, t_start))
            t_res = max(0.0, min(args.t_max - t_pre, t_end - t_start))
            t_post = max(0.0, args.t_max - (t_pre + t_res))

            # Phase 1 (pre): original adjacency
            times_pre, S_pre, I_pre, R_pre = simulate_sir_on_graph(
                adjacency=Ax, beta=args.beta, gamma=args.gamma, S0=S0x, I0=I0x, R0=R0x, t_max=t_pre, dt=args.dt
            )

            # Phase 2 (restriction): masked adjacency
            Ax_mask = mask_adjacency_per_node(Ax, drop_fraction=drop_frac, seed=rseed)
            S0_mid, I0_mid, R0_mid = S_pre[-1].astype(np.float32), I_pre[-1].astype(np.float32), R_pre[-1].astype(np.float32)
            times_res, S_res, I_res, R_res = simulate_sir_on_graph(
                adjacency=Ax_mask, beta=args.beta, gamma=args.gamma, S0=S0_mid, I0=I0_mid, R0=R0_mid, t_max=t_res, dt=args.dt
            )

            # Phase 3 (post): back to original adjacency
            S0_end, I0_end, R0_end = S_res[-1].astype(np.float32), I_res[-1].astype(np.float32), R_res[-1].astype(np.float32)
            times_post, S_post, I_post, R_post = simulate_sir_on_graph(
                adjacency=Ax, beta=args.beta, gamma=args.gamma, S0=S0_end, I0=I0_end, R0=R0_end, t_max=t_post, dt=args.dt
            )

            # Stitch histories (skip first element of each subsequent phase to avoid duplication)
            times_x = np.concatenate([
                times_pre,
                t_pre + times_res[1:],
                (t_pre + t_res) + times_post[1:],
            ], axis=0)
            S_hist_x = np.concatenate([S_pre, S_res[1:, :], S_post[1:, :]], axis=0)
            I_hist_x = np.concatenate([I_pre, I_res[1:, :], I_post[1:, :]], axis=0)
            R_hist_x = np.concatenate([R_pre, R_res[1:, :], R_post[1:, :]], axis=0)

            # Sanity: if all links disabled during restriction, I(t) must be non-increasing over that segment
            if drop_frac >= 1.0 and t_res > 0:
                i_tail = I_res[:, :].sum(axis=1)
                if np.any(np.diff(i_tail) > 1e-9):
                    print("Warning: I(t) increased during full restriction; check dynamics or dt.")

            # NN simulations with three models (adjacency-specific edge lists)
            model_pre = GraphRHSNet(adjacency=Ax, hidden_dim=args.hidden_dim, num_hidden=args.num_hidden, gamma=gamma_ckpt).to(device_t)
            model_res = GraphRHSNet(adjacency=Ax_mask, hidden_dim=args.hidden_dim, num_hidden=args.num_hidden, gamma=gamma_ckpt).to(device_t)
            model_post = GraphRHSNet(adjacency=Ax, hidden_dim=args.hidden_dim, num_hidden=args.num_hidden, gamma=gamma_ckpt).to(device_t)
            try:
                model_pre.phi1.load_state_dict(state_dict["phi1"])  # type: ignore[index]
                model_pre.phi2.load_state_dict(state_dict["phi2"])  # type: ignore[index]
                model_res.phi1.load_state_dict(state_dict["phi1"])  # type: ignore[index]
                model_res.phi2.load_state_dict(state_dict["phi2"])  # type: ignore[index]
                model_post.phi1.load_state_dict(state_dict["phi1"])  # type: ignore[index]
                model_post.phi2.load_state_dict(state_dict["phi2"])  # type: ignore[index]
            except Exception:
                model_pre.load_state_dict(state_dict, strict=False)
                model_res.load_state_dict(state_dict, strict=False)
                model_post.load_state_dict(state_dict, strict=False)

            t_nn_pre, S_nn_pre, I_nn_pre, R_nn_pre = simulate_with_model_torch(
                model_pre, S0=S0x, I0=I0x, R0=R0x, t_max=t_pre, dt=args.dt, device=device_t
            )
            S_mid_nn = S_nn_pre[-1].detach().cpu().numpy()
            I_mid_nn = I_nn_pre[-1].detach().cpu().numpy()
            R_mid_nn = R_nn_pre[-1].detach().cpu().numpy()
            t_nn_res, S_nn_res, I_nn_res, R_nn_res = simulate_with_model_torch(
                model_res, S0=S_mid_nn, I0=I_mid_nn, R0=R_mid_nn, t_max=t_res, dt=args.dt, device=device_t
            )
            S_end_nn = S_nn_res[-1].detach().cpu().numpy()
            I_end_nn = I_nn_res[-1].detach().cpu().numpy()
            R_end_nn = R_nn_res[-1].detach().cpu().numpy()
            t_nn_post, S_nn_post, I_nn_post, R_nn_post = simulate_with_model_torch(
                model_post, S0=S_end_nn, I0=I_end_nn, R0=R_end_nn, t_max=t_post, dt=args.dt, device=device_t
            )

            t_nn_x = np.concatenate([
                t_nn_pre,
                t_pre + t_nn_res[1:],
                (t_pre + t_res) + t_nn_post[1:],
            ], axis=0)
            S_nn_x = torch.cat([S_nn_pre, S_nn_res[1:, :], S_nn_post[1:, :]], dim=0)
            I_nn_x = torch.cat([I_nn_pre, I_nn_res[1:, :], I_nn_post[1:, :]], dim=0)
            R_nn_x = torch.cat([R_nn_pre, R_nn_res[1:, :], R_nn_post[1:, :]], dim=0)

            return Gx, Ax, times_x, S_hist_x, I_hist_x, R_hist_x, t_nn_x, S_nn_x, I_nn_x, R_nn_x

    # Run two graphs
    G1, A1, t1, S1, I1, R1, t1n, S1n, I1n, R1n = run_one(args.nodes1, args.p1, seed_offset=0, apply_restriction=False)
    G2, A2, t2, S2, I2, R2, t2n, S2n, I2n, R2n = run_one(args.nodes2, args.p2, seed_offset=17, apply_restriction=True)

    # Aggregates
    s1_true, i1_true, r1_true = S1.sum(axis=1), I1.sum(axis=1), R1.sum(axis=1)
    s1_nn = S1n.sum(dim=1).detach().cpu().numpy()
    i1_nn = I1n.sum(dim=1).detach().cpu().numpy()
    r1_nn = R1n.sum(dim=1).detach().cpu().numpy()

    s2_true, i2_true, r2_true = S2.sum(axis=1), I2.sum(axis=1), R2.sum(axis=1)
    s2_nn = S2n.sum(dim=1).detach().cpu().numpy()
    i2_nn = I2n.sum(dim=1).detach().cpu().numpy()
    r2_nn = R2n.sum(dim=1).detach().cpu().numpy()

    # Plot: 2 rows x 2 cols. Left: s/i/r curves; Right: degree dist with inset graph
    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.6, 1.4], height_ratios=[1, 1], wspace=0.30, hspace=0.35)

    # Row 1 (n1)
    ax_l1 = fig.add_subplot(gs[0, 0])
    ax_l1.plot(t1, s1_true, label="s true", color="#1f77b4")
    ax_l1.plot(t1n, s1_nn, label="s nn", linestyle=":", color="#1f77b4")
    ax_l1.plot(t1, i1_true, label="i true", color="#ff7f0e")
    ax_l1.plot(t1n, i1_nn, label="i nn", linestyle=":", color="#ff7f0e")
    ax_l1.plot(t1, r1_true, label="r true", color="#2ca02c")
    ax_l1.plot(t1n, r1_nn, label="r nn", linestyle=":", color="#2ca02c")
    # No restriction visual marker for row 1 (first case)
    ax_l1.set_ylabel("aggregate")
    ax_l1.set_title(f"Out-of-sample n={args.nodes1}, p={args.p1}")
    ax_l1.legend(loc="best", ncol=3, fontsize=8)

    try:
        import networkx as nx
        ax_r1 = fig.add_subplot(gs[0, 1])
        largest_cc_nodes_1 = max(nx.connected_components(G1), key=len) if G1.number_of_nodes() > 0 else []
        G1_vis = G1.subgraph(largest_cc_nodes_1).copy() if G1.number_of_nodes() > 0 else G1
        degrees1 = [d for _, d in G1_vis.degree()] if G1_vis.number_of_nodes() > 0 else []
        bins1 = max(5, min(25, int(np.sqrt(max(1, len(degrees1))))))
        ax_r1.hist(degrees1, bins=bins1, color="#888888", edgecolor="black")
        exp_deg1 = args.nodes1 * args.p1
        ax_r1.set_title(f"Degree dist (n={args.nodes1}, E[k]\u2248n*p={exp_deg1:.1f})")
        ax_r1.set_xlabel("Degree")
        ax_r1.set_ylabel("Count")
        ax_r1.set_ylim(0, 50)
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        ax_in1 = inset_axes(ax_r1, width="38%", height="38%", loc="upper right", borderpad=1.0)
        if G1_vis.number_of_nodes() > 0:
            pos1 = compute_layout_no_overlap(G1_vis, seed=args.seed)
            deg_map1 = dict(G1_vis.degree())
            node_colors1 = [deg_map1[n] for n in G1_vis.nodes()]
            vmin1 = min(node_colors1) if len(node_colors1) > 0 else 0
            vmax1 = max(node_colors1) if len(node_colors1) > 0 else 1
            nx.draw_networkx(
                G1_vis,
                pos=pos1,
                ax=ax_in1,
                node_size=14,
                width=0.4,
                with_labels=False,
                edge_color="#aaaaaa",
                node_color=node_colors1,
                cmap="viridis",
                vmin=vmin1,
                vmax=vmax1,
            )
        ax_in1.set_axis_off()
    except Exception:
        ax_r1 = fig.add_subplot(gs[0, 1])
        ax_r1.axis("off")

    # Row 2 (n2)
    ax_l2 = fig.add_subplot(gs[1, 0], sharex=ax_l1)
    ax_l2.plot(t2, s2_true, label="s true", color="#1f77b4")
    ax_l2.plot(t2n, s2_nn, label="s nn", linestyle=":", color="#1f77b4")
    ax_l2.plot(t2, i2_true, label="i true", color="#ff7f0e")
    ax_l2.plot(t2n, i2_nn, label="i nn", linestyle=":", color="#ff7f0e")
    ax_l2.plot(t2, r2_true, label="r true", color="#2ca02c")
    ax_l2.plot(t2n, r2_nn, label="r nn", linestyle=":", color="#2ca02c")
    # Draw restriction window markers for second case only if both times provided
    if args.restrict_time is not None and args.restrict_end_time is not None:
        try:
            x1 = float(args.restrict_time)
            x2 = float(args.restrict_end_time)
            if x2 > x1:
                ax_l2.axvline(x1, color="red", linestyle="--", linewidth=1.2, alpha=0.9)
                ax_l2.axvline(x2, color="red", linestyle="--", linewidth=1.2, alpha=0.9)
        except Exception:
            pass
    ax_l2.set_xlabel("time")
    ax_l2.set_ylabel("aggregate")
    ax_l2.set_title(f"Out-of-sample n={args.nodes2}, p={args.p2}")
    ax_l2.legend(loc="best", ncol=3, fontsize=8)

    try:
        import networkx as nx
        ax_r2 = fig.add_subplot(gs[1, 1])
        largest_cc_nodes_2 = max(nx.connected_components(G2), key=len) if G2.number_of_nodes() > 0 else []
        G2_vis = G2.subgraph(largest_cc_nodes_2).copy() if G2.number_of_nodes() > 0 else G2
        degrees2 = [d for _, d in G2_vis.degree()] if G2_vis.number_of_nodes() > 0 else []
        bins2 = max(5, min(25, int(np.sqrt(max(1, len(degrees2))))))
        ax_r2.hist(degrees2, bins=bins2, color="#888888", edgecolor="black")
        exp_deg2 = args.nodes2 * args.p2
        ax_r2.set_title(f"Degree dist (n={args.nodes2}, E[k]\u2248n*p={exp_deg2:.1f})")
        ax_r2.set_xlabel("Degree")
        ax_r2.set_ylabel("Count")
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        ax_in2 = inset_axes(ax_r2, width="38%", height="38%", loc="upper right", borderpad=1.0)
        if G2_vis.number_of_nodes() > 0:
            pos2 = compute_layout_no_overlap(G2_vis, seed=args.seed + 17)
            deg_map2 = dict(G2_vis.degree())
            node_colors2 = [deg_map2[n] for n in G2_vis.nodes()]
            vmin2 = min(node_colors2) if len(node_colors2) > 0 else 0
            vmax2 = max(node_colors2) if len(node_colors2) > 0 else 1
            nx.draw_networkx(
                G2_vis,
                pos=pos2,
                ax=ax_in2,
                node_size=14,
                width=0.4,
                with_labels=False,
                edge_color="#aaaaaa",
                node_color=node_colors2,
                cmap="viridis",
                vmin=vmin2,
                vmax=vmax2,
            )
        ax_in2.set_axis_off()
    except Exception:
        ax_r2 = fig.add_subplot(gs[1, 1])
        ax_r2.axis("off")

    plt.tight_layout()
    ts = datetime.datetime.now().strftime("%d_%m_%Y_%H:%M")
    default_dir = os.path.dirname(os.path.abspath(args.load_model))
    default_name = f"oos_n{args.nodes1}_{args.nodes2}_{ts}.png"
    out_path = args.out or os.path.join(default_dir, default_name)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved out-of-sample comparison to: {out_path}")


if __name__ == "__main__":
    main()


