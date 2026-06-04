# Agent-based-model informed neural networks

Code and experiments for the paper:

**Antulov-Fantulin, N.** Towards agent-based-model informed neural networks. *EPJ Data Science* **15**, 13 (2026).  
https://doi.org/10.1140/epjds/s13688-025-00616-z

The repository implements **ABM-informed neural networks**: neural ODEs whose right-hand sides decompose into **self dynamics** and **restricted graph interactions**, with inductive biases (conservation laws, nonnegativity, etc.) taken from agent-based and mean-field models. Case studies mirror the paper: Hamiltonian baseline, SIR on graphs, analytical GLV recovery, and macroeconomic GLV with latent channels.

LaTeX source: [`main.tex`](main.tex).

---

## Installation

**Python:** 3.13+ (see [`pyproject.toml`](pyproject.toml)).

Dependencies are declared in [`pyproject.toml`](pyproject.toml) and locked in [`uv.lock`](uv.lock). Top-level pins are also listed in [`requirements.txt`](requirements.txt); fully pinned versions are in [`requirements-lock.txt`](requirements-lock.txt).

| Package | Used for |
|---------|----------|
| `torch` | Neural ODEs, HNN, epidemic & GLV training |
| `torch-geometric` | Graph baselines / extensions |
| `numpy`, `scipy` | Simulation & numerics |
| `matplotlib`, `tqdm` | Plots & training progress |
| `pandas`, `pandas-datareader` | Macro GDP case study |
| `networkx` | Epidemic graph generation & layouts |

### Recommended: [uv](https://docs.astral.sh/uv/)

[uv](https://docs.astral.sh/uv/) installs the locked environment from `uv.lock` (reproducible across machines).

```bash
# Install uv (macOS / Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh

# From the repository root: create .venv and install all dependencies
uv sync

# Run scripts with the project interpreter
uv run python code/hnn_mass_spring_demo.py
```

After `uv sync`, activate the venv if you prefer a plain `python` command:

```bash
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python code/epidemic_demo_macro.py --epochs 1000 --exp_name macro_sir
```

### Alternatives (pip)

If you do not use uv:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt          # minimum versions (matches pyproject.toml)
# pip install -r requirements-lock.txt   # exact pins (closest to uv.lock)
```

With uv but without syncing the whole project:

```bash
uv venv
uv pip install -r requirements.txt
# uv pip install -r requirements-lock.txt
```

**PyTorch note:** `torch` wheels are platform-specific. If `uv sync` or pip fails on your OS/CUDA setup, install PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/) first, then run `uv sync` again or install the remaining requirements.

**Note:** The epidemic scripts import `epidemic_demo_micro` (`generate_erdos_renyi_adjacency_networkx`, `simulate_sir_on_graph`). Place that module alongside the other scripts under `code/` (it is **not** bundled in this repo).

### Running scripts

Run from the **repository root** so paths to `data/` and `data_cache/` resolve correctly:

```bash
uv run python code/<script>.py [args...]
# or, with .venv activated:  python code/<script>.py [args...]
```

Artifacts are written relative to the working directory or script location:

| Output | Typical location |
|--------|------------------|
| HNN figures | `plots/` (repo root) |
| Epidemic training / OOS | `code/experiments/` |
| GLV runs | `experiments/` (repo root) |

---

## Scripts in `code/` (overview)

| Script | Role |
|--------|------|
| [`code/hnn_mass_spring_demo.py`](code/hnn_mass_spring_demo.py) | HNN vs unconstrained Neural ODE on a mass–spring system |
| [`code/epidemic_demo_macro.py`](code/epidemic_demo_macro.py) | Train ABM-informed SIR RHS from aggregate curves only |
| [`code/epidemic_demo_macro_out_of_sample.py`](code/epidemic_demo_macro_out_of_sample.py) | Evaluate a saved macro model on new graphs / interventions |
| [`code/epidemic_demo_macro_functionals.py`](code/epidemic_demo_macro_functionals.py) | Same as macro training, with learnable $F,G,H$ functionals |
| [`code/GLV_learning_micro_explicit.py`](code/GLV_learning_micro_explicit.py) | Recover explicit GLV growth rates (Three-Body case study) |
| [`code/GLV_learning_micro_gdp_extended_universal_out.py`](code/GLV_learning_micro_gdp_extended_universal_out.py) | Shared-parameter GLV + macro latent ODE on real GDP data |

**Paper figures reproduced by these scripts** (see `main.tex` comments for the exact runs):

| Script | Paper figure(s) |
|--------|-----------------|
| `hnn_mass_spring_demo.py` | Fig. neural-ODE vs HNN (`figs/spring-4.pdf`, export from script PNGs) |
| `epidemic_demo_macro.py` | Appendix Fig. macro SIR training |
| `epidemic_demo_macro_out_of_sample.py` | Fig. macro SIR OOS + intervention |
| `epidemic_demo_macro_functionals.py` | Appendix Fig. separate-LR / functional SIR |
| `GLV_learning_micro_explicit.py` | Appendix Fig. three-body GLV |
| `GLV_learning_micro_gdp_extended_universal_out.py` | Fig. macro trajectory + interaction matrix $A$ |

---

## 1. `code/hnn_mass_spring_demo.py`

Illustrates **structure-preserving** learning (Hamiltonian neural network) versus a **black-box** neural ODE on the same data—motivation for conservation-style biases used later in ABM-informed models (see Fig. neural-ODE vs HNN in the paper).

### Dynamics

Unit mass, unit spring constant. State $(q,p)$ with Hamiltonian

$$
H(q,p) = \frac{1}{2}(q^2 + p^2).
$$

Hamilton’s equations:

$$
\dot{q} = \frac{\partial H}{\partial p} = p, \qquad
\dot{p} = -\frac{\partial H}{\partial q} = -q.
$$

Trajectories are circles in phase space (energy conserved).

### Model

- **Ground truth:** analytic RHS above; integration with RK4 (`dt=0.01`, rollout up to `t=1000`).
- **HNN:** MLP $H_\theta(q,p)$; derivatives via autograd:

$$
\dot{q} = \frac{\partial H_\theta}{\partial p}, \quad
\dot{p} = -\frac{\partial H_\theta}{\partial q}.
$$

- **Neural ODE baseline:** MLP directly predicts $(\dot{q},\dot{p})$ (same architecture width, no Hamiltonian structure).

### Learning

Supervised **derivative matching** on $4096$ samples $(q,p)$ with $r \sim \mathrm{Uniform}(0.2,2)$, $\theta \sim \mathrm{Uniform}(0,2\pi)$:

$$
\mathcal{L} = \frac{1}{N}\sum_{k=1}^{N} \left\| f_\theta(q_k,p_k) - (\dot{q}_k,\dot{p}_k)_{\mathrm{true}} \right\|_2^2.
$$

Adam, `lr=1e-3`, 2000 epochs. After training, both models are rolled out from ICs on a circle $r=1.2$.

### Data

**Synthetic only**—no files. Outputs under `plots/`:

- `neuralode_vs_hnn_side_by_side.png`
- `mse_vs_time.png`

### Paper reproduction

**Figure:** Introduction, HNN vs Neural ODE on mass–spring (`fig:neural-ode-vs-hnn`).

No CLI flags; hyperparameters are fixed in `TrainingConfig` inside the script (aligned with the paper text):

| Setting | Value |
|---------|-------|
| Training samples | 4096 |
| Batch size | 256 |
| Adam LR | `1e-3` |
| Epochs | 2000 |
| RK4 `dt` / rollout | `0.01` / `1000` |
| Plot IC radius | `1.2` |

```bash
uv run python code/hnn_mass_spring_demo.py
```

The paper PDF `figs/spring-4.pdf` was produced from the side-by-side phase-space panel (`plots/neuralode_vs_hnn_side_by_side.png`); convert or compose manually if you need the exact layout.

---

## 2. `code/epidemic_demo_macro.py` and `code/epidemic_demo_macro_out_of_sample.py`

**Case study 2 (contagion):** learn node-level SIR dynamics on a graph while observing only **aggregated** compartment sums $s(t)=\sum_i S_i$, $i(t)=\sum_i I_i$, $r(t)=\sum_i R_i$.

### Dynamics (ground truth)

On adjacency $A$, per-node SIR with rates $\beta,\gamma$:

$$
\dot{S}_i = -\sum_j \beta A_{ij} S_i I_j, \qquad
\dot{I}_i = \sum_j \beta A_{ij} S_i I_j - \gamma I_i, \qquad
\dot{R}_i = \gamma I_i.
$$

Each node obeys **mass conservation** $\dot{S}_i+\dot{I}_i+\dot{R}_i=0$; with $S_i+I_i+R_i=1$ initially, states stay on the probability simplex.

### Model (`GraphRHSNet` in `code/epidemic_demo_macro.py`)

ABM-informed decomposition with **hard-wired** functionals (paper Eqs. for $F,G,H$):

$$
\psi_1^{(i)} = \sum_{j:\, A_{ij}>0} \phi_1(S_i, I_j), \qquad
\phi_2^{(i)} = \phi_2(I_i),
$$

$$
\dot{S}_i = -\psi_1^{(i)}, \quad
\dot{I}_i = \psi_1^{(i)} - \phi_2^{(i)}, \quad
\dot{R}_i = \phi_2^{(i)}.
$$

$\phi_1:\mathbb{R}^2 \to \mathbb{R}_{\geq 0}$ and $\phi_2:\mathbb{R} \to \mathbb{R}_{\geq 0}$ are small MLPs (LeakyReLU hidden, ReLU output). So $\dot{S}_i+\dot{I}_i+\dot{R}_i=0$ by construction. Simulation uses differentiable RK4; after each step, $S,I,R$ are clamped to $[0,1]$ and renormalized per node.

### Learning (`code/epidemic_demo_macro.py`)

**Macro loss** (normalized L1 on aggregate curves over curriculum horizon $t_{\mathrm{curr}}$):

$$
\mathcal{L}_{\mathrm{macro}} = \frac{1}{L N}\sum_{t \leq t_{\mathrm{curr}}}
\left(
|s(t)-\tilde{s}(t)| + |i(t)-\tilde{i}(t)| + |r(t)-\tilde{r}(t)|
\right).
$$

**Regularizers** (optional after warmup):

$$
\mathcal{R}_{\phi_1,\mathrm{axis}} = \mathbb{E}_{S \sim \mathrm{Uniform}[0,1]}\left[|\phi_1(S,0)|\right], \qquad
\mathcal{R}_{\phi_2,\mathrm{origin}} = |\phi_2(0)|,
$$

$$
\mathcal{L} = \mathcal{L}_{\mathrm{macro}} + \lambda_1 \mathcal{R}_{\phi_1,\mathrm{axis}} + \lambda_2 \mathcal{R}_{\phi_2,\mathrm{origin}}.
$$

Curriculum: horizon grows from `horizon_base` by `horizon_increment` every `horizon_step_epochs`. Default training graph: Erdős–Rényi $n=100$, $p=0.05$; truth $\beta=0.4$, $\gamma=0.2$; RK4 $\Delta t=0.1$.

Checkpoints save `phi1`, `phi2`, `adjacency`, `gamma` under `code/experiments/<exp_name>_<timestamp>/macro_rhs.ckpt`.

### Paper reproduction — training (`epidemic_demo_macro.py`)

**Figure:** Appendix macro SIR training (`fig:macro-sir-training`).

| Parameter | Paper value |
|-----------|-------------|
| Graph | Erdős–Rényi $n{=}100$, $p{=}0.05$ |
| SIR truth | $\beta{=}0.4$, $\gamma{=}0.2$ |
| Simulation | $t_{\max}{=}30$, $\Delta t{=}0.1$ |
| Training | 1000 epochs, Adam $\eta{=}10^{-4}$, `clip_grad=10` |
| MLP $\phi_1,\phi_2$ | `hidden_dim=32`, `num_hidden=3` |
| Curriculum | `horizon_base=1`, `horizon_increment=1`, every 50 epochs, cap `t_train_max=30` |
| Regularizers | $\lambda_{\phi_1,\mathrm{axis}}{=}\lambda_{\phi_2,0}{=}1$, warmup 100 epochs |
| Seed | `7` |

Most values are script defaults; explicit paper command:

```bash
uv run python code/epidemic_demo_macro.py \
  --epochs 1000 \
  --lr 1e-4 \
  --nodes 100 \
  --p 0.05 \
  --beta 0.4 \
  --gamma 0.2 \
  --t_max 30 \
  --dt 0.1 \
  --hidden_dim 32 \
  --num_hidden 3 \
  --horizon_base 1 \
  --horizon_increment 1 \
  --horizon_step_epochs 50 \
  --t_train_max 30 \
  --clip_grad 10 \
  --lambda_phi1_axis 1 \
  --lambda_phi2_zero 1 \
  --reg_warmup_epochs 100 \
  --seed 7 \
  --exp_name macro_sir
```

Checkpoint: `code/experiments/macro_sir_<timestamp>/macro_rhs.ckpt` (plus `sir_on_graph_macro_experiment.png` in that folder).

### Data

- **Synthetic:** one ER graph + micro SIR simulation (`epidemic_demo_micro.simulate_sir_on_graph`).
- **Initial conditions:** up to 50 infected nodes in the largest connected component; remainder susceptible.
- **No external dataset files** for training.

### `code/epidemic_demo_macro_out_of_sample.py`

Loads a trained checkpoint and evaluates on **new** graphs (different $n,p$) without retraining. Compares learned rollout $\tilde{s},\tilde{i},\tilde{r}$ to ground-truth micro simulation.

**Intervention (optional):** between `restrict_time` and `restrict_end_time`, a fraction `restrict_drop_fraction` of each node’s outgoing edges is zeroed (`mask_adjacency_per_node`), modeling social distancing while $\phi_1,\phi_2$ stay fixed.

### Paper reproduction — OOS (`epidemic_demo_macro_out_of_sample.py`)

**Figure:** Main-text OOS SIR (`fig:macro-sir-model-out-of-sample`). Uses a **trained** `macro_rhs.ckpt` (from the training command above or your own run). Evaluation ground truth uses $\beta{=}0.3$, $\gamma{=}0.2$ (caption); architecture must match training (`hidden_dim=32`, `num_hidden=3`).

| Panel | Settings |
|-------|----------|
| Top | $n{=}150$, $p{=}0.05$, no intervention |
| Bottom | $n{=}250$, $p{=}0.05$, drop 90% of outgoing links per node for $t \in [1.5,\,10.0]$ |

```bash
uv run python code/epidemic_demo_macro_out_of_sample.py \
  --load_model code/experiments/macro_sir_<timestamp>/macro_rhs.ckpt \
  --nodes1 150 --p1 0.05 \
  --nodes2 250 --p2 0.05 \
  --beta 0.3 --gamma 0.2 \
  --t_max 30 --dt 0.1 --seed 7 \
  --hidden_dim 32 --num_hidden 3 \
  --restrict_time 1.5 \
  --restrict_end_time 10.0 \
  --restrict_drop_fraction 0.9
```

Replace `<timestamp>` with your experiment folder name (e.g. `26_09_2025_17:09` in `main.tex`).

---

## 3. `code/epidemic_demo_macro_functionals.py`

Same epidemic setup as section 2, but the map from $(\psi_1,\phi_2)$ to $(\dot{S},\dot{I},\dot{R})$ uses **learnable linear functionals** $F,G,H$ on the pair $(\psi_1^{(i)}, \phi_2^{(i)})$:

$$
\dot{S}_i = F(\psi_1^{(i)}, \phi_2^{(i)}), \quad
\dot{I}_i = G(\psi_1^{(i)}, \phi_2^{(i)}), \quad
\dot{R}_i = H(\psi_1^{(i)}, \phi_2^{(i)}),
$$

implemented as `nn.Linear(2,1)` coefficients. $F$ is **initialized and frozen** to match SIR structure $F=-\psi_1$; $G,H$ can be pretrained and/or learned with **separate learning rates** from $\phi_1,\phi_2$.

### Conservation during training

Penalty on violation of node-wise mass balance in the predicted RHS:

$$
\mathcal{R}_{\mathrm{cons}} = \mathbb{E}\left[\left|\dot{S}_i+\dot{I}_i+\dot{R}_i\right|\right], \qquad
\mathcal{L} = \mathcal{L}_{\mathrm{macro}} + \lambda_{\mathrm{cons}}\mathcal{R}_{\mathrm{cons}} + \cdots
$$

(plus the same $\phi_1,\phi_2$ axis/origin regularizers as section 2). Optional **stage-1 pretrain** fits $F,G,H$ on random states to minimize $\mathcal{R}_{\mathrm{cons}}$ before macro trajectory loss.

### Data & usage

Identical synthetic pipeline to `code/epidemic_demo_macro.py`. Use when studying **interpretable functionals** vs fully hard-wired SIR wiring (Appendix functional-learning figures in the paper).

### Paper reproduction

**Figure:** Appendix functional SIR / separate learning rates (`fig:SI-separate-lr-training`). Same graph and SIR simulation as section 2; $F$ is fixed to $-\psi_1$, $G,H$ learned with higher LR; conservation penalty $\lambda_{\mathrm{cons}}{=}10$.

| Parameter | Paper / appendix text |
|-----------|------------------------|
| Graph & SIR | $n{=}100$, $p{=}0.05$, $\beta{=}0.4$, $\gamma{=}0.2$, $t_{\max}{=}30$, $\Delta t{=}0.1$ |
| Epochs | 700 |
| $\phi_1,\phi_2$ LR | `0.01` |
| $G,H$ LR | `0.1` (`--lr_coeff_G`, `--lr_coeff_H`) |
| Curriculum | start $t{=}1$, $+1$ every **10** epochs (`horizon_step_epochs=10`) |
| $\lambda_{\mathrm{cons}}$ | 10 |
| MLP | `hidden_dim=32`, `num_hidden=3` |

```bash
uv run python code/epidemic_demo_macro_functionals.py \
  --epochs 700 \
  --lr 0.01 \
  --lr_coeff_G 0.1 \
  --lr_coeff_H 0.1 \
  --nodes 100 --p 0.05 \
  --beta 0.4 --gamma 0.2 \
  --t_max 30 --dt 0.1 \
  --hidden_dim 32 --num_hidden 3 \
  --horizon_base 1 \
  --horizon_increment 1 \
  --horizon_step_epochs 10 \
  --t_train_max 30 \
  --clip_grad 10 \
  --lambda_phi1_axis 1 \
  --lambda_phi2_zero 1 \
  --lambda_conservation 10 \
  --reg_warmup_epochs 100 \
  --seed 7 \
  --exp_name macro_sir_functionals
```

Optional (not required for the published figure): `--pretrain_coefficients` to warm-start $G,H$ on the mass-conservation penalty.

---

## 4. `code/GLV_learning_micro_explicit.py`

**Case study 1 (analytical GLV recovery):** “Three-Body” faction capacities with known $r_i$, $A_{ij}$, and exogenous schedules $S_i(t)$, $\tau_i(t)$.

### Dynamics

$$
\frac{dX_i}{dt} = X_i\left(r_i S_i(t) + \tau_i(t)\right) + X_i \sum_{j=1}^{N} a_{ij} X_j.
$$

Five factions; piecewise $S(t)$ (Sophon lock) and $\tau(t)$ (regime shocks) defined in-script (`get_S`, `get_tau`). Ground truth integrated with RK4; states clamped to $\geq 0$.

### Model

ABM-informed split matching the paper:

$$
\dot{x}_i = \phi_{\mathrm{self}}(x_i;\, r_i, S_i,\tau_i) + \sum_j a_{ij}\, \phi_{\mathrm{pair}}(x_i,x_j),
\qquad \phi_{\mathrm{pair}}(x_i,x_j) = x_i x_j \ \text{(fixed)}.
$$

with $\phi_{\mathrm{self}}(x_i) = x_i \exp(\log r_i)\, S_i(t) + x_i \tau_i(t)$ (learnable $\log r_i$; $A$ fixed from ground truth).

### Learning

Trajectory L1 on a **curriculum window** $t \in [0, T_{\mathrm{curr}}]$, $T_{\mathrm{curr}} \leq$ `train_end` (default 50 years) while evaluation runs to `t_max` (e.g. 250):

$$
\mathcal{L} = \frac{1}{NT}\sum_{i,\, t \leq T_{\mathrm{curr}}} \left|X_i^{\mathrm{pred}}(t) - X_i^{\mathrm{true}}(t)\right|.
$$

AdamW + optional CyclicLR / CosineAnnealing; curriculum expands by `curriculum_increment` on schedule or stagnation.

### Data

**Synthetic** initial vector $X_0 = (100,2,5,1,8)$; no external files. Writes `experiments/GLV_learning_micro_<timestamp>/` (trajectory, $r_i$ convergence plots).

### Paper reproduction

**Figure:** Appendix three-body GLV (`fig:three-body-explicit`). Train only on $t \in [0,50]$; evaluate rollout to $t_{\max}{=}250$.

| Parameter | Paper value |
|-----------|-------------|
| Epochs | 500 |
| LR / schedule | `5e-4`, CyclicLR max `2e-2`, step 75 |
| `t-max` / `train-end` | 250 / 50 |
| Curriculum | start 5, $+4$ every 50 epochs, stagnation 50 |
| Grad clip | 2.0 |

```bash
uv run python code/GLV_learning_micro_explicit.py \
  --device cpu \
  --epochs 500 \
  --lr 5e-4 \
  --cyclical-max-lr 2e-2 \
  --cyclical-step 75 \
  --t-max 250 \
  --train-end 50 \
  --curriculum-start 5 \
  --curriculum-increment 4 \
  --curriculum-period 50 \
  --grad-clip 2.0 \
  --stagnation-epochs 50 \
  --exp-prefix GLV_learning_micro
```

Outputs: `experiments/GLV_learning_micro_<timestamp>/trajectory.png`, `r_progress.png` (copy to `figs/glv_trajectory.png`, `figs/glv_r_progress.png` for the paper if needed).

---

## 5. `code/GLV_learning_micro_gdp_extended_universal_out.py`

**Case study 3 (macroeconomics):** coupled GDP and six macro channels per country; shared neural maps + learned interaction matrix $A$; **out-of-sample** years after `train_cut_year`.

### Dynamics (conceptual)

For each country $i$, normalized GDP $x_i(t)$ and latent macro vector $y_i(t) \in \mathbb{R}^6$:

$$
\dot{x}_i = \phi_x(x_i, y_i, e_i) + \sum_j A_{ij}\, x_i\, (\max(x_j,0)+\varepsilon)^{\beta},
$$

$$
\dot{y}_i = \phi_y(y_i, x_i, e_i),
$$

with $\phi_x(x) = x\,\phi_g(\cdot) + \phi_d(\cdot)$ (growth + drift MLPs) and **FiLM** adapters on hidden layers from country embedding $e_i$:

$$
h_{\ell+1} = \gamma_\ell(e_i) \odot h_\ell + \beta_\ell(e_i).
$$

The interaction exponent $\beta$ is learned as $\sigma(z)$, where $z$ is the trainable scalar `interaction_beta_raw` in the model.

### Model

`GDPGLVExtendedModel`: shared `PhiSelfCoupled`, `PhiYIndividual`, learnable `A`, optional per-row interaction scales; RK4 with `steps` substeps per year.

### Learning

Train only on years **strictly before** `--train-cut-year` (paper: 1995–2020 train, 2021+ holdout). Loss on rolled-out trajectories:

$$
\mathcal{L}_{\mathrm{GDP}} = \mathbb{E}_{i,t}\frac{\left| \hat{x}_{i,t} x^{\mathrm{base}}_i - x^{\mathrm{true}}_{i,t} x^{\mathrm{base}}_i \right|}{\left|x^{\mathrm{true}}_{i,t} x^{\mathrm{base}}_i\right| + \varepsilon}, \qquad
\mathcal{L}_{y} = \mathrm{MSE}(\hat{y}, y),
$$

$$
\mathcal{L} = \mathcal{L}_{\mathrm{GDP}} + w_y \mathcal{L}_{y} + \lambda_A \|A\|_2^2.
$$

Optional **teacher-forcing pretrain** matches finite-difference targets of $(\dot{x},\dot{y})$ from data. Interaction term ramps in after `interaction_warmup_epochs`; curriculum on number of time steps mirrors the GLV script.

### Data

- Default CSV: `data_cache/macro_dataset_kalman_processed.csv` (override with `--macro-csv`).
- Features per country–year: Unemployment, Interest_Rate, Debt, Working_Age_Pop, Inflation, Current_Account.
- GDP normalized by 1995 level; top `--top-n` economies by GDP at `--end-year`.
- Bundled raw tables also under `data_cache/` and `data/gdp_fixed_1970_2023.json`.

### Paper reproduction

**Figure:** Main-text macro GDP + $A$ (`fig:macro-traj-interaction`). Train on years before 2021; plot through 2024 (holdout from 2021). Command matches `main.tex`.

| Parameter | Paper value |
|-----------|-------------|
| Data | 1995–2024, top 10 economies |
| Holdout | `--train-cut-year 2021` |
| Epochs / RK4 | 500 / `steps=1` per year |
| LR | `5e-4`, cyclical max `2e-3`, step 75 |
| MLP | `hidden-dim 8`, `num-hidden 2`, `--phi-normalize-input` |
| Pretrain | 500 epochs, batch 256, lr `1e-3` |
| Interaction | warmup 25, ramp 50, $\lambda_A=10^{-5}$, `\|A\| \leq 0.6` |
| Curriculum | start 5 years, $+4$ every 10 epochs |

```bash
uv run python code/GLV_learning_micro_gdp_extended_universal_out.py \
  --device cpu \
  --macro-csv data_cache/macro_dataset_kalman_processed.csv \
  --start-year 1995 --end-year 2024 --top-n 10 \
  --train-cut-year 2021 \
  --epochs 500 --steps 1 \
  --lr 5e-4 \
  --cyclical-max-lr 2e-3 --cyclical-step 75 \
  --weight-decay 1e-4 --grad-clip 2.0 \
  --curriculum-start 5 --curriculum-increment 4 --curriculum-period 10 \
  --stagnation-epochs 50 \
  --hidden-dim 8 --num-hidden 2 --phi-normalize-input \
  --pretrain-epochs 500 --pretrain-batch-size 256 --pretrain-lr 1e-3 \
  --relative-eps 1e-2 \
  --interaction-warmup-epochs 25 --interaction-ramp-epochs 50 \
  --lambda-a 1e-5 --a-max-abs 0.6 --interaction-gain 1.0 \
  --exp-prefix GLV_learning_micro_gdp_extended_out_run \
  --preview-interval 50
```

Outputs: `experiments/GLV_learning_micro_gdp_extended_out_run_<timestamp>/` (`trajectory.png`, `A_matrix.png`, `beta_history.png`, `command.txt`). Copy figures to `figs/macro_trajectory.png` and `figs/macro_A_matrix.png` for LaTeX if needed.

---

## Repository layout

```
abm-nn/
├── README.md
├── pyproject.toml                    # Project metadata & dependencies (source of truth)
├── uv.lock                           # Locked versions for uv sync
├── requirements.txt                  # Pip-friendly dependency list
├── requirements-lock.txt             # Fully pinned pip export
├── main.tex                          # Paper source
├── code/                             # Experiment scripts
│   ├── hnn_mass_spring_demo.py
│   ├── epidemic_demo_macro.py
│   ├── epidemic_demo_macro_out_of_sample.py
│   ├── epidemic_demo_macro_functionals.py
│   ├── GLV_learning_micro_explicit.py
│   ├── GLV_learning_micro_gdp_extended_universal_out.py
│   └── experiments/                  # Epidemic checkpoints & plots (created at run time)
├── data_cache/                       # Macro time series (CSV)
├── data/                             # Additional GDP JSON
├── plots/                            # HNN figures (created at run time)
└── experiments/                      # GLV runs (created at run time)
```

---

## Citation

```bibtex
@article{AntulovFantulin2026ABMNN,
  author  = {Antulov-Fantulin, Nino},
  title   = {Towards agent-based-model informed neural networks},
  journal = {EPJ Data Science},
  volume  = {15},
  number  = {13},
  year    = {2026},
  doi     = {10.1140/epjds/s13688-025-00616-z}
}
```
