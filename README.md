# Agent-based-model informed neural networks

Code and experiments for the paper:

**Antulov-Fantulin, N.** Towards agent-based-model informed neural networks. *EPJ Data Science* **15**, 13 (2026).  
https://doi.org/10.1140/epjds/s13688-025-00616-z

The repository implements **ABM-informed neural networks**: neural ODEs whose right-hand sides decompose into **self dynamics** and **restricted graph interactions**, with inductive biases (conservation laws, nonnegativity, etc.) taken from agent-based and mean-field models. Case studies mirror the paper: Hamiltonian baseline, SIR on graphs, analytical GLV recovery, and macroeconomic GLV with latent channels.

LaTeX source: [`main.tex`](main.tex).

---

## Requirements

- Python 3.10+
- `torch`, `numpy`, `matplotlib`, `tqdm`
- `pandas` (macro GLV scripts)
- `networkx` (epidemic graph generation and layouts)

```bash
pip install torch numpy matplotlib tqdm pandas networkx
```

**Note:** The epidemic scripts import `epidemic_demo_micro` (`generate_erdos_renyi_adjacency_networkx`, `simulate_sir_on_graph`). That module is **not** in this repository; add it locally (or vendor it) before running `epidemic_demo_macro*.py`.

---

## Root scripts (overview)

| Script | Role |
|--------|------|
| `hnn_mass_spring_demo.py` | HNN vs unconstrained Neural ODE on a mass–spring system |
| `epidemic_demo_macro.py` | Train ABM-informed SIR RHS from aggregate curves only |
| `epidemic_demo_macro_out_of_sample.py` | Evaluate a saved macro model on new graphs / interventions |
| `epidemic_demo_macro_functionals.py` | Same as macro training, with learnable \(F,G,H\) functionals |
| `GLV_learning_micro_explicit.py` | Recover explicit GLV growth rates (Three-Body case study) |
| `GLV_learning_micro_gdp_extended_universal_out.py` | Shared-parameter GLV + macro latent ODE on real GDP data |

---

## 1. `hnn_mass_spring_demo.py`

Illustrates **structure-preserving** learning (Hamiltonian neural network) versus a **black-box** neural ODE on the same data—motivation for conservation-style biases used later in ABM-informed models (see Fig. neural-ODE vs HNN in the paper).

### Dynamics

Unit mass, unit spring constant. State \((q,p)\) with Hamiltonian

\[
H(q,p) = \tfrac{1}{2}(q^2 + p^2).
\]

Hamilton’s equations:

\[
\dot{q} = \frac{\partial H}{\partial p} = p, \qquad
\dot{p} = -\frac{\partial H}{\partial q} = -q.
\]

Trajectories are circles in phase space (energy conserved).

### Model

- **Ground truth:** analytic RHS above; integration with RK4 (`dt=0.01`, rollout up to `t=1000`).
- **HNN:** MLP \(H_\theta(q,p)\); derivatives via autograd:
  \[
  \dot{q} = \frac{\partial H_\theta}{\partial p}, \quad
  \dot{p} = -\frac{\partial H_\theta}{\partial q}.
  \]
- **Neural ODE baseline:** MLP directly predicts \((\dot{q},\dot{p})\) (same architecture width, no Hamiltonian structure).

### Learning

Supervised **derivative matching** on \(4096\) samples \((q,p)\) with \(r \sim \mathcal{U}(0.2,2)\), \(\theta \sim \mathcal{U}(0,2\pi)\):

\[
\mathcal{L} = \frac{1}{N}\sum_{k=1}^{N} \big\| f_\theta(q_k,p_k) - (\dot{q}_k,\dot{p}_k)_{\text{true}} \big\|_2^2.
\]

Adam, `lr=10^{-3}`, 2000 epochs. After training, both models are rolled out from ICs on a circle \(r=1.2\).

### Data

**Synthetic only**—no files. Outputs under `plots/`:

- `neuralode_vs_hnn_side_by_side.png`
- `mse_vs_time.png`

```bash
python hnn_mass_spring_demo.py
```

---

## 2. `epidemic_demo_macro.py` and `epidemic_demo_macro_out_of_sample.py`

**Case study 2 (contagion):** learn node-level SIR dynamics on a graph while observing only **aggregated** compartment sums \(s(t)=\sum_i S_i\), \(i(t)=\sum_i I_i\), \(r(t)=\sum_i R_i\).

### Dynamics (ground truth)

On adjacency \(A\), per-node SIR with rates \(\beta,\gamma\):

\[
\dot{S}_i = -\sum_j \beta A_{ij} S_i I_j, \qquad
\dot{I}_i = \sum_j \beta A_{ij} S_i I_j - \gamma I_i, \qquad
\dot{R}_i = \gamma I_i.
\]

Each node obeys **mass conservation** \(\dot{S}_i+\dot{I}_i+\dot{R}_i=0\); with \(S_i+I_i+R_i=1\) initially, states stay on the probability simplex.

### Model (`GraphRHSNet` in `epidemic_demo_macro.py`)

ABM-informed decomposition with **hard-wired** functionals (paper Eqs. for \(F,G,H\)):

\[
\psi_1^{(i)} = \sum_{j: A_{ij}>0} \phi_1(S_i, I_j), \qquad
\phi_2^{(i)} = \phi_2(I_i),
\]

\[
\dot{S}_i = -\psi_1^{(i)}, \quad
\dot{I}_i = \psi_1^{(i)} - \phi_2^{(i)}, \quad
\dot{R}_i = \phi_2^{(i)}.
\]

\(\phi_1:\mathbb{R}^2\to\mathbb{R}_{\ge 0}\) and \(\phi_2:\mathbb{R}\to\mathbb{R}_{\ge 0}\) are small MLPs (LeakyReLU hidden, ReLU output). So \(\dot{S}_i+\dot{I}_i+\dot{R}_i=0\) by construction. Simulation uses differentiable RK4; after each step, \(S,I,R\) are clamped to \([0,1]\) and renormalized per node.

### Learning (`epidemic_demo_macro.py`)

**Macro loss** (normalized L1 on aggregate curves over curriculum horizon \(t_{\text{curr}}\)):

\[
\mathcal{L}_{\text{macro}} = \frac{1}{L \cdot N}\sum_{t\le t_{\text{curr}}} \Big(
|s(t)-\tilde{s}(t)| + |i(t)-\tilde{i}(t)| + |r(t)-\tilde{r}(t)|
\Big).
\]

**Regularizers** (optional after warmup):

\[
\mathcal{R}_{\phi_1,\text{axis}} = \mathbb{E}_{S\sim\mathcal{U}[0,1]}\big[|\phi_1(S,0)|\big], \qquad
\mathcal{R}_{\phi_2,\text{origin}} = |\phi_2(0)|,
\]

\[
\mathcal{L} = \mathcal{L}_{\text{macro}} + \lambda_1 \mathcal{R}_{\phi_1,\text{axis}} + \lambda_2 \mathcal{R}_{\phi_2,\text{origin}}.
\]

Curriculum: horizon grows from `horizon_base` by `horizon_increment` every `horizon_step_epochs`. Default training graph: Erdős–Rényi \(n=100\), \(p=0.05\); truth \(\beta=0.4\), \(\gamma=0.2\); RK4 \(\Delta t=0.1\).

Checkpoints save `phi1`, `phi2`, `adjacency`, `gamma` under `experiments/`.

```bash
python epidemic_demo_macro.py --epochs 1000 --exp_name macro_sir
```

### Data

- **Synthetic:** one ER graph + micro SIR simulation (`epidemic_demo_micro.simulate_sir_on_graph`).
- **Initial conditions:** up to 50 infected nodes in the largest connected component; remainder susceptible.
- **No external dataset files** for training.

### `epidemic_demo_macro_out_of_sample.py`

Loads a trained checkpoint and evaluates on **new** graphs (different \(n,p\)) without retraining. Compares learned rollout \(\tilde{s},\tilde{i},\tilde{r}\) to ground-truth micro simulation.

**Intervention (optional):** between `restrict_time` and `restrict_end_time`, a fraction `restrict_drop_fraction` of each node’s outgoing edges is zeroed (`mask_adjacency_per_node`), modeling social distancing while \(\phi_1,\phi_2\) stay fixed.

```bash
python epidemic_demo_macro_out_of_sample.py \
  --load_model experiments/<run>/macro_rhs.ckpt \
  --nodes1 150 --nodes2 250 --beta 0.3 --gamma 0.2 \
  --restrict_time 1.5 --restrict_end_time 10.0 --restrict_drop_fraction 0.9
```

---

## 3. `epidemic_demo_macro_functionals.py`

Same epidemic setup as §2, but the map from \((\psi_1,\phi_2)\) to \((\dot{S},\dot{I},\dot{R})\) uses **learnable linear functionals** \(F,G,H\) on the pair \((\psi_1^{(i)}, \phi_2^{(i)})\):

\[
\dot{S}_i = F(\psi_1^{(i)}, \phi_2^{(i)}), \quad
\dot{I}_i = G(\psi_1^{(i)}, \phi_2^{(i)}), \quad
\dot{R}_i = H(\psi_1^{(i)}, \phi_2^{(i)}),
\]

implemented as `nn.Linear(2,1)` coefficients. \(F\) is **initialized and frozen** to match SIR structure \(F=-\psi_1\); \(G,H\) can be pretrained and/or learned with **separate learning rates** from \(\phi_1,\phi_2\).

### Conservation during training

Penalty on violation of node-wise mass balance in the predicted RHS:

\[
\mathcal{R}_{\text{cons}} = \mathbb{E}\Big[\big|\dot{S}_i+\dot{I}_i+\dot{R}_i\big|\Big], \qquad
\mathcal{L} = \mathcal{L}_{\text{macro}} + \lambda_{\text{cons}}\mathcal{R}_{\text{cons}} + \cdots
\]

(plus the same \(\phi_1,\phi_2\) axis/origin regularizers as §2). Optional **stage-1 pretrain** fits \(F,G,H\) on random states to minimize \(\mathcal{R}_{\text{cons}}\) before macro trajectory loss.

### Data & usage

Identical synthetic pipeline to `epidemic_demo_macro.py`. Use when studying **interpretable functionals** vs fully hard-wired SIR wiring (Appendix functional-learning figures in the paper).

```bash
python epidemic_demo_macro_functionals.py --epochs 1000 --pretrain_coefficients
```

---

## 4. `GLV_learning_micro_explicit.py`

**Case study 1 (analytical GLV recovery):** “Three-Body” faction capacities with known \(r_i\), \(A_{ij}\), and exogenous schedules \(S_i(t)\), \(\tau_i(t)\).

### Dynamics

\[
\frac{dX_i}{dt} = X_i\big(r_i S_i(t) + \tau_i(t)\big) + X_i \sum_{j=1}^{N} a_{ij} X_j.
\]

Five factions; piecewise \(S(t)\) (Sophon lock) and \(\tau(t)\) (regime shocks) defined in-script (`get_S`, `get_tau`). Ground truth integrated with RK4; states clamped to \(\ge 0\).

### Model

ABM-informed split matching the paper:

\[
\dot{x}_i = \underbrace{\phi_{\text{self}}(x_i; r_i, S_i,\tau_i)}_{\text{learned}} + \underbrace{\sum_j a_{ij}\, \phi_{\text{pair}}(x_i,x_j)}_{\text{fixed } \phi_{\text{pair}}=x_i x_j},
\]

with \(\phi_{\text{self}}(x_i) = x_i \exp(\log r_i)\, S_i(t) + x_i \tau_i(t)\) (learnable \(\log r_i\); \(A\) fixed from ground truth).

### Learning

Trajectory L1 on a **curriculum window** \(t \in [0, T_{\text{curr}}]\), \(T_{\text{curr}}\le\) `train_end` (default 50 years) while evaluation runs to `t_max` (e.g. 250):

\[
\mathcal{L} = \frac{1}{NT}\sum_{i,t\le T_{\text{curr}}} |X_i^{\text{pred}}(t) - X_i^{\text{true}}(t)|.
\]

AdamW + optional CyclicLR / CosineAnnealing; curriculum expands by `curriculum_increment` on schedule or stagnation.

### Data

**Synthetic** initial vector \(X_0 = (100,2,5,1,8)\); no external files. Writes `experiments/GLV_learning_micro_<timestamp>/` (trajectory, \(r_i\) convergence plots).

```bash
python GLV_learning_micro_explicit.py --epochs 500 --train-end 50 --t-max 250
```

---

## 5. `GLV_learning_micro_gdp_extended_universal_out.py`

**Case study 3 (macroeconomics):** coupled GDP and six macro channels per country; shared neural maps + learned interaction matrix \(A\); **out-of-sample** years after `train_cut_year`.

### Dynamics (conceptual)

For each country \(i\), normalized GDP \(x_i(t)\) and latent macro vector \(y_i(t)\in\mathbb{R}^6\):

\[
\dot{x}_i = \phi_x(x_i, y_i, e_i) + \sum_j A_{ij}\, x_i\, (\max(x_j,0)+\varepsilon)^{\beta},
\]

\[
\dot{y}_i = \phi_y(y_i, x_i, e_i),
\]

with \(\phi_x(x) = x\,\phi_g(\cdot) + \phi_d(\cdot)\) (growth + drift MLPs) and **FiLM** adapters on hidden layers from country embedding \(e_i\):

\[
h_{\ell+1} = \gamma_\ell(e_i)\odot h_\ell + \beta_\ell(e_i).
\]

\(\beta\) (interaction exponent) is learned via \(\sigma(\text{interaction\_beta\_raw})\).

### Model

`GDPGLVExtendedModel`: shared `PhiSelfCoupled`, `PhiYIndividual`, learnable `A`, optional per-row interaction scales; RK4 with `steps` substeps per year.

### Learning

Train only on years **strictly before** `--train-cut-year` (paper: 1995–2020 train, 2021+ holdout). Loss on rolled-out trajectories:

\[
\mathcal{L}_{\text{GDP}} = \mathbb{E}_{i,t}\frac{| \hat{x}_{i,t} x^{\text{base}}_i - x^{\text{true}}_{i,t} x^{\text{base}}_i |}{|x^{\text{true}}_{i,t} x^{\text{base}}_i| + \varepsilon}, \qquad
\mathcal{L}_{y} = \mathrm{MSE}(\hat{y}, y),
\]

\[
\mathcal{L} = \mathcal{L}_{\text{GDP}} + w_y \mathcal{L}_{y} + \lambda_A \|A\|_2^2.
\]

Optional **teacher-forcing pretrain** matches finite-difference targets of \((\dot{x},\dot{y})\) from data. Interaction term ramps in after `interaction_warmup_epochs`; curriculum on number of time steps mirrors the GLV script.

### Data

- Default CSV: `data_cache/macro_dataset_kalman_processed.csv` (override with `--macro-csv`).
- Features per country–year: Unemployment, Interest_Rate, Debt, Working_Age_Pop, Inflation, Current_Account.
- GDP normalized by 1995 level; top `--top-n` economies by GDP at `--end-year`.
- Bundled raw tables also under `data_cache/` and `data/gdp_fixed_1970_2023.json`.

```bash
python GLV_learning_micro_gdp_extended_universal_out.py \
  --device cpu --start-year 1995 --end-year 2024 --top-n 10 \
  --train-cut-year 2021 --epochs 500 --macro-csv data_cache/macro_dataset_kalman_processed.csv
```

Outputs: `experiments/GLV_learning_micro_gdp_extended_<timestamp>/` (`trajectory.png`, `A_matrix.png`, `beta_history.png`, `command.txt`).

---

## Repository layout

```
abm-nn/
├── main.tex                          # Paper source
├── hnn_mass_spring_demo.py
├── epidemic_demo_macro.py
├── epidemic_demo_macro_out_of_sample.py
├── epidemic_demo_macro_functionals.py
├── GLV_learning_micro_explicit.py
├── GLV_learning_micro_gdp_extended_universal_out.py
├── data_cache/                       # Macro time series (CSV)
├── data/                             # Additional GDP JSON
└── experiments/                      # Created at run time (checkpoints, plots)
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
