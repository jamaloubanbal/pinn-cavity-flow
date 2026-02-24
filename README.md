# PINN Lid-Driven Cavity Flow

A Physics-Informed Neural Network (PINN) that solves the 2-D incompressible
Navier-Stokes equations for the classical **lid-driven cavity** benchmark at
Re = 100, using pure PyTorch — no external CFD library.

---

## Governing equations

The steady, incompressible Navier-Stokes equations on the unit square
$\Omega = [0,1]^2$:

$$
\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0
\quad \text{(continuity)}
$$

$$
u\frac{\partial u}{\partial x} + v\frac{\partial u}{\partial y} + \frac{\partial p}{\partial x} - \nu\!\left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right) = 0
\quad \text{(x-momentum)}
$$

$$
u\frac{\partial v}{\partial x} + v\frac{\partial v}{\partial y} + \frac{\partial p}{\partial y} - \nu\!\left(\frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2}\right) = 0
\quad \text{(y-momentum)}
$$

with $\nu = 0.01$ (Re = 100).

**Boundary conditions**

| Wall       | Condition        |
|------------|-----------------|
| Left  x=0  | u = 0,  v = 0   |
| Right x=1  | u = 0,  v = 0   |
| Bottom y=0 | u = 0,  v = 0   |
| Top   y=1  | u = 1,  v = 0   |

---

## PINN approach

A multi-layer perceptron (MLP) maps spatial coordinates $(x, y)$ to the
flow fields $(u, v, p)$.  Training minimises a composite loss:

$$
\mathcal{L} = \underbrace{\mathcal{L}_\text{PDE}}_{\text{physics residual}}
            + 10\,\underbrace{\mathcal{L}_\text{BC}}_{\text{boundary residual}}
$$

* **Physics residual** $\mathcal{L}_\text{PDE}$: mean-squared sum of the three
  Navier-Stokes residuals evaluated at 10 000 random collocation points inside
  $\Omega$.  All spatial derivatives are computed analytically via
  `torch.autograd.grad` with `create_graph=True`, so gradients flow through
  the derivative operators during back-propagation.

* **Boundary residual** $\mathcal{L}_\text{BC}$: mean-squared deviation from
  the Dirichlet boundary conditions on 200 uniformly spaced points per wall
  (800 points total).

**Network architecture** — `src/model.py`

```
Input (2) → [Linear(64) → tanh] × 6 → Linear(3) → Output (u, v, p)
```

Xavier-uniform initialisation (gain = 5/3 for tanh).  All parameters stored
in `torch.float64` for numerical stability.

**Training** — `src/train.py`

| Phase  | Optimiser | Iterations | Learning rate |
|--------|-----------|-----------|---------------|
| 1      | Adam      | 20 000    | 1 × 10⁻³     |
| 2      | L-BFGS    | ≤ 1 000   | (line search) |

---

## Project structure

```
pinn-cavity-flow/
├── cfd/             # OpenFOAM and Code_Saturne case setup
├── src/
│   ├── model.py     # MLP architecture
│   ├── pinn.py      # PDE residuals + PINN loss class
│   ├── train.py     # Training loop (Adam → L-BFGS)
│   ├── utils.py     # PINN post-processing & visualisation
│   └── compare.py   # PINN vs OpenFOAM vs Code_Saturne comparison
├── notebooks/
│   └── explore.ipynb
├── results/         # Generated outputs (ignored in git)
├── tests/
│   └── test_pde_residuals.py
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# 1. Create and activate a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

---

## Running

**Train the model** (saves `results/model.pt` and `results/loss_history.npy`):

```bash
python src/train.py
```

**Generate visualisations** (saves five PNG files to `results/`):

```bash
python src/utils.py
```

**Compare against OpenFOAM and Code_Saturne** (expects CFD outputs present):

```bash
python src/compare.py
```

**Run the PDE residual sanity tests** (no trained model needed):

```bash
pytest tests/test_pde_residuals.py -v
```

**Interactive exploration** (requires Jupyter):

```bash
jupyter notebook notebooks/explore.ipynb
```

---

## Results

> _Training plots and comparison figures will be added here after a full run._

Expected outputs in `results/`:

| File | Description |
|------|-------------|
| `model.pt` | Trained network state-dict |
| `loss_history.npy` | Loss per iteration (Adam + L-BFGS) |
| `u_field.png` | u-velocity colour map |
| `v_field.png` | v-velocity colour map |
| `p_field.png` | Pressure colour map |
| `streamlines.png` | Streamlines coloured by speed |
| `centerline_profiles.png` | PINN vs Ghia et al. (1982) centreline data |
| `comparison/*.png` | PINN/OpenFOAM/Code_Saturne comparison figures |

The PINN solution is validated against the widely-used reference data of
Ghia, Ghia & Shin (1982) *"High-Re solutions for incompressible flow using
the Navier-Stokes equations and a multigrid method"*,
Journal of Computational Physics, 48, 387–411.

---

## References

- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019).
  Physics-informed neural networks. *Journal of Computational Physics*, 378, 686–707.
- Ghia, U., Ghia, K. N., & Shin, C. T. (1982).
  High-Re solutions for incompressible flow using the Navier-Stokes equations
  and a multigrid method. *Journal of Computational Physics*, 48(3), 387–411.
