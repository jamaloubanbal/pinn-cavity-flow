"""
Training script for the lid-driven cavity PINN.

Run from the project root:
    python src/train.py

Phases
------
1. Adam  : lr = 1e-3, 20 000 iterations  (fast progress from random init)
2. L-BFGS: max 1 000 iterations          (fine-grained convergence)

Outputs
-------
results/model.pt          — final model state-dict
results/loss_history.npy  — concatenated Adam + L-BFGS loss values
"""

import os
import sys

# Make sibling modules importable when run as  python src/train.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from model import MLP
from pinn import PINN


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sample_collocation(n: int):
    """
    Sample *n* random points uniformly in [0, 1]², returned as two **separate**
    leaf tensors (x_col, y_col) each of shape (n, 1) with requires_grad=True.

    Why separate tensors?  The PINN computes spatial derivatives via
    torch.autograd.grad(u, x_col).  For this to trace correctly, x_col and
    y_col must be *leaves* in the autograd graph.  If we sliced a single
    tensor (x = xy[:, 0:1]), the slice would not be a leaf and grad() would
    return zeros even though the model output depends on x.
    """
    x = torch.rand(n, 1, dtype=torch.float64, requires_grad=True)
    y = torch.rand(n, 1, dtype=torch.float64, requires_grad=True)
    return x, y


def sample_boundary(n_per_wall: int):
    """
    Uniformly sample *n_per_wall* points on each of the four cavity walls.

    Returns
    -------
    xy_left, xy_right, xy_bottom, xy_top — each (n_per_wall, 2), float64
    """
    t = torch.linspace(0.0, 1.0, n_per_wall, dtype=torch.float64).reshape(-1, 1)

    xy_left   = torch.cat([torch.zeros_like(t), t              ], dim=1)  # x=0
    xy_right  = torch.cat([torch.ones_like(t),  t              ], dim=1)  # x=1
    xy_bottom = torch.cat([t,                   torch.zeros_like(t)], dim=1)  # y=0
    xy_top    = torch.cat([t,                   torch.ones_like(t) ], dim=1)  # y=1

    return xy_left, xy_right, xy_bottom, xy_top


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def train() -> None:
    torch.set_default_dtype(torch.float64)

    os.makedirs("results", exist_ok=True)

    # ------------------------------------------------------------------ model
    model = MLP()
    pinn  = PINN(model, nu=0.01)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters : {total_params:,}")

    # ------------------------------------------------------------------ data
    N_COL      = 10_000
    N_PER_WALL = 200

    x_col, y_col  = sample_collocation(N_COL)
    xy_left, xy_right, xy_bottom, xy_top = sample_boundary(N_PER_WALL)

    loss_history: list[float] = []

    # ==================================================================
    # Phase 1 — Adam
    # ==================================================================
    print()
    print("=" * 62)
    print("Phase 1 : Adam  (20 000 iterations, lr = 1e-3)")
    print("=" * 62)

    optimizer_adam = torch.optim.Adam(model.parameters(), lr=1e-3)

    for step in range(1, 20_001):
        optimizer_adam.zero_grad()

        loss, loss_pde, loss_bc = pinn.total_loss(
            x_col, y_col, xy_left, xy_right, xy_bottom, xy_top
        )
        loss.backward()
        optimizer_adam.step()

        loss_history.append(loss.item())

        if step % 500 == 0:
            print(
                f"  step {step:6d} | total {loss.item():.4e} "
                f"| pde {loss_pde.item():.4e} | bc {loss_bc.item():.4e}"
            )

    # ==================================================================
    # Phase 2 — L-BFGS
    # ==================================================================
    print()
    print("=" * 62)
    print("Phase 2 : L-BFGS  (max 1 000 iterations, strong Wolfe)")
    print("=" * 62)

    optimizer_lbfgs = torch.optim.LBFGS(
        model.parameters(),
        max_iter=1_000,
        tolerance_grad=1e-9,
        tolerance_change=1e-11,
        history_size=50,
        line_search_fn="strong_wolfe",
    )

    lbfgs_losses: list[float] = []

    def closure() -> torch.Tensor:
        optimizer_lbfgs.zero_grad()
        loss, _pde, _bc = pinn.total_loss(
            x_col, y_col, xy_left, xy_right, xy_bottom, xy_top
        )
        loss.backward()
        lbfgs_losses.append(loss.item())
        return loss

    optimizer_lbfgs.step(closure)

    loss_history.extend(lbfgs_losses)

    n_evals = len(lbfgs_losses)
    print(f"  L-BFGS finished — {n_evals} function evaluations")
    print(f"  Final loss : {lbfgs_losses[-1]:.4e}")

    # ==================================================================
    # Save
    # ==================================================================
    torch.save(model.state_dict(), "results/model.pt")
    np.save("results/loss_history.npy", np.array(loss_history))

    print()
    print("Saved  results/model.pt")
    print("Saved  results/loss_history.npy")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train()
