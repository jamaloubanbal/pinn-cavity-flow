"""
PDE residual sanity checks — no trained model required.

Poiseuille flow is an exact solution to the 2-D incompressible Navier-Stokes
equations:

    u(x, y) = y (1 - y)
    v(x, y) = 0
    p(x, y) = -2 ν x

Verification (by hand):
  Continuity : du/dx + dv/dy = 0 + 0 = 0  ✓
  x-momentum : u·(0) + 0·(1-2y) + (-2ν) - ν·(0 + (-2)) = -2ν + 2ν = 0  ✓
  y-momentum : u·(0) + 0·(0) + 0 - ν·(0 + 0) = 0  ✓

All three residuals must be within tolerance 1e-5 (float64 arithmetic is
close to machine precision here).
"""

import os
import sys

# Ensure src/ is on the path regardless of how pytest is invoked
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import torch

from pinn import compute_pde_residuals

NU     = 0.01
TOL    = 1e-5
N_PTS  = 64         # grid points per axis  (64² = 4096 total)


# ---------------------------------------------------------------------------
# Fixture: analytical Poiseuille solution on a grid
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def poiseuille():
    """
    Build an interior grid and evaluate the Poiseuille analytical solution.

    x and y are leaf tensors with requires_grad=True so that autograd can
    compute spatial derivatives of the analytical expressions.
    """
    # Avoid boundaries (corners) — use a strictly interior grid
    coords = torch.linspace(0.05, 0.95, N_PTS, dtype=torch.float64)
    X_grid, Y_grid = torch.meshgrid(coords, coords, indexing="ij")

    # Clone into new leaf tensors so that requires_grad can be set
    x = X_grid.reshape(-1, 1).clone().detach().requires_grad_(True)
    y = Y_grid.reshape(-1, 1).clone().detach().requires_grad_(True)

    u = y * (1.0 - y)                                # shape (N², 1)
    v = torch.zeros(x.shape[0], 1, dtype=torch.float64)  # constant zero, no graph link
    p = -2.0 * NU * x                                # shape (N², 1)

    return x, y, u, v, p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_continuity_residual(poiseuille):
    """du/dx + dv/dy must vanish everywhere."""
    x, y, u, v, p = poiseuille
    res_cont, _, _ = compute_pde_residuals(x, y, u, v, p, NU)
    max_err = res_cont.abs().max().item()
    assert max_err < TOL, (
        f"Continuity residual too large: max |res| = {max_err:.2e}  (tol {TOL:.0e})"
    )


def test_x_momentum_residual(poiseuille):
    """u·du/dx + v·du/dy + dp/dx − ν(∂²u/∂x² + ∂²u/∂y²) must vanish."""
    x, y, u, v, p = poiseuille
    _, res_x, _ = compute_pde_residuals(x, y, u, v, p, NU)
    max_err = res_x.abs().max().item()
    assert max_err < TOL, (
        f"x-momentum residual too large: max |res| = {max_err:.2e}  (tol {TOL:.0e})"
    )


def test_y_momentum_residual(poiseuille):
    """u·dv/dx + v·dv/dy + dp/dy − ν(∂²v/∂x² + ∂²v/∂y²) must vanish."""
    x, y, u, v, p = poiseuille
    _, _, res_y = compute_pde_residuals(x, y, u, v, p, NU)
    max_err = res_y.abs().max().item()
    assert max_err < TOL, (
        f"y-momentum residual too large: max |res| = {max_err:.2e}  (tol {TOL:.0e})"
    )


# ---------------------------------------------------------------------------
# Standalone runner (no pytest required)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    coords  = torch.linspace(0.05, 0.95, N_PTS, dtype=torch.float64)
    X_grid, Y_grid = torch.meshgrid(coords, coords, indexing="ij")
    x = X_grid.reshape(-1, 1).clone().detach().requires_grad_(True)
    y = Y_grid.reshape(-1, 1).clone().detach().requires_grad_(True)
    u = y * (1.0 - y)
    v = torch.zeros(x.shape[0], 1, dtype=torch.float64)
    p = -2.0 * NU * x

    rc, rx, ry = compute_pde_residuals(x, y, u, v, p, NU)
    print(f"max |continuity residual|  : {rc.abs().max().item():.3e}")
    print(f"max |x-momentum residual|  : {rx.abs().max().item():.3e}")
    print(f"max |y-momentum residual|  : {ry.abs().max().item():.3e}")
    print(f"All within tolerance {TOL:.0e}? "
          f"{'YES' if max(rc.abs().max(), rx.abs().max(), ry.abs().max()) < TOL else 'NO'}")
