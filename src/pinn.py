"""
Physics-Informed Neural Network for 2-D lid-driven cavity flow.

Governing equations (incompressible Navier-Stokes, Re = 100, nu = 0.01):
  Continuity  :  du/dx + dv/dy = 0
  x-momentum  :  u·du/dx + v·du/dy + dp/dx - nu·(d²u/dx² + d²u/dy²) = 0
  y-momentum  :  u·dv/dx + v·dv/dy + dp/dy - nu·(d²v/dx² + d²v/dy²) = 0

All derivatives are computed via torch.autograd.grad with create_graph=True
so that the loss can be differentiated a second time w.r.t. network weights.
"""

import torch
from typing import Tuple


# ---------------------------------------------------------------------------
# Private helper
# ---------------------------------------------------------------------------

def _grad(output: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """
    Compute elementwise partial derivative d(output)/d(inputs) via autograd.

    Uses grad_outputs=ones to obtain the elementwise (diagonal Jacobian) result.
    Returns a zero tensor of the same shape as *inputs* in two cases:
      1. The output has no grad_fn and requires_grad=False (e.g. a constant
         zero tensor used for v in the Poiseuille test).  Calling
         autograd.grad on such tensors raises a RuntimeError in PyTorch ≥ 2.x.
      2. The output has a grad_fn but does not depend on *inputs*
         (allow_unused=True returns None).
    """
    # Fast-path: output is not part of any autograd graph — gradient is zero
    if not output.requires_grad and output.grad_fn is None:
        return torch.zeros_like(inputs)

    g = torch.autograd.grad(
        output,
        inputs,
        grad_outputs=torch.ones_like(output),
        create_graph=True,
        allow_unused=True,
    )[0]
    if g is None:
        return torch.zeros_like(inputs)
    return g


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_pde_residuals(
    x: torch.Tensor,
    y: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    nu: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the three Navier-Stokes residuals at given points.

    Args:
        x : (N, 1) x-coordinates, leaf tensor with requires_grad=True
        y : (N, 1) y-coordinates, leaf tensor with requires_grad=True
        u : (N, 1) x-velocity prediction (must be in the autograd graph of x, y)
        v : (N, 1) y-velocity prediction
        p : (N, 1) pressure prediction
        nu: kinematic viscosity

    Returns:
        (res_continuity, res_x_momentum, res_y_momentum), each (N, 1)
    """
    # --- first-order derivatives ---
    u_x = _grad(u, x)
    u_y = _grad(u, y)
    v_x = _grad(v, x)
    v_y = _grad(v, y)
    p_x = _grad(p, x)
    p_y = _grad(p, y)

    # --- second-order derivatives ---
    u_xx = _grad(u_x, x)
    u_yy = _grad(u_y, y)
    v_xx = _grad(v_x, x)
    v_yy = _grad(v_y, y)

    # --- residuals ---
    res_continuity  = u_x + v_y
    res_x_momentum  = u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
    res_y_momentum  = u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)

    return res_continuity, res_x_momentum, res_y_momentum


class PINN:
    """
    Wraps an MLP model with physics-informed loss functions.

    Loss:  L = 1.0 * L_pde + 10.0 * L_bc

    Boundary conditions (lid-driven cavity on [0,1]²):
      - Left   wall (x=0): u = v = 0   (no-slip)
      - Right  wall (x=1): u = v = 0   (no-slip)
      - Bottom wall (y=0): u = v = 0   (no-slip)
      - Top    lid  (y=1): u = 1, v = 0 (moving lid)
    """

    def __init__(self, model: torch.nn.Module, nu: float = 0.01) -> None:
        self.model = model
        self.nu = nu

    # ------------------------------------------------------------------
    def compute_pde_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        PDE residual loss at collocation points.

        Args:
            x: (N, 1) leaf tensor with requires_grad=True
            y: (N, 1) leaf tensor with requires_grad=True

        x and y must be **separate leaf tensors** (not slices of a shared
        tensor).  The model is called on torch.cat([x, y], dim=1), so that
        the autograd graph runs  u → model(xy) → cat([x, y]) → x, y.
        Slicing a single requires_grad tensor would cut that path.
        """
        xy = torch.cat([x, y], dim=1)  # CatBackward0 — keeps x, y as leaves
        uvp = self.model(xy)
        u = uvp[:, 0:1]
        v = uvp[:, 1:2]
        p = uvp[:, 2:3]

        res_cont, res_x, res_y = compute_pde_residuals(x, y, u, v, p, self.nu)

        return (res_cont ** 2).mean() + (res_x ** 2).mean() + (res_y ** 2).mean()

    # ------------------------------------------------------------------
    def compute_bc_loss(
        self,
        xy_left  : torch.Tensor,
        xy_right : torch.Tensor,
        xy_bottom: torch.Tensor,
        xy_top   : torch.Tensor,
    ) -> torch.Tensor:
        """
        Boundary-condition loss on the four walls.

        No requires_grad needed for BC tensors (no spatial derivatives taken).
        """
        # Left wall  (x=0): u=v=0
        out_l = self.model(xy_left)
        loss_left = (out_l[:, 0] ** 2 + out_l[:, 1] ** 2).mean()

        # Right wall (x=1): u=v=0
        out_r = self.model(xy_right)
        loss_right = (out_r[:, 0] ** 2 + out_r[:, 1] ** 2).mean()

        # Bottom wall (y=0): u=v=0
        out_b = self.model(xy_bottom)
        loss_bottom = (out_b[:, 0] ** 2 + out_b[:, 1] ** 2).mean()

        # Top lid    (y=1): u=1, v=0
        out_t = self.model(xy_top)
        loss_top = ((out_t[:, 0] - 1.0) ** 2 + out_t[:, 1] ** 2).mean()

        return loss_left + loss_right + loss_bottom + loss_top

    # ------------------------------------------------------------------
    def total_loss(
        self,
        x_col    : torch.Tensor,
        y_col    : torch.Tensor,
        xy_left  : torch.Tensor,
        xy_right : torch.Tensor,
        xy_bottom: torch.Tensor,
        xy_top   : torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (total_loss, pde_loss, bc_loss).
        Weights: L = 1.0 * L_pde + 10.0 * L_bc

        Args:
            x_col, y_col: separate (N, 1) leaf tensors, requires_grad=True
            xy_left/right/bottom/top: (M, 2) boundary point tensors
        """
        loss_pde = self.compute_pde_loss(x_col, y_col)
        loss_bc  = self.compute_bc_loss(xy_left, xy_right, xy_bottom, xy_top)
        total    = 1.0 * loss_pde + 10.0 * loss_bc
        return total, loss_pde, loss_bc
