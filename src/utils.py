"""
Post-processing and visualisation for the lid-driven cavity PINN.

Run from the project root after training:
    python src/utils.py

Outputs (all saved to results/)
--------------------------------
u_field.png            — u-velocity  (imshow + colorbar)
v_field.png            — v-velocity  (imshow + colorbar)
p_field.png            — pressure    (imshow + colorbar)
streamlines.png        — streamline plot
centerline_profiles.png — u(x=0.5,y) and v(x,y=0.5) vs Ghia et al. (1982)
"""

import os
import sys

# Make sibling modules importable when run as  python src/utils.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import MLP


# ---------------------------------------------------------------------------
# Ghia et al. (1982) reference data for Re = 100
# "High-Re solutions for incompressible flow using the Navier-Stokes equations
#  and a multigrid method", Journal of Computational Physics, 48, 387-411.
# ---------------------------------------------------------------------------

# u-velocity along the vertical centreline  (x = 0.5)
GHIA_Y = np.array([
    0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813,
    0.4531, 0.5000, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609,
    0.9688, 0.9766, 1.0000,
])
GHIA_U = np.array([
     0.00000, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150, -0.15662,
    -0.21090, -0.20581, -0.13641,  0.00332,  0.23151,  0.68717,  0.73722,
     0.78871,  0.84123,  1.00000,
])

# v-velocity along the horizontal centreline  (y = 0.5)
GHIA_X = np.array([
    0.0000, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563, 0.2266,
    0.2344, 0.5000, 0.8047, 0.8594, 0.9063, 0.9453, 0.9531,
    0.9609, 0.9688, 1.0000,
])
GHIA_V = np.array([
     0.00000,  0.09233,  0.10091,  0.10890,  0.12317,  0.16077,  0.17507,
     0.17527,  0.05454, -0.24533, -0.22445, -0.16914, -0.10313, -0.08864,
    -0.07391, -0.05906,  0.00000,
])


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(path: str = "results/model.pt") -> MLP:
    """Load a trained MLP from *path* and set it to eval mode."""
    model = MLP()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Grid evaluation
# ---------------------------------------------------------------------------

def evaluate_on_grid(model: MLP, n: int = 100):
    """
    Evaluate *model* on an n×n uniform grid over [0, 1]².

    Returns
    -------
    X, Y : (n, n) numpy arrays of coordinates  (indexing='ij')
    U, V, P : (n, n) numpy arrays of field values
    """
    x_lin = np.linspace(0.0, 1.0, n)
    y_lin = np.linspace(0.0, 1.0, n)
    X, Y  = np.meshgrid(x_lin, y_lin, indexing="ij")

    xy_flat = torch.tensor(
        np.stack([X.ravel(), Y.ravel()], axis=1),
        dtype=torch.float64,
    )

    with torch.no_grad():
        uvp = model(xy_flat).numpy()

    U = uvp[:, 0].reshape(n, n)
    V = uvp[:, 1].reshape(n, n)
    P = uvp[:, 2].reshape(n, n)

    return X, Y, U, V, P


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _save_field(data: np.ndarray, title: str, path: str, cmap: str = "RdBu_r") -> None:
    """
    Render a 2-D field stored with indexing='ij' as an imshow plot.

    data shape : (n_x, n_y)  →  transposed to (n_y, n_x) for imshow
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        data.T,
        extent=[0, 1, 0, 1],
        origin="lower",
        cmap=cmap,
        aspect="equal",
    )
    plt.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  saved  {path}")


def plot_u_field(X, Y, U, out_dir: str = "results") -> None:
    _save_field(U, "u-velocity", os.path.join(out_dir, "u_field.png"))


def plot_v_field(X, Y, V, out_dir: str = "results") -> None:
    _save_field(V, "v-velocity", os.path.join(out_dir, "v_field.png"))


def plot_p_field(X, Y, P, out_dir: str = "results") -> None:
    _save_field(P, "Pressure", os.path.join(out_dir, "p_field.png"), cmap="viridis")


def plot_streamlines(X, Y, U, V, out_dir: str = "results") -> None:
    """
    Streamline plot coloured by speed magnitude.

    With indexing='ij':
      X[:, 0] = x_lin,   Y[0, :] = y_lin
      streamplot expects u/v as (n_y, n_x), so we transpose.
    """
    x_lin = X[:, 0]
    y_lin = Y[0, :]
    speed = np.sqrt(U ** 2 + V ** 2)

    fig, ax = plt.subplots(figsize=(6, 5))
    strm = ax.streamplot(
        x_lin, y_lin,
        U.T, V.T,
        color=speed.T,
        cmap="viridis",
        density=1.5,
        linewidth=0.8,
        arrowsize=0.8,
    )
    plt.colorbar(strm.lines, ax=ax, label="Speed  |u|")
    ax.set_title("Streamlines")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    path = os.path.join(out_dir, "streamlines.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  saved  {path}")


def plot_centerline_profiles(X, Y, U, V, out_dir: str = "results") -> None:
    """
    Left panel  : u(x=0.5, y)  overlaid with Ghia 1982 Re=100 data
    Right panel : v(x, y=0.5)  overlaid with Ghia 1982 Re=100 data

    With indexing='ij' and n=100 points:
      mid = 50  →  x_lin[50] ≈ 0.505  (closest to 0.5)
      U[mid, :] = u values at x≈0.5 for all y
      V[:, mid] = v values at y≈0.5 for all x
    """
    n   = X.shape[0]
    mid = n // 2

    y_lin    = Y[mid, :]    # y values  (same for every row)
    u_center = U[mid, :]    # u at x ≈ 0.5

    x_lin    = X[:, mid]    # x values  (same for every column)
    v_center = V[:, mid]    # v at y ≈ 0.5

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # --- u centreline ---
    ax1.plot(u_center, y_lin, "b-", linewidth=2, label="PINN")
    ax1.scatter(GHIA_U, GHIA_Y, color="red", s=40, zorder=5,
                label="Ghia et al. (1982)")
    ax1.set_xlabel("u")
    ax1.set_ylabel("y")
    ax1.set_title("u-velocity at x = 0.5")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- v centreline ---
    ax2.plot(x_lin, v_center, "b-", linewidth=2, label="PINN")
    ax2.scatter(GHIA_X, GHIA_V, color="red", s=40, zorder=5,
                label="Ghia et al. (1982)")
    ax2.set_xlabel("x")
    ax2.set_ylabel("v")
    ax2.set_title("v-velocity at y = 0.5")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "centerline_profiles.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  saved  {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)

    print("Loading model …")
    model = load_model(os.path.join(out_dir, "model.pt"))

    print("Evaluating on 100×100 grid …")
    X, Y, U, V, P = evaluate_on_grid(model, n=100)

    print("Generating plots …")
    plot_u_field(X, Y, U, out_dir)
    plot_v_field(X, Y, V, out_dir)
    plot_p_field(X, Y, P, out_dir)
    plot_streamlines(X, Y, U, V, out_dir)
    plot_centerline_profiles(X, Y, U, V, out_dir)

    print("Done.")


if __name__ == "__main__":
    main()
