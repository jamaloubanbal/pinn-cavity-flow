"""
Four-way comparison: PINN  vs  OpenFOAM (icoFoam)  vs  Code_Saturne  vs  Ghia et al. (1982)
Lid-driven cavity, Re = 100.

Run from the project root after all solvers have finished:
    python src/compare.py

Outputs (saved to results/comparison/)
---------------------------------------
centreline_comparison.png  — u(x=0.5,y) and v(x,y=0.5)  all four sources
field_comparison.png       — u, v, p fields side-by-side (PINN, OF, CS)
error_map.png              — |PINN - OF| and |PINN - CS| on the [0,1]² grid
streamline_comparison.png  — streamlines for PINN, OF, and CS
"""

import os
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

from model import MLP
from utils import load_model, evaluate_on_grid, GHIA_Y, GHIA_U, GHIA_X, GHIA_V


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OF_CASE  = os.path.join("cfd", "openfoam_cavity")
CS_RESU  = os.path.join("cfd", "cs_cavity", "cavity", "RESU")
OUT_DIR  = os.path.join("results", "comparison")


# ---------------------------------------------------------------------------
# OpenFOAM CSV reader
# ---------------------------------------------------------------------------

def _griddata_with_nearest(points, values, X, Y):
    """Linear interpolation with nearest fallback to fill boundary NaNs."""
    arr = griddata(points, values, (X, Y), method="linear")
    if np.isnan(arr).any():
        arr_nn = griddata(points, values, (X, Y), method="nearest")
        arr = np.where(np.isnan(arr), arr_nn, arr)
    return arr

def _latest_sample_dir(base: str) -> str:
    """Return path to the highest-numbered time directory under base."""
    dirs = glob.glob(os.path.join(base, "*"))
    time_dirs = []
    for d in dirs:
        try:
            time_dirs.append((float(os.path.basename(d)), d))
        except ValueError:
            pass
    if not time_dirs:
        raise FileNotFoundError(f"No time directories found in {base}")
    return max(time_dirs)[1]


def read_of_centrelines(of_case: str):
    """
    Read centreline CSV files written by OpenFOAM v2512 'sets' function object.

    OF v2512 writes a single combined file per set containing ALL requested
    fields.  With fields=(U p) and axis=y/x the format is:

      verticalCentreline_p_U.csv   →  header row, then: y, p, U_0, U_1, U_2
      horizontalCentreline_p_U.csv →  header row, then: x, p, U_0, U_1, U_2

    Returns
    -------
    y_vc, u_vc : arrays along vertical centreline   (x = 0.5)
    x_hc, v_hc : arrays along horizontal centreline (y = 0.5)
    """
    sample_base = os.path.join(of_case, "postProcessing", "sample")
    t_dir = _latest_sample_dir(sample_base)

    # Vertical centreline  (axis=y)
    vc_file = os.path.join(t_dir, "verticalCentreline_p_U.csv")
    data_vc = np.loadtxt(vc_file, delimiter=",", skiprows=1)
    # columns: y, p, U_0(=u), U_1(=v), U_2
    y_vc = data_vc[:, 0]
    u_vc = data_vc[:, 2]   # Ux = u-component

    # Horizontal centreline  (axis=x)
    hc_file = os.path.join(t_dir, "horizontalCentreline_p_U.csv")
    data_hc = np.loadtxt(hc_file, delimiter=",", skiprows=1)
    # columns: x, p, U_0(=u), U_1(=v), U_2
    x_hc = data_hc[:, 0]
    v_hc = data_hc[:, 3]   # Uy = v-component

    print(f"  OF sample dir : {t_dir}")
    print(f"  Vertical CL   : {len(y_vc)} points")
    print(f"  Horizontal CL : {len(x_hc)} points")
    return y_vc, u_vc, x_hc, v_hc


# ---------------------------------------------------------------------------
# Code_Saturne centreline reader
# ---------------------------------------------------------------------------

def _latest_cs_resu_dir(resu_base: str) -> str:
    """Return path to the most-recently modified RESU sub-directory."""
    dirs = [
        d for d in glob.glob(os.path.join(resu_base, "*"))
        if os.path.isdir(d) and
           os.path.exists(os.path.join(d, "vertical_centreline.csv"))
    ]
    if not dirs:
        raise FileNotFoundError(
            f"No completed CS RESU directory (with centreline CSVs) under {resu_base}"
        )
    return max(dirs, key=os.path.getmtime)


def read_cs_centrelines(cs_resu: str):
    """
    Read centreline CSV files written by cs_user_extra_operations_finalize.

    Each file has a header row and two rows per coordinate value (from the
    pair of x-columns / y-rows captured within tolerance of the centreline).
    We average the duplicates.

    vertical_centreline.csv   : y, u, v
    horizontal_centreline.csv : x, u, v

    Returns
    -------
    y_vc, u_vc : arrays along vertical centreline   (x ≈ 0.5)
    x_hc, v_hc : arrays along horizontal centreline (y ≈ 0.5)
    """
    run_dir = _latest_cs_resu_dir(cs_resu)

    def _read_avg(path, coord_col, val_col):
        data = np.loadtxt(path, delimiter=",", skiprows=1)
        coords = data[:, coord_col]
        vals   = data[:, val_col]
        u_coords = np.unique(coords)
        v_avg = np.array([vals[coords == c].mean() for c in u_coords])
        return u_coords, v_avg

    vc_path = os.path.join(run_dir, "vertical_centreline.csv")
    hc_path = os.path.join(run_dir, "horizontal_centreline.csv")

    y_vc, u_vc = _read_avg(vc_path, 0, 1)   # col 0=y, col 1=u
    x_hc, v_hc = _read_avg(hc_path, 0, 2)   # col 0=x, col 2=v

    print(f"  CS RESU dir   : {run_dir}")
    print(f"  Vertical CL   : {len(y_vc)} points")
    print(f"  Horizontal CL : {len(x_hc)} points")
    return y_vc, u_vc, x_hc, v_hc


# ---------------------------------------------------------------------------
# Code_Saturne 2-D field reader (EnSight Gold C Binary)
# ---------------------------------------------------------------------------

def _read_ensight_c80(fh) -> str:
    """Read one fixed-width 80-byte C binary string field."""
    raw = fh.read(80)
    if len(raw) != 80:
        raise EOFError("Unexpected EOF while reading EnSight C binary string")
    return raw.split(b"\x00", 1)[0].decode("ascii", errors="ignore").strip()


def _read_cs_geometry(geo_path: str):
    """
    Read EnSight Gold geometry and return cell centres for hexa8 elements.

    Returns
    -------
    x_cc, y_cc, n_elem
    """
    with open(geo_path, "rb") as fh:
        _ = _read_ensight_c80(fh)  # C Binary
        _ = _read_ensight_c80(fh)  # geometry title
        _ = _read_ensight_c80(fh)  # description
        _ = _read_ensight_c80(fh)  # node id handling
        _ = _read_ensight_c80(fh)  # element id handling

        if _read_ensight_c80(fh).lower() != "part":
            raise ValueError(f"Invalid EnSight geometry (missing 'part') in {geo_path}")
        _ = np.fromfile(fh, dtype="<i4", count=1)[0]  # part id
        _ = _read_ensight_c80(fh)  # part name

        if _read_ensight_c80(fh).lower() != "coordinates":
            raise ValueError(f"Invalid EnSight geometry (missing 'coordinates') in {geo_path}")
        n_nodes = int(np.fromfile(fh, dtype="<i4", count=1)[0])

        x = np.fromfile(fh, dtype="<f4", count=n_nodes)
        y = np.fromfile(fh, dtype="<f4", count=n_nodes)
        _ = np.fromfile(fh, dtype="<f4", count=n_nodes)  # z not needed

        elem_type = _read_ensight_c80(fh).lower()
        if elem_type != "hexa8":
            raise ValueError(f"Unsupported CS element type '{elem_type}' in {geo_path}")
        n_elem = int(np.fromfile(fh, dtype="<i4", count=1)[0])

        conn = np.fromfile(fh, dtype="<i4", count=n_elem * 8).reshape(n_elem, 8) - 1

    x_cc = x[conn].mean(axis=1)
    y_cc = y[conn].mean(axis=1)
    return x_cc, y_cc, n_elem


def _read_cs_scalar_per_elem(path: str, n_elem: int):
    """Read EnSight Gold scalar-per-element field."""
    with open(path, "rb") as fh:
        _ = _read_ensight_c80(fh)  # variable name + time
        if _read_ensight_c80(fh).lower() != "part":
            raise ValueError(f"Invalid EnSight scalar file (missing 'part') in {path}")
        _ = np.fromfile(fh, dtype="<i4", count=1)[0]  # part id
        _ = _read_ensight_c80(fh)  # element type
        vals = np.fromfile(fh, dtype="<f4", count=n_elem)

    if len(vals) != n_elem:
        raise ValueError(f"Expected {n_elem} values in {path}, got {len(vals)}")
    return vals


def _read_cs_vector_per_elem(path: str, n_elem: int):
    """Read EnSight Gold vector-per-element field."""
    with open(path, "rb") as fh:
        _ = _read_ensight_c80(fh)  # variable name + time
        if _read_ensight_c80(fh).lower() != "part":
            raise ValueError(f"Invalid EnSight vector file (missing 'part') in {path}")
        _ = np.fromfile(fh, dtype="<i4", count=1)[0]  # part id
        _ = _read_ensight_c80(fh)  # element type

        u = np.fromfile(fh, dtype="<f4", count=n_elem)
        v = np.fromfile(fh, dtype="<f4", count=n_elem)
        w = np.fromfile(fh, dtype="<f4", count=n_elem)

    if len(u) != n_elem or len(v) != n_elem or len(w) != n_elem:
        raise ValueError(f"Expected {n_elem} vectors in {path}")
    return u, v, w


def read_cs_2d_fields(cs_resu: str, n_grid: int = 100):
    """
    Read Code_Saturne EnSight fields and interpolate onto a uniform n_grid grid.

    Returns
    -------
    X_cs, Y_cs : (n_grid, n_grid) coordinate arrays
    U_cs, V_cs, P_cs : (n_grid, n_grid) field arrays
    """
    run_dir = _latest_cs_resu_dir(cs_resu)
    post_dir = os.path.join(run_dir, "postprocessing")

    geo_path = os.path.join(post_dir, "results_fluid_domain.geo")
    vel_path = os.path.join(post_dir, "results_fluid_domain.velocity.00001")
    pre_path = os.path.join(post_dir, "results_fluid_domain.pressure.00001")

    x_cc, y_cc, n_elem = _read_cs_geometry(geo_path)
    u_cc, v_cc, _ = _read_cs_vector_per_elem(vel_path, n_elem)
    p_cc = _read_cs_scalar_per_elem(pre_path, n_elem)

    x_lin = np.linspace(0, 1, n_grid)
    y_lin = np.linspace(0, 1, n_grid)
    X, Y = np.meshgrid(x_lin, y_lin, indexing="ij")
    pts = np.column_stack([x_cc, y_cc])

    U_cs = _griddata_with_nearest(pts, u_cc, X, Y)
    V_cs = _griddata_with_nearest(pts, v_cc, X, Y)
    P_cs = _griddata_with_nearest(pts, p_cc, X, Y)

    print(f"  CS field run   : {run_dir}")
    print(f"  CS elements    : {n_elem}")
    return X, Y, U_cs, V_cs, P_cs


# ---------------------------------------------------------------------------
# OpenFOAM 2-D field reader
# ---------------------------------------------------------------------------

def _parse_of_internal_field(filepath: str, field_type: str):
    """
    Parse an OpenFOAM ASCII internal field (scalar or vector).

    Parameters
    ----------
    filepath   : path to the field file (e.g. "50/U")
    field_type : "vector" or "scalar"

    Returns
    -------
    numpy array of shape (N,) for scalar or (N, 3) for vector.
    """
    with open(filepath) as fh:
        text = fh.read()

    # Find the internalField block
    idx = text.find("internalField")
    if idx == -1:
        raise ValueError(f"internalField not found in {filepath}")

    block = text[idx:]

    # Extract number of entries
    import re
    m = re.search(r"(\d+)\s*\n\s*\(", block)
    if not m:
        raise ValueError(f"Cannot parse cell count in {filepath}")
    n_cells = int(m.group(1))

    # Extract the parenthesised data block
    paren_start = block.index("(", m.start()) + 1
    paren_end   = block.index("\n)", paren_start)
    raw = block[paren_start:paren_end].strip()

    if field_type == "vector":
        # Each entry is "(vx vy vz)"
        entries = re.findall(r"\(([^)]+)\)", raw)
        data = np.array([list(map(float, e.split())) for e in entries])
    else:
        data = np.array(list(map(float, raw.split())))

    if len(data) != n_cells:
        raise ValueError(
            f"Expected {n_cells} entries, got {len(data)} in {filepath}"
        )
    return data


def read_of_2d_fields(of_case: str, n_grid: int = 100):
    """
    Read U and p from the latest OpenFOAM time directory and interpolate
    onto a uniform n_grid × n_grid mesh over [0,1]².

    Returns
    -------
    X_of, Y_of : (n_grid, n_grid) coordinate arrays  (indexing='ij')
    U_of, V_of, P_of : (n_grid, n_grid) field arrays
    """
    # Find latest time directory
    time_dirs = []
    for entry in os.listdir(of_case):
        try:
            t = float(entry)
            if t > 0:
                time_dirs.append(t)
        except ValueError:
            pass
    if not time_dirs:
        raise FileNotFoundError(f"No solution time directories in {of_case}")
    t_latest = max(time_dirs)
    t_dir    = os.path.join(of_case, f"{t_latest:g}")
    print(f"  OF latest time : {t_latest}  →  {t_dir}")

    # Cell centres written by writeCellCentres function object
    cc_dir = os.path.join(of_case, "postProcessing", "writeCellCentres",
                          f"{t_latest:g}")
    if not os.path.isdir(cc_dir):
        # Fall back to time directory itself (older OF versions put C there)
        cc_dir = t_dir

    cc_path = os.path.join(cc_dir, "C")
    if not os.path.exists(cc_path):
        raise FileNotFoundError(
            f"Cell centre file not found at {cc_path}\n"
            "Make sure writeCellCentres function object ran successfully."
        )

    # Read cell centres and fields
    cc  = _parse_of_internal_field(cc_path, "vector")   # (N, 3)
    uvw = _parse_of_internal_field(os.path.join(t_dir, "U"), "vector")
    p   = _parse_of_internal_field(os.path.join(t_dir, "p"), "scalar")

    x_cc = cc[:, 0]
    y_cc = cc[:, 1]
    u_cc = uvw[:, 0]
    v_cc = uvw[:, 1]
    p_cc = p

    # Interpolate onto uniform grid
    x_lin = np.linspace(0, 1, n_grid)
    y_lin = np.linspace(0, 1, n_grid)
    X, Y  = np.meshgrid(x_lin, y_lin, indexing="ij")
    pts   = np.column_stack([x_cc, y_cc])

    U_of = _griddata_with_nearest(pts, u_cc, X, Y)
    V_of = _griddata_with_nearest(pts, v_cc, X, Y)
    P_of = _griddata_with_nearest(pts, p_cc, X, Y)

    print(f"  OF cells read  : {len(x_cc)}")
    return X, Y, U_of, V_of, P_of


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_centreline_comparison(
    y_vc, u_vc,             # OF vertical centreline
    x_hc, v_hc,             # OF horizontal centreline
    X_pinn, Y_pinn,         # PINN grid
    U_pinn, V_pinn,
    out_dir: str,
    y_cs=None, u_cs=None,   # CS vertical centreline (optional)
    x_cs=None, v_cs=None,   # CS horizontal centreline (optional)
) -> None:
    n = X_pinn.shape[0]
    mid = n // 2

    y_pinn = Y_pinn[mid, :]
    u_pinn = U_pinn[mid, :]
    x_pinn = X_pinn[:, mid]
    v_pinn = V_pinn[:, mid]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- u centreline ---
    ax1.plot(u_pinn, y_pinn, "b-",  lw=2,   label="PINN")
    ax1.plot(u_vc,   y_vc,   "g--", lw=2,   label="OpenFOAM (icoFoam)")
    if u_cs is not None:
        ax1.plot(u_cs, y_cs, "m-.", lw=2,   label="Code_Saturne")
    ax1.scatter(GHIA_U, GHIA_Y, color="red", s=60, zorder=5,
                label="Ghia et al. (1982)", marker="o")
    ax1.set_xlabel("u", fontsize=12)
    ax1.set_ylabel("y", fontsize=12)
    ax1.set_title("u-velocity at x = 0.5  (Re = 100)", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # --- v centreline ---
    ax2.plot(x_pinn, v_pinn, "b-",  lw=2,   label="PINN")
    ax2.plot(x_hc,   v_hc,   "g--", lw=2,   label="OpenFOAM (icoFoam)")
    if v_cs is not None:
        ax2.plot(x_cs, v_cs, "m-.", lw=2,   label="Code_Saturne")
    ax2.scatter(GHIA_X, GHIA_V, color="red", s=60, zorder=5,
                label="Ghia et al. (1982)", marker="o")
    ax2.set_xlabel("x", fontsize=12)
    ax2.set_ylabel("v", fontsize=12)
    ax2.set_title("v-velocity at y = 0.5  (Re = 100)", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.suptitle("PINN vs OpenFOAM vs Code_Saturne vs Ghia et al. (1982) — Re = 100",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    path = os.path.join(out_dir, "centreline_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved  {path}")


def plot_field_comparison(
    X, Y,
    U_pinn, V_pinn, P_pinn,
    U_of,   V_of,   P_of,
    out_dir: str,
    U_cs=None, V_cs=None, P_cs=None,
) -> None:
    """Side-by-side field comparison for PINN/OpenFOAM/(optional) Code_Saturne."""

    methods = [("PINN", U_pinn, V_pinn, P_pinn), ("OpenFOAM", U_of, V_of, P_of)]
    if U_cs is not None and V_cs is not None and P_cs is not None:
        methods.append(("Code_Saturne", U_cs, V_cs, P_cs))

    fields = [
        ("u-velocity", "RdBu_r", [m[1] for m in methods]),
        ("v-velocity", "RdBu_r", [m[2] for m in methods]),
        ("Pressure", "viridis", [m[3] for m in methods]),
    ]

    n_cols = len(methods)
    fig, axes = plt.subplots(3, n_cols, figsize=(6 * n_cols, 14))
    if n_cols == 1:
        axes = np.array(axes).reshape(3, 1)

    for row, (label, cmap, arrs) in enumerate(fields):
        kw = dict(extent=[0, 1, 0, 1], origin="lower", aspect="equal", cmap=cmap)

        # Shared colour scale
        vmin = min(np.nanmin(a) for a in arrs)
        vmax = max(np.nanmax(a) for a in arrs)
        kw.update(vmin=vmin, vmax=vmax)

        for col, (method_name, u_m, v_m, p_m) in enumerate(methods):
            arr = [u_m, v_m, p_m][row]
            im = axes[row, col].imshow(arr.T, **kw)
            plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
            axes[row, col].set_title(f"{method_name} — {label}", fontsize=11)
            axes[row, col].set_xlabel("x")
            axes[row, col].set_ylabel("y")

    title = "Field comparison: PINN vs OpenFOAM (Re = 100)"
    if n_cols == 3:
        title = "Field comparison: PINN vs OpenFOAM vs Code_Saturne (Re = 100)"
    plt.suptitle(title,
                 fontsize=13, y=1.002)
    plt.tight_layout()
    path = os.path.join(out_dir, "field_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved  {path}")


def plot_error_maps(
    X, Y,
    U_pinn, V_pinn, P_pinn,
    U_of,   V_of,   P_of,
    out_dir: str,
    U_cs=None, V_cs=None, P_cs=None,
) -> None:
    """Absolute pointwise error maps against OpenFOAM and optional Code_Saturne."""
    refs = [("OpenFOAM", U_of, V_of, P_of)]
    if U_cs is not None and V_cs is not None and P_cs is not None:
        refs.append(("Code_Saturne", U_cs, V_cs, P_cs))

    n_rows = len(refs)
    fig, axes = plt.subplots(n_rows, 3, figsize=(16, 4 * n_rows))
    if n_rows == 1:
        axes = np.array([axes])

    kw = dict(extent=[0, 1, 0, 1], origin="lower", aspect="equal", cmap="hot_r")

    P_pinn_n = P_pinn - np.nanmean(P_pinn)

    for row, (ref_name, U_ref, V_ref, P_ref) in enumerate(refs):
        P_ref_n = P_ref - np.nanmean(P_ref)
        errs = [
            np.abs(U_pinn - U_ref),
            np.abs(V_pinn - V_ref),
            np.abs(P_pinn_n - P_ref_n),
        ]
        titles = [f"|Δu| vs {ref_name}", f"|Δv| vs {ref_name}", f"|Δp| vs {ref_name} (zero-mean)"]

        for col, (err, title) in enumerate(zip(errs, titles)):
            im = axes[row, col].imshow(err.T, **kw)
            plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
            axes[row, col].set_title(title, fontsize=11)
            axes[row, col].set_xlabel("x")
            axes[row, col].set_ylabel("y")

        for name, err in [("u", errs[0]), ("v", errs[1]), ("p", errs[2])]:
            print(f"  |PINN-{ref_name}| {name}: L∞ = {np.nanmax(err):.4f}  "
                  f"L² = {np.sqrt(np.nanmean(err**2)):.4f}")

    sup = "|PINN − OpenFOAM| error maps (Re = 100)"
    if n_rows == 2:
        sup = "|PINN − OpenFOAM| and |PINN − Code_Saturne| error maps (Re = 100)"
    plt.suptitle(sup, fontsize=13, y=1.01)
    plt.tight_layout()
    path = os.path.join(out_dir, "error_map.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved  {path}")


def plot_streamline_comparison(
    X, Y,
    U_pinn, V_pinn,
    U_of,   V_of,
    out_dir: str,
    U_cs=None, V_cs=None,
) -> None:
    x_lin = X[:, 0]
    y_lin = Y[0, :]

    fields = [
        (U_pinn, V_pinn, "PINN"),
        (U_of, V_of, "OpenFOAM (icoFoam)"),
    ]
    if U_cs is not None and V_cs is not None:
        fields.append((U_cs, V_cs, "Code_Saturne"))

    fig, axes = plt.subplots(1, len(fields), figsize=(6.5 * len(fields), 5))
    if len(fields) == 1:
        axes = [axes]

    for ax, (U, V, title) in zip(np.atleast_1d(axes), fields):
        speed = np.sqrt(U**2 + V**2)
        strm = ax.streamplot(
            x_lin, y_lin, U.T, V.T,
            color=speed.T, cmap="viridis",
            density=1.5, linewidth=0.8, arrowsize=0.8,
        )
        plt.colorbar(strm.lines, ax=ax, label="Speed |u|")
        ax.set_title(f"Streamlines — {title}", fontsize=11)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    stitle = "Streamline comparison: PINN vs OpenFOAM (Re = 100)"
    if len(fields) == 3:
        stitle = "Streamline comparison: PINN vs OpenFOAM vs Code_Saturne (Re = 100)"
    plt.suptitle(stitle, fontsize=13, y=1.01)
    plt.tight_layout()
    path = os.path.join(out_dir, "streamline_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved  {path}")


# ---------------------------------------------------------------------------
# Quantitative summary table
# ---------------------------------------------------------------------------

def print_summary_table(
    y_vc, u_vc, x_hc, v_hc,
    X_pinn, Y_pinn, U_pinn, V_pinn,
    y_cs=None, u_cs=None, x_cs=None, v_cs=None,
) -> None:
    """Print L∞ errors vs Ghia for PINN, OpenFOAM, and Code_Saturne."""
    n = X_pinn.shape[0]; mid = n // 2
    y_pinn = Y_pinn[mid, :]; u_pinn = U_pinn[mid, :]
    x_pinn = X_pinn[:, mid]; v_pinn = V_pinn[:, mid]

    def interp_err(x_ref, f_ref, x_pred, f_pred):
        f_interp = np.interp(x_ref, x_pred, f_pred)
        return np.max(np.abs(f_interp - f_ref))

    err_u_pinn = interp_err(GHIA_Y, GHIA_U, y_pinn, u_pinn)
    err_v_pinn = interp_err(GHIA_X, GHIA_V, x_pinn, v_pinn)
    err_u_of   = interp_err(GHIA_Y, GHIA_U, y_vc,   u_vc)
    err_v_of   = interp_err(GHIA_X, GHIA_V, x_hc,   v_hc)

    print()
    print("=" * 56)
    print("  L∞ error vs Ghia et al. (1982) — Re = 100")
    print("=" * 56)
    print(f"  {'Method':<22} {'u centreline':>14} {'v centreline':>14}")
    print(f"  {'-'*22} {'-'*14} {'-'*14}")
    print(f"  {'PINN':<22} {err_u_pinn:>14.4f} {err_v_pinn:>14.4f}")
    print(f"  {'OpenFOAM (icoFoam)':<22} {err_u_of:>14.4f} {err_v_of:>14.4f}")
    if u_cs is not None and v_cs is not None:
        err_u_cs = interp_err(GHIA_Y, GHIA_U, y_cs, u_cs)
        err_v_cs = interp_err(GHIA_X, GHIA_V, x_cs, v_cs)
        print(f"  {'Code_Saturne':<22} {err_u_cs:>14.4f} {err_v_cs:>14.4f}")
    print("=" * 56)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading PINN model …")
    pinn_model = load_model("results/model.pt")
    X_p, Y_p, U_p, V_p, P_p = evaluate_on_grid(pinn_model, n=100)

    print("Reading OpenFOAM centreline data …")
    y_vc, u_vc, x_hc, v_hc = read_of_centrelines(OF_CASE)

    print("Reading OpenFOAM 2-D fields …")
    X_of, Y_of, U_of, V_of, P_of = read_of_2d_fields(OF_CASE, n_grid=100)

    # Code_Saturne data (optional — skip gracefully if not available)
    y_cs = u_cs = x_cs = v_cs = None
    U_cs = V_cs = P_cs = None
    try:
        print("Reading Code_Saturne centreline data …")
        y_cs, u_cs, x_cs, v_cs = read_cs_centrelines(CS_RESU)
        print("Reading Code_Saturne 2-D fields …")
        X_cs, Y_cs, U_cs, V_cs, P_cs = read_cs_2d_fields(CS_RESU, n_grid=100)
    except FileNotFoundError as exc:
        print(f"  [skip] Code_Saturne data not found: {exc}")

    print("Generating plots …")
    plot_centreline_comparison(
        y_vc, u_vc, x_hc, v_hc, X_p, Y_p, U_p, V_p, OUT_DIR,
        y_cs=y_cs, u_cs=u_cs, x_cs=x_cs, v_cs=v_cs,
    )
    plot_field_comparison(
        X_p, Y_p, U_p, V_p, P_p, U_of, V_of, P_of, OUT_DIR,
        U_cs=U_cs, V_cs=V_cs, P_cs=P_cs,
    )
    plot_error_maps(
        X_p, Y_p, U_p, V_p, P_p, U_of, V_of, P_of, OUT_DIR,
        U_cs=U_cs, V_cs=V_cs, P_cs=P_cs,
    )
    plot_streamline_comparison(
        X_p, Y_p, U_p, V_p, U_of, V_of, OUT_DIR,
        U_cs=U_cs, V_cs=V_cs,
    )

    print_summary_table(
        y_vc, u_vc, x_hc, v_hc, X_p, Y_p, U_p, V_p,
        y_cs=y_cs, u_cs=u_cs, x_cs=x_cs, v_cs=v_cs,
    )
    print("Done.  Results saved to", OUT_DIR)


if __name__ == "__main__":
    main()
