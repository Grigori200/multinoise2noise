"""Batch-render cryo-EM density maps as isosurfaces colored by local resolution.

Produces:
  - One combined comparison-grid PNG (rows = datasets grouped by
    Simulated/Real, columns = methods)
  - One 360-degree rotation GIF per dataset/method combo

Memory-safe: processes one dataset/method at a time, writing intermediate
cell images to disk, then assembles the grid from PNGs.

Requires: pyvista, mrcfile, numpy, matplotlib, Pillow.
Runs headless (pyvista off_screen).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import warnings
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import mrcfile
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from PIL import Image

os.environ["DISPLAY"] = ""
warnings.filterwarnings("ignore")

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]

from scipy.ndimage import affine_transform
from scipy.spatial.transform import Rotation

import pyvista as pv

REPO_ROOT = Path(__file__).resolve().parents[1]


def _get_cryo_datasets_dir() -> Path:
    """Resolve cryo_datasets directory from env var or default sibling location."""
    env = os.environ.get("CRYO_DATASETS_DIR")
    if env:
        return Path(env)
    return REPO_ROOT.parent / "cryo_datasets"


# ── custom colormap: green (high res) → white → blue (low res) ────────
_CMAP_COLORS = [(0.2, 0.7, 0.2), (1.0, 1.0, 1.0), (0.2, 0.3, 0.8)]
CMAP_OBJ = LinearSegmentedColormap.from_list("green_white_blue", _CMAP_COLORS, N=256)
CMAP_NAME = "green_white_blue"
try:
    plt.colormaps.register(CMAP_OBJ, name=CMAP_NAME)
except ValueError:
    pass

# ── paths ──────────────────────────────────────────────────────────────
RESULTS_BASE = str(_get_cryo_datasets_dir() / "results")
MASK_BASE = str(_get_cryo_datasets_dir())
LOCRES_BASE = str(REPO_ROOT / "storage" / "tmp" / "locres_maps_refine4")
LOCRES_SYM_BASE = str(_get_cryo_datasets_dir() / "results" / "local_resolution")

# ── datasets & methods ─────────────────────────────────────────────────
SIMULATED_DATASETS = ["10025_sim", "10474_sim", "10948_sim"]
REAL_DATASETS = ["10025", "10474", "10948"]
DATASETS = SIMULATED_DATASETS + REAL_DATASETS

METHODS: OrderedDict[str, str] = OrderedDict(
    [
        ("base", "noise2noise"),
        ("base_ctf", "noise2noise_ctf"),
        ("default", "multi_noise2noise"),
        ("default_ctf", "multi_noise2noise_ctf"),
        ("dose_weighted", "multi_noise2noise_dw"),
        ("dose_weighted_ctf", "multi_noise2noise_ctf_dw"),
    ]
)

# ── display names for paper ────────────────────────────────────────────
DATASET_DISPLAY_NAMES: dict[str, str] = {
    "10025": "T20S Proteasome",
    "10025_sim": "T20S Proteasome",
    "10474": "Apoferritin",
    "10474_sim": "Apoferritin",
    "10948": "GroEL",
    "10948_sim": "GroEL",
}

METHOD_DISPLAY_NAMES: dict[str, str] = {
    "base": "Topaz Denoise",
    "base_ctf": "Topaz Denoise\n+ CTF",
    "default": "Multi-Noise2Noise",
    "default_ctf": "Multi-Noise2Noise\n+ CTF",
    "dose_weighted": "Multi-Noise2Noise\n+ DW",
    "dose_weighted_ctf": "Multi-Noise2Noise\n+ CTF + DW",
}

# ── rendering defaults ─────────────────────────────────────────────────
THRESHOLD_PERCENTILE = 85
GIF_ZOOM = 0.9

THRESHOLD_OVERRIDES: dict[str, float] = {}
N_FRAMES = 72
IMAGE_SIZE = (800, 800)
OUTPUT_DIR = str(REPO_ROOT / "results" / "locres_figures")

# ── mask datasets that share a mask directory ──────────────────────────
MASK_DATASET_MAP: dict[str, str] = {
    "10025_sim": "10025",
    "10474_sim": "10474",
    "10948_sim": "10948_sim",
}


# ── symmetry operations ────────────────────────────────────────────────

DATASET_SYMMETRY: dict[str, str] = {
    "10025": "D7", "10025_sim": "D7",
    "10474": "O",  "10474_sim": "O",
    "10948": "D7", "10948_sim": "D7",
}


def _d_n_rotations(n: int) -> list[np.ndarray]:
    """Generate rotation matrices for D_n (dihedral) symmetry.

    D_n = n-fold rotation about X + n 2-fold rotations about axes in YZ plane.
    Total: 2n elements.
    """
    rots = []
    for k in range(n):
        angle = 2 * np.pi * k / n
        rots.append(Rotation.from_rotvec([angle, 0, 0]).as_matrix())
    for k in range(n):
        angle = 2 * np.pi * k / n
        flip_axis = np.array([0.0, np.cos(angle / 2), np.sin(angle / 2)])
        flip_axis /= np.linalg.norm(flip_axis)
        rots.append(Rotation.from_rotvec(flip_axis * np.pi).as_matrix())
    return rots


def _octahedral_rotations() -> list[np.ndarray]:
    """Generate 24 rotation matrices for octahedral (O) symmetry."""
    rots = []
    axes_4fold = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    for ax in axes_4fold:
        for k in range(4):
            rots.append(Rotation.from_rotvec(np.array(ax, dtype=float) * k * np.pi / 2).as_matrix())

    axes_3fold = [
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
    ]
    for ax in axes_3fold:
        ax_norm = np.array(ax, dtype=float)
        ax_norm /= np.linalg.norm(ax_norm)
        for k in (1, 2):
            rots.append(Rotation.from_rotvec(ax_norm * k * 2 * np.pi / 3).as_matrix())

    axes_2fold = [
        [1, 1, 0], [1, -1, 0], [1, 0, 1], [1, 0, -1],
        [0, 1, 1], [0, 1, -1],
    ]
    for ax in axes_2fold:
        ax_norm = np.array(ax, dtype=float)
        ax_norm /= np.linalg.norm(ax_norm)
        rots.append(Rotation.from_rotvec(ax_norm * np.pi).as_matrix())

    return rots


def _get_symmetry_rotations(sym: str) -> list[np.ndarray]:
    if sym.startswith("D"):
        n = int(sym[1:])
        return _d_n_rotations(n)
    elif sym == "O":
        return _octahedral_rotations()
    else:
        return [np.eye(3)]


def _deduplicate_rotations(rots: list[np.ndarray], tol: float = 1e-6) -> list[np.ndarray]:
    unique = []
    for r in rots:
        is_dup = False
        for u in unique:
            if np.allclose(r, u, atol=tol):
                is_dup = True
                break
        if not is_dup:
            unique.append(r)
    return unique


def symmetrize_volume(vol: np.ndarray, symmetry: str) -> np.ndarray:
    """Average a 3D volume over all rotations of the given symmetry group.

    Rotations are applied about the center of the volume using
    scipy.ndimage.affine_transform with linear interpolation.
    """
    rots = _deduplicate_rotations(_get_symmetry_rotations(symmetry))
    center = np.array(vol.shape, dtype=float) / 2.0

    result = np.zeros_like(vol, dtype=np.float64)
    for R in rots:
        offset = center - R @ center
        result += affine_transform(vol.astype(np.float64), R, offset=offset, order=1)
    result /= len(rots)
    return result.astype(vol.dtype)


def mrc_path(dataset: str, method_dir: str) -> Path:
    return Path(
        RESULTS_BASE, dataset, method_dir,
        "Refine_4", "1", "denoised_mean", "half_1_sharp.mrc",
    )


def mask_path(dataset: str) -> Path:
    """Return symmetrized mask if available, otherwise original."""
    ds = MASK_DATASET_MAP.get(dataset, dataset)
    sym_mask = Path(LOCRES_SYM_BASE, ds, "mask_sym.mrc")
    if sym_mask.exists():
        return sym_mask
    return Path(MASK_BASE, ds, "mask.mrc")


def locres_path(dataset: str, method_dir: str) -> Path:
    return Path(LOCRES_BASE, dataset, method_dir, "locres_avg.mrc")


def files_exist(dataset: str, method_dir: str) -> bool:
    mp = mrc_path(dataset, method_dir)
    mkp = mask_path(dataset)
    lrp = locres_path(dataset, method_dir)
    return all(p.exists() for p in (mp, mkp, lrp))


# ── per-dataset color scale ───────────────────────────────────────────

def compute_dataset_clims(
    datasets: list[str],
) -> dict[str, tuple[float, float]]:
    """Compute locres color limits individually per dataset row.

    Each dataset gets its own scale so color variation is maximised.
    """
    clims: dict[str, tuple[float, float]] = {}

    for ds in datasets:
        mkp = mask_path(ds)
        if not mkp.exists():
            continue
        with mrcfile.open(str(mkp), mode="r") as f:
            mask = f.data > 0.5

        all_lr: list[np.ndarray] = []
        for _label, method_dir in METHODS.items():
            sym_lrp = Path(LOCRES_SYM_BASE, ds, method_dir, "locres_avg_sym.mrc")
            lrp = locres_path(ds, method_dir)
            use_path = sym_lrp if sym_lrp.exists() else lrp
            if not use_path.exists():
                continue
            with mrcfile.open(str(use_path), mode="r") as f:
                lr = f.data[mask]
            lr = lr[lr > 0]
            if lr.size > 0:
                all_lr.append(lr)

        del mask
        gc.collect()

        if not all_lr:
            clims[ds] = (3.0, 6.0)
            continue

        combined = np.concatenate(all_lr)
        del all_lr
        lo = float(np.percentile(combined, 5))
        hi = float(np.percentile(combined, 80))
        margin = (hi - lo) * 0.05
        lo = max(0, lo - margin)
        hi = hi + margin
        clims[ds] = (lo, hi)
        print(f"  clim[{ds}] = [{lo:.2f}, {hi:.2f}] A")
        del combined
        gc.collect()

    return clims


# ── isosurface extraction ─────────────────────────────────────────────

def _load_raw(dataset: str, method_dir: str):
    """Load the three volumes and return (vol, mask_bool, locres, voxel_size).

    Prefers pre-symmetrized locres maps from LOCRES_SYM_BASE; falls back
    to on-the-fly symmetrization if the pre-computed file is missing.
    """
    mp = mrc_path(dataset, method_dir)
    mkp = mask_path(dataset)

    sym_lrp = Path(LOCRES_SYM_BASE, dataset, method_dir, "locres_avg_sym.mrc")
    lrp = locres_path(dataset, method_dir)

    with mrcfile.open(str(mp), mode="r") as f:
        vol = f.data.copy()
        voxel_size = float(f.voxel_size.x)
    with mrcfile.open(str(mkp), mode="r") as f:
        mask_bool = f.data > 0.5

    if sym_lrp.exists():
        print(f"[sym] Loading pre-symmetrized locres: {sym_lrp.name}")
        with mrcfile.open(str(sym_lrp), mode="r") as f:
            locres = f.data.copy()
    else:
        with mrcfile.open(str(lrp), mode="r") as f:
            locres = f.data.copy()
        sym = DATASET_SYMMETRY.get(dataset, "C1")
        if sym != "C1":
            print(f"[sym] Applying {sym} symmetry to locres on-the-fly...", end=" ", flush=True)
            locres = symmetrize_volume(locres, sym)
            print("done")

    return vol, mask_bool, locres, voxel_size


def create_isosurface(
    dataset: str,
    method_dir: str,
    threshold_pct: float,
) -> pv.PolyData | None:
    """Simple fixed-threshold isosurface (used for grid cells)."""
    vol, mask_bool, locres, voxel_size = _load_raw(dataset, method_dir)

    vol_masked = vol * mask_bool
    del vol
    positive = vol_masked[mask_bool]
    positive = positive[positive > 0]
    if positive.size == 0:
        return None
    threshold = float(np.percentile(positive, threshold_pct))
    del positive

    grid = pv.ImageData(dimensions=vol_masked.shape, spacing=[voxel_size] * 3)
    grid.point_data["density"] = vol_masked.flatten(order="F")
    del vol_masked
    grid.point_data["locres"] = locres.flatten(order="F")
    del locres, mask_bool
    gc.collect()

    contour = grid.contour([threshold], scalars="density")
    del grid
    gc.collect()

    if contour.n_points == 0:
        return None
    return contour


def create_isosurface_adaptive(
    dataset: str,
    method_dir: str,
    threshold_pct: float,
    min_fragment_frac: float = 0.005,
) -> pv.PolyData | None:
    """Resolution-adaptive isosurface for GIFs.

    The density threshold is relaxed in lower-resolution regions so the
    surface stays connected there, while high-resolution regions keep
    fine detail.  Small disconnected fragments are removed afterwards.
    """
    vol, mask_bool, locres, voxel_size = _load_raw(dataset, method_dir)

    vol_masked = vol * mask_bool
    del vol
    positive = vol_masked[mask_bool]
    positive = positive[positive > 0]
    if positive.size == 0:
        return None
    threshold = float(np.percentile(positive, threshold_pct))
    del positive

    # Resolution-adaptive factor: lower threshold in low-res regions.
    # Normalise locres into [0, 1] where 0 = best resolution, 1 = worst.
    lr_valid = locres[mask_bool & (locres > 0)]
    lr_lo = float(np.percentile(lr_valid, 5))
    lr_hi = float(np.percentile(lr_valid, 95))
    del lr_valid
    lr_range = max(lr_hi - lr_lo, 1e-6)
    lr_norm = np.clip((locres - lr_lo) / lr_range, 0.0, 1.0)

    # Factor: 1.0 at best resolution → 0.4 at worst resolution.
    adaptive_factor = np.where(mask_bool, 1.0 - 0.6 * lr_norm, 1.0)
    threshold_map = threshold * adaptive_factor
    del adaptive_factor, lr_norm

    # Normalised density: value 1.0 means "exactly at local threshold".
    safe_thresh = np.where(threshold_map > 0, threshold_map, 1.0)
    density_norm = vol_masked / safe_thresh
    del threshold_map, safe_thresh, vol_masked

    grid = pv.ImageData(dimensions=density_norm.shape, spacing=[voxel_size] * 3)
    grid.point_data["density"] = density_norm.flatten(order="F")
    del density_norm
    grid.point_data["locres"] = locres.flatten(order="F")
    del locres, mask_bool
    gc.collect()

    contour = grid.contour([1.0], scalars="density")
    del grid
    gc.collect()

    if contour.n_points == 0:
        return None

    # Remove small disconnected fragments.
    labeled = contour.connectivity("all")
    region_ids = labeled.point_data["RegionId"]
    unique, counts = np.unique(region_ids, return_counts=True)
    min_size = max(int(contour.n_points * min_fragment_frac), 1)
    keep = set(unique[counts >= min_size])
    keep_mask = np.isin(region_ids, list(keep))
    filtered = labeled.extract_points(keep_mask, adjacent_cells=True)
    del labeled, contour
    gc.collect()

    if filtered.n_points == 0:
        return None
    return filtered


# ── rendering ──────────────────────────────────────────────────────────

def render_view(
    mesh: pv.PolyData,
    clim: tuple[float, float],
    camera_position: str | list = "xy",
    window_size: tuple[int, int] = IMAGE_SIZE,
    show_scalar_bar: bool = False,
    zoom: float = 1.35,
) -> np.ndarray:
    pl = pv.Plotter(off_screen=True, window_size=list(window_size))
    pl.add_mesh(
        mesh,
        scalars="locres",
        cmap=CMAP_OBJ,
        clim=list(clim),
        show_scalar_bar=show_scalar_bar,
    )
    pl.camera_position = camera_position
    pl.camera.zoom(zoom)
    img = pl.screenshot(return_img=True)
    pl.close()
    return img


def render_and_save_cell(
    dataset: str,
    method_key: str,
    method_dir: str,
    threshold_pct: float,
    clim: tuple[float, float],
    cell_dir: Path,
) -> bool:
    """Render a single grid cell (no fragment filtering)."""
    out_png = cell_dir / f"{dataset}_{method_key}.png"
    if out_png.exists():
        print(f"    [cached] {out_png.name}")
        return True

    if not files_exist(dataset, method_dir):
        print(f"    SKIP {dataset}/{method_dir}: missing files")
        return False

    print(f"    {dataset}/{method_key}...", end=" ", flush=True)
    mesh = create_isosurface(dataset, method_dir, threshold_pct)
    if mesh is None:
        print("empty isosurface")
        return False

    lr = mesh.point_data["locres"]
    lr_valid = lr[lr > 0]
    print(
        f"{mesh.n_points} pts, "
        f"locres [{lr_valid.min():.2f}, {lr_valid.max():.2f}] A",
        end=" ",
        flush=True,
    )

    img = render_view(mesh, clim=clim, camera_position="xy", show_scalar_bar=False)
    del mesh
    gc.collect()

    Image.fromarray(img).save(out_png)
    del img
    gc.collect()

    print(f"-> {out_png.name}")
    return True


def render_and_save_gif(
    dataset: str,
    method_key: str,
    method_dir: str,
    threshold_pct: float,
    clim: tuple[float, float],
    output_dir: Path,
    n_frames: int,
) -> None:
    """Render a rotation GIF."""
    gif_path = output_dir / dataset / f"{method_key}.gif"
    if gif_path.exists():
        print(f"    [cached] {gif_path}")
        return

    if not files_exist(dataset, method_dir):
        return

    print(f"    GIF {dataset}/{method_key}...", end=" ", flush=True)
    mesh = create_isosurface(dataset, method_dir, threshold_pct)
    if mesh is None:
        print("empty isosurface")
        return

    gif_path.parent.mkdir(parents=True, exist_ok=True)
    frames: list[Image.Image] = []
    center = mesh.center
    radius = mesh.length * 1.2

    for i in range(n_frames):
        angle = 2 * np.pi * i / n_frames
        cam_x = center[0] + radius * np.sin(angle)
        cam_z = center[2] + radius * np.cos(angle)
        camera_pos = [
            (cam_x, center[1], cam_z),
            tuple(center),
            (0, 1, 0),
        ]
        img = render_view(
            mesh, clim=clim, camera_position=camera_pos,
            show_scalar_bar=(i == 0), zoom=GIF_ZOOM,
        )
        frames.append(Image.fromarray(img))

    del mesh
    gc.collect()

    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=80,
        loop=0,
        optimize=True,
    )
    del frames
    gc.collect()

    size_mb = gif_path.stat().st_size / (1024 * 1024)
    print(f"saved ({size_mb:.1f} MB)")


# ── grid assembly ─────────────────────────────────────────────────────

METHODS_SIM_KEYS = [
    "base", "base_ctf", "default", "default_ctf",
]
METHODS_REAL_KEYS = list(METHODS.keys())


def _assemble_single_grid(
    cell_dir: Path,
    output_path: Path,
    row_datasets: list[str],
    method_keys: list[str],
    clims: dict[str, tuple[float, float]],
    title: str = "",
) -> None:
    """Assemble one grid: rows = methods, columns = datasets, colorbar per column."""
    ncols = len(row_datasets)
    nrows = len(method_keys)

    cell_w, cell_h = 2.0, 2.0
    left_margin = 0.85
    right_margin = 0.05
    title_extra = 0.2 if title else 0.0
    top_margin = 0.4 + title_extra
    bot_margin = 0.5
    col_gap = 0.02

    fig_w = left_margin + ncols * cell_w + (ncols - 1) * col_gap + right_margin
    fig_h = top_margin + nrows * cell_h + bot_margin

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=200, facecolor="white")

    col_step = cell_w + col_gap

    if title:
        content_cx = (left_margin + left_margin + ncols * cell_w + (ncols - 1) * col_gap) / 2 / fig_w
        fig.text(content_cx, 1.0 - (title_extra * 0.45) / fig_h, title,
                 ha="center", va="center", fontsize=13, fontweight="bold")

    for col_idx, ds in enumerate(row_datasets):
        ds_label = DATASET_DISPLAY_NAMES[ds]
        cx = (left_margin + col_idx * col_step + cell_w / 2) / fig_w
        cy = 1.0 - (title_extra + 0.4 * 0.35) / fig_h
        fig.text(cx, cy, ds_label, ha="center", va="center",
                 fontsize=11, fontweight="bold")

    for row_idx, method_key in enumerate(method_keys):
        label = METHOD_DISPLAY_NAMES[method_key]
        y_off = top_margin + row_idx * cell_h
        cy = 1.0 - (y_off + cell_h / 2) / fig_h
        cx = (left_margin * 0.55) / fig_w
        fig.text(cx, cy, label, ha="center", va="center",
                 fontsize=10, fontweight="bold", rotation=90)

    for col_idx, ds in enumerate(row_datasets):
        clim = clims.get(ds, (3.0, 6.0))
        for row_idx, method_key in enumerate(method_keys):
            y_off = top_margin + row_idx * cell_h
            x0 = (left_margin + col_idx * col_step) / fig_w
            y0 = 1.0 - (y_off + cell_h) / fig_h
            w = cell_w / fig_w
            h = cell_h / fig_h
            ax = fig.add_axes([x0, y0, w, h])

            png_path = cell_dir / f"{ds}_{method_key}.png"
            if png_path.exists():
                cell_img = np.array(Image.open(png_path))
                ax.imshow(cell_img)
            ax.axis("off")

        cbar_pad = 0.15 * cell_w
        cbar_x = (left_margin + col_idx * col_step + cbar_pad) / fig_w
        cbar_y_bottom = top_margin + nrows * cell_h + 0.08
        cbar_y = 1.0 - (cbar_y_bottom + 0.2) / fig_h
        cbar_h = 0.2 / fig_h
        cbar_width = (cell_w - 2 * cbar_pad) / fig_w

        cbar_ax = fig.add_axes([cbar_x, cbar_y, cbar_width, cbar_h])
        norm = Normalize(vmin=clim[0], vmax=clim[1])
        sm = plt.cm.ScalarMappable(cmap=CMAP_OBJ, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
        cbar.set_label("Local Res. (\u00c5)", fontsize=9)
        step = (clim[1] - clim[0]) / 4
        ticks = np.arange(clim[0], clim[1] + step * 0.1, step)
        cbar.set_ticks(ticks)
        cbar.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}"))
        cbar.ax.tick_params(labelsize=8)

    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved grid: {output_path}")


def _assemble_single_grid_transposed(
    cell_dir: Path,
    output_path: Path,
    row_datasets: list[str],
    method_keys: list[str],
    clims: dict[str, tuple[float, float]],
) -> None:
    """Transposed grid: rows = datasets, columns = methods, colorbar per row."""
    ncols = len(method_keys)
    nrows = len(row_datasets)

    cell_w, cell_h = 2.0, 2.0
    left_margin = 0.85
    right_margin = 0.5
    top_margin = 0.85
    bot_margin = 0.05
    col_gap = 0.02
    row_gap = 0.02

    fig_w = left_margin + ncols * cell_w + (ncols - 1) * col_gap + right_margin
    fig_h = top_margin + nrows * cell_h + (nrows - 1) * row_gap + bot_margin

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=200, facecolor="white")

    col_step = cell_w + col_gap
    row_step = cell_h + row_gap

    for col_idx, method_key in enumerate(method_keys):
        label = METHOD_DISPLAY_NAMES[method_key]
        cx = (left_margin + col_idx * col_step + cell_w / 2) / fig_w
        cy = 1.0 - (top_margin * 0.35) / fig_h
        fig.text(cx, cy, label, ha="center", va="center",
                 fontsize=10, fontweight="bold")

    for row_idx, ds in enumerate(row_datasets):
        ds_label = DATASET_DISPLAY_NAMES[ds]
        y_off = top_margin + row_idx * row_step
        cy = 1.0 - (y_off + cell_h / 2) / fig_h
        cx = (left_margin * 0.55) / fig_w
        fig.text(cx, cy, ds_label, ha="center", va="center",
                 fontsize=11, fontweight="bold", rotation=90)

    for row_idx, ds in enumerate(row_datasets):
        clim = clims.get(ds, (3.0, 6.0))
        for col_idx, method_key in enumerate(method_keys):
            y_off = top_margin + row_idx * row_step
            x0 = (left_margin + col_idx * col_step) / fig_w
            y0 = 1.0 - (y_off + cell_h) / fig_h
            w = cell_w / fig_w
            h = cell_h / fig_h
            ax = fig.add_axes([x0, y0, w, h])

            png_path = cell_dir / f"{ds}_{method_key}.png"
            if png_path.exists():
                cell_img = np.array(Image.open(png_path))
                ax.imshow(cell_img)
            ax.axis("off")

        cbar_x_left = left_margin + ncols * col_step + 0.08
        cbar_pad = 0.15 * cell_h
        cbar_y = 1.0 - (y_off + cell_h - cbar_pad) / fig_h
        cbar_w = 0.2 / fig_w
        cbar_h_frac = (cell_h - 2 * cbar_pad) / fig_h
        cbar_x = cbar_x_left / fig_w

        cbar_ax = fig.add_axes([cbar_x, cbar_y, cbar_w, cbar_h_frac])
        norm = Normalize(vmin=clim[0], vmax=clim[1])
        sm = plt.cm.ScalarMappable(cmap=CMAP_OBJ, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation="vertical")
        cbar.set_label("Local Res. (\u00c5)", fontsize=9)
        step = (clim[1] - clim[0]) / 4
        ticks = np.arange(clim[0], clim[1] + step * 0.1, step)
        cbar.set_ticks(ticks)
        cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}"))
        cbar.ax.tick_params(labelsize=8)

    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved grid: {output_path}")


def assemble_grids(
    cell_dir: Path,
    output_dir: Path,
    datasets: list[str],
    clims: dict[str, tuple[float, float]],
) -> None:
    """Assemble separate sim and real grids (original + transposed)."""
    sim_ds = [ds for ds in SIMULATED_DATASETS if ds in datasets]
    real_ds = [ds for ds in REAL_DATASETS if ds in datasets]

    if sim_ds:
        _assemble_single_grid(
            cell_dir, output_dir / "locres_grid_sim.png",
            sim_ds, METHODS_SIM_KEYS, clims, title="Simulated",
        )
        _assemble_single_grid_transposed(
            cell_dir, output_dir / "locres_grid_sim_T.png",
            sim_ds, METHODS_SIM_KEYS, clims,
        )
    if real_ds:
        _assemble_single_grid(
            cell_dir, output_dir / "locres_grid_real.png",
            real_ds, METHODS_REAL_KEYS, clims, title="Real",
        )
        _assemble_single_grid_transposed(
            cell_dir, output_dir / "locres_grid_real_T.png",
            real_ds, METHODS_REAL_KEYS, clims,
        )


# ── main ───────────────────────────────────────────────────────────────

def main() -> None:
    global THRESHOLD_PERCENTILE, N_FRAMES  # noqa: PLW0603

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=str, default=OUTPUT_DIR,
        help="Root output directory for figures.",
    )
    parser.add_argument(
        "--datasets", nargs="*", default=None,
        help="Datasets to process (default: all).",
    )
    parser.add_argument(
        "--threshold-pct", type=float, default=THRESHOLD_PERCENTILE,
        help="Percentile for isosurface threshold.",
    )
    parser.add_argument(
        "--n-frames", type=int, default=N_FRAMES,
        help="Number of frames in rotation GIF.",
    )
    parser.add_argument(
        "--skip-grid", action="store_true",
        help="Skip the combined grid figure.",
    )
    parser.add_argument(
        "--skip-gifs", action="store_true",
        help="Skip individual rotation GIFs.",
    )
    args = parser.parse_args()

    THRESHOLD_PERCENTILE = args.threshold_pct
    N_FRAMES = args.n_frames

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    cell_dir = output_root / "_cells"
    cell_dir.mkdir(parents=True, exist_ok=True)

    datasets = args.datasets if args.datasets else DATASETS

    print("Computing per-dataset color scales...")
    clims = compute_dataset_clims(datasets)
    clims_path = output_root / "clims.json"
    with open(clims_path, "w") as f:
        json.dump({k: list(v) for k, v in clims.items()}, f, indent=2)
    print(f"  Saved clims to {clims_path}")

    if not args.skip_grid:
        print("\nPhase 1: Rendering individual cell images...")
        for ds in datasets:
            clim = clims.get(ds, (3.0, 6.0))
            thresh = THRESHOLD_OVERRIDES.get(ds, THRESHOLD_PERCENTILE)
            for method_key, method_dir in METHODS.items():
                render_and_save_cell(
                    ds, method_key, method_dir,
                    thresh, clim, cell_dir,
                )

        print("\nPhase 2: Assembling grid figures...")
        assemble_grids(cell_dir, output_root, datasets, clims)

    if not args.skip_gifs:
        print("\nPhase 3: Rendering rotation GIFs...")
        for ds in datasets:
            clim = clims.get(ds, (3.0, 6.0))
            thresh = THRESHOLD_OVERRIDES.get(ds, THRESHOLD_PERCENTILE)
            for method_key, method_dir in METHODS.items():
                render_and_save_gif(
                    ds, method_key, method_dir,
                    thresh, clim, output_root,
                    N_FRAMES,
                )

    print(f"\nDone. All outputs in: {output_root}")


if __name__ == "__main__":
    main()
