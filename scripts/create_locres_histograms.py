"""Create a 3x3 composite figure of local-resolution histograms.

Layout (3 rows x 3 columns):
  Row 0 — Simulated datasets (all 4 methods)
  Row 1 — Real datasets, CTF comparison
  Row 2 — Real datasets, DW comparison

Each row has one subplot per dataset (T20S, Apoferritin, GroEL).
Histograms are plotted as spatial-frequency distributions (1/Å)
matching the style of the existing locres_hist_*.png assets.

Uses pre-symmetrized locres maps when available.
"""

from __future__ import annotations

import gc
import os
from collections import OrderedDict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mrcfile
import numpy as np
from scipy.ndimage import gaussian_filter1d

REPO_ROOT = Path(__file__).resolve().parents[1]


def _get_cryo_datasets_dir() -> Path:
    """Resolve cryo_datasets directory from env var or default sibling location."""
    env = os.environ.get("CRYO_DATASETS_DIR")
    if env:
        return Path(env)
    return REPO_ROOT.parent / "cryo_datasets"


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.15,
    "grid.linewidth": 0.4,
})

LOCRES_BASE = REPO_ROOT / "storage" / "tmp" / "locres_maps_refine4"
LOCRES_SYM_BASE = _get_cryo_datasets_dir() / "results" / "local_resolution"
MASK_BASE = _get_cryo_datasets_dir()
OUTPUT_DIR = REPO_ROOT / "results" / "paper_assets"

MASK_DATASET_MAP = {"10025_sim": "10025", "10474_sim": "10474", "10948_sim": "10948_sim"}

DATASETS_SIM = ["10025_sim", "10474_sim", "10948_sim"]
DATASETS_REAL = ["10025", "10474", "10948"]

DATASET_LABELS = {
    "10025": "T20S Proteasome", "10025_sim": "T20S Proteasome",
    "10474": "Apoferritin",     "10474_sim": "Apoferritin",
    "10948": "GroEL",           "10948_sim": "GroEL",
}

DATASET_ANGPIX = {
    "10025": 0.9856, "10025_sim": 0.9856,
    "10474": 0.706667, "10474_sim": 0.706667,
    "10948": 0.813, "10948_sim": 0.813,
}

DATASET_BOX = {
    "10025": 300, "10025_sim": 300,
    "10474": 384, "10474_sim": 384,
    "10948": 360, "10948_sim": 360,
}

METHODS_DIR: OrderedDict[str, str] = OrderedDict([
    ("base",             "noise2noise"),
    ("base_ctf",         "noise2noise_ctf"),
    ("default",          "multi_noise2noise"),
    ("default_ctf",      "multi_noise2noise_ctf"),
    ("dose_weighted",    "multi_noise2noise_dw"),
    ("dose_weighted_ctf","multi_noise2noise_ctf_dw"),
])

METHOD_STYLE = {
    "base":             {"color": "#7BAFD4", "ls": "-",  "label": "Noise2Noise"},
    "base_ctf":         {"color": "#4A6FA5", "ls": "--", "label": "Noise2Noise + CTF"},
    "default":          {"color": "#5FB87A", "ls": "-",  "label": "Multi-Noise2Noise"},
    "default_ctf":      {"color": "#2D8B4E", "ls": "--", "label": "Multi-Noise2Noise + CTF"},
    "dose_weighted":    {"color": "#E8963E", "ls": "-",  "label": "Multi-Noise2Noise + DW"},
    "dose_weighted_ctf":{"color": "#D4722A", "ls": "--", "label": "Multi-Noise2Noise + CTF + DW"},
}

METHODS_SIM = ["base", "base_ctf", "default", "default_ctf"]
METHODS_CTF = ["base", "base_ctf", "default", "default_ctf"]
METHODS_DW  = ["base", "default", "dose_weighted", "dose_weighted_ctf"]


def load_mask(dataset: str) -> np.ndarray:
    """Return symmetrized mask if available, otherwise original."""
    ds = MASK_DATASET_MAP.get(dataset, dataset)
    sym_mask = LOCRES_SYM_BASE / ds / "mask_sym.mrc"
    mkp = MASK_BASE / ds / "mask.mrc"
    use_path = sym_mask if sym_mask.exists() else mkp
    with mrcfile.open(str(use_path), mode="r") as f:
        return f.data > 0.5


def load_locres(dataset: str, method_key: str) -> np.ndarray | None:
    method_dir = METHODS_DIR[method_key]
    sym_path = LOCRES_SYM_BASE / dataset / method_dir / "locres_avg_sym.mrc"
    raw_path = LOCRES_BASE / dataset / method_dir / "locres_avg.mrc"
    use_path = sym_path if sym_path.exists() else raw_path
    if not use_path.exists():
        return None
    with mrcfile.open(str(use_path), mode="r") as f:
        return f.data.copy()


def locres_to_spatial_freq(locres_vals: np.ndarray) -> np.ndarray:
    """Convert local resolution in Å to spatial frequency in 1/Å."""
    return 1.0 / locres_vals


def compute_percentage(
    vals_angstrom: np.ndarray,
    edges: np.ndarray,
    sigma: float = 2.0,
):
    """Compute smooth histogram in spatial frequency space, as % of masked voxels.

    Uses pre-computed bin edges (shared across methods within a panel)
    so that percentages are comparable.
    """
    freq = locres_to_spatial_freq(vals_angstrom)
    counts, _ = np.histogram(freq, bins=edges)
    total = counts.sum()
    pct = counts / total * 100.0 if total > 0 else counts
    centers = 0.5 * (edges[:-1] + edges[1:])
    smoothed = gaussian_filter1d(pct, sigma=sigma)
    bin_width = centers[1] - centers[0]
    centers = np.concatenate([[centers[0] - bin_width], centers, [centers[-1] + bin_width]])
    smoothed = np.concatenate([[0.0], smoothed, [0.0]])
    return centers, smoothed


def plot_panel(
    ax: plt.Axes,
    dataset: str,
    method_keys: list[str],
    n_bins: int = 150,
) -> None:
    mask = load_mask(dataset)

    all_vals: dict[str, np.ndarray] = {}
    all_freq_vals: list[np.ndarray] = []
    for mk in method_keys:
        locres = load_locres(dataset, mk)
        if locres is None:
            continue
        vals = locres[mask]
        vals = vals[vals > 0]
        del locres
        gc.collect()
        if vals.size == 0:
            continue
        all_vals[mk] = vals
        all_freq_vals.append(locres_to_spatial_freq(vals))
    del mask
    gc.collect()

    if not all_freq_vals:
        return

    combined_freq = np.concatenate(all_freq_vals)
    lo = float(np.percentile(combined_freq, 2))
    hi = float(np.percentile(combined_freq, 98))
    margin = (hi - lo) * 0.03
    edges = np.linspace(lo - margin, hi + margin, n_bins + 1)
    del combined_freq, all_freq_vals
    gc.collect()

    for mk in method_keys:
        if mk not in all_vals:
            continue
        style = METHOD_STYLE[mk]
        freq, pct = compute_percentage(all_vals[mk], edges)
        ax.plot(freq, pct, color=style["color"], ls=style["ls"],
                lw=1.6, label=style["label"], zorder=3)
    del all_vals
    gc.collect()

    ax.set_xlabel("Spatial frequency (1/Å)", fontsize=8)
    ax.set_ylabel("Voxels (%)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=5.5, loc="upper left", framealpha=0.85,
              edgecolor="#cccccc", fancybox=False, handlelength=2.5)


def main() -> None:
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))

    row_titles = ["Simulated", "Real — CTF comparison", "Real — DW comparison"]
    col_labels = [DATASET_LABELS[ds] for ds in DATASETS_SIM]

    for col_idx, (ds_sim, ds_real) in enumerate(
        zip(DATASETS_SIM, DATASETS_REAL)
    ):
        plot_panel(axes[0, col_idx], ds_sim, METHODS_SIM)
        plot_panel(axes[1, col_idx], ds_real, METHODS_CTF)
        plot_panel(axes[2, col_idx], ds_real, METHODS_DW)

    # Per-column shared x limits; independent y per subplot
    for col_idx in range(3):
        col_xranges = [axes[r, col_idx].get_xlim() for r in range(3)]
        x_lo = min(r[0] for r in col_xranges)
        x_hi = max(r[1] for r in col_xranges)
        for r in range(3):
            axes[r, col_idx].set_xlim(x_lo, x_hi)
            axes[r, col_idx].set_ylim(0, None)

    for row_idx in range(3):
        for col_idx in range(3):
            if row_idx < 2:
                axes[row_idx, col_idx].set_xlabel("")
            if col_idx > 0:
                axes[row_idx, col_idx].set_ylabel("")

    for col_idx, label in enumerate(col_labels):
        axes[0, col_idx].set_title(
            label, fontsize=13, fontweight="bold", pad=8,
        )

    for row_idx, rt in enumerate(row_titles):
        axes[row_idx, 0].annotate(
            rt, xy=(-0.25, 0.5), xycoords="axes fraction",
            ha="center", va="center", fontsize=11, fontweight="bold",
            rotation=90,
        )

    fig.tight_layout(pad=0.8, h_pad=1.0, w_pad=1.0)
    fig.subplots_adjust(top=0.93, left=0.08)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "locres_histograms_3x3.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
