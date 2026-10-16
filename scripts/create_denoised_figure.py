"""Create a WACV-style composite figure of denoised particle images."""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

SAMPLES_DIR = REPO_ROOT / "results" / "sample_particles"
WACV_DIR = REPO_ROOT / "results" / "paper_assets" / "wacv_particle_examples"
OUTPUT_DIR = REPO_ROOT / "results" / "paper_assets"

DATASETS_SIM = [
    ("10025_sim", "Proteasome\nT20S"),
    ("10474_sim", "Human Light Chain\nApoferritin"),
    ("10948_sim", "GroEL"),
]
DATASETS_REAL = [
    ("10025", "Proteasome\nT20S"),
    ("10474", "Human Light Chain\nApoferritin"),
    ("10948", "GroEL"),
]

METHODS_SIM = [
    ("raw",                "Noisy"),
    ("base",               "Noise2Noise"),
    ("base_ctf",           "Noise2Noise\n+ CTF"),
    ("default",            "Multi-Noise2Noise"),
    ("default_ctf",        "Multi-Noise2Noise\n+ CTF"),
    ("clean",              "Clean"),
]

METHODS_REAL = [
    ("raw",                "Noisy"),
    ("base",               "Noise2Noise"),
    ("base_ctf",           "Noise2Noise\n+ CTF"),
    ("default",            "Multi-Noise2Noise"),
    ("default_ctf",        "Multi-Noise2Noise\n+ CTF"),
    ("dose_weighted",      "Multi-Noise2Noise\n+ DW"),
    ("dose_weighted_ctf",  "Multi-Noise2Noise\n+ CTF + DW"),
]

METHODS = METHODS_SIM + [(k, l) for k, l in METHODS_REAL if k not in dict(METHODS_SIM)]

WACV_PARTICLE_IDS = [
    "018149", "070629", "096156", "127233", "144117",
    "148065", "160745", "160838", "178558", "185698",
]


def crop_white_border(img_arr: np.ndarray, threshold: int = 245) -> np.ndarray:
    """Remove white border from image array (H, W) or (H, W, C)."""
    if img_arr.ndim == 3:
        gray = img_arr.mean(axis=2)
    else:
        gray = img_arr
    
    mask = gray < threshold
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if not rows.any() or not cols.any():
        return img_arr
    
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    return img_arr[rmin:rmax+1, cmin:cmax+1]


def crop_wacv_image(img_arr: np.ndarray) -> np.ndarray:
    """Crop wacv images: skip text at top, remove white borders."""
    skip_top = 27
    cropped = img_arr[skip_top:]
    return crop_white_border(cropped)


def load_sample_particle_image(dataset: str, method: str, particle_idx: int) -> np.ndarray | None:
    """Load from sample_particles directory."""
    path = SAMPLES_DIR / dataset / "denoised" / f"particle_{particle_idx}_{method}.png"
    if not path.exists():
        return None
    img = Image.open(path).convert('L')
    arr = np.array(img)
    return crop_white_border(arr)


def load_wacv_image(method: str, wacv_id: str) -> np.ndarray | None:
    """Load 10474 real image from wacv_particle_examples."""
    if method == "raw":
        path = WACV_DIR / "raw_average" / f"particle_{wacv_id}_raw_average.png"
    elif method == "clean":
        return None
    else:
        path = WACV_DIR / "denoised_average" / method / f"particle_{wacv_id}_{method}.png"
    
    if not path.exists():
        return None
    img = Image.open(path).convert('L')
    arr = np.array(img)
    return crop_wacv_image(arr)


def load_image(dataset: str, method: str, particle_idx: int) -> np.ndarray | None:
    """Load image for a dataset+method combination."""
    if dataset == "10474":
        wacv_id = WACV_PARTICLE_IDS[particle_idx]
        return load_wacv_image(method, wacv_id)
    return load_sample_particle_image(dataset, method, particle_idx)


def normalize_contrast(img: np.ndarray, plow: float = 1, phigh: float = 99) -> np.ndarray:
    """Percentile-based contrast normalization."""
    vmin = np.percentile(img, plow)
    vmax = np.percentile(img, phigh)
    if vmax <= vmin:
        return np.zeros_like(img, dtype=np.float32)
    out = (img.astype(np.float32) - vmin) / (vmax - vmin)
    return np.clip(out, 0, 1)


def create_split_figure(
    particle_idx: int,
    datasets: list[tuple[str, str]],
    methods: list[tuple[str, str]],
    output_path: str,
    title: str = "",
):
    """Create a single-column figure: methods as rows, datasets as columns."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 8,
        'axes.linewidth': 0.5,
    })

    n_methods = len(methods)
    n_datasets = len(datasets)

    images = {}
    for ds, _label in datasets:
        for method_key, _method_label in methods:
            img = load_image(ds, method_key, particle_idx)
            if img is not None:
                images[(ds, method_key)] = normalize_contrast(img)

    cell_size = 1.15
    label_width = 1.15
    title_height = 0.18 if title else 0.0
    header_height = 0.55 + title_height

    fig_width = label_width + n_datasets * cell_size + 0.05
    fig_height = header_height + n_methods * cell_size + 0.05

    fig = plt.figure(figsize=(fig_width, fig_height), facecolor='white')

    left_margin = label_width / fig_width
    top_margin = 1.0 - header_height / fig_height

    if title:
        content_center_x = left_margin + (1.0 - left_margin) * 0.5
        fig.text(
            content_center_x, 1.0 - (title_height * 0.45) / fig_height,
            title,
            ha='center', va='center',
            fontsize=10, fontweight='bold',
            transform=fig.transFigure,
        )
    col_frac = cell_size / fig_width
    row_frac = cell_size / fig_height

    for col_idx, (ds, label) in enumerate(datasets):
        x_start = left_margin + col_idx * col_frac

        for row_idx, (method_key, method_label) in enumerate(methods):
            y_start = top_margin - (row_idx + 1) * row_frac

            key = (ds, method_key)
            if key not in images:
                continue

            ax = fig.add_axes([x_start, y_start, col_frac * 0.97, row_frac * 0.97])
            ax.imshow(images[key], cmap='gray', vmin=0, vmax=1, aspect='equal')
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.3)
                spine.set_color('#888888')

    for row_idx, (method_key, method_label) in enumerate(methods):
        y_center = top_margin - (row_idx + 0.5) * row_frac
        fig.text(
            left_margin * 0.83, y_center,
            method_label,
            ha='center', va='center',
            fontsize=7, fontweight='bold',
            rotation=90,
            transform=fig.transFigure,
        )

    for col_idx, (ds, label) in enumerate(datasets):
        x_center = left_margin + (col_idx + 0.5) * col_frac
        fig.text(
            x_center, top_margin + 0.01,
            label,
            ha='center', va='bottom',
            fontsize=7, fontweight='bold',
            transform=fig.transFigure,
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.02, facecolor='white')
    plt.close(fig)
    print(f"  Saved {output_path}")


def create_split_figure_transposed(
    particle_idx: int,
    datasets: list[tuple[str, str]],
    methods: list[tuple[str, str]],
    output_path: str,
):
    """Transposed figure: datasets as rows, methods as columns."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 8,
        'axes.linewidth': 0.5,
    })

    n_methods = len(methods)
    n_datasets = len(datasets)

    images = {}
    for ds, _label in datasets:
        for method_key, _method_label in methods:
            img = load_image(ds, method_key, particle_idx)
            if img is not None:
                images[(ds, method_key)] = normalize_contrast(img)

    cell_size = 1.15
    label_height = 0.55
    header_width = 1.15

    fig_width = header_width + n_methods * cell_size + 0.05
    fig_height = label_height + n_datasets * cell_size + 0.05

    fig = plt.figure(figsize=(fig_width, fig_height), facecolor='white')

    left_margin = header_width / fig_width
    top_margin = 1.0 - label_height / fig_height
    col_frac = cell_size / fig_width
    row_frac = cell_size / fig_height

    for col_idx, (method_key, method_label) in enumerate(methods):
        x_start = left_margin + col_idx * col_frac

        for row_idx, (ds, ds_label) in enumerate(datasets):
            y_start = top_margin - (row_idx + 1) * row_frac

            key = (ds, method_key)
            if key not in images:
                continue

            ax = fig.add_axes([x_start, y_start, col_frac * 0.97, row_frac * 0.97])
            ax.imshow(images[key], cmap='gray', vmin=0, vmax=1, aspect='equal')
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.3)
                spine.set_color('#888888')

    for row_idx, (ds, ds_label) in enumerate(datasets):
        y_center = top_margin - (row_idx + 0.5) * row_frac
        fig.text(
            left_margin * 0.83, y_center,
            ds_label,
            ha='center', va='center',
            fontsize=7, fontweight='bold',
            rotation=90,
            transform=fig.transFigure,
        )

    for col_idx, (method_key, method_label) in enumerate(methods):
        x_center = left_margin + (col_idx + 0.5) * col_frac
        fig.text(
            x_center, top_margin + 0.01,
            method_label,
            ha='center', va='bottom',
            fontsize=7, fontweight='bold',
            transform=fig.transFigure,
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.02, facecolor='white')
    plt.close(fig)
    print(f"  Saved {output_path}")


def main():
    for pidx in range(10):
        create_split_figure(
            pidx, DATASETS_SIM, METHODS_SIM,
            str(OUTPUT_DIR / f"denoised_sim_particle_{pidx}.png"),
            title="Simulated",
        )
        create_split_figure(
            pidx, DATASETS_REAL, METHODS_REAL,
            str(OUTPUT_DIR / f"denoised_real_particle_{pidx}.png"),
            title="Real",
        )
        create_split_figure_transposed(
            pidx, DATASETS_SIM, METHODS_SIM,
            str(OUTPUT_DIR / f"denoised_sim_particle_{pidx}_T.png"),
        )
        create_split_figure_transposed(
            pidx, DATASETS_REAL, METHODS_REAL,
            str(OUTPUT_DIR / f"denoised_real_particle_{pidx}_T.png"),
        )
    print("\nDone!")


if __name__ == "__main__":
    main()
