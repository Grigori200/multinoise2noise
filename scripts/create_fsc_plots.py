"""Create per-dataset FSC-true plots for all methods, WACV paper style."""

import os
import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = REPO_ROOT / "results"
OUTPUT_DIR = RESULTS_DIR / "paper_assets"

SUBSET_POWER = 4

DATASETS = {
    "10025_sim": {"pkl": "10025_sim_ref.pkl", "angpix": 0.9856,   "size": 300, "title": "Proteasome T20S (Simulated)"},
    "10474_sim": {"pkl": "10474_sim_ref.pkl", "angpix": 0.706667, "size": 384, "title": "Human Light Chain Apoferritin (Simulated)"},
    "10948_sim": {"pkl": "10948_sim_ref.pkl", "angpix": 0.813,    "size": 360, "title": "GroEL (Simulated)"},
    "10025":     {"pkl": "10025_ref_v4.pkl",  "angpix": 0.9856,   "size": 300, "title": "Proteasome T20S (Real)"},
    "10474":     {"pkl": "10474_ref_v4.pkl",  "angpix": 0.706667, "size": 384, "title": "Human Light Chain Apoferritin (Real)"},
    "10948":     {"pkl": "10948_ref_v4.pkl",  "angpix": 0.813,    "size": 360, "title": "GroEL (Real)"},
}

METHOD_STYLE = {
    "raw":                      {"color": "#888888", "ls": "-",  "lw": 1.5, "label": "Raw"},
    "noise2noise":              {"color": "#7BAFD4", "ls": "-",  "lw": 1.5, "label": "Noise2Noise"},
    "noise2noise_ctf":          {"color": "#4A6FA5", "ls": "--", "lw": 1.5, "label": "Noise2Noise + CTF"},
    "multi_noise2noise":        {"color": "#5FB87A", "ls": "-",  "lw": 1.5, "label": "Multi-Noise2Noise"},
    "multi_noise2noise_ctf":    {"color": "#2D8B4E", "ls": "--", "lw": 1.8, "label": "Multi-Noise2Noise + CTF"},
    "multi_noise2noise_dw":     {"color": "#E8A94D", "ls": "-",  "lw": 1.5, "label": "Multi-Noise2Noise + DW"},
    "multi_noise2noise_ctf_dw": {"color": "#D4770B", "ls": "--", "lw": 1.8, "label": "Multi-Noise2Noise + CTF + DW"},
}

METHODS_SIM = [
    "noise2noise", "noise2noise_ctf",
    "multi_noise2noise", "multi_noise2noise_ctf",
]

METHODS_REAL_CTF = [
    "noise2noise", "noise2noise_ctf",
    "multi_noise2noise", "multi_noise2noise_ctf",
]

METHODS_REAL_DW = [
    "noise2noise",
    "multi_noise2noise",
    "multi_noise2noise_dw",
    "multi_noise2noise_ctf_dw",
]


def setup_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 9,
        'axes.linewidth': 0.6,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': True,
        'grid.alpha': 0.15,
        'grid.linewidth': 0.4,
    })


def plot_fsc_panel(ax, df, methods, angpix, size):
    n_bins = size // 2 + 1
    freqs = np.arange(n_bins) / (size * angpix)

    for method in methods:
        if method not in df.task_name.values:
            continue
        style = METHOD_STYLE[method]
        sub = df[(df.task_name == method) & (df.subset_power == SUBSET_POWER)]
        if len(sub) == 0:
            continue

        fscs = torch.stack([r.fsc.detach().clone() for _, r in sub.iterrows()])
        mean_fsc = fscs.mean(dim=0).numpy()
        std_fsc = fscs.std(dim=0).numpy() if len(sub) > 1 else np.zeros_like(mean_fsc)

        ax.plot(freqs, mean_fsc,
                color=style["color"], ls=style["ls"], lw=style["lw"],
                label=style["label"], zorder=3)

        if len(sub) > 1:
            ax.fill_between(freqs,
                            np.clip(mean_fsc - std_fsc, 0, 1),
                            np.clip(mean_fsc + std_fsc, 0, 1),
                            color=style["color"], alpha=0.1, zorder=2)

    ax.axhline(y=0.5, color='#CC4444', ls=':', lw=0.8, alpha=0.7, zorder=1)
    ax.axhline(y=0.143, color='#CC4444', ls=':', lw=0.8, alpha=0.7, zorder=1)
    ax.text(freqs[-1] * 0.98, 0.52, '0.5', ha='right', va='bottom',
            fontsize=7, color='#CC4444', fontstyle='italic')
    ax.text(freqs[-1] * 0.98, 0.16, '0.143', ha='right', va='bottom',
            fontsize=7, color='#CC4444', fontstyle='italic')

    ax.set_xlim(0, freqs[-1])
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Spatial frequency (1/\u00C5)", fontsize=9)
    ax.set_ylabel("FSC$_{true}$", fontsize=9)
    ax.legend(fontsize=6.5, loc='upper right', framealpha=0.85,
              edgecolor='#cccccc', fancybox=False, handlelength=2.5)


def plot_simulated(ds_key: str, ds_info: dict):
    df = pd.read_pickle(str(RESULTS_DIR / ds_info["pkl"]))
    fig, ax = plt.subplots(figsize=(5, 3.2))
    plot_fsc_panel(ax, df, METHODS_SIM, ds_info["angpix"], ds_info["size"])
    ax.set_title(ds_info["title"], fontsize=10, fontweight='bold', pad=8)
    fig.tight_layout(pad=0.5)
    out = str(OUTPUT_DIR / f"fsc_true_{ds_key}.png")
    fig.savefig(out, dpi=300, bbox_inches='tight', pad_inches=0.05, facecolor='white')
    plt.close(fig)
    print(f"  Saved {out}")


def plot_real(ds_key: str, ds_info: dict):
    df = pd.read_pickle(str(RESULTS_DIR / ds_info["pkl"]))
    angpix, size = ds_info["angpix"], ds_info["size"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 6), sharex=True)

    plot_fsc_panel(ax1, df, METHODS_REAL_CTF, angpix, size)
    ax1.set_title("CTF comparison", fontsize=9, fontweight='bold', pad=6)
    ax1.set_xlabel("")

    plot_fsc_panel(ax2, df, METHODS_REAL_DW, angpix, size)
    ax2.set_title("DW comparison", fontsize=9, fontweight='bold', pad=6)

    fig.suptitle(ds_info["title"], fontsize=10, fontweight='bold', y=1.01)
    fig.tight_layout(pad=0.5)
    out = str(OUTPUT_DIR / f"fsc_true_{ds_key}.png")
    fig.savefig(out, dpi=300, bbox_inches='tight', pad_inches=0.05, facecolor='white')
    plt.close(fig)
    print(f"  Saved {out}")


def main():
    setup_style()
    os.makedirs(str(OUTPUT_DIR), exist_ok=True)
    for ds_key, ds_info in DATASETS.items():
        if ds_key.endswith("_sim"):
            plot_simulated(ds_key, ds_info)
        else:
            plot_real(ds_key, ds_info)
    print("\nDone!")


if __name__ == "__main__":
    main()
