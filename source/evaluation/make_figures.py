"""Figures of the manuscript from the evaluation outputs.

- ``frame_sweep``: mean FRC against frames averaged into the input (Fig. 2), values in
  ``figures/frame_sweep.csv``,
- ``fsc_true_<dataset>``: model-to-map FSC of ``figures.fsc_main_dataset`` (Fig. 5),
- ``fsc_true_3x3``: model-to-map FSC of all datasets (Supp. Fig. S2),
- ``locres_histograms_3x3``: local-resolution distributions (Fig. 6).

FSC curves are the mean ± standard deviation over all (subset, half-map) rows written by
:mod:`source.evaluation.evaluate_reconstruction`; histograms are read from
``locres/locres_histograms.csv`` of :mod:`source.evaluation.evaluate_local_resolution`.

Usage::

    python -m source.evaluation.make_figures
"""

import json
from pathlib import Path

import hydra
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

THRESHOLD_COLOR = "#d62728"


def load_fsc_curves(
    cfg: DictConfig, dataset: str, method: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return (spatial frequency in 1/Å, curves of shape [n_rows, n_shells])."""
    rows_dir = Path(cfg.output_dir) / "rows" / dataset
    expected = len(cfg.reconstruction.subsets) * len(cfg.reconstruction.halves)
    paths = sorted(rows_dir.glob(f"{method}__s*_h*.json"))
    if len(paths) != expected:
        raise FileNotFoundError(
            f"{dataset}/{method}: expected {expected} FSC rows, found {len(paths)}"
        )
    curves, angpix = [], set()
    for path in paths:
        angpix.add(json.loads(path.read_text())["angpix"])
        curves.append(np.load(path.with_suffix(".npy")))
    if len(angpix) != 1 or len({c.shape for c in curves}) != 1:
        raise ValueError(
            f"{dataset}/{method}: rows differ in pixel size or shell count"
        )
    curves = np.stack(curves)
    n_shells = curves.shape[1]
    frequency = np.arange(n_shells) / ((n_shells - 1) * 2 * angpix.pop())
    return frequency, curves


def label(cfg: DictConfig, method: str) -> str:
    return str(cfg.methods[method])


def draw_thresholds(ax, thresholds, fmt: str = "{}") -> None:
    for threshold in thresholds:
        ax.axhline(threshold, color=THRESHOLD_COLOR, lw=0.8, ls=":", alpha=0.8)
        ax.text(
            0.995,
            threshold + 0.015,
            fmt.format(threshold),
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=7,
            style="italic",
            color=THRESHOLD_COLOR,
        )


def plot_fsc(
    ax, cfg: DictConfig, dataset: str, methods, legend_size: float | None = 6
) -> dict:
    """Plot mean ± std FSC curves; return the line of each method by method name."""
    nyquist = 0.0
    lines = {}
    for method in methods:
        frequency, curves = load_fsc_curves(cfg, dataset, method)
        style = cfg.figures.styles[method]
        mean, std = curves.mean(0), curves.std(0, ddof=1)
        (lines[method],) = ax.plot(
            frequency, mean, lw=1.4, label=label(cfg, method), **style
        )
        ax.fill_between(
            frequency, mean - std, mean + std, color=style.color, alpha=0.2, lw=0
        )
        nyquist = max(nyquist, float(frequency[-1]))
    draw_thresholds(ax, cfg.fsc.thresholds)
    ax.set_xlim(0, nyquist)
    ax.set_ylim(-0.03, 1.05)
    ax.grid(alpha=0.2)
    if legend_size is not None:
        ax.legend(fontsize=legend_size, loc="upper right")
    return lines


def figure_fsc_main(cfg: DictConfig, out_dir: Path) -> list[Path]:
    dataset = str(cfg.figures.fsc_main_dataset)
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_fsc(ax, cfg, dataset, [str(m) for m in cfg.figures.fsc_main_methods], 10)
    ax.set_xlabel("Spatial frequency (1/Å)")
    ax.set_ylabel("Model-to-map FSC (FSC$_{true}$)")
    top = ax.secondary_xaxis("top")
    ticks_angstrom = [10.0, 5.0, 4.0, 3.0, 2.5, 2.0]
    lo, hi = ax.get_xlim()
    ticks = [1 / t for t in ticks_angstrom if lo < 1 / t <= hi]
    top.set_xticks(ticks, [f"{1 / t:.1f} Å" for t in ticks])
    return save(fig, out_dir / f"fsc_true_{dataset}", cfg)


def grid(cfg: DictConfig, draw, ylabel: str):
    columns, rows = cfg.figures.columns, cfg.figures.rows
    fig, axes = plt.subplots(
        len(rows),
        len(columns),
        figsize=(4.6 * len(columns), 3.2 * len(rows)),
        squeeze=False,
    )
    for r, row in enumerate(rows):
        for c, column in enumerate(columns):
            ax = axes[r, c]
            dataset = str(column[row.kind])
            draw(ax, dataset, [str(m) for m in row.methods])
            if r == 0:
                ax.set_title(column.title, fontsize=12, fontweight="bold")
            if c == 0:
                ax.set_ylabel(ylabel)
                ax.annotate(
                    row.title,
                    xy=(-0.28, 0.5),
                    xycoords="axes fraction",
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=11,
                    fontweight="bold",
                )
            if r == len(rows) - 1:
                ax.set_xlabel("Spatial frequency (1/Å)")
    return fig


def figure_legend(fig, cfg: DictConfig, handles: dict) -> None:
    """One legend below the grid, one column per method, in order of first appearance."""
    fig.legend(
        list(handles.values()),
        [label(cfg, m) for m in handles],
        loc="lower center",
        ncol=len(handles),
        fontsize=9,
        frameon=True,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))


def figure_fsc_grid(cfg: DictConfig, out_dir: Path) -> list[Path]:
    handles = {}

    def draw(ax, dataset, methods):
        for method, line in plot_fsc(ax, cfg, dataset, methods, None).items():
            handles.setdefault(method, line)

    fig = grid(cfg, draw, "FSC$_{true}$")
    figure_legend(fig, cfg, handles)
    return save(fig, out_dir / "fsc_true_3x3", cfg)


def figure_locres_histograms(cfg: DictConfig, out_dir: Path) -> list[Path]:
    table = pd.read_csv(
        Path(cfg.output_dir) / "locres" / "locres_histograms.csv",
        dtype={"dataset": str},
    )
    handles = {}

    def draw(ax, dataset, methods):
        for method in methods:
            part = table[(table.dataset == dataset) & (table.method == method)]
            if part.empty:
                raise KeyError(f"No local-resolution histogram for {dataset}/{method}")
            centres = (part.frequency_low + part.frequency_high) / 2
            (line,) = ax.plot(
                centres, part.voxel_percent, lw=1.2, **cfg.figures.styles[method]
            )
            handles.setdefault(method, line)
        part = table[table.dataset == dataset]
        reached = part[part.method.isin(methods) & (part.voxel_percent > 0)]
        ax.set_xlim(part.frequency_low.min(), reached.frequency_high.max())
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.2)

    fig = grid(cfg, draw, "Voxels (%)")
    figure_legend(fig, cfg, handles)
    return save(fig, out_dir / "locres_histograms_3x3", cfg)


def frc_mean_beyond(frc: np.ndarray, angpix: float, cutoff_angstrom: float) -> float:
    """Mean FRC over the rings i > 0 whose resolution 2 * angpix * n_rings / i is <= cutoff."""
    frc = np.asarray(frc, dtype=float)
    rings = np.arange(1, len(frc))
    selected = rings[2 * angpix * len(frc) / rings <= cutoff_angstrom]
    if selected.size == 0:
        raise ValueError(f"No FRC ring reaches {cutoff_angstrom} Å at {angpix} Å/px")
    return float(frc[selected].mean())


def load_frame_sweep(cfg: DictConfig) -> pd.DataFrame:
    """One row per (dataset, frames averaged into the input, inference variant)."""
    sweep = cfg.frame_sweep
    rows = []
    for dataset in sweep.datasets:
        dataset = str(dataset)
        root = Path(cfg.datasets[dataset].results_dir) / "frame_sweep"
        paths = sorted(root.glob(f"split_*/{sweep.csv_name}"))
        if not paths:
            raise FileNotFoundError(f"No {root}/split_*/{sweep.csv_name}")
        for path in paths:
            frames = int(path.parent.name.removeprefix("split_"))
            curves = pd.read_csv(path)
            for variant, spec in sweep.variants.items():
                if spec.column not in curves:
                    raise KeyError(f"{path}: missing column {spec.column}")
                value = frc_mean_beyond(
                    curves[spec.column].dropna().to_numpy(),
                    float(cfg.datasets[dataset].angpix),
                    float(sweep.cutoff_angstrom),
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "frames": frames,
                        "variant": str(variant),
                        "mean_frc": value,
                        "source": str(path),
                    }
                )
    return pd.DataFrame(rows).sort_values(["dataset", "variant", "frames"])


def figure_frame_sweep(cfg: DictConfig, out_dir: Path) -> list[Path]:
    sweep = cfg.frame_sweep
    table = load_frame_sweep(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "frame_sweep.csv", index=False)
    fig, axes = plt.subplots(
        1, len(sweep.datasets), figsize=(6.5 * len(sweep.datasets), 4.8), squeeze=False
    )
    for ax, (dataset, spec) in zip(axes[0], sweep.datasets.items()):
        part = table[table.dataset == str(dataset)]
        for variant, style in sweep.variants.items():
            line = part[part.variant == str(variant)]
            ax.plot(
                line.frames,
                line.mean_frc,
                color=style.color,
                marker=style.marker,
                markersize=5,
                lw=1.6,
                label=style.label,
            )
        lo, hi = ax.get_ylim()
        n_frames = int(spec.n_frames)
        for frames, name in (
            (n_frames / 2, "Topaz-Denoise"),
            (n_frames - 1, "Multi-Noise2Noise"),
        ):
            ax.axvline(frames, color="#2e7d32", ls="--", lw=1.0, alpha=0.8)
            ax.text(
                frames - 0.4,
                lo + 0.04 * (hi - lo),
                name,
                rotation=90,
                ha="right",
                va="bottom",
                fontsize=9,
                style="italic",
                color="#2e7d32",
            )
        ax.set_ylim(lo, hi)
        ax.set_title(spec.title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Amount of frames averaged into input")
        ax.grid(alpha=0.2)
    axes[0, 0].set_ylabel("Mean FRC")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return save(fig, out_dir / "frame_sweep", cfg)


def save(fig, stem: Path, cfg: DictConfig) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = [stem.with_suffix(".png"), stem.with_suffix(".pdf")]
    for path in paths:
        fig.savefig(path, dpi=int(cfg.figures.dpi), bbox_inches="tight")
    plt.close(fig)
    return paths


@hydra.main(version_base="1.3", config_path="../../configs_eval", config_name="paper")
def main(cfg: DictConfig) -> None:
    out_dir = Path(cfg.output_dir) / "figures"
    for make in (
        figure_frame_sweep,
        figure_fsc_main,
        figure_fsc_grid,
        figure_locres_histograms,
    ):
        for path in make(cfg, out_dir):
            print(path)


if __name__ == "__main__":
    main()
