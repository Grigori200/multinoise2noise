"""Isosurfaces coloured by local resolution.

``locres_grid_real`` is Fig. 7 and ``locres_grid_sim`` Fig. 8 of the manuscript.

For every (dataset, method) of a figure, the isosurface of
``<results_dir>/<method>/Refine_<r>/<subset>/<image_mode>/<density>`` inside the symmetrised
mask core is drawn at the ``threshold_percentile`` of the positive masked density and coloured
by the averaged, symmetrised local-resolution map of
:mod:`source.evaluation.evaluate_local_resolution` (``locres/averages``). The colour range of
a dataset is the ``clim_percentiles`` of the local resolution inside the mask core, pooled over
the methods of the figure and widened by ``clim_margin`` of its width. Rows are methods,
columns are datasets, one colour bar per column.

Requires ``pyvista`` (``pip install -e .[render]``); renders off screen.

Usage::

    python -m source.evaluation.render_locres
"""

import json
from pathlib import Path

import hydra
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from source.evaluation.make_figures import label  # noqa: E402
from source.evaluation.maps import read_map  # noqa: E402

CMAP = LinearSegmentedColormap.from_list(
    "green_white_blue", [(0.2, 0.7, 0.2), (1.0, 1.0, 1.0), (0.2, 0.3, 0.8)], N=256
)


def density_path(cfg: DictConfig, dataset: str, method: str) -> Path:
    rc, rec = cfg.locres_render, cfg.reconstruction
    return (
        Path(cfg.datasets[dataset].results_dir)
        / method
        / f"Refine_{rec.refine}"
        / str(rc.subset)
        / rec.image_mode
        / rc.density
    )


def averages_dir(cfg: DictConfig) -> Path:
    return Path(cfg.output_dir) / "locres" / "averages"


def load_core(cfg: DictConfig, dataset: str) -> np.ndarray:
    path = averages_dir(cfg) / f"{dataset}__mask_sym.mrc"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} (run evaluate_local_resolution stage=aggregate)"
        )
    return read_map(path)[0].numpy() > float(cfg.locres.mask_core)


def load_locres(cfg: DictConfig, dataset: str, method: str) -> np.ndarray:
    path = averages_dir(cfg) / f"{dataset}__{method}.mrc"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} (run evaluate_local_resolution stage=aggregate)"
        )
    return read_map(path)[0].numpy()


def colour_limits(cfg: DictConfig, dataset: str, methods) -> tuple[float, float]:
    """Percentile range of the local resolution in the mask core, pooled over methods."""
    rc = cfg.locres_render
    core = load_core(cfg, dataset)
    values = []
    for method in methods:
        locres = load_locres(cfg, dataset, method)[core]
        values.append(locres[np.isfinite(locres) & (locres > 0)])
    values = np.concatenate(values)
    if values.size == 0:
        raise ValueError(f"{dataset}: no finite local resolution inside the mask core")
    lo, hi = (float(np.percentile(values, q)) for q in rc.clim_percentiles)
    margin = (hi - lo) * float(rc.clim_margin)
    return max(0.0, lo - margin), hi + margin


def render_cell(
    cfg: DictConfig, dataset: str, method: str, clim: tuple[float, float], out: Path
) -> dict:
    """Render one isosurface to ``out`` (PNG) and return its statistics."""
    import pyvista as pv

    rc = cfg.locres_render
    volume, voxel_size = read_map(density_path(cfg, dataset, method))
    volume = volume.numpy()
    core = load_core(cfg, dataset)
    locres = load_locres(cfg, dataset, method)
    if not volume.shape == core.shape == locres.shape:
        raise ValueError(
            f"{dataset}/{method}: shapes differ (density {volume.shape}, "
            f"mask {core.shape}, local resolution {locres.shape})"
        )
    masked = volume * core
    positive = masked[core & (masked > 0)]
    if positive.size == 0:
        raise ValueError(
            f"{dataset}/{method}: no positive density inside the mask core"
        )
    threshold = float(np.percentile(positive, float(rc.threshold_percentile)))

    grid = pv.ImageData(dimensions=masked.shape, spacing=[voxel_size] * 3)
    grid.point_data["density"] = masked.flatten(order="F")
    grid.point_data["locres"] = locres.flatten(order="F")
    surface = grid.contour([threshold], scalars="density")
    if surface.n_points == 0:
        raise ValueError(f"{dataset}/{method}: empty isosurface at {threshold}")

    plotter = pv.Plotter(off_screen=True, window_size=[int(rc.window_px)] * 2)
    plotter.add_mesh(
        surface, scalars="locres", cmap=CMAP, clim=list(clim), show_scalar_bar=False
    )
    plotter.camera_position = str(rc.camera)
    plotter.camera.zoom(float(rc.zoom))
    out.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(out))
    plotter.close()
    on_surface = surface.point_data["locres"]
    return {
        "dataset": dataset,
        "method": method,
        "density": str(density_path(cfg, dataset, method)),
        "threshold": threshold,
        "n_points": int(surface.n_points),
        "surface_locres_min": float(on_surface.min()),
        "surface_locres_max": float(on_surface.max()),
    }


def tick_decimals(ticks: np.ndarray, max_decimals: int = 3) -> int:
    """Fewest decimals (at least one) that keep all tick labels distinct."""
    for decimals in range(1, max_decimals + 1):
        if len({f"{t:.{decimals}f}" for t in ticks}) == len(ticks):
            return decimals
    return max_decimals


def method_label(cfg: DictConfig, method: str) -> str:
    return label(cfg, method).replace(" + ", "\n+ ", 1)


def assemble(
    cfg: DictConfig,
    spec: DictConfig,
    cells: dict,
    clims: dict,
    out_dir: Path,
    stem: str,
) -> list[Path]:
    """Rows = methods, columns = datasets, one horizontal colour bar per column."""
    columns = [str(column[spec.kind]) for column in cfg.figures.columns]
    titles = [column.title for column in cfg.figures.columns]
    methods = [str(m) for m in spec.methods]
    cell, left, top, bottom = 2.0, 0.85, 0.6, 0.5
    fig = plt.figure(
        figsize=(left + cell * len(columns), top + cell * len(methods) + bottom),
        facecolor="white",
    )
    width, height = fig.get_size_inches()
    fig.text(
        (left + cell * len(columns) / 2) / width,
        1 - 0.12 / height,
        spec.title,
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
    )
    for c, (dataset, title) in enumerate(zip(columns, titles)):
        x0 = left + c * cell
        fig.text(
            (x0 + cell / 2) / width,
            1 - 0.4 / height,
            title,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
        )
        for r, method in enumerate(methods):
            ax = fig.add_axes(
                [
                    x0 / width,
                    1 - (top + (r + 1) * cell) / height,
                    cell / width,
                    cell / height,
                ]
            )
            ax.imshow(plt.imread(cells[dataset, method]))
            ax.axis("off")
        pad = 0.15 * cell
        bar = fig.add_axes(
            [
                (x0 + pad) / width,
                1 - (top + len(methods) * cell + 0.28) / height,
                (cell - 2 * pad) / width,
                0.2 / height,
            ]
        )
        lo, hi = clims[dataset]
        colourbar = fig.colorbar(
            plt.cm.ScalarMappable(norm=Normalize(lo, hi), cmap=CMAP),
            cax=bar,
            orientation="horizontal",
        )
        ticks = np.linspace(lo, hi, 5)
        colourbar.set_ticks(ticks)
        colourbar.set_ticklabels([f"{t:.{tick_decimals(ticks)}f}" for t in ticks])
        colourbar.ax.tick_params(labelsize=8)
        colourbar.set_label("Local Res. (Å)", fontsize=9)
    for r, method in enumerate(methods):
        fig.text(
            left * 0.55 / width,
            1 - (top + (r + 0.5) * cell) / height,
            method_label(cfg, method),
            ha="center",
            va="center",
            rotation=90,
            fontsize=10,
            fontweight="bold",
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{stem}.png", out_dir / f"{stem}.pdf"]
    for path in paths:
        fig.savefig(path, dpi=int(cfg.figures.dpi), bbox_inches="tight")
    plt.close(fig)
    return paths


@hydra.main(version_base="1.3", config_path="../../configs_eval", config_name="paper")
def main(cfg: DictConfig) -> None:
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
    out_dir = Path(cfg.output_dir) / "figures"
    cell_dir = out_dir / "locres_render_cells"
    for stem, spec in cfg.locres_render.figures.items():
        datasets = [str(column[spec.kind]) for column in cfg.figures.columns]
        methods = [str(m) for m in spec.methods]
        clims = {d: colour_limits(cfg, d, methods) for d in datasets}
        cells, stats = {}, []
        for dataset in datasets:
            for method in methods:
                path = cell_dir / f"{dataset}__{method}.png"
                stats.append(render_cell(cfg, dataset, method, clims[dataset], path))
                cells[dataset, method] = path
                print(json.dumps(stats[-1]), flush=True)
        (out_dir / f"{stem}.json").write_text(
            json.dumps({"clims": clims, "cells": stats}, indent=1)
        )
        for path in assemble(cfg, spec, cells, clims, out_dir, stem):
            print(path)


if __name__ == "__main__":
    main()
