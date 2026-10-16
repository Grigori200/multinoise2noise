"""Local model-to-map resolution of half-map reconstructions from denoised test subsets.

Stage ``items`` (default): for every (dataset, method, subset, half-map), the local FSC between
the half-map and the reference map is computed in raised-cosine windows as in
:func:`source.reconstruction.postprocessing.locres` (window radius ``sampling / 2``, step and
soft edge ``sampling``, in Å). The local resolution of a window is taken at the first shell
where its phase-randomisation corrected FSC falls below ``locres.threshold``, and windows are
combined by a weighted harmonic mean. Each item writes ``locres/maps/<dataset>/<method>__s<s>_h<h>.mrc``
and a JSON row.

Stage ``aggregate``: the item maps of each (dataset, method) are averaged and symmetrised; the
mask is symmetrised by a minimum over the symmetry rotations. Voxels with symmetrised mask
``> locres.mask_core`` enter ``locres_summary.csv`` and ``locres_histograms.csv``. Histograms
are over spatial frequency (1/Å) in percent of voxels; the range of a dataset starts at the
``histogram.lower_percentile`` of the frequencies of all its methods and ends at the finest
resolution reached by any of them. Existing averages and symmetrised masks are reused when
``skip_existing`` is set.

Usage::

    python -m source.evaluation.evaluate_local_resolution shard.index=3 shard.count=72
    python -m source.evaluation.evaluate_local_resolution locres.stage=aggregate
"""

import json
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from tqdm import tqdm

from source.evaluation.evaluate_reconstruction import half_map_path, work_items
from source.evaluation.fsc_metrics import resolution_at_threshold
from source.evaluation.make_mask import selected_datasets
from source.evaluation.maps import read_map, write_map
from source.reconstruction.postprocessing import (
    calculate_fsc_true,
    determine_randomization_threshold,
    filter_coordinates_by_mask,
    generate_3d_coordinates,
    randomize_phases,
    raised_cosine_mask,
    symmetrise_results,
)
from source.reconstruction.symmetry import Symmetry
from source.reconstruction.utils import calculate_fsc, real_round

SYMMETRY_DIR = Path(__file__).resolve().parents[1] / "reconstruction" / "symmetry_files"


def window_at(
    shape: tuple[int, int, int], radius: int, edge: int, centre: torch.Tensor
) -> torch.Tensor:
    """Raised-cosine window centred on array index ``centre`` = (z, y, x).

    ``raised_cosine_mask`` offsets the array axes in argument order, so the centre is passed
    in array-axis order.
    """
    return raised_cosine_mask(shape, radius, radius + edge, *centre.tolist()).float()


def local_resolution_map(
    half_map: torch.Tensor,
    reference: torch.Tensor,
    mask: torch.Tensor,
    angpix: float,
    sampling: float,
    threshold: float,
    randomize_fsc_at: float,
    vs_reference: bool,
    seed: int,
) -> tuple[torch.Tensor, dict]:
    """Return the local resolution map (Å) and window statistics."""
    if half_map.shape != reference.shape or mask.shape != reference.shape:
        raise ValueError(
            f"Shape mismatch: map {tuple(half_map.shape)}, reference {tuple(reference.shape)}, "
            f"mask {tuple(mask.shape)}"
        )
    torch.manual_seed(seed)
    randomize_at = determine_randomization_threshold(
        calculate_fsc(half_map, reference), randomize_fsc_at
    )
    half_r = randomize_phases(half_map, randomize_at)
    ref_r = reference if vs_reference else randomize_phases(reference, randomize_at)

    step = real_round(torch.tensor(sampling / angpix))
    radius = real_round(torch.tensor(sampling / angpix / 2))
    coords = filter_coordinates_by_mask(
        generate_3d_coordinates(half_map.shape, step), mask > 0
    )
    if coords.shape[0] == 0:
        raise ValueError("No window centre lies inside the mask")
    inv_res = weights = 0
    resolutions, uncrossed = [], 0
    for centre in tqdm(coords, total=coords.shape[0], desc="locres"):
        window = window_at(half_map.shape, radius, step, centre)
        window = window * mask
        fsc_true = calculate_fsc_true(
            calculate_fsc(half_map * window, reference * window),
            calculate_fsc(half_r * window, ref_r * window),
            randomize_at,
        )
        res = resolution_at_threshold(fsc_true, angpix, threshold)
        uncrossed += not res["crossed"]
        resolutions.append(res["resolution"])
        inv_res = inv_res + window / res["resolution"]
        weights = weights + window
    local = torch.where(
        weights > 0, weights / inv_res.clamp_min(1e-12), torch.zeros_like(weights)
    )
    stats = {
        "n_windows": int(coords.shape[0]),
        "n_windows_uncrossed": int(uncrossed),
        "n_windows_shell0": int(sum(np.isinf(resolutions))),
        "window_step_px": int(step),
        "window_radius_px": int(radius),
        "randomize_at_shell": int(randomize_at),
    }
    return local, stats


def rotated_copies(volume: torch.Tensor, symmetry: Symmetry):
    """Yield ``volume`` resampled under every non-identity rotation of ``symmetry``.

    Uses the grid convention of :func:`source.reconstruction.postprocessing.symmetrise_results`.
    """
    D, H, W = volume.shape
    z, y, x = torch.meshgrid(
        torch.linspace(-1, 1, D, dtype=volume.dtype),
        torch.linspace(-1, 1, H, dtype=volume.dtype),
        torch.linspace(-1, 1, W, dtype=volume.dtype),
        indexing="ij",
    )
    grid = torch.stack([x, y, z], dim=-1).reshape(-1, 3)
    for r in symmetry.rr:
        rotation = r[:3, :3].to(dtype=volume.dtype)
        rotated = (grid @ rotation.T).reshape(1, D, H, W, 3)
        yield F.grid_sample(
            volume[None, None],
            rotated,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )[0, 0]


def symmetrise_mask(mask: torch.Tensor, symmetry: Symmetry) -> torch.Tensor:
    result = mask.clone()
    for rotated in rotated_copies(mask, symmetry):
        result = torch.minimum(result, rotated)
    return result


def load_symmetry(name: str) -> Symmetry:
    symmetry = Symmetry(name, str(SYMMETRY_DIR))
    symmetry.create_rotation_matrices()
    return symmetry


def item_paths(cfg: DictConfig, dataset: str, method: str, subset: int, half: int):
    root = Path(cfg.output_dir) / "locres"
    name = f"{method}__s{subset}_h{half}"
    return (
        root / "maps" / dataset / f"{name}.mrc",
        root / "rows" / dataset / f"{name}.json",
    )


def value_summary(values: np.ndarray) -> dict:
    finite = values[np.isfinite(values)]
    return {
        "n_voxels": int(values.size),
        "n_voxels_infinite": int(values.size - finite.size),
        "median": float(np.median(finite)),
        "mean": float(finite.mean()),
        "p2": float(np.percentile(finite, 2)),
        "p98": float(np.percentile(finite, 98)),
        "finest": float(finite.min()),
        "coarsest": float(finite.max()),
    }


def run_items(cfg: DictConfig) -> None:
    lc = cfg.locres
    items = work_items(cfg)
    count, index = int(cfg.shard.count), int(cfg.shard.index)
    if not 0 <= index < count:
        raise ValueError(f"shard.index must be in [0, {count}), got {index}")
    cache: dict = {}
    for dataset, method, subset, half in items[index::count]:
        map_out, json_out = item_paths(cfg, dataset, method, subset, half)
        if cfg.skip_existing and map_out.exists() and json_out.exists():
            continue
        ds = cfg.datasets[dataset]
        if dataset not in cache:
            reference, ref_vs = read_map(ds.reference_map)
            mask, mask_vs = read_map(ds.mask)
            for label, vs in (("reference", ref_vs), ("mask", mask_vs)):
                if abs(vs - ds.angpix) > 1e-3:
                    raise ValueError(
                        f"{dataset}: {label} voxel size {vs} != angpix {ds.angpix}"
                    )
            cache.clear()
            cache[dataset] = (reference, mask)
        reference, mask = cache[dataset]
        path = half_map_path(cfg, dataset, method, subset, half)
        half_map, _ = read_map(path)
        seed = int(lc.seed) + subset * 10 + half
        local, stats = local_resolution_map(
            half_map,
            reference,
            mask,
            float(ds.angpix),
            float(lc.sampling_angstrom),
            float(lc.threshold),
            float(cfg.fsc.randomize_fsc_at),
            bool(lc.vs_reference),
            seed,
        )
        write_map(map_out, local, float(ds.angpix))
        row = {
            "dataset": dataset,
            "method": method,
            "subset": subset,
            "half": half,
            "half_map": str(path),
            "reference_map": str(ds.reference_map),
            "mask": str(ds.mask),
            "local_resolution_map": str(map_out),
            "sampling_angstrom": float(lc.sampling_angstrom),
            "threshold": float(lc.threshold),
            "vs_reference": bool(lc.vs_reference),
            "seed": seed,
            **stats,
            **value_summary(local[mask > float(lc.mask_core)].numpy()),
        }
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(row, indent=1))
        print(json.dumps(row))


def run_aggregate(cfg: DictConfig) -> None:
    lc = cfg.locres
    root = Path(cfg.output_dir) / "locres"
    items = work_items(cfg)
    missing = [i for i in items if not item_paths(cfg, *i)[0].exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} item maps missing, e.g. {missing[:3]}")
    summary_rows, histogram_rows = [], []
    for dataset in selected_datasets(cfg):
        ds = cfg.datasets[dataset]
        angpix = float(ds.angpix)
        symmetry = load_symmetry(str(ds.symmetry))
        mask_path = root / "averages" / f"{dataset}__mask_sym.mrc"
        if cfg.skip_existing and mask_path.exists():
            mask_sym, _ = read_map(mask_path)
        else:
            mask_sym = symmetrise_mask(read_map(ds.mask)[0], symmetry)
            write_map(mask_path, mask_sym, angpix)
        core = mask_sym > float(lc.mask_core)
        values = {}
        for method in [str(m) for m in ds.methods]:
            keys = [i for i in items if i[0] == dataset and i[1] == method]
            avg_path = root / "averages" / f"{dataset}__{method}.mrc"
            if cfg.skip_existing and avg_path.exists():
                average, _ = read_map(avg_path)
            else:
                maps = [read_map(item_paths(cfg, *k)[0])[0] for k in keys]
                average = symmetrise_results(torch.stack(maps).mean(0), symmetry)
                write_map(avg_path, average, angpix)
            values[method] = average[core].numpy()
            summary_rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "n_maps": len(keys),
                    **value_summary(values[method]),
                }
            )
        frequencies = {m: 1.0 / v[np.isfinite(v) & (v > 0)] for m, v in values.items()}
        pooled = np.concatenate(list(frequencies.values()))
        low = float(np.percentile(pooled, float(lc.histogram.lower_percentile)))
        high = float(pooled.max())
        edges = np.linspace(low, high, int(lc.histogram.n_bins) + 1)
        for method, f in frequencies.items():
            counts, _ = np.histogram(f, bins=edges)
            for lo, hi, c in zip(edges[:-1], edges[1:], counts):
                histogram_rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "frequency_low": float(lo),
                        "frequency_high": float(hi),
                        "voxel_percent": 100.0 * float(c) / f.size,
                    }
                )
    pd.DataFrame(summary_rows).to_csv(root / "locres_summary.csv", index=False)
    pd.DataFrame(histogram_rows).to_csv(root / "locres_histograms.csv", index=False)


@hydra.main(version_base="1.3", config_path="../../configs_eval", config_name="paper")
def main(cfg: DictConfig) -> None:
    stage = str(cfg.locres.stage)
    if stage == "items":
        run_items(cfg)
    elif stage == "aggregate":
        run_aggregate(cfg)
    else:
        raise ValueError(f"locres.stage must be 'items' or 'aggregate', got {stage}")


if __name__ == "__main__":
    main()
