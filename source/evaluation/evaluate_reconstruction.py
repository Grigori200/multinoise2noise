"""Model-to-map FSC of half-map reconstructions from denoised test subsets.

For every (dataset, method, subset, half-map) the masked, phase-randomisation corrected FSC
(``fsc true`` of :func:`source.reconstruction.postprocessing.postprocess`) against the
reference map is computed. Each work item writes ``rows/<dataset>/<method>__s<subset>_h<half>``
``.json`` (scalars) and ``.npy`` (curve); :func:`collect` merges the rows into
``reconstruction_fsc.csv``.

Usage::

    python -m source.evaluation.evaluate_reconstruction 'select=[10025]'
    python -m source.evaluation.evaluate_reconstruction shard.index=3 shard.count=70
    python -m source.evaluation.evaluate_reconstruction collect_only=true
"""

import itertools
import json
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from source.evaluation.fsc_metrics import mean_fsc, resolution_at_threshold
from source.evaluation.make_mask import selected_datasets
from source.evaluation.maps import read_map
from source.reconstruction.postprocessing import postprocess


def fsc_true_vs_reference(
    half_map: torch.Tensor,
    reference: torch.Tensor,
    mask: torch.Tensor,
    angpix: float,
    randomize_fsc_at: float,
    vs_reference: bool,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Return the FSC curves of ``postprocess`` with a seeded phase randomisation."""
    if half_map.shape != reference.shape or mask.shape != reference.shape:
        raise ValueError(
            f"Shape mismatch: map {tuple(half_map.shape)}, reference {tuple(reference.shape)}, "
            f"mask {tuple(mask.shape)}"
        )
    torch.manual_seed(seed)
    result = postprocess(
        half_map,
        reference,
        angpix,
        999,
        mask=mask,
        randomize_fsc_at=randomize_fsc_at,
        local_resolution=False,
        vs_reference=vs_reference,
    )
    return result["fsc_postprocess"]


def work_items(cfg: DictConfig) -> list[tuple[str, str, int, int]]:
    items = []
    for dataset in selected_datasets(cfg):
        methods = [str(m) for m in cfg.datasets[dataset].methods]
        unknown = sorted(set(methods) - {str(m) for m in cfg.methods.keys()})
        if unknown:
            raise KeyError(f"{dataset}: methods {unknown} have no label in cfg.methods")
        items += itertools.product(
            [dataset],
            methods,
            [int(s) for s in cfg.reconstruction.subsets],
            [int(h) for h in cfg.reconstruction.halves],
        )
    return items


def half_map_path(
    cfg: DictConfig, dataset: str, method: str, subset: int, half: int
) -> Path:
    rec = cfg.reconstruction
    return (
        Path(cfg.datasets[dataset].results_dir)
        / method
        / f"Refine_{rec.refine}"
        / str(subset)
        / rec.image_mode
        / f"half_{half}_reconstruction.mrc"
    )


def row_paths(
    cfg: DictConfig, dataset: str, method: str, subset: int, half: int
) -> tuple[Path, Path]:
    stem = Path(cfg.output_dir) / "rows" / dataset / f"{method}__s{subset}_h{half}"
    return stem.with_suffix(".json"), stem.with_suffix(".npy")


def evaluate_item(
    cfg: DictConfig, dataset: str, method: str, subset: int, half: int, cache: dict
) -> tuple[dict, np.ndarray]:
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
    seed = int(cfg.fsc.seed) + subset * 10 + half
    curves = fsc_true_vs_reference(
        half_map,
        reference,
        mask,
        float(ds.angpix),
        float(cfg.fsc.randomize_fsc_at),
        bool(cfg.fsc.vs_reference),
        seed,
    )
    fsc_true = curves["fsc true"].double()
    row = {
        "dataset": dataset,
        "method": method,
        "subset": subset,
        "half": half,
        "refine": int(cfg.reconstruction.refine),
        "half_map": str(path),
        "reference_map": str(ds.reference_map),
        "mask": str(ds.mask),
        "angpix": float(ds.angpix),
        "randomize_fsc_at": float(cfg.fsc.randomize_fsc_at),
        "vs_reference": bool(cfg.fsc.vs_reference),
        "seed": seed,
        "n_shells": int(fsc_true.numel()),
        "mean_fsc": mean_fsc(fsc_true),
    }
    for threshold in cfg.fsc.thresholds:
        res = resolution_at_threshold(fsc_true, float(ds.angpix), float(threshold))
        row[f"resolution_{threshold}"] = res["resolution"]
        row[f"shell_{threshold}"] = res["shell"]
        row[f"crossed_{threshold}"] = res["crossed"]
    return row, fsc_true.numpy()


def collect(cfg: DictConfig) -> pd.DataFrame:
    rows = [
        json.loads(p.read_text())
        for p in sorted((Path(cfg.output_dir) / "rows").glob("*/*.json"))
    ]
    if not rows:
        raise FileNotFoundError(f"No rows under {cfg.output_dir}/rows")
    frame = pd.DataFrame(rows).sort_values(["dataset", "method", "subset", "half"])
    frame.to_csv(Path(cfg.output_dir) / "reconstruction_fsc.csv", index=False)
    return frame


@hydra.main(version_base="1.3", config_path="../../configs_eval", config_name="paper")
def main(cfg: DictConfig) -> None:
    if not cfg.get("collect_only", False):
        items = work_items(cfg)
        count, index = int(cfg.shard.count), int(cfg.shard.index)
        if not 0 <= index < count:
            raise ValueError(f"shard.index must be in [0, {count}), got {index}")
        cache: dict = {}
        for dataset, method, subset, half in items[index::count]:
            json_path, npy_path = row_paths(cfg, dataset, method, subset, half)
            if cfg.skip_existing and json_path.exists() and npy_path.exists():
                continue
            row, curve = evaluate_item(cfg, dataset, method, subset, half, cache)
            json_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(npy_path, curve)
            json_path.write_text(json.dumps(row, indent=1))
            print(json.dumps(row))
    if cfg.get("collect_only", False) or int(cfg.shard.count) == 1:
        collect(cfg)


if __name__ == "__main__":
    main()
