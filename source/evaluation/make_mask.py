"""Tight molecular masks from reference maps.

The mask is ``auto_mask(reference, threshold_fraction * max(reference), [extend_px],
soft_edge_px)``: binary threshold, binary extension by ``extend_px`` voxels and a raised-cosine
edge of ``soft_edge_px`` voxels added outside the binary core.

Usage::

    python -m source.evaluation.make_mask                      # all datasets in paper.yaml
    python -m source.evaluation.make_mask 'select=[10025]'     # one dataset
"""

import json
from pathlib import Path

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig

from source.evaluation.maps import read_map, write_map
from source.reconstruction.mask import auto_mask


def make_tight_mask(
    reference: torch.Tensor,
    threshold_fraction: float,
    extend_px: int,
    soft_edge_px: int,
) -> torch.Tensor:
    """Return a soft mask in [0, 1] with the same shape as ``reference``."""
    if not 0 < threshold_fraction < 1:
        raise ValueError(
            f"threshold_fraction must be in (0, 1), got {threshold_fraction}"
        )
    if extend_px < 0 or soft_edge_px < 0:
        raise ValueError("extend_px and soft_edge_px must be non-negative")
    threshold = threshold_fraction * float(reference.max())
    mask = auto_mask(reference, threshold, [int(extend_px)], int(soft_edge_px)).float()
    return mask.clamp(0.0, 1.0)


def mask_statistics(mask: torch.Tensor, reference: torch.Tensor) -> dict:
    """Coverage of the reference density by the mask."""
    core = mask >= 0.999
    support = mask > 0
    strong = reference >= reference.mean() + reference.std()
    n_strong = int(strong.sum())
    return {
        "box_fraction_support": float(support.float().mean()),
        "box_fraction_core": float(core.float().mean()),
        "reference_1sigma_voxels": n_strong,
        "reference_1sigma_in_core": float((strong & core).sum()) / max(n_strong, 1),
        "reference_1sigma_in_support": float((strong & support).sum())
        / max(n_strong, 1),
    }


def selected_datasets(cfg: DictConfig) -> list[str]:
    names = [str(name) for name in cfg.datasets.keys()]
    if cfg.get("select"):
        wanted = [str(name) for name in cfg.select]
        unknown = sorted(set(wanted) - set(names))
        if unknown:
            raise KeyError(f"Unknown datasets {unknown}; known: {names}")
        names = wanted
    return names


@hydra.main(version_base="1.3", config_path="../../configs_eval", config_name="paper")
def main(cfg: DictConfig) -> None:
    out_dir = Path(cfg.output_dir) / "masks"
    rows = []
    for name in selected_datasets(cfg):
        dataset = cfg.datasets[name]
        reference, voxel_size = read_map(dataset.reference_map)
        if abs(voxel_size - dataset.angpix) > 1e-3:
            raise ValueError(
                f"{name}: reference voxel size {voxel_size} != angpix {dataset.angpix}"
            )
        mask = make_tight_mask(reference, **cfg.mask)
        path = out_dir / f"{name}_mask_tight.mrc"
        write_map(path, mask, voxel_size)
        row = {
            "dataset": name,
            "reference_map": str(dataset.reference_map),
            "mask": str(path),
        }
        row.update({k: cfg.mask[k] for k in cfg.mask})
        row.update(mask_statistics(mask, reference))
        rows.append(row)
        print(json.dumps(row))
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "masks.csv"
    frame = pd.DataFrame(rows)
    if csv_path.exists():
        previous = pd.read_csv(csv_path, dtype={"dataset": str})
        frame = pd.concat(
            [previous[~previous["dataset"].isin(frame["dataset"])], frame]
        )
    frame.to_csv(csv_path, index=False)


if __name__ == "__main__":
    main()
