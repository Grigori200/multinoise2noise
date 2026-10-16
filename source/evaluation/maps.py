"""Reading and writing 3D maps."""

from pathlib import Path

import mrcfile
import numpy as np
import torch


def read_map(path: str | Path) -> tuple[torch.Tensor, float]:
    """Return the map as a float32 tensor of shape (Z, Y, X) and its voxel size in Å."""
    with mrcfile.open(str(path), permissive=True) as mrc:
        data = torch.tensor(np.asarray(mrc.data, dtype=np.float32))
        voxel_size = float(mrc.voxel_size.x)
    if data.ndim != 3:
        raise ValueError(f"{path}: expected a 3D map, got shape {tuple(data.shape)}")
    return data, voxel_size


def write_map(path: str | Path, data: torch.Tensor, voxel_size: float) -> None:
    """Write a 3D tensor as float32 MRC with the given voxel size in Å."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(str(path), overwrite=True) as mrc:
        mrc.set_data(data.detach().cpu().numpy().astype(np.float32))
        mrc.voxel_size = voxel_size
