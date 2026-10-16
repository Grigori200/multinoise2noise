import numpy as np
import torch

from source.evaluation.evaluate_local_resolution import (
    load_symmetry,
    local_resolution_map,
    symmetrise_mask,
    value_summary,
    window_at,
)
from source.evaluation.make_mask import make_tight_mask

ANGPIX = 2.5


def blob_volume(size: int, sigma: float) -> torch.Tensor:
    grid = torch.arange(size, dtype=torch.float32) - size // 2
    z, y, x = torch.meshgrid(grid, grid, grid, indexing="ij")
    return torch.exp(-(x**2 + y**2 + z**2) / (2 * sigma**2))


def test_window_is_centred_on_array_index():
    centre = torch.tensor([5, 12, 20])
    window = window_at((32, 32, 32), radius=3, edge=2, centre=centre)
    peak = torch.nonzero(window == window.max())
    assert window[5, 12, 20] == 1.0
    assert peak.float().mean(0).tolist() == [5.0, 12.0, 20.0]


def test_identical_maps_reach_nyquist():
    reference = blob_volume(32, sigma=5.0) + 0.1 * torch.randn(
        32, 32, 32, generator=torch.Generator().manual_seed(0)
    )
    mask = make_tight_mask(blob_volume(32, sigma=5.0), 0.05, 2, 2)
    local, stats = local_resolution_map(
        reference, reference, mask, ANGPIX, 25.0, 0.5, 0.8, True, seed=0
    )
    assert stats["n_windows"] > 0
    assert stats["n_windows_uncrossed"] == stats["n_windows"]
    values = local[mask > 0.5]
    torch.testing.assert_close(values, torch.full_like(values, 2 * ANGPIX))


def test_noisier_map_is_coarser_and_deterministic():
    gen = torch.Generator().manual_seed(1)
    reference = blob_volume(32, sigma=5.0) + 0.1 * torch.randn(
        32, 32, 32, generator=gen
    )
    noisy = reference + 0.5 * torch.randn(32, 32, 32, generator=gen)
    mask = make_tight_mask(blob_volume(32, sigma=5.0), 0.05, 2, 2)
    a, _ = local_resolution_map(noisy, reference, mask, ANGPIX, 25.0, 0.5, 0.8, True, 3)
    b, _ = local_resolution_map(noisy, reference, mask, ANGPIX, 25.0, 0.5, 0.8, True, 3)
    torch.testing.assert_close(a, b)
    assert float(a[mask > 0.5].median()) > 2 * ANGPIX


def test_symmetric_mask_is_kept_by_symmetrisation():
    mask = make_tight_mask(blob_volume(33, sigma=5.0), 0.05, 2, 2).double()
    sym = symmetrise_mask(mask, load_symmetry("D7"))
    grid = torch.arange(33, dtype=torch.float64) - 16
    z, y, x = torch.meshgrid(grid, grid, grid, indexing="ij")
    radius = (x**2 + y**2 + z**2).sqrt()
    core_radius = float(radius[mask > 0.99].max())
    interior = radius <= core_radius - 2
    assert float(sym[interior].min()) > 0.999
    assert float(sym[radius > core_radius + 4].max()) < 1e-9
    assert bool((sym <= mask + 1e-9).all())


def test_value_summary_ignores_infinite():
    summary = value_summary(np.array([3.0, 4.0, np.inf, 5.0]))
    assert summary["n_voxels_infinite"] == 1 and summary["finest"] == 3.0
    assert summary["median"] == 4.0
