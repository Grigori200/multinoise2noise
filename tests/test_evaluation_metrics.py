import math

import pytest
import torch

from source.evaluation.evaluate_reconstruction import fsc_true_vs_reference
from source.evaluation.fsc_metrics import mean_fsc, resolution_at_threshold
from source.evaluation.make_mask import make_tight_mask, mask_statistics
from source.reconstruction.utils import calculate_fsc, calculate_resolution

ANGPIX = 1.25


def blob_volume(size: int = 32, sigma: float = 4.0) -> torch.Tensor:
    grid = torch.arange(size, dtype=torch.float32) - size / 2
    z, y, x = torch.meshgrid(grid, grid, grid, indexing="ij")
    return torch.exp(-(x**2 + y**2 + z**2) / (2 * sigma**2))


def test_fsc_of_identical_volumes_is_one():
    torch.manual_seed(0)
    volume = torch.randn(24, 24, 24)
    fsc = calculate_fsc(volume, volume)
    assert fsc.shape == (13,)
    torch.testing.assert_close(fsc.double(), torch.ones(13, dtype=torch.float64))


def test_first_shell_below_rule():
    fsc = torch.tensor([1.0, 0.9, 0.6, 0.4, 0.2, 0.1])
    res = resolution_at_threshold(fsc, ANGPIX, 0.5)
    assert res["shell"] == 3 and res["crossed"]
    assert res["resolution"] == pytest.approx((len(fsc) - 1) * 2 * ANGPIX / 3)
    legacy = calculate_resolution(fsc, ANGPIX, 0.5)
    assert float(legacy) == pytest.approx((len(fsc) - 1) * 2 * ANGPIX / 2)


def test_threshold_equal_value_is_not_below():
    fsc = torch.tensor([1.0, 0.5, 0.5, 0.49])
    assert resolution_at_threshold(fsc, ANGPIX, 0.5)["shell"] == 3


def test_never_crossing_reports_nyquist():
    fsc = torch.ones(10)
    res = resolution_at_threshold(fsc, ANGPIX, 0.143)
    assert not res["crossed"]
    assert res["resolution"] == pytest.approx(2 * ANGPIX)


def test_mean_fsc_includes_all_shells():
    fsc = torch.tensor([1.0, 0.5, 0.0, -0.5])
    assert mean_fsc(fsc) == pytest.approx(0.25)


def test_tight_mask_shape_and_range():
    reference = blob_volume()
    mask = make_tight_mask(
        reference, threshold_fraction=0.05, extend_px=3, soft_edge_px=3
    )
    assert mask.shape == reference.shape
    assert float(mask.min()) >= 0.0 and float(mask.max()) <= 1.0
    assert bool((mask[reference >= 0.05] >= 0.999).all())
    stats = mask_statistics(mask, reference)
    assert stats["reference_1sigma_in_core"] == pytest.approx(1.0)
    assert 0 < stats["box_fraction_core"] <= stats["box_fraction_support"] < 1


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1])
def test_tight_mask_rejects_bad_threshold(bad):
    with pytest.raises(ValueError):
        make_tight_mask(
            blob_volume(), threshold_fraction=bad, extend_px=3, soft_edge_px=3
        )


def test_seeded_fsc_true_is_deterministic():
    torch.manual_seed(1)
    reference = blob_volume() + 0.05 * torch.randn(32, 32, 32)
    half_map = reference + 0.3 * torch.randn(32, 32, 32)
    mask = make_tight_mask(blob_volume(), 0.05, 3, 3)
    a = fsc_true_vs_reference(half_map, reference, mask, ANGPIX, 0.8, False, seed=7)
    b = fsc_true_vs_reference(half_map, reference, mask, ANGPIX, 0.8, False, seed=7)
    torch.testing.assert_close(a["fsc true"], b["fsc true"])
    assert math.isclose(float(a["fsc true"][0]), 1.0, abs_tol=1e-6)
