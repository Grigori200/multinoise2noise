import pytest
import torch

import source.reconstruction.postprocessing as pp

SHAPE = (32, 32, 32)
ANGPIX = 2.5


def off_centre_mask() -> torch.Tensor:
    grid = torch.arange(32, dtype=torch.float32)
    z, y, x = torch.meshgrid(grid, grid, grid, indexing="ij")
    return (((z - 10) ** 2 + (y - 16) ** 2 + (x - 22) ** 2) <= 36).float()


@pytest.fixture
def recorded_centres(monkeypatch):
    centres = []
    original = pp.raised_cosine_mask

    def recording(shape, radius, radius_p, c0, c1, c2):
        centres.append((int(c0), int(c1), int(c2)))
        return original(shape, radius, radius_p, c0, c1, c2)

    monkeypatch.setattr(pp, "raised_cosine_mask", recording)
    return centres


def maps():
    gen = torch.Generator().manual_seed(0)
    return [torch.randn(SHAPE, generator=gen) for _ in range(4)]


def test_locres_windows_are_centred_inside_mask(recorded_centres):
    mask = off_centre_mask()
    h1, h2, h1r, h2r = maps()
    pp.locres(h1, h2, h1r, h2r, 0.0, ANGPIX, 3, mask=mask, locres_sampling=10)
    assert recorded_centres
    assert all(mask[c] > 0 for c in recorded_centres)


def test_local_sharpen_windows_are_centred_inside_mask(recorded_centres):
    mask = off_centre_mask()
    h1, h2, h1r, h2r = maps()
    pp.local_sharpen(h1, h2, h1r, h2r, ANGPIX, 3, 10.0, mask=mask, loc_sampling=10)
    assert recorded_centres
    assert all(mask[c] > 0 for c in recorded_centres)
