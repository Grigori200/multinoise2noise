import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from source.evaluation import render_locres as rl
from source.evaluation.maps import write_map


def test_tick_decimals_keeps_labels_distinct():
    assert rl.tick_decimals(np.linspace(3.5, 4.9, 5)) == 1
    assert rl.tick_decimals(np.linspace(2.96, 3.16, 5)) == 2


def test_method_label_breaks_before_first_plus():
    cfg = OmegaConf.create({"methods": {"m": "Multi-Noise2Noise + CTF + DW"}})
    assert rl.method_label(cfg, "m") == "Multi-Noise2Noise\n+ CTF + DW"


def make_cfg(tmp_path):
    return OmegaConf.create(
        {
            "output_dir": str(tmp_path),
            "locres": {"mask_core": 0.5},
            "locres_render": {"clim_percentiles": [0, 100], "clim_margin": 0.1},
        }
    )


def write_inputs(tmp_path, maps):
    root = tmp_path / "locres" / "averages"
    mask = torch.zeros(4, 4, 4)
    mask[1:3, 1:3, 1:3] = 1.0
    write_map(root / "ds__mask_sym.mrc", mask, 1.0)
    for method, value in maps.items():
        write_map(root / f"ds__{method}.mrc", value, 1.0)
    return mask.bool()


def test_colour_limits_pool_methods_inside_core_only(tmp_path):
    core = torch.zeros(4, 4, 4, dtype=torch.bool)
    core[1:3, 1:3, 1:3] = True
    a = torch.where(core, torch.tensor(3.0), torch.tensor(100.0))
    b = torch.where(core, torch.tensor(5.0), torch.tensor(0.5))
    write_inputs(tmp_path, {"a": a, "b": b})
    lo, hi = rl.colour_limits(make_cfg(tmp_path), "ds", ["a", "b"])
    assert lo == pytest.approx(3.0 - 0.2)
    assert hi == pytest.approx(5.0 + 0.2)


def test_colour_limits_missing_average_raises(tmp_path):
    write_inputs(tmp_path, {"a": torch.full((4, 4, 4), 3.0)})
    with pytest.raises(FileNotFoundError):
        rl.colour_limits(make_cfg(tmp_path), "ds", ["a", "missing"])
