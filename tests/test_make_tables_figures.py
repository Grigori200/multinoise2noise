import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from source.evaluation import make_figures as mf
from source.evaluation import make_tables as mt

METHODS = ["noise2noise", "multi_noise2noise"]


def make_cfg(tmp_path):
    return OmegaConf.create(
        {
            "output_dir": str(tmp_path / "out"),
            "reconstruction": {"subsets": [1, 2], "halves": [1, 2]},
            "fsc": {"thresholds": [0.143, 0.5]},
            "methods": {
                "noise2noise": "Topaz-Denoise",
                "multi_noise2noise": "Multi-Noise2Noise",
            },
            "datasets": {
                "A": {"angpix": 1.0, "results_dir": str(tmp_path / "results" / "A")},
                "A_sim": {
                    "angpix": 1.0,
                    "results_dir": str(tmp_path / "results" / "A_sim"),
                },
            },
            "tables": {
                "fsc_recon": {
                    "sections": [
                        {"title": "Real", "datasets": {"A": "Ds"}, "methods": METHODS}
                    ]
                }
            },
            "figures": {
                "dpi": 50,
                "fsc_main_dataset": "A",
                "fsc_main_methods": METHODS,
                "columns": [{"title": "Ds", "real": "A", "sim": "A_sim"}],
                "rows": [
                    {"title": "Simulated", "kind": "sim", "methods": METHODS},
                    {"title": "Real", "kind": "real", "methods": METHODS},
                ],
                "styles": {
                    "noise2noise": {"color": "#6baed6", "linestyle": "-"},
                    "multi_noise2noise": {"color": "#41ab5d", "linestyle": "-"},
                },
            },
            "frame_sweep": {
                "cutoff_angstrom": 8.0,
                "csv_name": "frc_metrics_test.csv",
                "variants": {
                    "singular": {
                        "column": "singular_FRC",
                        "label": "single",
                        "color": "#1f77b4",
                        "marker": "s",
                    },
                    "mean": {
                        "column": "mean_FRC",
                        "label": "mean",
                        "color": "#ff7f0e",
                        "marker": "o",
                    },
                },
                "datasets": {"A_sim": {"title": "Ds", "n_frames": 10}},
            },
        }
    )


def fsc_frame():
    rows = []
    for method, offset in (("noise2noise", 0.5), ("multi_noise2noise", 0.0)):
        for i, (s, h) in enumerate([(1, 1), (1, 2), (2, 1), (2, 2)]):
            rows.append(
                {
                    "dataset": "A",
                    "method": method,
                    "subset": s,
                    "half": h,
                    "resolution_0.143": 3.0 + offset + 0.1 * i,
                    "resolution_0.5": 4.0 + offset + 0.1 * i,
                    "mean_fsc": 0.5 - offset / 10 + 0.01 * i,
                }
            )
    return pd.DataFrame(rows)


def test_summarise_mean_and_sample_std():
    stats = mt.summarise(fsc_frame(), expected_rows=4).set_index("method")
    values = 3.0 + 0.1 * np.arange(4)
    assert stats.at["multi_noise2noise", "resolution_0.143_mean"] == pytest.approx(
        values.mean()
    )
    assert stats.at["multi_noise2noise", "resolution_0.143_std"] == pytest.approx(
        values.std(ddof=1)
    )


def test_summarise_rejects_missing_rows():
    with pytest.raises(ValueError, match="Expected 4 rows"):
        mt.summarise(fsc_frame().iloc[1:], expected_rows=4)


def test_fsc_recon_rows_bold_marks_best(tmp_path):
    cfg = make_cfg(tmp_path)
    lines = mt.fsc_recon_rows(cfg, mt.summarise(fsc_frame(), 4))
    topaz = next(line for line in lines if "Topaz-Denoise" in line)
    multi = next(line for line in lines if "Multi-Noise2Noise" in line)
    assert multi.count("\\mathbf") == 3
    assert topaz.count("\\mathbf") == 0


def test_fsc_recon_rows_ties_are_all_bold(tmp_path):
    frame = fsc_frame()
    frame.loc[
        frame.method == "noise2noise",
        ["resolution_0.143", "resolution_0.5", "mean_fsc"],
    ] = frame.loc[
        frame.method == "multi_noise2noise",
        ["resolution_0.143", "resolution_0.5", "mean_fsc"],
    ].to_numpy()
    lines = mt.fsc_recon_rows(make_cfg(tmp_path), mt.summarise(frame, 4))
    assert all(
        line.count("\\mathbf") == 3
        for line in lines
        if "Noise2Noise" in line or "Topaz" in line
    )


def test_fsc_recon_rows_missing_method(tmp_path):
    frame = fsc_frame()
    with pytest.raises(KeyError, match="no FSC rows"):
        mt.fsc_recon_rows(
            make_cfg(tmp_path), mt.summarise(frame[frame.method == "noise2noise"], 4)
        )


def test_frc_mean_beyond_selects_rings_finer_than_cutoff():
    frc = np.arange(11, dtype=float)
    # Ring i has resolution 2 * 1 * 11 / i <= 8 for i >= 2.75, i.e. rings 3..10.
    assert mf.frc_mean_beyond(frc, angpix=1.0, cutoff_angstrom=8.0) == pytest.approx(
        np.arange(3, 11).mean()
    )


def test_frc_mean_beyond_unreachable_cutoff():
    with pytest.raises(ValueError, match="No FRC ring"):
        mf.frc_mean_beyond(np.ones(5), angpix=1.0, cutoff_angstrom=1.0)


def write_fsc_rows(cfg, dataset, n_shells=21, n_rows=4):
    rows_dir = Path(cfg.output_dir) / "rows" / dataset
    rows_dir.mkdir(parents=True, exist_ok=True)
    for method in METHODS:
        for k, (s, h) in enumerate([(1, 1), (1, 2), (2, 1), (2, 2)][:n_rows]):
            stem = rows_dir / f"{method}__s{s}_h{h}"
            stem.with_suffix(".json").write_text(json.dumps({"angpix": 1.0}))
            np.save(stem.with_suffix(".npy"), np.linspace(1, 0, n_shells) - 0.01 * k)


def test_load_fsc_curves_frequency_axis(tmp_path):
    cfg = make_cfg(tmp_path)
    write_fsc_rows(cfg, "A")
    frequency, curves = mf.load_fsc_curves(cfg, "A", "noise2noise")
    assert curves.shape == (4, 21)
    assert frequency[0] == 0 and frequency[-1] == pytest.approx(0.5)


def test_load_fsc_curves_rejects_missing_rows(tmp_path):
    cfg = make_cfg(tmp_path)
    write_fsc_rows(cfg, "A", n_rows=3)
    with pytest.raises(FileNotFoundError, match="expected 4 FSC rows"):
        mf.load_fsc_curves(cfg, "A", "noise2noise")


def test_figures_render(tmp_path):
    cfg = make_cfg(tmp_path)
    for dataset in ("A", "A_sim"):
        write_fsc_rows(cfg, dataset)
    locres_dir = tmp_path / "out" / "locres"
    locres_dir.mkdir(parents=True)
    bins = np.linspace(0.1, 0.4, 7)
    pd.DataFrame(
        [
            {
                "dataset": d,
                "method": m,
                "frequency_low": lo,
                "frequency_high": hi,
                "voxel_percent": 10.0,
            }
            for d in ("A", "A_sim")
            for m in METHODS
            for lo, hi in zip(bins[:-1], bins[1:])
        ]
    ).to_csv(locres_dir / "locres_histograms.csv", index=False)
    for frames in (1, 5, 9):
        split = tmp_path / "results" / "A_sim" / "frame_sweep" / f"split_{frames}"
        split.mkdir(parents=True)
        pd.DataFrame(
            {"singular_FRC": np.linspace(1, 0, 11), "mean_FRC": np.linspace(1, 0.1, 11)}
        ).to_csv(split / "frc_metrics_test.csv", index=False)
    out_dir = tmp_path / "out" / "figures"
    paths = []
    for make in (
        mf.figure_frame_sweep,
        mf.figure_fsc_main,
        mf.figure_fsc_grid,
        mf.figure_locres_histograms,
    ):
        paths += make(cfg, out_dir)
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)
    assert {p.name for p in paths} >= {
        "frame_sweep.png",
        "fsc_true_A.pdf",
        "fsc_true_3x3.png",
    }
    sweep = pd.read_csv(out_dir / "frame_sweep.csv")
    assert len(sweep) == 3 * 2
    assert sorted(sweep.frames.unique()) == [1, 5, 9]


def test_frame_sweep_missing_column(tmp_path):
    cfg = make_cfg(tmp_path)
    split = tmp_path / "results" / "A_sim" / "frame_sweep" / "split_1"
    split.mkdir(parents=True)
    pd.DataFrame({"singular_FRC": np.ones(11)}).to_csv(
        split / "frc_metrics_test.csv", index=False
    )
    with pytest.raises(KeyError, match="mean_FRC"):
        mf.load_frame_sweep(cfg)
