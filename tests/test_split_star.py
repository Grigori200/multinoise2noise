import numpy as np
import pandas as pd
import pytest

from source.datasets.split_star import build_split_star


def make_stars(n_micrographs: int = 4, per_micrograph: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    rows = []
    for m in range(n_micrographs):
        for p in range(per_micrograph):
            rows.append(
                {
                    "rlnMicrographName": f"mic_{m}.mrc",
                    "rlnCoordinateX": float(10 * p),
                    "rlnCoordinateY": float(m),
                    "rlnAngleRot": float(rng.uniform(0, 360)),
                    "particle_index": len(rows),
                    "rlnRandomSubset": 1 + (p % 2),
                }
            )
    full = pd.DataFrame(rows)
    subset = full[full["rlnRandomSubset"] == 2].copy()
    subset["rlnRandomSubset"] = 1 + (np.arange(len(subset)) // 2) % 2
    test = subset[subset["rlnRandomSubset"] == 2].copy()
    test["rlnRandomSubset"] = 1 + np.arange(len(test)) % 2
    optics = pd.DataFrame({"rlnOpticsGroup": [1]})
    return (
        {"optics": optics, "particles": full},
        {"optics": optics, "particles": subset},
        test,
    )


def test_split_assignment_and_halves():
    full, subset, test = make_stars()
    star, summary = build_split_star(full, subset, test, train_half=1)
    out = star["particles"]
    assert len(out) == len(full["particles"])
    assert summary["counts"] == {"train": 8, "val": 16, "test": 8}
    key = ["rlnMicrographName", "rlnCoordinateX", "rlnCoordinateY"]
    sub = subset["particles"]
    train_keys = set(
        map(tuple, sub.loc[sub["rlnRandomSubset"] == 1, key].values.tolist())
    )
    assert (
        set(map(tuple, out.loc[out["split"] == "train", key].values.tolist()))
        == train_keys
    )
    assert summary["test_halves"] == {1: 4, 2: 4}
    merged = out[out["split"] == "test"].merge(test, on=key, suffixes=("", "_t"))
    assert (merged["rlnRandomSubset"] == merged["rlnRandomSubset_t"]).all()
    pd.testing.assert_series_equal(
        out["particle_index"], full["particles"]["particle_index"]
    )
    assert summary["micrographs_missing_split"] == {"train": 0, "val": 0, "test": 0}


def test_train_half_two_swaps_train_and_test():
    full, subset, _ = make_stars()
    sub = subset["particles"]
    test = sub[sub["rlnRandomSubset"] == 1].copy()
    test["rlnRandomSubset"] = 1 + np.arange(len(test)) % 2
    star, summary = build_split_star(full, subset, test, train_half=2)
    assert summary["counts"] == {"train": 8, "val": 16, "test": 8}


def test_rejects_wrong_test_halves():
    full, subset, test = make_stars()
    with pytest.raises(ValueError, match="exactly the test particles"):
        build_split_star(full, subset, test.iloc[:-1], train_half=1)


def test_rejects_subset_outside_full():
    full, subset, test = make_stars()
    subset["particles"].loc[subset["particles"].index[0], "rlnCoordinateX"] = -1.0
    with pytest.raises(ValueError, match="not in the full STAR"):
        build_split_star(full, subset, test, train_half=1)


def test_duplicate_test_rows_must_agree():
    full, subset, test = make_stars()
    dup = pd.concat([test, test.iloc[:1]])
    _, summary = build_split_star(full, subset, dup, train_half=1)
    assert summary["test_halves_duplicate_rows"] == 1
    bad = dup.copy()
    bad.iloc[-1, bad.columns.get_loc("rlnRandomSubset")] = (
        3 - bad.iloc[0]["rlnRandomSubset"]
    )
    with pytest.raises(ValueError, match="different rlnRandomSubset"):
        build_split_star(full, subset, bad, train_half=1)


def test_conflicting_duplicate_keeps_first_occurrence():
    full, subset, test = make_stars()
    first_half = int(test.iloc[0]["rlnRandomSubset"])
    extra = test.iloc[:1].copy()
    extra["rlnRandomSubset"] = 3 - first_half
    star, summary = build_split_star(
        full, subset, pd.concat([test, extra]), train_half=1, on_conflict="first"
    )
    out = star["particles"]
    key = ["rlnMicrographName", "rlnCoordinateX", "rlnCoordinateY"]
    row = out.merge(extra[key], on=key)
    assert len(row) == 1 and int(row["rlnRandomSubset"].iloc[0]) == first_half
    assert summary["test_halves_conflicts_kept_first"][0]["halves"] == [
        first_half,
        3 - first_half,
    ]
    assert summary["counts"]["test"] == len(test)
