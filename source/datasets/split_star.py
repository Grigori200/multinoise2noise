"""Build the training STAR with an explicit train / val / test ``split`` column.

Inputs (all from the dataset repository):

- ``<ds>.star``: every particle of the dataset.
- ``<ds>_subset.star``: the ``rlnRandomSubset == 2`` half of ``<ds>.star``, re-split into two
  new halves. Half ``train_half`` of it is the training quarter, the other half was used for
  validation in the original runs.
- ``subsets/denoised_1_1.star``: the test quarter with its own two reconstruction halves.
  It is only read. A particle listed in both halves is an error unless
  ``--on-conflict first`` is given, which keeps its first occurrence; such particles are
  listed in the summary.

Output: all particles of ``<ds>.star`` with ``split`` set to

- ``train``: ``<ds>_subset.star`` particles with ``rlnRandomSubset == train_half``,
- ``test``: the remaining ``<ds>_subset.star`` particles,
- ``val``: particles of ``<ds>.star`` that are not in ``<ds>_subset.star``.

``rlnRandomSubset`` is taken from ``<ds>_subset.star`` for train particles, from the
test-halves STAR for test particles and from ``<ds>.star`` for validation particles. All other
columns are copied from ``<ds>.star``. Particles are matched on micrograph name and
coordinates.

Usage::

    python -m source.datasets.split_star --full 10948.star --subset 10948_subset.star \
        --test-halves subsets/denoised_1_1.star --out 10948_split.star
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import starfile

KEY = ["rlnMicrographName", "rlnCoordinateX", "rlnCoordinateY"]
SPLITS = ("train", "val", "test")


def _particles(star) -> pd.DataFrame:
    return star["particles"] if isinstance(star, dict) else star


def _index(
    df: pd.DataFrame,
    name: str,
    allow_duplicates: bool = False,
    on_conflict: str = "error",
) -> tuple[pd.DataFrame, list[dict]]:
    """Index ``df`` by ``KEY``; return it with the duplicated particles whose halves differ.

    Duplicated rows are only accepted with ``allow_duplicates``. Duplicates with different
    ``rlnRandomSubset`` raise unless ``on_conflict == 'first'``, which keeps the first row.
    """
    if on_conflict not in ("error", "first"):
        raise ValueError(f"on_conflict must be 'error' or 'first', got {on_conflict}")
    missing = [c for c in KEY + ["rlnRandomSubset"] if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns {missing}")
    if df[KEY].isna().any().any():
        raise ValueError(f"{name} has empty values in {KEY}")
    keyed = df.set_index(KEY)
    conflicts = []
    if keyed.index.has_duplicates:
        if not allow_duplicates:
            raise ValueError(
                f"{name} has duplicated particles (same micrograph and coordinates)"
            )
        halves = keyed.groupby(level=KEY)["rlnRandomSubset"].agg(list)
        for key, values in halves[halves.map(lambda v: len(set(v)) > 1)].items():
            conflicts.append(
                {**dict(zip(KEY, key)), "halves": [int(v) for v in values]}
            )
        if conflicts and on_conflict == "error":
            raise ValueError(
                f"{name} has {len(conflicts)} duplicated particles with different "
                f"rlnRandomSubset: {conflicts}; pass on_conflict='first' to keep the "
                "first occurrence"
            )
        keyed = keyed[~keyed.index.duplicated(keep="first")]
    return keyed, conflicts


def build_split_star(
    full: dict,
    subset: dict,
    test_halves,
    train_half: int = 1,
    on_conflict: str = "error",
) -> tuple[dict, dict]:
    """Return the split STAR (dict with 'optics' and 'particles') and a summary dict."""
    if train_half not in (1, 2):
        raise ValueError(f"train_half must be 1 or 2, got {train_half}")
    full_p = _particles(full)
    full_k, _ = _index(full_p, "full STAR")
    sub_k, _ = _index(_particles(subset), "subset STAR")
    test_k, conflicts = _index(
        _particles(test_halves),
        "test-halves STAR",
        allow_duplicates=True,
        on_conflict=on_conflict,
    )

    if not sub_k.index.isin(full_k.index).all():
        raise ValueError("subset STAR contains particles that are not in the full STAR")
    in_subset = full_k.index.isin(sub_k.index)
    sub_half = sub_k["rlnRandomSubset"].reindex(full_k.index)
    split = pd.Series("val", index=full_k.index)
    split[in_subset & (sub_half == train_half)] = "train"
    split[in_subset & (sub_half != train_half)] = "test"

    test_index = split.index[split == "test"]
    if set(test_index) != set(test_k.index):
        raise ValueError(
            "test-halves STAR does not contain exactly the test particles "
            f"({len(test_k)} vs {len(test_index)})"
        )
    new_half = full_k["rlnRandomSubset"].copy()
    new_half[split == "train"] = sub_half[split == "train"]
    new_half[split == "test"] = test_k["rlnRandomSubset"].reindex(test_index)

    out = full_p.copy()
    out["rlnRandomSubset"] = new_half.to_numpy().astype(int)
    out["split"] = split.to_numpy()

    counts = out["split"].value_counts().to_dict()
    per_micrograph = out.groupby("rlnMicrographName")["split"].agg(set)
    summary = {
        "n_particles": int(len(out)),
        "counts": {s: int(counts.get(s, 0)) for s in SPLITS},
        "fractions": {s: round(counts.get(s, 0) / len(out), 4) for s in SPLITS},
        "test_halves": {
            int(k): int(v)
            for k, v in out.loc[out["split"] == "test", "rlnRandomSubset"]
            .value_counts()
            .items()
        },
        "n_micrographs": int(len(per_micrograph)),
        "micrographs_missing_split": {
            s: int(sum(s not in v for v in per_micrograph)) for s in SPLITS
        },
        "test_halves_duplicate_rows": int(len(_particles(test_halves)) - len(test_k)),
        "test_halves_conflicts_kept_first": conflicts,
    }
    star = (
        {"optics": full["optics"], "particles": out} if isinstance(full, dict) else out
    )
    return star, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--test-halves", type=Path, required=True)
    parser.add_argument("--train-half", type=int, default=1)
    parser.add_argument(
        "--on-conflict",
        choices=("error", "first"),
        default="error",
        help="Particles listed in both test halves: fail, or keep the first occurrence.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--summary", type=Path, default=None, help="Optional JSON summary path."
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.out.exists() and not args.overwrite:
        raise FileExistsError(f"{args.out} exists; pass --overwrite to replace it")

    star, summary = build_split_star(
        starfile.read(args.full),
        starfile.read(args.subset),
        starfile.read(args.test_halves),
        train_half=args.train_half,
        on_conflict=args.on_conflict,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    starfile.write(star, args.out, overwrite=True)
    summary.update(
        {
            "full_star": str(args.full),
            "subset_star": str(args.subset),
            "test_halves_star": str(args.test_halves),
            "out": str(args.out),
        }
    )
    text = json.dumps(summary, indent=1)
    print(text)
    if args.summary is not None:
        args.summary.write_text(text)


if __name__ == "__main__":
    main()
