"""LaTeX tables of the manuscript from the evaluation outputs.

``tables/fsc_recon.tex``: body of Table 4 (FSC = 0.143 and FSC = 0.5 resolution, mean FSC;
mean ± standard deviation over all (subset, half-map) rows of ``reconstruction_fsc.csv``).
The best mean of every dataset block and column is set in bold (lowest resolution, highest
mean FSC; ties are all bold at the printed precision). ``tables/fsc_recon.csv`` holds the
unrounded values.

Usage::

    python -m source.evaluation.make_tables
"""

from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

COLUMNS = (
    ("resolution_0.143", 2, "min"),
    ("resolution_0.5", 2, "min"),
    ("mean_fsc", 4, "max"),
)


def summarise(frame: pd.DataFrame, expected_rows: int) -> pd.DataFrame:
    counts = frame.groupby(["dataset", "method"]).size()
    wrong = counts[counts != expected_rows]
    if not wrong.empty:
        raise ValueError(
            f"Expected {expected_rows} rows per method, got {wrong.to_dict()}"
        )
    stats = frame.groupby(["dataset", "method"])[[c for c, _, _ in COLUMNS]].agg(
        ["mean", "std"]
    )
    stats.columns = [f"{c}_{s}" for c, s in stats.columns]
    return stats.reset_index()


def cell(mean: float, std: float, digits: int, bold: bool) -> str:
    text = f"{mean:.{digits}f} \\pm {std:.{digits}f}"
    return f"$\\mathbf{{{text}}}$" if bold else f"${text}$"


def fsc_recon_rows(cfg: DictConfig, stats: pd.DataFrame) -> list[str]:
    lines = []
    for section in cfg.tables.fsc_recon.sections:
        lines += [
            "\\midrule",
            f"\\multicolumn{{5}}{{c}}{{{section.title}}} \\\\",
            "\\midrule",
        ]
        methods = [str(m) for m in section.methods]
        for d, (dataset, short) in enumerate(section.datasets.items()):
            block = stats[stats.dataset == str(dataset)].set_index("method")
            missing = sorted(set(methods) - set(block.index))
            if missing:
                raise KeyError(f"{dataset}: no FSC rows for {missing}")
            block = block.loc[methods]
            best = {}
            for column, digits, rule in COLUMNS:
                rounded = block[f"{column}_mean"].round(digits)
                best[column] = rounded.min() if rule == "min" else rounded.max()
            for i, method in enumerate(methods):
                first = (
                    f"\\multirow{{{len(methods)}}}{{*}}"
                    f"{{\\rotatebox[origin=c]{{90}}{{{short}}}}}"
                    if i == 0
                    else ""
                )
                cells = [
                    cell(
                        block.at[method, f"{column}_mean"],
                        block.at[method, f"{column}_std"],
                        digits,
                        round(block.at[method, f"{column}_mean"], digits)
                        == best[column],
                    )
                    for column, digits, _ in COLUMNS
                ]
                lines.append(
                    " & ".join([first, str(cfg.methods[method]), *cells]) + " \\\\"
                )
            if d < len(section.datasets) - 1:
                lines.append("\\midrule")
    return lines


@hydra.main(version_base="1.3", config_path="../../configs_eval", config_name="paper")
def main(cfg: DictConfig) -> None:
    out_dir = Path(cfg.output_dir) / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(
        Path(cfg.output_dir) / "reconstruction_fsc.csv", dtype={"dataset": str}
    )
    expected = len(cfg.reconstruction.subsets) * len(cfg.reconstruction.halves)
    stats = summarise(frame, expected)
    stats.to_csv(out_dir / "fsc_recon.csv", index=False)
    header = [
        "\\toprule",
        "Dataset & Method & FSC=0.143 (\\AA)$\\downarrow$ & FSC=0.5 (\\AA)$\\downarrow$ "
        "& Mean FSC$\\uparrow$ \\\\",
    ]
    body = fsc_recon_rows(cfg, stats)
    text = "\n".join(header + body + ["\\bottomrule"]) + "\n"
    (out_dir / "fsc_recon.tex").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
