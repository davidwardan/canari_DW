"""Plot the warmup / train / validation / test split of every benchmark series.

    python scripts/plot_splits.py [--config_path config/full_chronos2.yaml]

The split boundaries come from `common.prepare_dataset`, so the figure shows
exactly what the benchmark uses.
"""

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common

DOUBLE_COL = (6.5, 3.5)
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "font.family": "serif",
    "text.usetex": True,
    "pgf.rcfonts": False,
    "pgf.preamble": r"\usepackage{amsfonts}\usepackage{amssymb}\usepackage{amsmath}",
    "lines.linewidth": 1,
    "figure.figsize": DOUBLE_COL,
    "font.size": 9,
    "savefig.dpi": 300,
})

REGIONS = [  # label, colour of the shaded span
    ("Warmup (LLM context)", "0.85"),
    ("Train", "tab:blue"),
    ("Validation", "tab:orange"),
    ("Test", "tab:green"),
]


def main(config_path):
    config = common.load_config(config_path)
    series_names = common.resolve_series(config)

    fig, axes = plt.subplots(
        len(series_names), 1, figsize=(DOUBLE_COL[0], 0.85 * len(series_names)),
    )
    for ax, series in zip(axes, series_names):
        frame = common.base.load_series(series)
        processor = common.prepare_dataset(series, config)["data_processor"]
        index = processor.data.index
        bounds = [
            frame.index[0],
            index[processor.train_start],
            index[processor.validation_start],
            index[processor.test_start],
            frame.index[-1],
        ]
        for (label, colour), start, end in zip(REGIONS, bounds[:-1], bounds[1:]):
            alpha = 1.0 if colour.startswith("0.") else 0.2
            ax.axvspan(start, end, color=colour, alpha=alpha, lw=0)
        ax.plot(frame.index, frame.iloc[:, 0], color="tab:red", lw=0.6)
        ax.set_xlim(frame.index[0], frame.index[-1])
        ax.set_ylabel(series, rotation=0, ha="right", va="center")
        ax.set_yticks([])
        print(f"{series}: warmup {bounds[0]:%Y-%m} | train {bounds[1]:%Y-%m} | "
              f"val {bounds[2]:%Y-%m} | test {bounds[3]:%Y-%m} -> {bounds[4]:%Y-%m}")
    axes[-1].set_xlabel("Date")

    handles = [
        Patch(color=c, alpha=1.0 if c.startswith("0.") else 0.2, label=l) for l, c in REGIONS
    ]
    fig.legend(handles=handles, loc="upper center", ncol=len(REGIONS), frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=0.2)

    out = common.EXP_DIR / "figures" / "data_splits"
    for ext in ("pgf", "pdf"):
        fig.savefig(out.with_suffix(f".{ext}"), bbox_inches="tight")
    print(f"saved {out}.pgf/.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default=str(common.EXP_DIR / "config" / "full_chronos2.yaml"))
    main(parser.parse_args().config_path)
