"""Hidden-state figures over the whole series: one figure per series, seed and arm.

Rows use the notation of the EWMA deck: $y_t$ with the one-step prediction, level
$z^L_t$, trend $z^T_t$, LSTM output $z^{(O)}_t$, white noise $z^V_t$, and for the EWMA
arms the slow memory $z^{M_s}_t$ and the slope reading $s_t$. Posterior means ± one std;
dashed black lines are the true baseline $b_t = m + 0.005\,t$ and trend 0.005; dotted grey
lines are ±tau. $s_t$ has no band because the memories' covariance is not saved. Vertical
lines mark the validation and test starts. Before the test split the states come from
the selected snapshot's frozen filter; RSR starts learning 52 weeks before the test.

Figures go to `figures/<tag>_states/<series>/`.

    PYTHONPATH=/Users/davidwardan/canari_DW/src python \
        experiments/ewma_real_series_rsr/scripts/plot_states.py --tag RUN_TAG
"""

import argparse
import json
import multiprocessing
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory

import run_experiment as rx
from metrics import PERIOD, slope_reading
from summarize import save  # also applies the shared plotting defaults

KAPPA_TEX = {"0.5": "1/2", "0.25": "1/4", "0.125": "1/8"}
STATES = (("level", r"$z^{\mathtt{L}}_t$"), ("trend", r"$z^{\mathtt{T}}_t$"),
          ("lstm", r"$z^{(\mathtt{O})}_t$"), ("white_noise", r"$z^{\mathtt{V}}_t$"))


def arm_text(arm: str) -> str:
    online = "frozen LSTM" if arm.startswith("frozen") else "RSR (replay smoothed)"
    if "_ewma_k" not in arm:
        return f"{online}, no constraint"
    label = arm.rsplit("_k", 1)[1]
    return rf"{online} + EWMA, $\kappa={KAPPA_TEX.get(label, label)}$"


def band(ax, x, mean, std) -> None:
    ax.fill_between(x, mean - std, mean + std, color="tab:blue", alpha=0.3, linewidth=0)
    ax.plot(x, mean, color="tab:blue")


def truth(ax, x, values) -> None:
    ax.plot(x, values, color="black", linestyle="--", linewidth=0.6)


def plot_run(job: dict, figures_dir: Path) -> Path:
    series, arm, seed = job["series"], job["arm"], job["seed"]
    states = pd.read_csv(Path(job["out_dir"]) / "full_states.csv")
    selection = json.loads((Path(job["snapshot_path"]).parent / "selection.json").read_text())
    y_std = rx.scale(rx.make_processor(rx.with_trend(rx.detrended_series(series))))[1]
    years = np.arange(len(states)) / PERIOD
    constrained = "lstm_ewma_slow_mean" in states

    num_rows = 1 + len(STATES) + (2 if constrained else 0)
    fig, axes = plt.subplots(num_rows, 1, sharex=True, figsize=(6.5, 0.85 * num_rows + 0.4))
    axes[0].plot(years, states["obs"], color="tab:red", linestyle="none", marker=".",
                 markersize=1)
    band(axes[0], years, states["mean"], states["std"])
    axes[0].set_ylabel(r"$y_t$")
    offset = (rf" $\mid$ level started {job['level_offset']:+g} off (std {job['level_offset_std']:g})"
              if "level_offset" in job else "")
    axes[0].set_title(rf"{series} + $0.005\,t$ $\mid$ {arm_text(arm)} $\mid$ seed {seed}{offset}",
                      loc="left", fontsize=8)
    for ax, (column, label) in zip(axes[1:], STATES):
        band(ax, years, states[f"{column}_mean"], states[f"{column}_std"])
        ax.set_ylabel(label)
    truth(axes[1], years, states["baseline"])
    truth(axes[2], years, np.full(len(states), rx.SLOPE))
    if constrained:
        mean_ax, slope_ax = axes[-2:]
        band(mean_ax, years, states["lstm_ewma_slow_mean"], states["lstm_ewma_slow_std"])
        slope_ax.plot(years, slope_reading(states["lstm_ewma_fast_mean"],
                                           states["lstm_ewma_slow_mean"],
                                           rx.FAST_SPAN, rx.SLOW_SPAN), color="tab:blue")
        for ax, tau in zip((mean_ax, slope_ax), selection["taus"]):
            for sign in (-1, 1):
                ax.axhline(sign * tau * y_std, color="tab:gray", linestyle=":", linewidth=0.8)
        mean_ax.set_ylabel(r"$z^{\mathtt{M}_{\mathtt{s}}}_t$")
        slope_ax.set_ylabel(r"$s_t$")

    boundaries = [rx.TRAIN_LEN, rx.TEST_START]
    for ax in axes:
        for boundary in boundaries:
            ax.axvline(boundary / PERIOD, color="tab:gray", linewidth=0.5)
        ax.tick_params(labelsize=7)
        ax.yaxis.get_offset_text().set_fontsize(7)
        ax.yaxis.label.set(rotation=0, ha="right", va="center")
    split_labels = blended_transform_factory(axes[0].transData, axes[0].transAxes)
    edges = [0, *boundaries, len(states)]
    for name, start, stop in zip(("train", "validation", "test"), edges, edges[1:]):
        axes[0].text((start + stop) / 2 / PERIOD, 0.96, name, transform=split_labels,
                     ha="center", va="top", fontsize=7, color="tab:gray")
    axes[-1].set_xlabel("time (years)")
    axes[-1].set_xlim(0, len(states) / PERIOD)

    handles = [Line2D([], [], color="tab:red", linestyle="none", marker="."),
               (Patch(color="tab:blue", alpha=0.3, linewidth=0), Line2D([], [], color="tab:blue")),
               Line2D([], [], color="black", linestyle="--", linewidth=0.6)]
    labels = ["observation", r"mean $\pm\sigma$", "true baseline / trend"]
    if constrained:
        handles.append(Line2D([], [], color="tab:gray", linestyle=":"))
        labels.append(r"$\pm\tau$")
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0),
               ncol=len(labels), frameon=False)
    fig.align_ylabels(axes)
    fig.tight_layout(h_pad=0.2)
    out_dir = figures_dir / series
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{series}_seed{seed}_{arm.replace('.', 'p')}"
    save(fig, out_dir, name)
    return out_dir / name


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", required=True, help="run tag, e.g. level_offset/TAG")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    parser.add_argument("--arms", nargs="+", choices=rx.ARMS, default=list(rx.ARMS))
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    jobs = json.loads((rx.EXP_DIR / "results" / args.tag / "jobs.json").read_text())
    figures_dir = rx.EXP_DIR / "figures" / f"{args.tag}_states"
    selected = [(job, figures_dir) for job in jobs["evaluation"]
                if job["seed"] in args.seeds and job["arm"] in args.arms]
    with multiprocessing.get_context("spawn").Pool(args.workers) as pool:
        for path in pool.starmap(plot_run, selected):
            print("saved", path.relative_to(rx.EXP_DIR))


if __name__ == "__main__":
    main()
