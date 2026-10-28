"""Hidden-state figures: one scenario and seed, one column per arm, one row per state.

Rows: observations with the one-step prediction, then the filtered level, trend, LSTM
and white noise (posterior mean ± one std), then the EWMA mean and slope readings the
constraint acts on. Dashed black lines are the true values of the toy; dotted grey
lines are the ±tau tolerances. Seed 1 is the default example; seed 5 is shown as well
because its offline snapshot left the trend at about 0.6 delta (chosen after the run).

    PYTHONPATH=/Users/davidwardan/canari_DW/src python \
        experiments/ewma_drift_attribution/scripts/plot_states.py --tag RUN_TAG
"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import run_experiment as rx
from metrics import PERIOD, ewma_readings, slope_reading
from summarize import save  # also applies the shared plotting defaults

ARMS = ("none", "rsr_kinf", "rsr_k1", "rsr_k0.25")
LABELS = {"none": "no constraint", "frozen": "frozen", "rsr_kinf": r"$\kappa=\infty$ online",
          "rsr_k1": r"$\kappa=1$", "rsr_k0.5": r"$\kappa=1/2$", "rsr_k0.25": r"$\kappa=1/4$"}
ROWS = (r"$y_t$", r"level", r"trend", r"LSTM", r"white noise", r"mean reading",
        r"slope reading")


def band(ax, x, mean, std) -> None:
    ax.fill_between(x, mean - std, mean + std, color="tab:blue", alpha=0.3, linewidth=0)
    ax.plot(x, mean, color="tab:blue")


def truth(ax, x, values) -> None:
    ax.plot(x, values, color="black", linestyle="--", linewidth=0.6)


def plot_states(tag: str, scenario: str, seed: int, arms) -> None:
    y_std = rx.make_processor(rx.toy_series(scenario, seed)).scale_const_std[0]
    time = np.arange(rx.TEST_START, rx.NUM_STEPS)
    years = time / PERIOD
    slope = rx.SCENARIOS[scenario]["slope"]
    true_mean, true_slope = (
        reading[rx.TEST_START:] * y_std
        for reading in ewma_readings(rx.true_pattern(scenario) / y_std,
                                     rx.FAST_SPAN, rx.SLOW_SPAN))

    fig, axes = plt.subplots(len(ROWS), len(arms), sharex=True, sharey="row",
                             figsize=(6.5, 8.5))
    for column, arm in enumerate(arms):
        run = rx.EXP_DIR / "results" / f"{tag}_{scenario}_{arm}_seed{seed}"
        states = pd.read_csv(run / "states.csv")
        taus = json.loads((run / "metrics.json").read_text())
        ax = axes[:, column]
        ax[0].plot(years, states["obs"], color="tab:red", linestyle="none", marker=".",
                   markersize=1)
        band(ax[0], years, states["mean"], states["std"])
        for row, name in enumerate(("level", "trend", "lstm", "white_noise"), start=1):
            band(ax[row], years, states[f"{name}_mean"], states[f"{name}_std"])
        truth(ax[1], years, slope * time)
        truth(ax[2], years, np.full(len(time), slope))
        truth(ax[3], years, states["pattern"])
        if "lstm_ewma_slow_mean" in states:
            band(ax[5], years, states["lstm_ewma_slow_mean"], states["lstm_ewma_slow_std"])
            # The fast/slow covariance is not saved, so the slope reading has no band.
            ax[6].plot(years, slope_reading(states["lstm_ewma_fast_mean"],
                                            states["lstm_ewma_slow_mean"],
                                            rx.FAST_SPAN, rx.SLOW_SPAN), color="tab:blue")
            truth(ax[5], years, true_mean)
            truth(ax[6], years, true_slope)
            for row, tau in ((5, taus["tau_mean"]), (6, taus["tau_slope"])):
                for sign in (-1, 1):
                    if tau is not None:
                        ax[row].axhline(sign * tau * y_std, color="tab:gray", linestyle=":",
                                        linewidth=0.8)
        else:
            for row in (5, 6):
                ax[row].text(0.5, 0.5, "no constraint memories", transform=ax[row].transAxes,
                             ha="center", va="center", fontsize=7, color="tab:gray")
        if rx.SCENARIOS[scenario]["phase_shift"]:
            for panel in ax:
                panel.axvline(rx.CHANGE_START / PERIOD, color="tab:gray", linewidth=0.5)
        ax[0].set_title(LABELS[arm], fontsize=9)
        ax[-1].set_xlabel("time (years)")
    for row, label in enumerate(ROWS):
        axes[row, 0].set_ylabel(label)
    for panel in axes.flat:
        panel.tick_params(labelsize=7)
        panel.yaxis.get_offset_text().set_fontsize(7)
    handles = [
        Line2D([], [], color="tab:red", linestyle="none", marker=".", label="observation"),
        (Patch(color="tab:blue", alpha=0.3, linewidth=0), Line2D([], [], color="tab:blue")),
        Line2D([], [], color="black", linestyle="--", linewidth=0.6, label="true value"),
        Line2D([], [], color="tab:gray", linestyle=":", label=r"$\pm\tau$"),
    ]
    labels = ["observation", r"mean $\pm\sigma$", "true value", r"$\pm\tau$"]
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4,
               frameon=False)
    fig.align_ylabels(axes[:, 0])
    fig.tight_layout(h_pad=0.3, w_pad=0.3)
    save(fig, rx.EXP_DIR / "figures", f"{tag}_states_{scenario}_seed{seed}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", required=True)
    parser.add_argument("--scenarios", nargs="+", choices=rx.SCENARIOS, default=list(rx.SCENARIOS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 5])
    parser.add_argument("--arms", nargs="+", choices=rx.ARMS, default=list(ARMS))
    args = parser.parse_args()
    for scenario in args.scenarios:
        for seed in args.seeds:
            plot_states(args.tag, scenario, seed, args.arms)
            print(f"saved {args.tag}_states_{scenario}_seed{seed}")


if __name__ == "__main__":
    main()
