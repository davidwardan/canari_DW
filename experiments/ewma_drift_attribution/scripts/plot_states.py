"""Hidden-state figures over the whole series: one figure per scenario, seed and arm.

Rows use the notation of the EWMA deck: observations $y_t$ with the one-step prediction,
level $z^L_t$, trend $z^T_t$, LSTM output $z^{(O)}_t$, white noise $z^V_t$, slow memory
$z^{M_s}_t$ (the mean reading) and the slope reading $s_t = (z^{M_f}_t - z^{M_s}_t) / d$.
Posterior means ± one std; dashed black lines are the true values of the toy; dotted grey
lines are ±tau. $s_t$ has no band because the memories' covariance is not saved. Vertical
lines mark the validation and test starts (dashed: the phase shift in D). Before the test
split the states come from the selected snapshot's frozen filter, not from the training
epochs; RSR starts learning one window (52 weeks) before the test split. Seed 5 is
included because its snapshot left the trend at about 0.6 delta (chosen after the run).

Reads the full-series runs written by `history.py`; figures go to
`figures/<history tag>_states/<scenario>/`.

    PYTHONPATH=/Users/davidwardan/canari_DW/src python \
        experiments/ewma_drift_attribution/scripts/plot_states.py --history-tag HISTORY_TAG
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory

import run_experiment as rx
from metrics import PERIOD, ewma_readings, slope_reading
from summarize import save  # also applies the shared plotting defaults

SCENARIO_TEXT = {
    "A": r"Scenario A: no slope, $y_t = p_t + v_t$",
    "B": r"Scenario B: slope, $y_t = 0.005\,t + p_t + v_t$",
    "C": r"Scenario C: as B, trend initialized at 0",
    "D": r"Scenario D: as B, $p_t$ shifted by a quarter cycle at week 416",
}
KAPPA_TEX = {"inf": r"\infty", "0.5": "1/2", "0.25": "1/4"}
STATES = (  # column, label, true value
    ("level", r"$z^{\mathtt{L}}_t$", "baseline"),
    ("trend", r"$z^{\mathtt{T}}_t$", "slope"),
    ("lstm", r"$z^{(\mathtt{O})}_t$", "pattern"),
    ("white_noise", r"$z^{\mathtt{V}}_t$", None),
)


def arm_text(arm: str) -> str:
    if arm == "none":
        return "no constraint, offline and online"
    if arm == "frozen":
        return r"frozen LSTM, trained at $\kappa=1$"
    label = arm.removeprefix("rsr_k")
    return rf"RSR, online $\kappa={KAPPA_TEX.get(label, label)}$, trained at $\kappa=1$"


def band(ax, x, mean, std) -> None:
    ax.fill_between(x, mean - std, mean + std, color="tab:blue", alpha=0.3, linewidth=0)
    ax.plot(x, mean, color="tab:blue")


def truth(ax, x, values) -> None:
    ax.plot(x, values, color="black", linestyle="--", linewidth=0.6)


def plot_run(job: dict, figures_dir: Path) -> Path:
    scenario, arm, seed = job["scenario"], job["arm"], job["seed"]
    states = pd.read_csv(Path(job["out_dir"]) / "full_states.csv")
    taus = json.loads((Path(job["original_run"]) / "metrics.json").read_text())
    y_std = rx.make_processor(rx.toy_series(scenario, seed)).scale_const_std[0]
    time = np.arange(rx.NUM_STEPS)
    years = time / PERIOD
    slope = rx.SCENARIOS[scenario]["slope"]
    true_values = {"baseline": slope * time, "slope": np.full(len(time), slope),
                   "pattern": states["pattern"].to_numpy()}
    constrained = "lstm_ewma_slow_mean" in states

    num_rows = 1 + len(STATES) + (2 if constrained else 0)
    fig, axes = plt.subplots(num_rows, 1, sharex=True, figsize=(6.5, 0.85 * num_rows + 0.4))
    axes[0].plot(years, states["obs"], color="tab:red", linestyle="none", marker=".",
                 markersize=1)
    band(axes[0], years, states["mean"], states["std"])
    truth(axes[0], years, slope * time + states["pattern"])
    axes[0].set_ylabel(r"$y_t$")
    axes[0].set_title(f"{SCENARIO_TEXT[scenario]} $\\mid$ {arm_text(arm)} $\\mid$ seed {seed}",
                      loc="left", fontsize=8)
    for ax, (column, label, true_key) in zip(axes[1:], STATES):
        band(ax, years, states[f"{column}_mean"], states[f"{column}_std"])
        if true_key is not None:
            truth(ax, years, true_values[true_key])
        ax.set_ylabel(label)
    if constrained:
        true_mean, true_slope = (reading * y_std for reading in ewma_readings(
            rx.true_pattern(scenario) / y_std, rx.FAST_SPAN, rx.SLOW_SPAN))
        mean_ax, slope_ax = axes[-2:]
        band(mean_ax, years, states["lstm_ewma_slow_mean"], states["lstm_ewma_slow_std"])
        slope_ax.plot(years, slope_reading(states["lstm_ewma_fast_mean"],
                                           states["lstm_ewma_slow_mean"],
                                           rx.FAST_SPAN, rx.SLOW_SPAN), color="tab:blue")
        truth(mean_ax, years, true_mean)
        truth(slope_ax, years, true_slope)
        for ax, tau in ((mean_ax, taus["tau_mean"]), (slope_ax, taus["tau_slope"])):
            if tau is not None:
                for sign in (-1, 1):
                    ax.axhline(sign * tau * y_std, color="tab:gray", linestyle=":",
                               linewidth=0.8)
        mean_ax.set_ylabel(r"$z^{\mathtt{M}_{\mathtt{s}}}_t$")
        slope_ax.set_ylabel(r"$s_t$")

    boundaries = [rx.TRAIN_LEN, rx.TEST_START]
    for ax in axes:
        for boundary in boundaries:
            ax.axvline(boundary / PERIOD, color="tab:gray", linewidth=0.5)
        if rx.SCENARIOS[scenario]["phase_shift"]:
            ax.axvline(rx.CHANGE_START / PERIOD, color="tab:gray", linewidth=0.5,
                       linestyle="--")
        ax.tick_params(labelsize=7)
        ax.yaxis.get_offset_text().set_fontsize(7)
        ax.yaxis.label.set(rotation=0, ha="right", va="center")
    split_labels = blended_transform_factory(axes[0].transData, axes[0].transAxes)
    edges = [0, *boundaries, rx.NUM_STEPS]
    for name, start, stop in zip(("train", "validation", "test"), edges, edges[1:]):
        axes[0].text((start + stop) / 2 / PERIOD, 0.96, name, transform=split_labels,
                     ha="center", va="top", fontsize=7, color="tab:gray")
    axes[-1].set_xlabel("time (years)")
    axes[-1].set_xlim(0, rx.NUM_STEPS / PERIOD)

    handles = [Line2D([], [], color="tab:red", linestyle="none", marker="."),
               (Patch(color="tab:blue", alpha=0.3, linewidth=0), Line2D([], [], color="tab:blue")),
               Line2D([], [], color="black", linestyle="--", linewidth=0.6)]
    labels = ["observation", r"mean $\pm\sigma$", "true value"]
    if constrained and taus["tau_mean"] is not None:
        handles.append(Line2D([], [], color="tab:gray", linestyle=":"))
        labels.append(r"$\pm\tau$")
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0),
               ncol=len(labels), frameon=False)
    fig.align_ylabels(axes)
    fig.tight_layout(h_pad=0.2)
    out_dir = figures_dir / scenario
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{scenario}_seed{seed}_{arm.replace('.', 'p')}"
    save(fig, out_dir, name)
    return out_dir / name


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--history-tag", required=True)
    args = parser.parse_args()
    config = json.loads((rx.EXP_DIR / "results" / args.history_tag / "config.json").read_text())
    figures_dir = rx.EXP_DIR / "figures" / f"{args.history_tag}_states"
    for job in config["jobs"]:
        print("saved", plot_run(job, figures_dir).relative_to(rx.EXP_DIR))


if __name__ == "__main__":
    main()
