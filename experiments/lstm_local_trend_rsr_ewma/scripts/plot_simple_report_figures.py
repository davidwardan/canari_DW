"""Teaching figures for the simplified mathematical report.

Produces two figures under figures/20261015_simple_report/:
  identification.{pgf,pdf}   same total signal, two different decompositions
  ewma_behaviour.{pgf,pdf}   fast/slow EWMA response to a ramp and to a yearly wave

Run from the repository root:
  python experiments/lstm_local_trend_rsr_ewma/scripts/plot_simple_report_figures.py
"""

from pathlib import Path

import matplotlib as mpl

mpl.use("pgf")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SINGLE_COL = (3.5, 2.5)
DOUBLE_COL = (6.5, 3.5)
PREAMBLE = r"\usepackage{amsfonts}\usepackage{amssymb}\usepackage{amsmath}"
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex", "font.family": "serif", "text.usetex": True,
    "pgf.rcfonts": False,
    "pgf.preamble": PREAMBLE,
    "text.latex.preamble": PREAMBLE,
    "lines.linewidth": 1, "figure.figsize": SINGLE_COL, "font.size": 9, "savefig.dpi": 300,
})

OUT = Path(__file__).resolve().parents[1] / "figures" / "20261015_simple_report"

# Fixed colours shared by both figures.
C_TOTAL = "black"
C_BASE = "tab:blue"
C_PATTERN = "tab:orange"
C_FAST = "tab:green"
C_SLOW = "tab:purple"


def ewma(x, h):
    """Zero-start EWMA, M_t = rho M_{t-1} + (1 - rho) x_t with rho = exp(-1/h)."""
    rho = np.exp(-1.0 / h)
    m, out = 0.0, np.empty_like(x, dtype=float)
    for t, xt in enumerate(x):
        m = rho * m + (1.0 - rho) * xt
        out[t] = m
    return out


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pgf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def identification_figure():
    t = np.arange(0, 157)
    slope = 0.02
    wave = np.sin(2 * np.pi * t / 52)
    total = slope * t + wave
    stories = [
        (slope * t, wave),                  # (a) trend in the baseline
        (np.zeros_like(t, dtype=float), wave + slope * t),  # (b) trend in the network
    ]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL[0], 2.4), sharey=True)
    for ax, (base, pattern), tag in zip(axes, stories, ("(a)", "(b)")):
        ax.plot(t, total, color=C_TOTAL, lw=2.2, alpha=0.25, label=r"total signal $L_t+O_t$")
        ax.plot(t, base, color=C_BASE, label=r"baseline $L_t$")
        ax.plot(t, pattern, color=C_PATTERN, label=r"network pattern $O_t$")
        ax.text(0.03, 0.93, tag, transform=ax.transAxes, va="top")
        ax.set_xlabel(r"Time $t$ (weeks)")
    axes[0].set_ylabel("Value")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0),
               ncol=3, frameon=False)
    fig.tight_layout()
    save(fig, "identification")


def ewma_figure():
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL[0], 2.4))

    # (a) ramp with short horizons so the lag is visible
    b, hf, hs = 0.1, 4, 8
    t = np.arange(0, 41)
    ramp = b * t
    mf, ms = ewma(ramp, hf), ewma(ramp, hs)
    ax = axes[0]
    ax.plot(t, ramp, color=C_PATTERN, label=r"network output $O_t$")
    ax.plot(t, mf, color=C_FAST, label=r"fast memory $M^{\mathrm{f}}_t$")
    ax.plot(t, ms, color=C_SLOW, label=r"slow memory $M^{\mathrm{s}}_t$")
    # Gaps b*ell between input and memories, drawn just after the last time step.
    tk, xf, xs = t[-1], 42.5, 47.0
    for m, x, color in ((ramp, xs, C_PATTERN), (mf, xf, C_FAST), (ms, xs, C_SLOW)):
        ax.plot([tk, x], [m[tk], m[tk]], color=color, lw=0.6, ls=":")
    for m, x, lab in ((mf, xf, r"$b\,\ell_{\mathrm{f}}$"), (ms, xs, r"$b\,\ell_{\mathrm{s}}$")):
        ax.annotate("", xy=(x, m[tk]), xytext=(x, ramp[tk]),
                    arrowprops=dict(arrowstyle="<->", lw=0.6, shrinkA=0, shrinkB=0))
        ax.text(x, ramp[tk] + 0.1, lab, ha="center", va="bottom", fontsize=8)
    ax.set_xlim(-2, 50)
    ax.set_xticks([0, 10, 20, 30, 40])
    ax.set_ylim(-0.2, 4.6)
    ax.text(0.03, 0.93, "(a)", transform=ax.transAxes, va="top")
    ax.set_xlabel(r"Time $t$ (weeks)")
    ax.set_ylabel("Value")

    # (b) yearly wave with the horizons used in the experiment
    t = np.arange(0, 521)
    wave = np.sin(2 * np.pi * t / 52)
    ax = axes[1]
    ax.plot(t, wave, color=C_PATTERN)
    ax.plot(t, ewma(wave, 104), color=C_FAST)
    ax.plot(t, ewma(wave, 208), color=C_SLOW)
    ax.text(0.03, 0.93, "(b)", transform=ax.transAxes, va="top")
    ax.set_xlabel(r"Time $t$ (weeks)")
    ax.set_ylim(-1.25, 1.45)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0),
               ncol=3, frameon=False)
    fig.tight_layout()
    save(fig, "ewma_behaviour")


if __name__ == "__main__":
    identification_figure()
    ewma_figure()
    print(f"Saved figures to {OUT}")
