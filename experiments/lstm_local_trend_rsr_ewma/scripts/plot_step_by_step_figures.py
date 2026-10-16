"""Extra teaching figures for the step-by-step mathematical report.

Produces, under figures/20261016_step_by_step_report/:
  ewma_weights.{pgf,pdf}      EWMA weights as bars, with their balance point (average delay)
  wave_survival.{pgf,pdf}     height of a wave left in each memory, as a function of its period
  gaussian_product.{pgf,pdf}  prior bell x measurement bell = narrower updated bell
  slice.{pgf,pdf}             a tilted cloud of two linked variables, sliced at the measured value

The identification and EWMA-response figures are reused from
figures/20261015_simple_report/ (plot_simple_report_figures.py).

Run from the repository root:
  python experiments/lstm_local_trend_rsr_ewma/scripts/plot_step_by_step_figures.py
"""

from pathlib import Path

import matplotlib as mpl

mpl.use("pgf")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402

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

OUT = Path(__file__).resolve().parents[1] / "figures" / "20261016_step_by_step_report"

# Colour code shared with the report (and plot_simple_report_figures.py):
# fast memory green, slow memory purple, network output orange, measured data red, gain gold.
# A generic belief (no specific quantity) is black: dashed before, solid after.
C_FAST = "tab:green"
C_SLOW = "tab:purple"
C_OUTPUT = "tab:orange"
C_MEAS = "tab:red"
C_GAIN = "#B8860B"
C_GENERIC = "black"


def normal_pdf(x, mu, var):
    return np.exp(-0.5 * (x - mu) ** 2 / var) / np.sqrt(2 * np.pi * var)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pgf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def ewma_weights_figure():
    k = np.arange(0, 31)
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL[0], 2.2), sharey=True)
    for ax, h, color, tag, name in zip(axes, (4, 8), (C_FAST, C_SLOW), ("(a)", "(b)"),
                                       (r"\mathrm{f}", r"\mathrm{s}")):
        rho = np.exp(-1.0 / h)
        w = (1 - rho) * rho ** k
        ell = rho / (1 - rho)
        ax.bar(k, w, width=0.75, color=color)
        ax.plot([ell], [-0.012], marker="^", color="black", markersize=7, clip_on=False)
        ax.text(ell + 0.8, -0.012, rf"$\ell_{name}={ell:.1f}$", va="center", fontsize=8)
        ax.text(0.97, 0.93, rf"{tag} $h={h}$", transform=ax.transAxes, ha="right", va="top")
        ax.set_xlabel(r"Age $k$ (weeks ago)")
        ax.set_ylim(-0.02, 0.24)
        ax.set_xlim(-1, 30.5)
    axes[0].set_ylabel(r"Weight $w_k=c\rho^k$")
    fig.tight_layout()
    save(fig, "ewma_weights")


def wave_survival_figure():
    period = np.logspace(np.log10(4), np.log10(3000), 400)
    omega = 2 * np.pi / period
    fig, ax = plt.subplots(figsize=SINGLE_COL)
    for h, color, label in ((104, C_FAST, r"fast memory, $h=104$"), (208, C_SLOW, r"slow memory, $h=208$")):
        rho = np.exp(-1.0 / h)
        height = (1 - rho) / np.sqrt(1 - 2 * rho * np.cos(omega) + rho ** 2)
        ax.plot(period, height, color=color, label=label)
    ax.axvline(52, color="black", lw=0.6, ls=":")
    ax.text(56, 0.9, "one year", fontsize=8, va="top")
    ax.set_xscale("log")
    ax.set_xlabel(r"Period of the wave $P$ (weeks)")
    ax.set_ylabel("Height left in memory")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
              handlelength=1.5, columnspacing=1.0)
    fig.tight_layout()
    save(fig, "wave_survival")


def gaussian_product_figure():
    x = np.linspace(-3.5, 7.5, 600)
    mu, var, u, tau2 = 3.0, 2.0, 0.0, 1.0
    K = var / (var + tau2)
    mu_new, var_new = mu + K * (u - mu), (1 - K) * var
    fig, ax = plt.subplots(figsize=SINGLE_COL)
    ax.plot(x, normal_pdf(x, mu, var), color=C_GENERIC, ls="--", label=r"before, $\mathcal{N}(x;3,2)$")
    ax.plot(x, normal_pdf(x, u, tau2), color=C_MEAS, label=r"measurement, $\mathcal{N}(0;x,1)$")
    ax.plot(x, normal_pdf(x, mu_new, var_new), color=C_GENERIC, lw=1.6,
            label=r"after, $\mathcal{N}(x;1,\tfrac{2}{3})$")
    ax.set_xlabel(r"Value $x$")
    ax.set_ylabel("Density")
    ax.set_ylim(0, 0.55)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1, frameon=False)
    fig.tight_layout()
    save(fig, "gaussian_product")


def slice_figure():
    # (U, H) with var[W] = 1, tau^2 = 0.25, so var[U] = 1.25, and cov(H, W) = 0.8.
    mu = np.array([1.0, 2.0])
    cov = np.array([[1.25, 0.8], [0.8, 1.0]])
    u_obs = 0.0
    K = cov[0, 1] / cov[0, 0]
    mu_h, var_h = mu[1] + K * (u_obs - mu[0]), cov[1, 1] - cov[0, 1] ** 2 / cov[0, 0]

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL[0], 2.6),
                             gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    rng = np.random.default_rng(1)
    pts = rng.multivariate_normal(mu, cov, size=1500)
    in_slice = np.abs(pts[:, 0] - u_obs) < 0.08
    ax.scatter(pts[~in_slice, 0], pts[~in_slice, 1], s=2, color="0.75", linewidths=0)
    ax.scatter(pts[in_slice, 0], pts[in_slice, 1], s=5, color=C_OUTPUT, linewidths=0, zorder=3)
    vals, vecs = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    for n in (1, 2):
        ax.add_patch(Ellipse(mu, 2 * n * np.sqrt(vals[1]), 2 * n * np.sqrt(vals[0]), angle=angle,
                             fill=False, color="black", lw=0.6))
    uu = np.linspace(-3.0, 5.0, 10)
    ax.plot(uu, mu[1] + K * (uu - mu[0]), color=C_GAIN, ls="--", lw=1.0,
            label=r"average $H$ for each $u$ (slope $K_H$)")
    ax.axvline(u_obs, color=C_MEAS, lw=0.8, label=r"measured $u=0$")
    ax.plot([u_obs], [mu_h], marker="o", color="black", markersize=4, zorder=4)
    ax.set_xlim(-3.0, 5.0)
    ax.set_ylim(-1.2, 5.2)
    ax.set_xlabel(r"Measurement $U$")
    ax.set_ylabel(r"Unmeasured $H$")
    ax.text(0.03, 0.95, "(a)", transform=ax.transAxes, va="top")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1, frameon=False)

    ax = axes[1]
    h = np.linspace(-1.5, 5.5, 500)
    ax.plot(h, normal_pdf(h, mu[1], cov[1, 1]), color=C_OUTPUT, ls="--", label=r"before, $\mathcal{N}(h;2,1)$")
    ax.plot(h, normal_pdf(h, mu_h, var_h), color=C_OUTPUT, lw=1.6,
            label=rf"after, $\mathcal{{N}}(h;{mu_h:.2f},{var_h:.3f})$")
    ax.set_xlabel(r"Value $h$")
    ax.set_ylabel("Density")
    ax.set_ylim(0, 0.65)
    ax.text(0.03, 0.95, "(b)", transform=ax.transAxes, va="top")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1, frameon=False)
    fig.tight_layout()
    save(fig, "slice")


if __name__ == "__main__":
    ewma_weights_figure()
    wave_survival_figure()
    gaussian_product_figure()
    slice_figure()
    print(f"Saved figures to {OUT}")
