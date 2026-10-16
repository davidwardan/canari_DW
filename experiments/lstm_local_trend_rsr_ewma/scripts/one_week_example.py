"""One worked week of the six-state model: every component before and after the wish,
and after the data. Used by mathematical_report_step_by_step.tex (section "Watching one week").

The example runs the report's own equations (prediction, wish, data update) with the
experiment's settings (h_f=104, h_s=208, tau_m=0.05, tau_d=0.0005*24/52, sigma_V=0.1) on a
synthetic signal. A stand-in replaces the LSTM: each week it returns the true yearly wave plus
a drift of 0.002 per week, with standard deviation 0.15, and it does not learn. The wish is
switched off for weeks 0..399 (so the drift builds up in the network output and the memories)
and switched on in week 400, the week we look at.

Outputs, under figures/20261016_step_by_step_report/:
  one_week.{pgf,pdf}            one panel per quantity: predicted -> after the wish -> after the data
  one_week_wish_table.tex       the wish step, row by row (gains, move, before/after)
  one_week_data_table.tex       the data step, row by row
  one_week_values.tex           \newcommand macros for the numbers quoted in the text

Run from the repository root:
  python experiments/lstm_local_trend_rsr_ewma/scripts/one_week_example.py
"""

from pathlib import Path

import matplotlib as mpl

mpl.use("pgf")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DOUBLE_COL = (6.5, 3.5)
PREAMBLE = r"\usepackage{amsfonts}\usepackage{amssymb}\usepackage{amsmath}"
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex", "font.family": "serif", "text.usetex": True,
    "pgf.rcfonts": False,
    "pgf.preamble": PREAMBLE,
    "text.latex.preamble": PREAMBLE,
    "lines.linewidth": 1, "figure.figsize": DOUBLE_COL, "font.size": 9, "savefig.dpi": 300,
})

OUT = Path(__file__).resolve().parents[1] / "figures" / "20261016_step_by_step_report"

# Colour code of the report.
COLOURS = {"L": "#1F77B4", "T": "#0F8B9A", "O": "#E06C00", "V": "#7F7F7F",
           "Mf": "#2CA02C", "Ms": "#9467BD", "S": "#8C564B", "Y": "#444444"}
C_WISH = "#C2399A"
C_DATA = "#D62728"

# Experiment settings.
H_F, H_S = 104, 208
TAU_M, TAU_D = 0.05, 0.0005 * 24 / 52
SIGMA_V = 0.1
# Synthetic signal and stand-in network.
SLOPE_TRUE, DRIFT, SIGMA_O = 0.004, 0.002, 0.15
WEEK, SEED = 400, 3


def model():
    rf, rs = np.exp(-1 / H_F), np.exp(-1 / H_S)
    cf, cs = 1 - rf, 1 - rs
    dl = rs / (1 - rs) - rf / (1 - rf)
    A = np.zeros((6, 6))
    A[0, 0] = A[0, 1] = A[1, 1] = 1
    A[4, 4], A[5, 5] = rf, rs
    g = np.array([0, 0, 1, 0, cf, cs])
    e = np.array([0, 0, 0, 1, 0, 0.0])
    Cc = np.array([[0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1 / dl, -1 / dl]])
    Gam = np.diag([TAU_M ** 2, TAU_D ** 2])
    C = np.array([1, 0, 1, 1, 0, 0.0])
    return A, g, e, Cc, Gam, C, dl


def simulate():
    A, g, e, Cc, Gam, C, dl = model()
    rng = np.random.default_rng(SEED)
    mu = np.zeros(6)
    S = np.diag([0.5 ** 2, 0.005 ** 2, 0, 0, 0, 0])
    for t in range(WEEK + 1):
        y = SLOPE_TRUE * t + np.sin(2 * np.pi * t / 52) + rng.normal(0, 0.1)
        mO = np.sin(2 * np.pi * t / 52) + DRIFT * t
        m0 = A @ mu + g * mO                                                    # predict
        S0 = A @ S @ A.T + SIGMA_O ** 2 * np.outer(g, g) + SIGMA_V ** 2 * np.outer(e, e)
        Om = Cc @ S0 @ Cc.T + Gam
        K = S0 @ Cc.T @ np.linalg.inv(Om)
        if t == WEEK:                                                           # the wish
            m1, S1 = m0 - K @ Cc @ m0, S0 - K @ Om @ K.T
        else:
            m1, S1 = m0, S0
        sY2 = C @ S1 @ C                                                        # the data
        k = S1 @ C
        m2, S2 = m1 + k / sY2 * (y - C @ m1), S1 - np.outer(k, k) / sY2
        mu, S = m2, S2
    return dict(m0=m0, S0=S0, m1=m1, S1=S1, m2=m2, S2=S2, y=y, K=K, Om=Om, k=k, sY2=sY2,
                Cc=Cc, C=C, dl=dl)


def check(r):
    """Row-by-row formulas of the report agree with the matrix formulas."""
    S0, Cc, Om, K = r["S0"], r["Cc"], r["Om"], r["K"]
    wires = S0 @ Cc.T                                  # row i = [a_i, b_i]
    surprise = -Cc @ r["m0"]
    for i in range(6):
        gain_i = wires[i] @ np.linalg.inv(Om)
        assert np.allclose(gain_i, K[i])
        assert np.isclose(r["m0"][i] + gain_i @ surprise, r["m1"][i])
        assert np.isclose(S0[i, i] - wires[i] @ np.linalg.inv(Om) @ wires[i], r["S1"][i, i])
    assert np.allclose(K[3], 0) and np.isclose(r["m1"][3], r["m0"][3])         # noise untouched
    assert np.isclose(r["C"] @ r["m2"], r["y"]) and abs(r["C"] @ r["S2"] @ r["C"]) < 1e-12


def stages(r):
    """Mean and std of every quantity at the three stages, including S and Y."""
    w_S = np.array([0, 0, 0, 0, 1 / r["dl"], -1 / r["dl"]])
    rows = {}
    for name, i in (("L", 0), ("T", 1), ("O", 2), ("V", 3), ("Mf", 4), ("Ms", 5)):
        rows[name] = [(r[f"m{s}"][i], np.sqrt(max(r[f"S{s}"][i, i], 0))) for s in (0, 1, 2)]
    for name, w in (("S", w_S), ("Y", r["C"])):
        rows[name] = [(w @ r[f"m{s}"], np.sqrt(max(w @ r[f"S{s}"] @ w, 0))) for s in (0, 1, 2)]
    # Gains of S and Y follow from the gains of the components.
    gains = {n: r["K"][i] for n, i in (("L", 0), ("T", 1), ("O", 2), ("V", 3), ("Mf", 4), ("Ms", 5))}
    gains["S"] = w_S @ r["K"]
    gains["Y"] = r["C"] @ r["K"]
    w_full = {n: np.eye(6)[i] for n, i in (("L", 0), ("T", 1), ("O", 2), ("V", 3), ("Mf", 4), ("Ms", 5))}
    w_full.update(S=w_S, Y=r["C"])
    data_gain = {n: (w_full[n] @ r["k"]) / r["sY2"] for n in w_full}
    return rows, gains, data_gain


def num(x, digits=3):
    """Plain decimals for ordinary numbers, a power of ten for very small ones."""
    if x == 0 or abs(x) >= 10 ** (-digits + 1) / 2:
        s = f"{x:.{digits}f}"
        return "0" if float(s) == 0 else s
    e = int(np.floor(np.log10(abs(x))))
    return rf"{x / 10 ** e:.2f}\times10^{{{e}}}"


def signed(x):
    """A move, with an explicit + sign so the direction is visible."""
    s = num(x)
    return s if s.startswith("-") or s == "0" else "+" + s


def pm(m, s):
    """mean +- std, sharing a power of ten when both are small."""
    s = 0.0 if s < 1e-8 else s          # the sum y = L + O + V is exact after the data
    big = max(abs(m), s)
    if big >= 0.01 or big == 0:
        return rf"{num(m)}\pm{num(s)}"
    e = int(np.floor(np.log10(big)))
    return rf"({m / 10 ** e:.3f}\pm{s / 10 ** e:.3f})\times10^{{{e}}}"


SYMBOL = {"L": r"\Lv_t", "T": r"\Tv_t", "O": r"\Ov_t", "V": r"\Vv_t", "Mf": r"\mf_t",
          "Ms": r"\ms_t", "S": r"\Sv_t", "Y": r"\Yv_t"}
NAME = {"L": "level", "T": "slope", "O": "network output", "V": "noise", "Mf": "fast memory",
        "Ms": "slow memory", "S": "slope signal", "Y": "forecast of $\\yv_t$"}
ORDER = ["L", "T", "O", "V", "Mf", "Ms"]
DERIVED = ["S", "Y"]


def write_tables(r, rows, gains, data_gain):
    s1, s2 = -(r["Cc"] @ r["m0"])
    lines = [r"\begin{tabular}{@{}lrccrr@{}}", r"\toprule",
             r" & & \multicolumn{2}{c}{\gain{Gain} for the reading of} & & \\",
             r"\cmidrule(lr){3-4}",
             r"Quantity & \multicolumn{1}{c}{Before the wish} & $\ms_t$ & $\Sv_t$"
             r" & \multicolumn{1}{c}{\wish{Move}} & \multicolumn{1}{c}{After the wish}\\", r"\midrule"]
    for n in ORDER + ["mid"] + DERIVED:
        if n == "mid":
            lines.append(r"\midrule")
            continue
        (m0, sd0), (m1, sd1), _ = rows[n]
        g1, g2 = gains[n]
        lines.append(rf"${SYMBOL[n]}$ {NAME[n]} & ${pm(m0, sd0)}$ & $\gain{{{num(g1)}}}$ & "
                     rf"$\gain{{{num(g2)}}}$ & $\wish{{{signed(m1 - m0)}}}$ & ${pm(m1, sd1)}$\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "one_week_wish_table.tex").write_text("\n".join(lines) + "\n")

    lines = [r"\begin{tabular}{@{}lrcrr@{}}", r"\toprule",
             r"Quantity & \multicolumn{1}{c}{After the wish} & \gain{Gain} & \multicolumn{1}{c}{\textcolor{cY}{Move}}"
             r" & \multicolumn{1}{c}{After the data}\\", r"\midrule"]
    for n in ORDER + ["mid"] + DERIVED:
        if n == "mid":
            lines.append(r"\midrule")
            continue
        _, (m1, sd1), (m2, sd2) = rows[n]
        lines.append(rf"${SYMBOL[n]}$ {NAME[n]} & ${pm(m1, sd1)}$ & $\gain{{{num(data_gain[n])}}}$ & "
                     rf"$\textcolor{{cY}}{{{signed(m2 - m1)}}}$ & ${pm(m2, sd2)}$\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "one_week_data_table.tex").write_text("\n".join(lines) + "\n")

    def mac(name, value):
        return rf"\newcommand{{\{name}}}{{{value}}}"
    vals = [
        mac("owWeek", WEEK), mac("owDrift", DRIFT), mac("owSlopeTrue", SLOPE_TRUE),
        mac("owSigmaO", SIGMA_O),
        mac("owSurpM", num(s1)), mac("owSurpS", num(s2)),
        mac("owY", num(r["y"])), mac("owSurpY", num(r["y"] - rows["Y"][1][0])),
        mac("owKMs", num(gains["Ms"][0])), mac("owKMsS", num(gains["Ms"][1])),
        mac("owKL", num(gains["L"][0])), mac("owKLS", num(gains["L"][1])),
        mac("owMoveLM", num(gains["L"][0] * s1)), mac("owMoveLS", num(gains["L"][1] * s2)),
        mac("owMoveL", num(rows["L"][1][0] - rows["L"][0][0])),
        mac("owTauD", num(TAU_D)),
    ]
    for n in ORDER + DERIVED:
        for stage, (m, sd) in zip(("Pred", "Wish", "Data"), rows[n]):
            vals.append(mac(f"ow{n}{stage}", num(m)))
            vals.append(mac(f"ow{n}{stage}Sd", num(sd)))
        vals.append(mac(f"ow{n}WishSigma", f"{(rows[n][1][0] - rows[n][0][0]) / rows[n][0][1]:+.1f}"
                        if rows[n][0][1] > 0 else "0"))
    (OUT / "one_week_values.tex").write_text("\n".join(vals) + "\n")


def figure(r, rows):
    fig, axes = plt.subplots(2, 4, figsize=(DOUBLE_COL[0], 3.6))
    labels = {"L": r"$L_t$, level", "T": r"$T_t$, slope ($\times10^{-3}$)", "O": r"$O_t$, network",
              "V": r"$V_t$, noise", "Mf": r"$M^{\mathrm{f}}_t$, fast memory",
              "Ms": r"$M^{\mathrm{s}}_t$, slow memory", "S": "$S_t$, slope signal\n" + r"($\times10^{-3}$)",
              "Y": r"$Y_t$, forecast"}
    scale = {"T": 1e3, "S": 1e3}
    for ax, n in zip(axes.ravel(), ORDER + DERIVED):
        f = scale.get(n, 1.0)
        m = np.array([s[0] for s in rows[n]]) * f
        sd = np.array([s[1] for s in rows[n]]) * f
        x = np.arange(3)
        ax.errorbar(x, m, yerr=sd, fmt="o", color=COLOURS[n], ms=3.5, capsize=2.5, elinewidth=1, zorder=3)
        for (x0, x1, c) in ((0, 1, C_WISH), (1, 2, C_DATA)):
            if abs(m[x1] - m[x0]) > 1e-12:
                ax.annotate("", xy=(x1 - 0.08, m[x1]), xytext=(x0 + 0.08, m[x0]),
                            arrowprops=dict(arrowstyle="-|>", color=c, lw=1.3, mutation_scale=8))
        if n == "Y":
            ax.axhline(r["y"], color=C_DATA, lw=0.8, ls="--")
            ax.text(-0.3, r["y"], r"measured $y_t$", color=C_DATA, va="bottom", fontsize=7)
        ax.set_xticks(x, ["pred.", "wish", "data"])
        ax.set_xlim(-0.4, 2.6)
        ax.text(0.04, 0.96, labels[n], transform=ax.transAxes, va="top", fontsize=7.5, color=COLOURS[n])
        lo, hi = (m - sd).min(), (m + sd).max()
        pad = 0.35 * (hi - lo)
        ax.set_ylim(lo - 0.1 * (hi - lo), hi + pad)
        ax.tick_params(labelsize=7)
    handles = [plt.Line2D([], [], color=C_WISH, lw=1.3, marker=">", ms=4, label="push from the wish"),
               plt.Line2D([], [], color=C_DATA, lw=1.3, marker=">", ms=4, label="push from the data"),
               plt.Line2D([], [], color="black", marker="o", ms=3.5, lw=0, label=r"mean $\pm1$ std")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "one_week.pgf", bbox_inches="tight")
    fig.savefig(OUT / "one_week.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    res = simulate()
    check(res)
    rows_, gains_, data_gain_ = stages(res)
    write_tables(res, rows_, gains_, data_gain_)
    figure(res, rows_)
    for n in ORDER + DERIVED:
        (a, sa), (b, sb), (c, sc) = rows_[n]
        print(f"{n:3s} {a: .6f}±{sa:.6f} -> {b: .6f}±{sb:.6f} -> {c: .6f}±{sc:.6f}   K={gains_[n]}")
    print(f"Saved to {OUT}")
