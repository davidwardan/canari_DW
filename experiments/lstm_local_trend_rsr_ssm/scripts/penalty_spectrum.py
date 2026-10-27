"""How strongly each constraint penalizes an offset, a ramp and periodic LSTM outputs.

Builds the exact batch Gaussian factor exp(-r' Lambda r / 2) that each mechanism puts on
an LSTM output sequence r (constraint factors only, no observation) and reports the
equivalent std sqrt(mean(r^2) / (r' Lambda r / T)). See `../planning/maths.md`.

Run from the repository root:

    python experiments/lstm_local_trend_rsr_ssm/scripts/penalty_spectrum.py
"""

from datetime import datetime
from zoneinfo import ZoneInfo

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_experiment import CONSTRAINT, EXP_DIR, PERIOD, SSM_STD_RATIO
from ssm_constraint import NOISE_SHAPE, TRANSITION, steady_state
from summarize import ARM_COLORS, SINGLE_COL, save

NUM_STEPS = 20 * PERIOD
STD_RESIDUAL = 0.95  # median auxiliary residual std of the ten benchmark series
JOINT_COLOR = "tab:purple"


def memory_maps(transition, gain):
    """Stack J_t with m_t = J_t r for m_t = transition m_{t-1} + gain r_t, m_0 = 0."""

    maps = np.zeros((NUM_STEPS, len(gain), NUM_STEPS))
    memory = np.zeros((len(gain), NUM_STEPS))
    for step in range(NUM_STEPS):
        memory = transition @ memory
        memory[:, step] += gain.ravel()
        maps[step] = memory
    return maps


def summary_precision(maps, rows, noise_var):
    summaries = np.einsum("ij,tjk->tik", rows, maps)
    return np.einsum("tik,ij,tjl->kl", summaries, np.linalg.inv(noise_var), summaries)


def ssm_precision():
    gain, posterior_var, memory_transition = steady_state(
        SSM_STD_RATIO * STD_RESIDUAL, STD_RESIDUAL
    )
    return summary_precision(memory_maps(memory_transition, gain), np.eye(2), posterior_var)


def ewma_precision():
    durations = np.array([CONSTRAINT["fast_span"], CONSTRAINT["slow_span"]], dtype=float)
    rhos = np.exp(-1 / durations)
    lags = rhos / (1 - rhos)
    rows = np.array([[0, 1], [1, -1]]) / np.array([[1], [lags[1] - lags[0]]])
    noise_var = np.diag([CONSTRAINT["tau_mean"] ** 2, CONSTRAINT["tau_slope"] ** 2])
    return summary_precision(memory_maps(np.diag(rhos), 1 - rhos), rows, noise_var)


def joint_precision():
    """Latent auxiliary level/slope linked by r_t = L_t + E_t, both pseudo-observed as 0.

    Uses the Kalman memories' sqrt(diag(P)) as level/slope tolerances.
    """

    _, posterior_var, _ = steady_state(SSM_STD_RATIO * STD_RESIDUAL, STD_RESIDUAL)
    process_noise = (SSM_STD_RATIO * STD_RESIDUAL) ** 2 * NOISE_SHAPE
    # Prior path covariance: Var(Z_t) recursion, Cov(Z_t, Z_s) = A^(t-s) Var(Z_s)
    variances = np.zeros((NUM_STEPS, 2, 2))
    variance = posterior_var  # same prior as the memories
    for step in range(NUM_STEPS):
        variance = TRANSITION @ variance @ TRANSITION.T + process_noise
        variances[step] = variance
    lag = np.subtract.outer(np.arange(NUM_STEPS), np.arange(NUM_STEPS))
    blocks = np.broadcast_to(variances[None], (NUM_STEPS, NUM_STEPS, 2, 2)).copy()
    blocks[:, :, 0, :] += np.maximum(lag, 0)[..., None] * variances[None, :, 1, :]
    lower = (lag >= 0)[..., None, None]
    blocks = np.where(lower, blocks, np.swapaxes(blocks, 0, 1).swapaxes(2, 3))
    prior = blocks.transpose(0, 2, 1, 3).reshape(2 * NUM_STEPS, 2 * NUM_STEPS)
    # Condition every L_t and S_t on zero with the memories' marginal tolerances
    noise = np.diag(np.tile(np.diag(posterior_var), NUM_STEPS))
    posterior = prior - prior @ np.linalg.solve(prior + noise, prior)
    covariance = posterior[::2, ::2]
    return np.linalg.inv(covariance + STD_RESIDUAL**2 * np.eye(NUM_STEPS))


def equivalent_std(precision, pattern):
    return float(np.sqrt(np.mean(pattern**2) / (pattern @ precision @ pattern / NUM_STEPS)))


def main():
    precisions = {
        "Joint alternative": joint_precision(),
        "EWMA": ewma_precision(),
        "Aux SSM": ssm_precision(),
    }
    time = np.arange(NUM_STEPS, dtype=float)
    patterns = {
        "offset": np.ones(NUM_STEPS),
        "ramp": time - time.mean(),
        f"seasonal (period {PERIOD})": np.sin(2 * np.pi * time / PERIOD),
    }
    table = pd.DataFrame(
        {name: [equivalent_std(precision, pattern) for pattern in patterns.values()]
         for name, precision in precisions.items()},
        index=list(patterns),
    )
    table.loc["seasonal / offset"] = table.iloc[2] / table.iloc[0]

    periods = np.geomspace(4, NUM_STEPS, 60)
    spectrum = pd.DataFrame(
        {name: [equivalent_std(precision, np.sin(2 * np.pi * time / period))
                for period in periods]
         for name, precision in precisions.items()},
        index=pd.Index(periods, name="period_weeks"),
    )

    tag = datetime.now(ZoneInfo("America/Toronto")).strftime("%Y%m%d-%H%M%S")
    out_dir = EXP_DIR / "results" / f"{tag}_penalty_spectrum"
    out_dir.mkdir(parents=True, exist_ok=False)
    table.to_csv(out_dir / "equivalent_std.csv")
    spectrum.to_csv(out_dir / "spectrum.csv")
    lines = [
        f"Equivalent std (smaller = stronger penalty), {NUM_STEPS} weeks, "
        f"sigma_E = {STD_RESIDUAL}.",
        "",
        "| Pattern | " + " | ".join(table.columns) + " |",
        "| :--- |" + " ---: |" * len(table.columns),
    ]
    lines += [f"| {name} | " + " | ".join(f"{value:.4g}" for value in row) + " |"
              for name, row in table.iterrows()]
    (out_dir / "equivalent_std.md").write_text("\n".join(lines) + "\n")
    print(table.to_string(float_format=lambda value: f"{value:.4g}"))

    colors = {"Joint alternative": JOINT_COLOR, "EWMA": ARM_COLORS["ewma"],
              "Aux SSM": ARM_COLORS["ssm"]}
    fig, ax = plt.subplots(figsize=SINGLE_COL)
    for name in precisions:
        ax.loglog(periods, spectrum[name], color=colors[name], label=name)
    ax.axvline(PERIOD, color="0.5", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Period of $r_t$ [weeks]")
    ax.set_ylabel("Equivalent std [std]")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1), ncol=3, frameon=False)
    fig.tight_layout()
    (EXP_DIR / "figures").mkdir(exist_ok=True)
    save(fig, EXP_DIR / "figures" / f"{tag}_penalty_spectrum")
    print(f"Saved {out_dir} and figures/{tag}_penalty_spectrum.pdf")


if __name__ == "__main__":
    main()
