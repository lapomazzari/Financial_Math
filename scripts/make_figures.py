"""Regenerate the figures in docs/figures used by the README.

Usage (from the repo root):  python scripts/make_figures.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bonus_certificate import (
    bonus_certificate_mc, bonus_certificate_payoff, bonus_certificate_price, bs_put,
    fit_gbm, load_prices, simulate_gbm_paths,
)

OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, BG = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.titlelocation": "left", "lines.linewidth": 2, "legend.frameon": False,
})

prices = load_prices()
params = fit_gbm(prices)
S0, B, H, R, T = float(prices.loc["2024-06-14"]), 24.0, 16.0, 0.02, 1.0
SIGMA = params.sigma
exact = bonus_certificate_price(S0, B, H, R, SIGMA, T)
print(f"S0={S0:.2f} sigma={SIGMA:.4f} mu={params.mu:.4f} price={exact:.4f}")

# 1) Payoff profile
x = np.linspace(0, 40, 801)
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.plot(x, x, color=INK2, lw=1, ls="--", label="Share (or certificate after the barrier was hit)")
ax.plot(x[x > H], np.maximum(x[x > H], B), color=BLUE, label="Certificate if the barrier was never hit")
ax.axvline(H, color=ORANGE, lw=1.2, ls=":")
ax.axhline(B, color=AQUA, lw=1.2, ls=":")
ax.annotate(f"barrier H = {H:.0f}", (H, 2), xytext=(4, 0), textcoords="offset points", color=INK2)
ax.annotate(f"bonus level B = {B:.0f}", (1, B), xytext=(0, 4), textcoords="offset points", color=INK2)
ax.set_xlim(0, 40); ax.set_ylim(0, 40)
ax.set_title("Bonus Certificate payoff at maturity")
ax.set_xlabel("Share price at maturity $P_T$ (EUR)"); ax.set_ylabel("Payoff (EUR)"); ax.legend(loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "payoff.png", dpi=150); plt.close(fig)

# 2) Monte Carlo: discrete barrier checks overprice, the Brownian-bridge correction does not
steps = [4, 12, 52, 252, 1000]
disc, bridge = [], []
for n in steps:
    disc.append(bonus_certificate_mc(S0, B, H, R, SIGMA, T, n_steps=n, n_paths=200_000, rng=10, bridge=False))
    bridge.append(bonus_certificate_mc(S0, B, H, R, SIGMA, T, n_steps=n, n_paths=200_000, rng=10, bridge=True))
fig, ax = plt.subplots(figsize=(9, 4.2))
for res, col, lab in [(disc, ORANGE, "Barrier checked only at time steps"), (bridge, BLUE, "With Brownian-bridge correction")]:
    m = np.array([p for p, _ in res]); se = np.array([s for _, s in res])
    ax.errorbar(steps, m, yerr=1.96 * se, color=col, marker="o", ms=5, capsize=3, label=lab)
    print(lab, dict(zip(steps, np.round(m, 4))))
ax.axhline(exact, color=INK, lw=1, ls="--", label=f"Closed form: {exact:.3f}")
ax.set_xscale("log")
ax.set_title("Monte Carlo price vs number of monitoring dates (200k paths, 95% CI)")
ax.set_xlabel("Time steps per year (log scale)"); ax.set_ylabel("Certificate price (EUR)"); ax.legend()
fig.tight_layout(); fig.savefig(OUT / "mc_convergence.png", dpi=150); plt.close(fig)

# 3) Value of the bonus feature vs barrier level and volatility
barriers = np.linspace(8, 22, 141)
fig, ax = plt.subplots(figsize=(9, 4.2))
for sig, col in [(0.15, AQUA), (SIGMA, BLUE), (0.30, ORANGE)]:
    ax.plot(barriers, bonus_certificate_price(S0, B, barriers, R, sig, T) - S0, color=col,
            label=f"σ = {sig:.0%}" + (" (Telekom estimate)" if sig == SIGMA else ""))
ax.axvline(H, color=INK2, lw=1, ls=":")
ax.annotate("H = 16", (H, 0.05), xytext=(4, 0), textcoords="offset points", color=INK2)
ax.set_title("Premium over the share (down-and-out put value) vs barrier level")
ax.set_xlabel("Barrier level H (EUR)"); ax.set_ylabel("Certificate price − share price (EUR)"); ax.legend()
fig.tight_layout(); fig.savefig(OUT / "sensitivity.png", dpi=150); plt.close(fig)
print("vanilla put value (H -> 0):", bs_put(S0, B, R, SIGMA, T))

# 4) Payoff distribution under the physical measure
paths = simulate_gbm_paths(S0, params.mu, SIGMA, T, 252, 100_000, rng=11)
hit = (paths.min(axis=1) <= H)
PT = paths[:, -1]
payoff = np.where(hit, PT, np.maximum(PT, B))
fig, ax = plt.subplots(figsize=(9, 4.2))
for data, col, lab in [(PT, INK2, "Share"), (payoff, BLUE, "Bonus Certificate")]:
    xs = np.sort(data)
    ax.plot(xs, np.arange(1, len(xs) + 1) / len(xs), color=col, label=lab)
ax.axvline(H, color=ORANGE, lw=1.2, ls=":")
ax.annotate(f"H = {H:.0f}", (H, 0.9), xytext=(4, 0), textcoords="offset points", color=INK2)
ax.annotate(f"{(payoff == B).mean():.0%} of paths pay exactly B = {B:.0f}", (B, 0.5),
            xytext=(8, 0), textcoords="offset points", color=INK2)
ax.set_xlim(8, 45)
ax.set_title(f"Value after one year, real-world drift μ = {params.mu:.1%} (100,000 paths)")
ax.set_xlabel("Value at maturity (EUR)"); ax.set_ylabel("Probability value ≤ x"); ax.legend(loc="lower right")
fig.tight_layout(); fig.savefig(OUT / "payoff_distribution.png", dpi=150); plt.close(fig)
print(f"P(barrier hit)={hit.mean():.3f}  P(payoff == B)={(payoff == B).mean():.3f}  "
      f"mean share={PT.mean():.2f} mean cert={payoff.mean():.2f}  "
      f"5% quantile share={np.quantile(PT, .05):.2f} cert={np.quantile(payoff, .05):.2f}")

# 5) Price history
fig, ax = plt.subplots(figsize=(9, 3.6))
ax.plot(prices.index, prices.values, color=BLUE, lw=1.2)
ax.set_title(f"Deutsche Telekom daily close, {prices.index[0]:%b %Y} – {prices.index[-1]:%b %Y}")
ax.set_ylabel("EUR")
fig.tight_layout(); fig.savefig(OUT / "telekom_prices.png", dpi=150); plt.close(fig)
