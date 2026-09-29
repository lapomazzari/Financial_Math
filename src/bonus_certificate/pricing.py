"""Closed-form Black–Scholes prices for vanilla, down-and-out and Bonus Certificate payoffs.

All rates, volatilities and maturities are in annual units, and the barrier is
monitored continuously.

A Bonus Certificate with bonus level B and barrier H < B pays at maturity T

    D = P_T + (B - P_T)^+ * 1{ min_{t<=T} P_t > H },

i.e. one share plus a down-and-out put with strike B and barrier H.
"""

import numpy as np
from scipy.stats import norm


def _d1(S, K, r, sigma, T):
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def bs_call(S, K, r, sigma, T):
    """Black–Scholes price of a European call."""
    d1 = _d1(S, K, r, sigma, T)
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put(S, K, r, sigma, T):
    """Black–Scholes price of a European put."""
    d1 = _d1(S, K, r, sigma, T)
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def knock_out_discount(S, K, H, r, sigma, T):
    """Amount by which a down barrier H <= K lowers the price of a call with strike K."""
    lam = r - 0.5 * sigma**2
    x1 = (np.log(H**2 / (S * K)) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    x2 = x1 - sigma * np.sqrt(T)
    term1 = S * (H / S) ** (2 + 2 * lam / sigma**2) * norm.cdf(x1)
    term2 = K * np.exp(-r * T) * (H / S) ** (2 * lam / sigma**2) * norm.cdf(x2)
    return term1 - term2


def down_and_out_call(S, K, H, r, sigma, T):
    """Down-and-out call with strike K and barrier H <= K (continuous monitoring)."""
    if np.any(np.asarray(H) > np.asarray(K)):
        raise ValueError("formula requires the barrier H to be at or below the strike K")
    return bs_call(S, K, r, sigma, T) - knock_out_discount(S, K, H, r, sigma, T)


def barrier_survival_probability(S, H, drift, sigma, T):
    """P(min_{t<=T} P_t > H) for a GBM whose log-price has drift `drift - sigma^2/2`.

    With drift = r this is the risk-neutral survival probability Q^(2); with
    drift = r + sigma^2 it is the survival probability under the share measure, Q^(1).
    """
    nu = drift - 0.5 * sigma**2
    a = np.log(S / H)
    s = sigma * np.sqrt(T)
    return norm.cdf((a + nu * T) / s) - (H / S) ** (2 * nu / sigma**2) * norm.cdf((-a + nu * T) / s)


def bonus_certificate_price(S, B, H, r, sigma, T):
    """Fair value of a Bonus Certificate with bonus level B and barrier H < B.

    Uses the decomposition
        V = DOC(S; K=B, H) + e^{-rT} B Q^(2) - S (Q^(1) - 1)
    where Q^(1), Q^(2) are the barrier survival probabilities under the share
    measure and the risk-neutral measure.
    """
    if np.any(np.asarray(H) >= np.asarray(B)):
        raise ValueError("barrier H must be below the bonus level B")
    if np.any(np.asarray(S) <= np.asarray(H)):
        return np.asarray(S, dtype=float) * 1.0  # barrier already hit: the certificate is one share
    q1 = barrier_survival_probability(S, H, r + sigma**2, sigma, T)
    q2 = barrier_survival_probability(S, H, r, sigma, T)
    return down_and_out_call(S, B, H, r, sigma, T) + np.exp(-r * T) * B * q2 - S * (q1 - 1)


def down_and_out_put(S, K, H, r, sigma, T):
    """Down-and-out put with strike K > H: the option part of the Bonus Certificate."""
    return bonus_certificate_price(S, K, H, r, sigma, T) - S
