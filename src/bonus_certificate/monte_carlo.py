"""Monte Carlo simulation of GBM paths and Bonus Certificate payoffs."""

import numpy as np


def simulate_gbm_paths(S0, drift, sigma, T, n_steps, n_paths, rng=None, antithetic=False):
    """Simulate GBM price paths, shape (n_paths, n_steps + 1), including S0 in column 0.

    With ``antithetic=True`` the second half of the paths uses the negated shocks
    of the first half (n_paths must be even).
    """
    rng = np.random.default_rng(rng)
    dt = T / n_steps
    if antithetic:
        if n_paths % 2:
            raise ValueError("n_paths must be even for antithetic sampling")
        Z = rng.standard_normal((n_paths // 2, n_steps))
        Z = np.vstack([Z, -Z])
    else:
        Z = rng.standard_normal((n_paths, n_steps))
    increments = (drift - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_paths = np.log(S0) + np.cumsum(increments, axis=1)
    return np.exp(np.hstack([np.full((n_paths, 1), np.log(S0)), log_paths]))


def barrier_survival(paths, H, sigma, dt, bridge=True):
    """Probability that each path stayed above H over the whole horizon.

    With ``bridge=False`` the barrier is only checked on the simulation grid (0 or 1).
    With ``bridge=True`` it also accounts for crossings *between* grid points, using the
    Brownian-bridge crossing probability exp(-2 ln(S_i/H) ln(S_{i+1}/H) / (sigma^2 dt)).
    This removes the upward bias of discretely monitored barrier prices.
    """
    above = paths > H
    if not bridge:
        return above.all(axis=1).astype(float)
    a = np.log(paths[:, :-1] / H)
    b = np.log(paths[:, 1:] / H)
    with np.errstate(over="ignore", invalid="ignore"):
        p_cross = np.where((a > 0) & (b > 0), np.exp(-2.0 * a * b / (sigma**2 * dt)), 1.0)
    return np.prod(1.0 - p_cross, axis=1)


def bonus_certificate_payoff(paths, B, H, sigma, T, bridge=True):
    """Expected payoff of a Bonus Certificate on each path, given the barrier survival probability."""
    dt = T / (paths.shape[1] - 1)
    PT = paths[:, -1]
    survival = barrier_survival(paths, H, sigma, dt, bridge=bridge)
    return PT + survival * np.maximum(B - PT, 0.0)


def bonus_certificate_mc(S0, B, H, r, sigma, T, n_steps=252, n_paths=100_000, rng=None,
                         antithetic=True, bridge=True):
    """Risk-neutral Monte Carlo price of a Bonus Certificate.

    Returns (price, standard_error). The standard error treats antithetic pairs
    as single samples, so it is valid with and without antithetic sampling.
    """
    paths = simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths, rng=rng, antithetic=antithetic)
    discounted = np.exp(-r * T) * bonus_certificate_payoff(paths, B, H, sigma, T, bridge=bridge)
    if antithetic:
        half = n_paths // 2
        samples = 0.5 * (discounted[:half] + discounted[half:])
    else:
        samples = discounted
    return samples.mean(), samples.std(ddof=1) / np.sqrt(len(samples))
