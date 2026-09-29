import numpy as np
import pandas as pd
import pytest

from bonus_certificate import (
    barrier_survival_probability,
    bonus_certificate_mc,
    bonus_certificate_price,
    bs_call,
    bs_put,
    down_and_out_call,
    fit_gbm,
    load_prices,
    simulate_gbm_paths,
)

S, B, H, R, SIGMA, T = 22.63, 24.0, 16.0, 0.02, 0.21, 1.0


def test_put_call_parity():
    assert bs_call(S, B, R, SIGMA, T) - bs_put(S, B, R, SIGMA, T) == pytest.approx(S - B * np.exp(-R * T))


def test_down_and_out_call_tends_to_vanilla_for_a_far_barrier():
    assert down_and_out_call(S, B, 1e-6, R, SIGMA, T) == pytest.approx(bs_call(S, B, R, SIGMA, T), rel=1e-10)
    assert down_and_out_call(S, B, H, R, SIGMA, T) < bs_call(S, B, R, SIGMA, T)


def test_bonus_certificate_limits():
    # Far barrier: share + vanilla put with strike B.
    assert bonus_certificate_price(S, B, 1e-6, R, SIGMA, T) == pytest.approx(S + bs_put(S, B, R, SIGMA, T), rel=1e-10)
    # Barrier already breached: the certificate is just the share.
    assert bonus_certificate_price(15.0, B, H, R, SIGMA, T) == pytest.approx(15.0)
    # In between: worth more than the share, less than share + vanilla put.
    price = bonus_certificate_price(S, B, H, R, SIGMA, T)
    assert S < price < S + bs_put(S, B, R, SIGMA, T)


def test_barrier_survival_probability_matches_simulation():
    paths = simulate_gbm_paths(S, R, SIGMA, T, n_steps=2000, n_paths=20_000, rng=0)
    simulated = (paths.min(axis=1) > H).mean()
    assert barrier_survival_probability(S, H, R, SIGMA, T) == pytest.approx(simulated, abs=0.01)


def test_monte_carlo_with_brownian_bridge_matches_closed_form():
    price, se = bonus_certificate_mc(S, B, H, R, SIGMA, T, n_steps=252, n_paths=200_000, rng=1)
    assert abs(price - bonus_certificate_price(S, B, H, R, SIGMA, T)) < 4 * se


def test_discrete_barrier_monitoring_overprices():
    exact = bonus_certificate_price(S, B, H, R, SIGMA, T)
    discrete, se = bonus_certificate_mc(S, B, H, R, SIGMA, T, n_steps=12, n_paths=200_000, rng=2, bridge=False)
    assert discrete - exact > 4 * se


def test_fit_gbm_recovers_parameters():
    paths = simulate_gbm_paths(100.0, 0.08, 0.25, T=40.0, n_steps=40 * 252, n_paths=1, rng=3)
    prices = pd.Series(paths[0])
    params = fit_gbm(prices)
    assert params.sigma == pytest.approx(0.25, rel=0.02)
    assert params.mu == pytest.approx(0.08, abs=0.1)  # drift is notoriously hard to estimate


def test_telekom_headline_price():
    prices = load_prices()
    params = fit_gbm(prices)
    assert prices.loc["2024-06-14"] == pytest.approx(22.63, abs=0.01)
    assert params.sigma == pytest.approx(0.2124, abs=1e-3)
    assert bonus_certificate_price(prices.loc["2024-06-14"], B, H, R, params.sigma, T) == pytest.approx(24.251, abs=1e-3)
