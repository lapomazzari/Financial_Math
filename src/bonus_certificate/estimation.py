"""Estimation of Black–Scholes parameters from historical prices."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TRADING_DAYS = 252
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def load_prices(path=DATA_DIR / "telekom.csv"):
    """Daily closing prices as a Series indexed by date."""
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    return pd.to_numeric(df["Close"], errors="coerce").dropna()


def log_returns(prices):
    return np.log(prices / prices.shift(1)).dropna()


@dataclass
class GBMParams:
    mu: float     # annual drift
    sigma: float  # annual volatility


def fit_gbm(prices, trading_days=TRADING_DAYS):
    """Estimate annual GBM drift and volatility from daily log-returns.

    Y_i ~ N((mu - sigma^2/2) dt, sigma^2 dt) with dt = 1/trading_days, so
    sigma = sd(Y) / sqrt(dt) and mu = mean(Y) / dt + sigma^2 / 2.
    """
    y = log_returns(prices)
    dt = 1.0 / trading_days
    sigma = y.std() / np.sqrt(dt)
    mu = y.mean() / dt + 0.5 * sigma**2
    return GBMParams(mu=float(mu), sigma=float(sigma))
