"""Pricing and risk analysis of Bonus Certificates under Black–Scholes."""

from .estimation import GBMParams, fit_gbm, load_prices, log_returns
from .monte_carlo import bonus_certificate_mc, bonus_certificate_payoff, simulate_gbm_paths
from .pricing import (
    barrier_survival_probability,
    bonus_certificate_price,
    bs_call,
    bs_put,
    down_and_out_call,
    down_and_out_put,
    knock_out_discount,
)

__all__ = [
    "GBMParams", "fit_gbm", "load_prices", "log_returns",
    "bonus_certificate_mc", "bonus_certificate_payoff", "simulate_gbm_paths",
    "barrier_survival_probability", "bonus_certificate_price", "bs_call", "bs_put",
    "down_and_out_call", "down_and_out_put", "knock_out_discount",
]
