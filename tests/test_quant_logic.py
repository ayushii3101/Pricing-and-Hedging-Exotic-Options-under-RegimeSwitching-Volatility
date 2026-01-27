"""
Quant logic tests for pricing correctness.
"""

import numpy as np
from scipy.stats import norm
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pricing.exotic_options import VanillaOption


def _black_scholes_delta(
    spot: float,
    strike: float,
    maturity: float,
    volatility: float,
    risk_free_rate: float,
    dividend_yield: float,
    option_type: str,
) -> float:
    d1 = (
        np.log(spot / strike)
        + (risk_free_rate - dividend_yield + 0.5 * volatility**2) * maturity
    ) / (volatility * np.sqrt(maturity))

    if option_type == "call":
        return np.exp(-dividend_yield * maturity) * norm.cdf(d1)
    return np.exp(-dividend_yield * maturity) * (norm.cdf(d1) - 1.0)


def test_put_call_parity_black_scholes():
    spot = 100.0
    strike = 100.0
    maturity = 1.0
    risk_free_rate = 0.05
    dividend_yield = 0.02
    volatility = 0.2

    call = VanillaOption(strike=strike, maturity=maturity, option_type="call")
    put = VanillaOption(strike=strike, maturity=maturity, option_type="put")

    call_price = call.black_scholes_price(
        spot, volatility, risk_free_rate, dividend_yield=dividend_yield
    )
    put_price = put.black_scholes_price(
        spot, volatility, risk_free_rate, dividend_yield=dividend_yield
    )

    lhs = call_price - put_price
    rhs = spot * np.exp(-dividend_yield * maturity) - strike * np.exp(
        -risk_free_rate * maturity
    )

    assert np.isclose(lhs, rhs, atol=1e-6)


def test_boundary_conditions_zero_volatility():
    spot = 110.0
    strike = 100.0
    maturity = 0.5
    risk_free_rate = 0.03
    dividend_yield = 0.0
    volatility = 1.0e-8

    call = VanillaOption(strike=strike, maturity=maturity, option_type="call")
    put = VanillaOption(strike=strike, maturity=maturity, option_type="put")

    call_price = call.black_scholes_price(
        spot, volatility, risk_free_rate, dividend_yield=dividend_yield
    )
    put_price = put.black_scholes_price(
        spot, volatility, risk_free_rate, dividend_yield=dividend_yield
    )

    expected_call = max(
        spot * np.exp(-dividend_yield * maturity)
        - strike * np.exp(-risk_free_rate * maturity),
        0.0,
    )
    expected_put = max(
        strike * np.exp(-risk_free_rate * maturity)
        - spot * np.exp(-dividend_yield * maturity),
        0.0,
    )

    assert np.isclose(call_price, expected_call, atol=1e-6)
    assert np.isclose(put_price, expected_put, atol=1e-6)


def test_greeks_delta_consistency():
    spot = 100.0
    strike = 95.0
    maturity = 0.75
    risk_free_rate = 0.01
    dividend_yield = 0.0
    volatility = 0.25

    call = VanillaOption(strike=strike, maturity=maturity, option_type="call")

    eps = 1.0e-4
    price_up = call.black_scholes_price(
        spot * (1.0 + eps),
        volatility,
        risk_free_rate,
        dividend_yield=dividend_yield,
    )
    price_down = call.black_scholes_price(
        spot * (1.0 - eps),
        volatility,
        risk_free_rate,
        dividend_yield=dividend_yield,
    )
    delta_fd = (price_up - price_down) / (2.0 * spot * eps)

    delta_analytic = _black_scholes_delta(
        spot,
        strike,
        maturity,
        volatility,
        risk_free_rate,
        dividend_yield,
        option_type="call",
    )

    assert np.isclose(delta_fd, delta_analytic, atol=1e-6)
