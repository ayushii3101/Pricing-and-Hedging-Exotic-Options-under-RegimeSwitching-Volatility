"""
Pydantic-based input validation helpers.
"""

from typing import Optional, Literal

import numpy as np
from pydantic import BaseModel, conint, confloat, validator


class _FiniteModel(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    @validator("*")
    def _ensure_finite(cls, value):
        if isinstance(value, (int, float)) and not np.isfinite(value):
            raise ValueError("value must be finite")
        return value


class OptionInputs(_FiniteModel):
    strike: confloat(gt=0)
    maturity: confloat(gt=0)
    option_type: Literal["call", "put"]


class BarrierOptionInputs(OptionInputs):
    barrier: confloat(gt=0)
    barrier_type: Literal["up-and-out", "down-and-out", "up-and-in", "down-and-in"]
    rebate: confloat(ge=0)


class AsianOptionInputs(OptionInputs):
    averaging_type: Literal["arithmetic", "geometric"]


class LookbackOptionInputs(OptionInputs):
    lookback_type: Literal["fixed", "floating"]


class DigitalOptionInputs(OptionInputs):
    payout: confloat(gt=0)


class SimulatorInputs(_FiniteModel):
    spot_price: confloat(gt=0)
    risk_free_rate: float
    dividend_yield: float


class MonteCarloInputs(_FiniteModel):
    n_simulations: conint(gt=0)
    seed: Optional[int]


class PDESolverInputs(_FiniteModel):
    S_max: confloat(gt=0)
    n_space: conint(gt=2)
    n_time: conint(gt=1)
    theta: confloat(ge=0, le=1)


class SimulationInputs(_FiniteModel):
    n_paths: conint(gt=0)
    n_steps: conint(gt=0)
    T: confloat(gt=0)
    initial_regime: conint(ge=0)


class RegimeParametersInputs(_FiniteModel):
    regime_id: conint(ge=0)
    name: str
    drift: float
    volatility: confloat(gt=0)
    mean_reversion: confloat(ge=0)
    vol_of_vol: confloat(ge=0)
    long_term_var: Optional[confloat(ge=0)]
    correlation: confloat(ge=-1, le=1)


class MarkovChainInputs(_FiniteModel):
    n_regimes: conint(gt=0)
    dt: confloat(gt=0)


class BlackScholesInputs(_FiniteModel):
    spot: confloat(gt=0)
    strike: confloat(gt=0)
    maturity: confloat(gt=0)
    volatility: confloat(gt=0)
    risk_free_rate: float
    dividend_yield: float


class BumpInputs(_FiniteModel):
    bump_size: confloat(gt=0)


def validate_option_inputs(strike: float, maturity: float, option_type: str) -> None:
    OptionInputs(strike=strike, maturity=maturity, option_type=option_type)


def validate_barrier_inputs(
    strike: float,
    maturity: float,
    option_type: str,
    barrier: float,
    barrier_type: str,
    rebate: float,
) -> None:
    BarrierOptionInputs(
        strike=strike,
        maturity=maturity,
        option_type=option_type,
        barrier=barrier,
        barrier_type=barrier_type,
        rebate=rebate,
    )


def validate_asian_inputs(
    strike: float,
    maturity: float,
    option_type: str,
    averaging_type: str,
) -> None:
    AsianOptionInputs(
        strike=strike,
        maturity=maturity,
        option_type=option_type,
        averaging_type=averaging_type,
    )


def validate_lookback_inputs(
    strike: float,
    maturity: float,
    option_type: str,
    lookback_type: str,
) -> None:
    LookbackOptionInputs(
        strike=strike,
        maturity=maturity,
        option_type=option_type,
        lookback_type=lookback_type,
    )


def validate_digital_inputs(
    strike: float,
    maturity: float,
    option_type: str,
    payout: float,
) -> None:
    DigitalOptionInputs(
        strike=strike,
        maturity=maturity,
        option_type=option_type,
        payout=payout,
    )


def validate_simulator_inputs(
    spot_price: float,
    risk_free_rate: float,
    dividend_yield: float,
) -> None:
    SimulatorInputs(
        spot_price=spot_price,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
    )


def validate_monte_carlo_inputs(n_simulations: int, seed: Optional[int]) -> None:
    MonteCarloInputs(n_simulations=n_simulations, seed=seed)


def validate_pde_inputs(
    S_max: float,
    n_space: int,
    n_time: int,
    theta: float,
) -> None:
    PDESolverInputs(S_max=S_max, n_space=n_space, n_time=n_time, theta=theta)


def validate_simulation_inputs(
    n_paths: int,
    n_steps: int,
    T: float,
    initial_regime: int,
) -> None:
    SimulationInputs(
        n_paths=n_paths, n_steps=n_steps, T=T, initial_regime=initial_regime
    )


def validate_regime_parameters(
    regime_id: int,
    name: str,
    drift: float,
    volatility: float,
    mean_reversion: float,
    vol_of_vol: float,
    long_term_var: Optional[float],
    correlation: float,
) -> None:
    RegimeParametersInputs(
        regime_id=regime_id,
        name=name,
        drift=drift,
        volatility=volatility,
        mean_reversion=mean_reversion,
        vol_of_vol=vol_of_vol,
        long_term_var=long_term_var,
        correlation=correlation,
    )


def validate_markov_chain_inputs(n_regimes: int, dt: float) -> None:
    MarkovChainInputs(n_regimes=n_regimes, dt=dt)


def validate_black_scholes_inputs(
    spot: float,
    strike: float,
    maturity: float,
    volatility: float,
    risk_free_rate: float,
    dividend_yield: float,
) -> None:
    BlackScholesInputs(
        spot=spot,
        strike=strike,
        maturity=maturity,
        volatility=volatility,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
    )


def validate_bump_size(bump_size: float) -> None:
    BumpInputs(bump_size=bump_size)
