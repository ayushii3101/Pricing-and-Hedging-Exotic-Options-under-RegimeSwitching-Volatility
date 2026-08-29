# Pricing and Hedging Exotic Options under Regime-Switching Volatility

## Overview

This project is a research-oriented Python framework for pricing and hedging path-dependent options when market volatility changes between distinct states. Market regimes evolve through a discrete-state Markov chain, while each regime defines its own drift and volatility dynamics. The default configuration uses three Heston-style regimes representing low-, medium-, and high-volatility markets.

The framework combines simulation, valuation, risk measurement, and hedge analysis in one reproducible workflow. It is intended for quantitative-finance experimentation and model validation, not direct production trading.

## Core capabilities

- Simulates risk-neutral asset, variance, and regime paths.
- Prices barrier, Asian, lookback, digital, and vanilla options with Monte Carlo methods.
- Provides a coupled finite-difference PDE solver for vanilla and simple barrier options.
- Estimates Delta, Gamma, Vega, Theta, and Rho using finite differences.
- Builds stock, cash, and vanilla-option hedge portfolios and evaluates dynamic mean-variance hedging.
- Checks the discounted-price martingale property, regime stationarity, and Monte Carlo convergence.
- Compares simulated vanilla prices with the Black-Scholes benchmark.

## Analysis workflow

The main analysis loads parameters from `config/model_config.yaml`, attempts to obtain the latest price for the configured ticker (SPY by default) from Yahoo Finance, constructs the regime-switching model, and validates its simulated dynamics. It then prices barrier, Asian, and vanilla options, calculates Greeks and convergence statistics, performs a hedging experiment, and saves the results as JSON and pickle files.

If live market data is unavailable, the analysis uses the configured fallback spot price. Market inputs, regime parameters, transition probabilities, option terms, and simulation settings are defined in the YAML configuration.

## Project layout

```text
config/             Model, market, option, simulation, and hedge settings
src/models/         Regime process, volatility models, and asset simulation
src/pricing/        Option payoffs, Monte Carlo pricing, and PDE pricing
src/hedging/        Greeks, hedge instruments, and hedge optimization
src/analytics/      Model validation and visualization utilities
src/utils/          Configuration, validation, math, and result helpers
tests/              Pricing, hedging, and regime-model tests
demo.py             Short end-to-end demonstration
main.py             Full configured analysis
results/            Generated analysis artifacts
```

## Run the project

Python 3.8 or later is required.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run the full configured analysis:

```powershell
python main.py
```

The full run writes its log to `results/analysis.log` and its serialized outputs to `results/comprehensive_analysis/`.

Run the test suite with:

```powershell
pytest tests -v
```
