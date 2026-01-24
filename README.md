# Regime-Switching Stochastic Volatility: Exotic Options Pricing and Hedging

## Project Overview

This project implements a comprehensive framework for pricing and hedging exotic derivative contracts (barrier options, Asian options) under regime-switching stochastic volatility models. The framework captures realistic market dynamics through a Markov chain representing different volatility regimes.

## Features

- **Multi-Regime Asset Modeling**: Stochastic volatility with regime-switching dynamics
- **Exotic Option Pricing**: Up-and-out barriers, Asian options via PDE and Monte Carlo
- **Optimal Hedging**: Dynamic mean-variance hedging portfolio construction
- **Risk-Neutral Simulation**: Martingale validation and extensive Monte Carlo analysis
- **Professional Visualizations**: Comprehensive plots and analytics dashboards
- **Robust Testing**: Unit tests and validation suite

## Project Structure

```
project2/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── setup.py                          # Package installation
├── config/
│   └── model_config.yaml             # Model parameters and configuration
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── regime_switching.py      # Markov chain regime dynamics
│   │   ├── stochastic_volatility.py # Volatility modeling
│   │   └── asset_dynamics.py        # Asset price simulation
│   ├── pricing/
│   │   ├── __init__.py
│   │   ├── pde_solver.py            # PDE-based pricing
│   │   ├── monte_carlo.py           # Monte Carlo pricing
│   │   └── exotic_options.py        # Exotic option payoffs
│   ├── hedging/
│   │   ├── __init__.py
│   │   ├── greeks.py                # Delta, Vega, Gamma computation
│   │   ├── portfolio.py             # Portfolio construction
│   │   └── optimization.py          # Mean-variance hedging
│   ├── analytics/
│   │   ├── __init__.py
│   │   ├── validation.py            # Martingale tests
│   │   ├── performance.py           # Hedging performance metrics
│   │   └── visualization.py         # Plotting and dashboards
│   └── utils/
│       ├── __init__.py
│       ├── math_utils.py            # Mathematical utilities
│       └── data_utils.py            # Data handling
├── notebooks/
│   └── 01_model_exploration.ipynb   # Interactive model exploration
├── tests/
│   ├── __init__.py
│   ├── test_regime_switching.py
│   ├── test_pricing.py
│   └── test_hedging.py
├── examples/
│   ├── basic_pricing.py             # Simple pricing example
│   ├── advanced_hedging.py          # Complete hedging workflow
│   └── sensitivity_analysis.py      # Parameter sensitivity
└── results/
    └── .gitkeep                     # Output folder for results

```

## Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Setup

```powershell
# Clone or navigate to project directory
cd "d:\(9) coding\New folder\project2"

# Create virtual environment (recommended)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Install project in development mode
pip install -e .
```

## Quick Start

### Basic Pricing Example

```python
from src.models.regime_switching import RegimeSwitchingModel
from src.pricing.exotic_options import BarrierOption
from src.pricing.monte_carlo import MonteCarloEngine

# Initialize model
model = RegimeSwitchingModel(n_regimes=3)
option = BarrierOption(option_type='up-and-out', barrier=120, strike=100)

# Price option
mc_engine = MonteCarloEngine(model, n_simulations=100000)
price = mc_engine.price_option(option)
print(f"Option Price: ${price:.4f}")
```

### Running Examples

```powershell
# Basic pricing
python examples/basic_pricing.py

# Advanced hedging
python examples/advanced_hedging.py

# Sensitivity analysis
python examples/sensitivity_analysis.py
```

## Mathematical Framework

### Regime-Switching Dynamics

The asset price follows:

```
dS_t = μ_i S_t dt + σ_i(S_t, t) S_t dW_t
```

where regime i ∈ {1, 2, 3} evolves via a Markov chain with transition matrix Q.

### PDE System

For each regime i, the option value V_i satisfies:

```
∂V_i/∂t + (1/2)σ_i² S² ∂²V_i/∂S² + r S ∂V_i/∂S - r V_i + Σ_j q_ij(V_j - V_i) = 0
```

### Mean-Variance Hedging

Optimal hedge weights minimize:

```
min_w Var[Π_T - H_T]  s.t.  E[Π_T - H_T] = 0
```

## Usage

### Jupyter Notebooks

Launch Jupyter and explore the interactive notebooks:

```powershell
jupyter notebook notebooks/
```

### Configuration

Edit `config/model_config.yaml` to customize:
- Regime parameters (drift, volatility)
- Transition probabilities
- Market parameters (risk-free rate, spot price)
- Simulation settings
- Option specifications

## Testing

Run the test suite:

```powershell
# Run all tests
pytest tests/ -v

# Run specific test module
pytest tests/test_pricing.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Results and Outputs

Results are saved to the `results/` directory:
- Option prices and Greeks
- Hedging performance metrics
- Visualization plots
- Performance reports

## Key Components

### Models

#### Markov Chain Regime Switching
The regime $r_t$ follows a continuous-time Markov chain with generator matrix $\mathbf{Q}$:

$$P(r_{t+\Delta t} = j \mid r_t = i) = Q_{ij}\Delta t + o(\Delta t), \quad i \neq j$$

#### Stochastic Volatility Models
**Heston Model**: Volatility follows a mean-reverting square-root process:

$$dv_t = \kappa(\theta - v_t)dt + \sigma_v\sqrt{v_t}dW_t^v$$

**SABR Model**: Stochastic $\alpha$ $\beta$ $\rho$ model for forward rates

**CEV Model**: Constant elasticity of variance with local volatility $\sigma(S_t) = \sigma S_t^{\beta-1}$

#### Asset Dynamics
Under risk-neutral measure $\mathbb{Q}$ with regime $r_t$:

$$dS_t = rS_tdt + \sigma(r_t, v_t)S_tdW_t$$

where $\sigma(r_t, v_t)$ is the regime-dependent stochastic volatility.

### Pricing

#### PDE Solver
Solves the coupled system of PDEs for option value $V_i(S,t)$ in regime $i$:

$$\frac{\partial V_i}{\partial t} + \frac{1}{2}\sigma_i^2S^2\frac{\partial^2 V_i}{\partial S^2} + rS\frac{\partial V_i}{\partial S} - rV_i + \sum_{j\neq i}Q_{ij}(V_j - V_i) = 0$$

#### Monte Carlo Pricing
Option price with payoff $\Phi(S_T)$:

$$V_0 = e^{-rT}\mathbb{E}^{\mathbb{Q}}[\Phi(S_T) \mid \mathcal{F}_0]$$

Implemented with variance reduction techniques (antithetic variates, control variates).

#### Exotic Options
- **Barrier Options**: Payoff depends on whether $S_t$ crosses barrier $B$: 
  - Up-and-out: $\Phi(S_T)\mathbb{1}_{\{\max_{0\leq t\leq T}S_t < B\}}$
- **Asian Options**: Payoff based on average price $\bar{S} = \frac{1}{T}\int_0^T S_t dt$
- **Lookback Options**: Payoff based on extrema $\max_{0\leq t\leq T}S_t$ or $\min_{0\leq t\leq T}S_t$

### Hedging

#### Greeks Calculation
First and second-order sensitivities:

$$\Delta = \frac{\partial V}{\partial S}, \quad \Gamma = \frac{\partial^2 V}{\partial S^2}, \quad \mathcal{V} = \frac{\partial V}{\partial \sigma}, \quad \Theta = \frac{\partial V}{\partial t}, \quad \rho = \frac{\partial V}{\partial r}$$

#### Dynamic Hedging
Minimize hedging error by maintaining portfolio $\Pi_t = V_t - \Delta_t S_t - \phi_t$ where $\phi_t$ is cash position.

#### Mean-Variance Optimization
Optimal hedge ratios $h^*$ minimize:

$$h^* = \arg\min_h \left\{ \mathbb{E}[(V_T - h^\top X_T)^2] - \lambda\text{Var}[h^\top X_T] \right\}$$

where $X_T$ are hedging instruments and $\lambda$ is risk-aversion parameter.

### Analytics

#### Martingale Testing
Verify risk-neutral measure: $\mathbb{E}^{\mathbb{Q}}[S_T e^{-rT} \mid \mathcal{F}_0] = S_0$ using t-test

#### Performance Metrics
- **Hedging Error**: $\epsilon_T = V_T - \Pi_T$
- **RMSE**: $\sqrt{\mathbb{E}[\epsilon_T^2]}$
- **P&L**: Cumulative profit/loss from dynamic hedging strategy

## Performance Optimization

- Vectorized NumPy operations
- Numba JIT compilation for critical paths
- Parallel Monte Carlo simulations
- Efficient sparse matrix solvers

## References

1. Hamilton, J.D. (1989). "A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle"
2. Heston, S. (1993). "A Closed-Form Solution for Options with Stochastic Volatility"
3. Elliott, R.J., Aggoun, L., and Moore, J.B. (1995). "Hidden Markov Models: Estimation and Control"
4. Föllmer, H. and Schweizer, M. (1991). "Hedging of Contingent Claims under Incomplete Information"

## License

MIT License - See LICENSE file for details

## Contributing

Contributions are welcome! Please see CONTRIBUTING.md for guidelines.

## Contact

For questions or issues, please open an issue on the project repository.

---

**Note**: This is a research and educational project. Use in production trading systems requires additional validation and risk management procedures.
