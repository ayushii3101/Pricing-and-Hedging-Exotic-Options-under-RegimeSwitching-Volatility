"""
Regime-Switching Stochastic Volatility Framework
==============================================

A comprehensive framework for pricing and hedging exotic options under
regime-switching stochastic volatility models.
"""

__version__ = "1.0.0"
__author__ = "Quantitative Finance Research Team"

from src.models.regime_switching import RegimeSwitchingModel, MarkovChain
from src.models.asset_dynamics import AssetSimulator
from src.pricing.exotic_options import BarrierOption, AsianOption, LookbackOption
from src.pricing.monte_carlo import MonteCarloEngine
from src.hedging.portfolio import HedgingPortfolio
from src.hedging.optimization import MeanVarianceHedger

__all__ = [
    "RegimeSwitchingModel",
    "MarkovChain",
    "AssetSimulator",
    "BarrierOption",
    "AsianOption",
    "LookbackOption",
    "MonteCarloEngine",
    "HedgingPortfolio",
    "MeanVarianceHedger",
]
