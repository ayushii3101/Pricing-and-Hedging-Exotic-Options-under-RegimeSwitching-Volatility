"""
Greeks Calculation
==================

Computes option Greeks (Delta, Gamma, Vega, Theta, Rho) for hedging.
"""

import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class GreeksCalculator:
    """
    Calculate option Greeks using finite difference methods.
    
    Parameters
    ----------
    monte_carlo_engine : MonteCarloEngine
        MC engine for pricing
    bump_size : float
        Bump size for finite differences
    """
    
    def __init__(self, monte_carlo_engine, bump_size: float = 0.01, fd_seed: Optional[int] = 12345):
        self.mc_engine = monte_carlo_engine
        self.bump_size = bump_size
        self.fd_seed = fd_seed

    def _price_with_seed(self, option, initial_regime: int = 0) -> float:
        """Price option using a fixed seed for finite-difference stability."""
        original_seed = getattr(self.mc_engine, "seed", None)
        if self.fd_seed is not None:
            self.mc_engine.seed = self.fd_seed
        try:
            return self.mc_engine.price_option(option, initial_regime, show_progress=False)['price']
        finally:
            self.mc_engine.seed = original_seed
        
    def calculate_all(
        self,
        option,
        initial_regime: int = 0
    ) -> Dict[str, float]:
        """Calculate all Greeks."""
        return self.mc_engine.price_with_greeks(option, initial_regime, self.bump_size)
    
    def delta(self, option, initial_regime: int = 0) -> float:
        """Calculate Delta: ∂V/∂S"""
        S0_original = self.mc_engine.simulator.S0

        self.mc_engine.simulator.S0 = S0_original * (1 + self.bump_size)
        price_up = self._price_with_seed(option, initial_regime)

        self.mc_engine.simulator.S0 = S0_original * (1 - self.bump_size)
        price_down = self._price_with_seed(option, initial_regime)

        self.mc_engine.simulator.S0 = S0_original

        return (price_up - price_down) / (2 * S0_original * self.bump_size)
    
    def gamma(self, option, initial_regime: int = 0) -> float:
        """Calculate Gamma: ∂²V/∂S²"""
        S0_original = self.mc_engine.simulator.S0

        base_price = self._price_with_seed(option, initial_regime)

        self.mc_engine.simulator.S0 = S0_original * (1 + self.bump_size)
        price_up = self._price_with_seed(option, initial_regime)

        self.mc_engine.simulator.S0 = S0_original * (1 - self.bump_size)
        price_down = self._price_with_seed(option, initial_regime)

        self.mc_engine.simulator.S0 = S0_original

        return (price_up - 2 * base_price + price_down) / ((S0_original * self.bump_size) ** 2)
    
    def vega(self, option, initial_regime: int = 0) -> float:
        """Calculate Vega: ∂V/∂σ"""
        base_price = self._price_with_seed(option, initial_regime)
        
        # Bump all regime volatilities
        original_vols = []
        for params in self.mc_engine.simulator.regime_model.regime_params:
            original_vols.append(params.volatility)
            params.volatility *= (1 + self.bump_size)

        self.mc_engine.simulator.refresh_parameters()
        price_bumped = self._price_with_seed(option, initial_regime)
        
        # Restore
        for i, params in enumerate(self.mc_engine.simulator.regime_model.regime_params):
            params.volatility = original_vols[i]
        self.mc_engine.simulator.refresh_parameters()
        
        return (price_bumped - base_price) / self.bump_size
    
    def theta(self, option, initial_regime: int = 0) -> float:
        """Calculate Theta: ∂V/∂t"""
        base_price = self._price_with_seed(option, initial_regime)
        
        T_original = option.maturity
        option.maturity = T_original - 1.0 / 252
        price_forward = self._price_with_seed(option, initial_regime)
        option.maturity = T_original
        
        return (price_forward - base_price) / (1.0 / 252)
    
    def rho(self, option, initial_regime: int = 0) -> float:
        """Calculate Rho: ∂V/∂r"""
        base_price = self._price_with_seed(option, initial_regime)
        
        r_original = self.mc_engine.simulator.r
        self.mc_engine.simulator.r = r_original + self.bump_size
        price_bumped = self._price_with_seed(option, initial_regime)
        self.mc_engine.simulator.r = r_original
        
        return (price_bumped - base_price) / self.bump_size
