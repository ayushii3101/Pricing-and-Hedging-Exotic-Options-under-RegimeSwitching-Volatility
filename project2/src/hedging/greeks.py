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
    
    def __init__(self, monte_carlo_engine, bump_size: float = 0.01):
        self.mc_engine = monte_carlo_engine
        self.bump_size = bump_size
        
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
        price_up = self.mc_engine.price_option(option, initial_regime, show_progress=False)['price']
        
        self.mc_engine.simulator.S0 = S0_original * (1 - self.bump_size)
        price_down = self.mc_engine.price_option(option, initial_regime, show_progress=False)['price']
        
        self.mc_engine.simulator.S0 = S0_original
        
        return (price_up - price_down) / (2 * S0_original * self.bump_size)
    
    def gamma(self, option, initial_regime: int = 0) -> float:
        """Calculate Gamma: ∂²V/∂S²"""
        S0_original = self.mc_engine.simulator.S0
        
        base_price = self.mc_engine.price_option(option, initial_regime, show_progress=False)['price']
        
        self.mc_engine.simulator.S0 = S0_original * (1 + self.bump_size)
        price_up = self.mc_engine.price_option(option, initial_regime, show_progress=False)['price']
        
        self.mc_engine.simulator.S0 = S0_original * (1 - self.bump_size)
        price_down = self.mc_engine.price_option(option, initial_regime, show_progress=False)['price']
        
        self.mc_engine.simulator.S0 = S0_original
        
        return (price_up - 2 * base_price + price_down) / ((S0_original * self.bump_size) ** 2)
    
    def vega(self, option, initial_regime: int = 0) -> float:
        """Calculate Vega: ∂V/∂σ"""
        base_price = self.mc_engine.price_option(option, initial_regime, show_progress=False)['price']
        
        # Bump all regime volatilities
        original_vols = []
        for params in self.mc_engine.simulator.regime_model.regime_params:
            original_vols.append(params.volatility)
            params.volatility *= (1 + self.bump_size)
        
        self.mc_engine.simulator.vol_models = self.mc_engine.simulator._create_volatility_models()
        price_bumped = self.mc_engine.price_option(option, initial_regime, show_progress=False)['price']
        
        # Restore
        for i, params in enumerate(self.mc_engine.simulator.regime_model.regime_params):
            params.volatility = original_vols[i]
        self.mc_engine.simulator.vol_models = self.mc_engine.simulator._create_volatility_models()
        
        return (price_bumped - base_price) / self.bump_size
    
    def theta(self, option, initial_regime: int = 0) -> float:
        """Calculate Theta: ∂V/∂t"""
        base_price = self.mc_engine.price_option(option, initial_regime, show_progress=False)['price']
        
        T_original = option.maturity
        option.maturity = T_original - 1.0 / 252
        price_forward = self.mc_engine.price_option(option, initial_regime, show_progress=False)['price']
        option.maturity = T_original
        
        return (price_forward - base_price) / (1.0 / 252)
    
    def rho(self, option, initial_regime: int = 0) -> float:
        """Calculate Rho: ∂V/∂r"""
        base_price = self.mc_engine.price_option(option, initial_regime, show_progress=False)['price']
        
        r_original = self.mc_engine.simulator.r
        self.mc_engine.simulator.r = r_original + self.bump_size
        price_bumped = self.mc_engine.price_option(option, initial_regime, show_progress=False)['price']
        self.mc_engine.simulator.r = r_original
        
        return (price_bumped - base_price) / self.bump_size
