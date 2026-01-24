"""
Stochastic Volatility Models
=============================

Implements various stochastic volatility models for use within each regime.
"""

import numpy as np
from typing import Optional, Tuple
from abc import ABC, abstractmethod
import logging
from numba import jit

logger = logging.getLogger(__name__)


# --- JIT Compiled Kernels (Production Optimization) ---

@jit(nopython=True, cache=True)
def jit_heston_update(variance: float, dt: float, kappa: float, 
                      theta: float, sigma_v: float) -> float:
    """
    Core Heston variance update using Euler-Maruyama with absorption.
    Compiled for speed using Numba.
    """
    dW = np.random.randn() * np.sqrt(dt)
    
    # Absorption at zero to prevent negative variance
    V_t = max(variance, 0.0)
    
    drift = kappa * (theta - V_t)
    diffusion = sigma_v * np.sqrt(V_t) * dW
    
    V_next = V_t + drift * dt + diffusion
    return max(V_next, 0.0)


# --- Class Definitions ---

class StochasticVolatilityModel(ABC):
    """Abstract base class for stochastic volatility models."""
    
    @abstractmethod
    def simulate_variance_path(
        self,
        n_steps: int,
        dt: float,
        initial_var: float,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """Simulate a variance path."""
        pass
    
    @abstractmethod
    def get_volatility(self, variance: float) -> float:
        """Get volatility from variance."""
        pass


class ConstantVolatility(StochasticVolatilityModel):
    """
    Constant volatility model (standard Black-Scholes).
    
    Parameters
    ----------
    volatility : float
        Constant volatility level
    """
    
    def __init__(self, volatility: float):
        self.volatility = volatility
        self.variance = volatility ** 2
    
    def simulate_variance_path(
        self,
        n_steps: int,
        dt: float,
        initial_var: float = None,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """Return constant variance path."""
        return np.full(n_steps + 1, self.variance)
    
    def get_volatility(self, variance: float = None) -> float:
        """Return constant volatility."""
        return self.volatility


class HestonVolatility(StochasticVolatilityModel):
    """
    Heston stochastic volatility model.
    
    The variance follows:
        dV_t = κ(θ - V_t)dt + σ_v sqrt(V_t) dW_t^v
    
    Parameters
    ----------
    kappa : float
        Mean reversion speed
    theta : float
        Long-term variance level
    sigma_v : float
        Volatility of volatility
    rho : float
        Correlation between asset and variance Brownian motions
    """
    
    def __init__(
        self,
        kappa: float,
        theta: float,
        sigma_v: float,
        rho: float = 0.0
    ):
        self.kappa = kappa
        self.theta = theta
        self.sigma_v = sigma_v
        self.rho = rho
        
        # Feller condition for non-negativity
        self.feller_condition = 2 * kappa * theta > sigma_v ** 2
        if not self.feller_condition:
            logger.warning(
                "Feller condition not satisfied: variance may hit zero. "
                f"2*kappa*theta = {2*kappa*theta:.4f}, sigma_v^2 = {sigma_v**2:.4f}"
            )
    
    def simulate_variance_path(
        self,
        n_steps: int,
        dt: float,
        initial_var: float,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Simulate variance path using JIT-compiled Euler-Maruyama scheme.
        """
        if seed is not None:
            np.random.seed(seed)
        
        variance = np.zeros(n_steps + 1)
        variance[0] = initial_var
        
        # Use JIT compiled kernel for loop execution speed
        for t in range(n_steps):
            variance[t + 1] = jit_heston_update(
                variance[t], dt, self.kappa, self.theta, self.sigma_v
            )
        
        return variance
    
    def simulate_correlated_paths(
        self,
        n_steps: int,
        dt: float,
        initial_var: float,
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate correlated variance path and asset Brownian increments.
        (Kept for testing/demo purposes; main simulation uses AssetSimulator)
        """
        if seed is not None:
            np.random.seed(seed)
        
        variance = np.zeros(n_steps + 1)
        variance[0] = initial_var
        dW_asset = np.zeros(n_steps)
        
        for t in range(n_steps):
            # Generate correlated Brownian motions
            dW_v = np.random.randn() * np.sqrt(dt)
            dW_indep = np.random.randn() * np.sqrt(dt)
            
            # Correlated increment for asset
            dW_asset[t] = self.rho * dW_v + np.sqrt(1 - self.rho**2) * dW_indep
            
            # Update variance (using standard python logic here for simplicity in this helper method)
            V_t = max(variance[t], 0.0)
            drift = self.kappa * (self.theta - V_t)
            diffusion = self.sigma_v * np.sqrt(V_t) * dW_v
            
            variance[t + 1] = V_t + drift * dt + diffusion
            variance[t + 1] = max(variance[t + 1], 0.0)
        
        return variance, dW_asset
    
    def get_volatility(self, variance: float) -> float:
        """Get volatility from variance."""
        return np.sqrt(max(variance, 0.0))


class SABRVolatility(StochasticVolatilityModel):
    """
    SABR (Stochastic Alpha Beta Rho) model.
    
    The volatility follows:
        dσ_t = α σ_t dW_t
    
    Parameters
    ----------
    alpha : float
        Volatility of volatility
    beta : float
        CEV parameter (typically 0, 0.5, or 1)
    rho : float
        Correlation between asset and volatility
    """
    
    def __init__(self, alpha: float, beta: float = 1.0, rho: float = 0.0):
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
    
    def simulate_variance_path(
        self,
        n_steps: int,
        dt: float,
        initial_var: float,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """Simulate variance path for SABR."""
        if seed is not None:
            np.random.seed(seed)
        
        # SABR models volatility, not variance
        vol = np.sqrt(initial_var)
        volatility = np.zeros(n_steps + 1)
        volatility[0] = vol
        
        for t in range(n_steps):
            dW = np.random.randn() * np.sqrt(dt)
            volatility[t + 1] = volatility[t] + self.alpha * volatility[t] * dW
            volatility[t + 1] = max(volatility[t + 1], 0.01)  # Floor for stability
        
        return volatility ** 2  # Return variance
    
    def get_volatility(self, variance: float) -> float:
        """Get volatility from variance."""
        return np.sqrt(max(variance, 0.0))


class CEVVolatility(StochasticVolatilityModel):
    """
    Constant Elasticity of Variance (CEV) model.
    
    Local volatility: σ(S) = σ * S^(β-1)
    
    Parameters
    ----------
    sigma : float
        Volatility parameter
    beta : float
        Elasticity parameter
    """
    
    def __init__(self, sigma: float, beta: float = 1.0):
        self.sigma = sigma
        self.beta = beta
    
    def simulate_variance_path(
        self,
        n_steps: int,
        dt: float,
        initial_var: float,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """CEV has deterministic local volatility."""
        return np.full(n_steps + 1, initial_var)
    
    def get_local_volatility(self, spot_price: float) -> float:
        """Get local volatility as function of spot price."""
        return self.sigma * (spot_price ** (self.beta - 1))
    
    def get_volatility(self, variance: float = None) -> float:
        """Get base volatility."""
        return self.sigma


def create_volatility_model(
    vol_type: str,
    base_volatility: float,
    **kwargs
) -> StochasticVolatilityModel:
    """
    Factory function to create volatility models.
    
    Parameters
    ----------
    vol_type : str
        Type of volatility model: 'constant', 'heston', 'sabr', 'cev'
    base_volatility : float
        Base volatility level
    **kwargs
        Additional parameters for specific models
        
    Returns
    -------
    StochasticVolatilityModel
        Instantiated volatility model
    """
    if vol_type == 'constant':
        return ConstantVolatility(base_volatility)
    
    elif vol_type == 'heston':
        kappa = kwargs.get('mean_reversion', 1.0)
        theta = kwargs.get('long_term_var', base_volatility ** 2)
        sigma_v = kwargs.get('vol_of_vol', 0.3)
        rho = kwargs.get('correlation', 0.0)
        return HestonVolatility(kappa, theta, sigma_v, rho)
    
    elif vol_type == 'sabr':
        alpha = kwargs.get('vol_of_vol', 0.3)
        beta = kwargs.get('beta', 1.0)
        rho = kwargs.get('correlation', 0.0)
        return SABRVolatility(alpha, beta, rho)
    
    elif vol_type == 'cev':
        beta = kwargs.get('beta', 1.0)
        return CEVVolatility(base_volatility, beta)
    
    else:
        raise ValueError(f"Unknown volatility type: {vol_type}")


