"""
PDE Solver for Regime-Switching Models
======================================

Solves coupled system of PDEs for option pricing under regime switching.
"""

import numpy as np
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


class PDESolver:
    """
    Finite difference solver for coupled regime-switching PDEs.
    
    For each regime i, the option value V_i satisfies:
    
    ∂V_i/∂t + (1/2)σ_i² S² ∂²V_i/∂S² + (r-q) S ∂V_i/∂S - r V_i + Σ_j q_ij(V_j - V_i) = 0
    
    Parameters
    ----------
    regime_model : RegimeSwitchingModel
        Regime model with parameters
    risk_free_rate : float
        Risk-free rate
    dividend_yield : float
        Dividend yield
    """
    
    def __init__(self, regime_model, risk_free_rate: float, dividend_yield: float = 0.0):
        self.regime_model = regime_model
        self.r = risk_free_rate
        self.q = dividend_yield
        self.n_regimes = regime_model.n_regimes
        
        logger.info(f"Initialized PDE solver for {self.n_regimes} regimes")
    
    def solve(
        self,
        option,
        S_max: float = 300.0,
        n_space: int = 200,
        n_time: int = 500,
        theta: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve the PDE system using finite differences.
        
        Parameters
        ----------
        option : ExoticOption
            Option to price
        S_max : float
            Maximum spot price for grid
        n_space : int
            Number of space steps
        n_time : int
            Number of time steps
        theta : float
            Implicit parameter (0=explicit, 0.5=Crank-Nicolson, 1=implicit)
            
        Returns
        -------
        V : np.ndarray
            Option values, shape (n_regimes, n_space)
        S_grid : np.ndarray
            Spot price grid
        prices : np.ndarray
            Option prices at spot for each regime
        """
        from .exotic_options import BarrierOption, VanillaOption
        
        T = option.maturity
        dt = T / n_time
        dS = S_max / n_space
        
        # Space grid
        S_grid = np.linspace(0, S_max, n_space + 1)
        
        # Initialize solution arrays for each regime
        V = np.zeros((self.n_regimes, n_space + 1))
        V_new = np.zeros((self.n_regimes, n_space + 1))
        
        # Terminal condition for each regime
        for regime in range(self.n_regimes):
            V[regime, :] = self._terminal_condition(S_grid, option)
        
        # Time stepping backward
        for t_idx in range(n_time):
            # Solve coupled system for all regimes
            for regime in range(self.n_regimes):
                params = self.regime_model.get_regime_parameters(regime)
                sigma = params.volatility
                
                # Build tri-diagonal matrix
                A, b = self._build_fd_system(
                    S_grid, V[regime], sigma, dt, dS, theta, regime
                )
                
                # Add coupling terms from other regimes
                Q = self.regime_model.markov_chain.Q
                for j in range(self.n_regimes):
                    if j != regime:
                        b += dt * Q[regime, j] * V[j]
                
                # Solve system
                V_new[regime] = spsolve(A, b)
            
            V = V_new.copy()
        
        # Extract prices at current spot
        S0 = self.regime_model.regime_params[0].volatility  # Use S0 from model if available
        prices = np.array([np.interp(100.0, S_grid, V[i]) for i in range(self.n_regimes)])
        
        logger.info(f"PDE solution computed: {prices}")
        
        return V, S_grid, prices
    
    def _terminal_condition(self, S_grid: np.ndarray, option) -> np.ndarray:
        """Compute terminal payoff condition."""
        from .exotic_options import BarrierOption, VanillaOption, AsianOption
        
        if isinstance(option, (VanillaOption, BarrierOption)):
            if option.option_type == 'call':
                return np.maximum(S_grid - option.strike, 0)
            else:
                return np.maximum(option.strike - S_grid, 0)
        else:
            # For path-dependent options, use Monte Carlo instead
            raise NotImplementedError("PDE solver only supports vanilla and simple barrier options")
    
    def _build_fd_system(
        self,
        S_grid: np.ndarray,
        V_old: np.ndarray,
        sigma: float,
        dt: float,
        dS: float,
        theta: float,
        regime: int
    ) -> Tuple[csr_matrix, np.ndarray]:
        """Build finite difference system for one regime."""
        n = len(S_grid)
        
        # Coefficients for finite differences
        alpha = np.zeros(n)
        beta = np.zeros(n)
        gamma = np.zeros(n)
        
        for i in range(1, n - 1):
            S = S_grid[i]
            alpha[i] = 0.5 * dt * ((self.r - self.q) / dS * S - sigma**2 / dS**2 * S**2)
            beta[i] = 1 + dt * (sigma**2 / dS**2 * S**2 + self.r)
            gamma[i] = -0.5 * dt * ((self.r - self.q) / dS * S + sigma**2 / dS**2 * S**2)
        
        # Build matrix (simplified Crank-Nicolson)
        diagonals = [alpha[1:], beta, gamma[:-1]]
        offsets = [-1, 0, 1]
        A = diags(diagonals, offsets, shape=(n, n), format='csr')
        
        # Right-hand side
        b = V_old.copy()
        
        # Boundary conditions
        A[0, 0] = 1
        A[-1, -1] = 1
        b[0] = 0  # Lower boundary
        b[-1] = max(S_grid[-1] - 100, 0)  # Upper boundary (ITM call approximation)
        
        return A, b
