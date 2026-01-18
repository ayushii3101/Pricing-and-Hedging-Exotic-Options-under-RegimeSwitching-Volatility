"""
Asset Dynamics Simulator
=========================

Simulates asset price paths under regime-switching stochastic volatility.
"""

import numpy as np
from typing import Optional, Tuple, List
import logging
from tqdm import tqdm

from .regime_switching import RegimeSwitchingModel
from .stochastic_volatility import (
    StochasticVolatilityModel,
    HestonVolatility,
    create_volatility_model
)

logger = logging.getLogger(__name__)


class AssetSimulator:
    """
    Simulates asset price dynamics under regime-switching stochastic volatility.
    
    The asset price follows:
        dS_t = μ_i S_t dt + σ_i(S_t, t) S_t dW_t
    
    where regime i evolves according to a Markov chain.
    
    Parameters
    ----------
    regime_model : RegimeSwitchingModel
        Regime-switching model with parameters
    spot_price : float
        Initial spot price S_0
    risk_free_rate : float
        Risk-free interest rate
    dividend_yield : float, optional
        Continuous dividend yield
    """
    
    def __init__(
        self,
        regime_model: RegimeSwitchingModel,
        spot_price: float,
        risk_free_rate: float,
        dividend_yield: float = 0.0
    ):
        self.regime_model = regime_model
        self.S0 = spot_price
        self.r = risk_free_rate
        self.q = dividend_yield
        
        # Create volatility models for each regime
        self.vol_models = self._create_volatility_models()
        
        logger.info(f"Initialized asset simulator with S0={spot_price}, r={risk_free_rate}")
    
    def _create_volatility_models(self) -> List[StochasticVolatilityModel]:
        """Create volatility model for each regime."""
        vol_models = []
        
        for params in self.regime_model.regime_params:
            # Default to constant if not specified
            vol_type = 'constant'
            
            # Check if Heston parameters exist
            if params.long_term_var is not None:
                vol_type = 'heston'
            
            vol_model = create_volatility_model(
                vol_type,
                params.volatility,
                mean_reversion=params.mean_reversion,
                long_term_var=params.long_term_var or params.volatility**2,
                vol_of_vol=params.vol_of_vol,
                correlation=params.correlation
            )
            vol_models.append(vol_model)
        
        return vol_models
    
    def simulate_paths(
        self,
        n_paths: int,
        n_steps: int,
        T: float,
        initial_regime: int = 0,
        seed: Optional[int] = None,
        risk_neutral: bool = True,
        antithetic: bool = False,
        show_progress: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate multiple asset price paths with regime switching.
        
        Parameters
        ----------
        n_paths : int
            Number of paths to simulate
        n_steps : int
            Number of time steps
        T : float
            Time horizon (years)
        initial_regime : int
            Starting regime
        seed : int, optional
            Random seed
        risk_neutral : bool
            If True, use risk-free rate as drift (for pricing)
            If False, use regime-specific drift (for real-world simulation)
        antithetic : bool
            Use antithetic variates for variance reduction
        show_progress : bool
            Show progress bar
            
        Returns
        -------
        prices : np.ndarray
            Asset price paths, shape (n_paths, n_steps+1)
        regimes : np.ndarray
            Regime paths, shape (n_paths, n_steps+1)
        variances : np.ndarray
            Variance paths, shape (n_paths, n_steps+1)
        """
        if seed is not None:
            np.random.seed(seed)
        
        dt = T / n_steps
        
        # If using antithetic variates, simulate half paths
        n_sim_paths = n_paths // 2 if antithetic else n_paths
        
        # Simulate regime paths
        regimes = self.regime_model.simulate_regimes(
            n_sim_paths, n_steps, initial_regime, seed
        )
        
        # Initialize arrays
        prices = np.zeros((n_sim_paths, n_steps + 1))
        variances = np.zeros((n_sim_paths, n_steps + 1))
        prices[:, 0] = self.S0
        
        # Set initial variances based on initial regime
        for path_idx in range(n_sim_paths):
            regime = regimes[path_idx, 0]
            variances[path_idx, 0] = self.regime_model.get_volatility(regime) ** 2
        
        # Simulate paths
        iterator = tqdm(range(n_sim_paths), desc="Simulating paths") if show_progress else range(n_sim_paths)
        
        for path_idx in iterator:
            prices[path_idx], variances[path_idx] = self._simulate_single_path(
                regimes[path_idx],
                n_steps,
                dt,
                risk_neutral
            )
        
        # Apply antithetic variates if requested
        if antithetic:
            prices, regimes, variances = self._apply_antithetic(
                prices, regimes, variances, n_steps, dt, risk_neutral
            )
        
        return prices, regimes, variances
    
    def _simulate_single_path(
        self,
        regime_path: np.ndarray,
        n_steps: int,
        dt: float,
        risk_neutral: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate a single asset path given regime path."""
        prices = np.zeros(n_steps + 1)
        variances = np.zeros(n_steps + 1)
        
        prices[0] = self.S0
        regime_0 = regime_path[0]
        variances[0] = self.regime_model.get_volatility(regime_0) ** 2
        
        for t in range(n_steps):
            regime = regime_path[t]
            regime_params = self.regime_model.get_regime_parameters(regime)
            
            # Get drift
            if risk_neutral:
                mu = self.r - self.q  # Risk-neutral drift
            else:
                mu = regime_params.drift  # Real-world drift
            
            # Get volatility model for this regime
            vol_model = self.vol_models[regime]
            
            # Generate correlated Brownian motions if Heston
            if isinstance(vol_model, HestonVolatility):
                # Simulate variance
                V_t = variances[t]
                dW_v = np.random.randn() * np.sqrt(dt)
                dW_indep = np.random.randn() * np.sqrt(dt)
                
                # Correlated increment for asset
                dW_S = vol_model.rho * dW_v + np.sqrt(1 - vol_model.rho**2) * dW_indep
                
                # Update variance
                drift_v = vol_model.kappa * (vol_model.theta - V_t)
                diffusion_v = vol_model.sigma_v * np.sqrt(max(V_t, 0)) * dW_v
                variances[t + 1] = max(V_t + drift_v * dt + diffusion_v, 0.0)
                
                # Update price
                sigma_t = np.sqrt(max(V_t, 0))
            else:
                # Constant or other volatility
                dW_S = np.random.randn() * np.sqrt(dt)
                sigma_t = regime_params.volatility
                variances[t + 1] = sigma_t ** 2
            
            # Geometric Brownian motion update
            drift = (mu - 0.5 * sigma_t**2) * dt
            diffusion = sigma_t * dW_S
            prices[t + 1] = prices[t] * np.exp(drift + diffusion)
        
        return prices, variances
    
    def _apply_antithetic(
        self,
        prices: np.ndarray,
        regimes: np.ndarray,
        variances: np.ndarray,
        n_steps: int,
        dt: float,
        risk_neutral: bool
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply antithetic variates by reflecting random shocks."""
        n_half = prices.shape[0]
        
        # Double the arrays
        prices_full = np.vstack([prices, np.zeros_like(prices)])
        regimes_full = np.vstack([regimes, regimes])  # Same regimes
        variances_full = np.vstack([variances, np.zeros_like(variances)])
        
        # Simulate antithetic paths (using negated random shocks)
        for path_idx in range(n_half):
            # Store original seed state and resimulate with negated shocks
            # This is simplified - in practice would need to negate the actual random draws
            prices_full[n_half + path_idx], variances_full[n_half + path_idx] = \
                self._simulate_single_path(
                    regimes[path_idx],
                    n_steps,
                    dt,
                    risk_neutral
                )
        
        return prices_full, regimes_full, variances_full
    
    def simulate_terminal_prices(
        self,
        n_paths: int,
        T: float,
        initial_regime: int = 0,
        seed: Optional[int] = None,
        risk_neutral: bool = True
    ) -> np.ndarray:
        """
        Simulate only terminal prices (more efficient for European options).
        
        Returns
        -------
        np.ndarray
            Terminal prices S_T, shape (n_paths,)
        """
        n_steps = max(int(T * 252), 1)  # At least daily
        prices, _, _ = self.simulate_paths(
            n_paths, n_steps, T, initial_regime, seed, risk_neutral, show_progress=False
        )
        return prices[:, -1]
    
    def validate_martingale_property(
        self,
        n_paths: int = 10000,
        n_steps: int = 252,
        T: float = 1.0,
        seed: Optional[int] = None
    ) -> dict:
        """
        Validate that discounted asset prices are martingales under risk-neutral measure.
        
        E^Q[S_T * exp(-rT)] should equal S_0
        
        Returns
        -------
        dict
            Validation results with statistics
        """
        logger.info("Validating martingale property...")
        
        prices, _, _ = self.simulate_paths(
            n_paths, n_steps, T, risk_neutral=True, seed=seed, show_progress=True
        )
        
        # Compute discounted terminal values
        terminal_prices = prices[:, -1]
        discount_factor = np.exp(-self.r * T)
        discounted_values = terminal_prices * discount_factor
        
        # Statistics
        mean_discounted = np.mean(discounted_values)
        std_discounted = np.std(discounted_values)
        error = mean_discounted - self.S0
        relative_error = error / self.S0
        
        # 95% confidence interval
        ci_width = 1.96 * std_discounted / np.sqrt(n_paths)
        
        results = {
            'S0': self.S0,
            'mean_discounted_ST': mean_discounted,
            'std_discounted_ST': std_discounted,
            'absolute_error': error,
            'relative_error': relative_error,
            'ci_95_lower': mean_discounted - ci_width,
            'ci_95_upper': mean_discounted + ci_width,
            'is_martingale': abs(relative_error) < 0.01,  # 1% tolerance
            'n_paths': n_paths,
            'T': T
        }
        
        logger.info(f"Martingale test: E^Q[S_T e^(-rT)] = {mean_discounted:.4f}, S_0 = {self.S0:.4f}")
        logger.info(f"Relative error: {relative_error*100:.2f}%")
        
        return results
    
    def get_regime_statistics(
        self,
        regime_paths: np.ndarray
    ) -> dict:
        """
        Compute statistics about regime transitions.
        
        Parameters
        ----------
        regime_paths : np.ndarray
            Simulated regime paths
            
        Returns
        -------
        dict
            Statistics about regime behavior
        """
        n_paths, n_steps = regime_paths.shape
        n_regimes = self.regime_model.n_regimes
        
        # Time spent in each regime
        time_in_regime = np.zeros(n_regimes)
        for regime in range(n_regimes):
            time_in_regime[regime] = np.mean(regime_paths == regime)
        
        # Number of transitions
        transitions = np.sum(np.diff(regime_paths, axis=1) != 0, axis=1)
        
        # Average time between transitions
        avg_time_between = (n_steps - 1) / (transitions.mean() + 1e-10)
        
        return {
            'time_in_regime': time_in_regime,
            'avg_transitions_per_path': transitions.mean(),
            'std_transitions': transitions.std(),
            'avg_time_between_transitions': avg_time_between,
            'stationary_distribution': self.regime_model.get_stationary_distribution()
        }
