"""
Asset Dynamics Simulator
=========================

Simulates asset price paths under regime-switching stochastic volatility.
"""

import numpy as np
from typing import Optional, Tuple, List
import logging
from tqdm import tqdm
from numba import jit

from .regime_switching import RegimeSwitchingModel
from .stochastic_volatility import (
    StochasticVolatilityModel,
    HestonVolatility,
    create_volatility_model
)
from ..utils.input_validation import (
    validate_simulator_inputs,
    validate_simulation_inputs,
)

logger = logging.getLogger(__name__)


# --- JIT Compiled Kernels (Production Optimization) ---

@jit(nopython=True, cache=True)
def _jit_simulate_paths(
    n_paths: int,
    n_steps: int,
    n_substeps: int,
    dt: float,
    S0: float,
    regime_paths: np.ndarray,
    param_matrix: np.ndarray,
    risk_neutral_drift: float,
    use_risk_neutral: bool,
    Z_S: np.ndarray,
    Z_V: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    High-performance Monte Carlo path generator using Numba with sub-stepping.
    Uses independent noise and Reflection scheme for variance stability.
    """
    prices = np.zeros((n_paths, n_steps + 1))
    variances = np.zeros((n_paths, n_steps + 1))
    
    prices[:, 0] = S0
    
    # Initialize variances based on initial regime
    for i in range(n_paths):
        r0 = regime_paths[i, 0]
        # If Heston (type=1), init at theta, else vol^2
        if param_matrix[r0, 6] == 1.0:
            variances[i, 0] = param_matrix[r0, 3] 
        else:
            variances[i, 0] = param_matrix[r0, 1] ** 2

    # Pre-calculate sub-step constants
    dt_sub = dt / n_substeps
    sqrt_dt_sub = np.sqrt(dt_sub)

    for i in range(n_paths):
        # Local running state
        current_S = prices[i, 0]
        current_V = variances[i, 0]

        for t in range(n_steps):
            regime = regime_paths[i, t]
            
            # Extract parameters
            mu_real = param_matrix[regime, 0]
            vol_const = param_matrix[regime, 1]
            kappa = param_matrix[regime, 2]
            theta = param_matrix[regime, 3]
            sigma_v = param_matrix[regime, 4]
            rho = param_matrix[regime, 5]
            model_type = param_matrix[regime, 6]
            
            mu = risk_neutral_drift if use_risk_neutral else mu_real
            
            # --- MICRO-STEPPING LOOP ---
            for k in range(n_substeps):
                flat_idx = t * n_substeps + k
                
                # Independent Random Numbers
                z_v = Z_S[i, flat_idx]      # Z1: Variance Driver
                z_ortho = Z_V[i, flat_idx]  # Z2: Orthogonal Asset Driver
                
                if model_type == 1.0: # Heston Model
                    # Reflection Scheme: Safer than Truncation for bias
                    V_pos = max(current_V, 0.0) # Use max for drift calculation stability
                    curr_vol = np.sqrt(V_pos)
                    
                    # 1. Update Variance
                    drift_v = kappa * (theta - V_pos) * dt_sub
                    diffusion_v = sigma_v * curr_vol * z_v * sqrt_dt_sub
                    
                    # Reflection: Take absolute value if it goes negative
                    current_V = np.abs(current_V + drift_v + diffusion_v)
                    
                    # 2. Update Asset (Log-Euler)
                    # Correlated noise: Z_s = rho*Z_v + sqrt(1-rho^2)*Z_ortho
                    rho_compl = np.sqrt(1.0 - rho*rho)
                    z_s_corr = rho * z_v + rho_compl * z_ortho
                    
                    drift_S = (mu - 0.5 * V_pos) * dt_sub
                    diffusion_S = curr_vol * z_s_corr * sqrt_dt_sub
                    
                    current_S = current_S * np.exp(drift_S + diffusion_S)
                    
                else: # Constant Volatility
                    curr_vol = vol_const
                    current_V = curr_vol ** 2
                    
                    drift_S = (mu - 0.5 * curr_vol**2) * dt_sub
                    diffusion_S = curr_vol * z_ortho * sqrt_dt_sub
                    
                    current_S = current_S * np.exp(drift_S + diffusion_S)
            
            # Store Daily
            prices[i, t+1] = current_S
            variances[i, t+1] = current_V
            
    return prices, variances


# --- Class Definitions ---

class AssetSimulator:
    """
    Simulates asset price dynamics under regime-switching stochastic volatility.
    """
    
    def __init__(
        self,
        regime_model: RegimeSwitchingModel,
        spot_price: float,
        risk_free_rate: float,
        dividend_yield: float = 0.0
    ):
        validate_simulator_inputs(spot_price, risk_free_rate, dividend_yield)
        self.regime_model = regime_model
        self.S0 = spot_price
        self.r = risk_free_rate
        self.q = dividend_yield
        
        # Create volatility models for each regime
        self.vol_models = self._create_volatility_models()
        
        # Pre-compute parameter matrix for JIT
        self._param_matrix = self._build_param_matrix()
        
        logger.info(f"Initialized optimized asset simulator with S0={spot_price}, r={risk_free_rate}")
    
    def _create_volatility_models(self) -> List[StochasticVolatilityModel]:
        """Create volatility model for each regime."""
        vol_models = []
        for params in self.regime_model.regime_params:
            vol_type = 'constant'
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

    def _build_param_matrix(self) -> np.ndarray:
        """Packs regime parameters into a numpy array for Numba JIT compatibility."""
        n_regimes = len(self.regime_model.regime_params)
        matrix = np.zeros((n_regimes, 7))
        
        for i, (params, model) in enumerate(zip(self.regime_model.regime_params, self.vol_models)):
            matrix[i, 0] = params.drift
            matrix[i, 1] = params.volatility
            
            if isinstance(model, HestonVolatility):
                matrix[i, 2] = model.kappa
                matrix[i, 3] = model.theta
                matrix[i, 4] = model.sigma_v
                matrix[i, 5] = model.rho
                matrix[i, 6] = 1.0 # Flag for Heston
            else:
                matrix[i, 2] = 0.0
                matrix[i, 3] = params.volatility**2
                matrix[i, 4] = 0.0
                matrix[i, 5] = 0.0
                matrix[i, 6] = 0.0 # Flag for Constant
                
        return matrix

    def refresh_parameters(self) -> None:
        """Rebuild volatility models and parameter matrix after parameter changes."""
        self.vol_models = self._create_volatility_models()
        self._param_matrix = self._build_param_matrix()
    
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
        Simulate multiple asset price paths with regime switching using JIT optimization.
        Uses 100x micro-stepping for maximum precision.
        """
        validate_simulation_inputs(n_paths, n_steps, T, initial_regime)
        if initial_regime >= self.regime_model.n_regimes:
            raise ValueError("initial_regime must be within available regime indices")

        if seed is not None:
            np.random.seed(seed)
        
        dt = T / n_steps
        
        # --- PRODUCTION SETTING: 100 MICRO-STEPS ---
        # Ensures stochastic integrals converge properly
        n_substeps = 100
        total_steps = n_steps * n_substeps
        # -------------------------------------------

        # 1. Simulate Regime Paths
        regimes = self.regime_model.simulate_regimes(
            n_paths, n_steps, initial_regime, seed
        )
        
        # 2. Generate Independent Random Numbers
        # Z_S: Variance Driver
        # Z_V: Orthogonal Asset Driver
        if antithetic:
            half_paths = n_paths // 2
            remainder = n_paths % 2
            
            Z_S_half = np.random.randn(half_paths, total_steps)
            Z_V_half = np.random.randn(half_paths, total_steps)
            
            Z_S = np.concatenate([Z_S_half, -Z_S_half])
            Z_V = np.concatenate([Z_V_half, -Z_V_half])
            
            if remainder > 0:
                Z_S = np.concatenate([Z_S, np.random.randn(remainder, total_steps)])
                Z_V = np.concatenate([Z_V, np.random.randn(remainder, total_steps)])
        else:
            Z_S = np.random.randn(n_paths, total_steps)
            Z_V = np.random.randn(n_paths, total_steps)
        
        # 3. Simulate Prices using JIT Kernel
        risk_neutral_drift = self.r - self.q
        
        if show_progress:
            logger.info(f"Simulating {n_paths} paths (JIT+Substep x{n_substeps}, Antithetic={antithetic})...")

        prices, variances = _jit_simulate_paths(
            n_paths, n_steps, n_substeps, dt, self.S0, 
            regimes, self._param_matrix, 
            risk_neutral_drift, risk_neutral,
            Z_S, Z_V
        )
        
        return prices, regimes, variances
    
    def simulate_terminal_prices(
        self,
        n_paths: int,
        T: float,
        initial_regime: int = 0,
        seed: Optional[int] = None,
        risk_neutral: bool = True
    ) -> np.ndarray:
        """Simulate only terminal prices."""
        n_steps = max(int(T * 252), 1)
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
        
        CRITICAL FIX: Temporarily sets dividend yield to 0.0 for this test.
        The Martingale Property E[S_T * e^-rT] = S_0 only holds if q=0.
        If q > 0, E[S_T * e^-rT] = S_0 * e^-qT.
        """
        logger.info("Validating martingale property...")
        
        # 1. Save original dividend yield
        original_q = self.q
        
        # 2. Set q=0 for the Martingale test (making drift = r)
        self.q = 0.0
        
        try:
            # 3. Run Simulation
            prices, _, _ = self.simulate_paths(
                n_paths, n_steps, T, risk_neutral=True, seed=seed, show_progress=True, antithetic=True
            )
            
            terminal_prices = prices[:, -1]
            discount_factor = np.exp(-self.r * T)
            discounted_values = terminal_prices * discount_factor
            
            mean_discounted = np.mean(discounted_values)
            std_discounted = np.std(discounted_values)
            error = mean_discounted - self.S0
            relative_error = error / self.S0
            
            ci_width = 1.96 * std_discounted / np.sqrt(n_paths)
            
            results = {
                'S0': self.S0,
                'mean_discounted_ST': mean_discounted,
                'std_discounted_ST': std_discounted,
                'absolute_error': error,
                'relative_error': relative_error,
                'ci_95_lower': mean_discounted - ci_width,
                'ci_95_upper': mean_discounted + ci_width,
                'is_martingale': abs(relative_error) < 0.01,
                'n_paths': n_paths,
                'T': T
            }
            
            logger.info(f"Martingale test: E^Q[S_T e^(-rT)] = {mean_discounted:.4f}, S_0 = {self.S0:.4f}")
            logger.info(f"Relative error: {relative_error*100:.2f}%")
            
            return results
            
        finally:
            # 4. Restore original dividend yield
            self.q = original_q
    
    def get_regime_statistics(self, regime_paths: np.ndarray) -> dict:
        """Compute statistics about regime transitions."""
        n_paths, n_steps = regime_paths.shape
        n_regimes = self.regime_model.n_regimes
        
        time_in_regime = np.zeros(n_regimes)
        for regime in range(n_regimes):
            time_in_regime[regime] = np.mean(regime_paths == regime)
        
        transitions = np.sum(np.diff(regime_paths, axis=1) != 0, axis=1)
        avg_time_between = (n_steps - 1) / (transitions.mean() + 1e-10)
        
        return {
            'time_in_regime': time_in_regime,
            'avg_transitions_per_path': transitions.mean(),
            'std_transitions': transitions.std(),
            'avg_time_between_transitions': avg_time_between,
            'stationary_distribution': self.regime_model.get_stationary_distribution()
        }
