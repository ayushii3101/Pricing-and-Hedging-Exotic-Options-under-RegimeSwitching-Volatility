"""
Regime-Switching Markov Chain Model
====================================

Implements discrete-state Markov chains for modeling regime transitions
in financial markets.
"""

import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass
import logging

from ..utils.input_validation import (
    validate_regime_parameters,
    validate_markov_chain_inputs,
)

logger = logging.getLogger(__name__)


@dataclass
class RegimeParameters:
    """Parameters for a single regime."""
    regime_id: int
    name: str
    drift: float
    volatility: float
    mean_reversion: float = 0.0
    vol_of_vol: float = 0.0
    long_term_var: Optional[float] = None
    correlation: float = 0.0

    def __post_init__(self) -> None:
        validate_regime_parameters(
            self.regime_id,
            self.name,
            self.drift,
            self.volatility,
            self.mean_reversion,
            self.vol_of_vol,
            self.long_term_var,
            self.correlation,
        )


class MarkovChain:
    """
    Discrete-state Markov chain for regime switching.
    
    The chain transitions between n_regimes states according to a 
    transition probability matrix Q.
    
    Parameters
    ----------
    n_regimes : int
        Number of regimes/states
    transition_matrix : np.ndarray
        Transition probability matrix Q where Q[i,j] is the probability
        of transitioning from regime i to regime j
    dt : float, optional
        Time step for continuous-time approximation
    """
    
    def __init__(
        self,
        n_regimes: int,
        transition_matrix: np.ndarray,
        dt: float = 1.0 / 252  # Daily time step
    ):
        validate_markov_chain_inputs(n_regimes, dt)
        self.n_regimes = n_regimes
        self.Q = np.array(transition_matrix)
        self.dt = dt
        
        self._validate_transition_matrix()
        self._compute_generator()
        
        logger.info(f"Initialized Markov chain with {n_regimes} regimes")
    
    def _validate_transition_matrix(self) -> None:
        """Validate that Q is a valid transition matrix."""
        if self.Q.shape != (self.n_regimes, self.n_regimes):
            raise ValueError(
                f"Transition matrix shape {self.Q.shape} doesn't match "
                f"n_regimes={self.n_regimes}"
            )
        
        # Check rows sum to 1
        row_sums = self.Q.sum(axis=1)
        if not np.allclose(row_sums, 1.0):
            raise ValueError(f"Transition matrix rows must sum to 1, got {row_sums}")
        
        # Check non-negative
        if np.any(self.Q < 0):
            raise ValueError("Transition matrix must have non-negative entries")
    
    def _compute_generator(self) -> None:
        """Compute the generator matrix (Q-matrix) for continuous time."""
        # For continuous time, we need the generator A such that Q(dt) ≈ I + A*dt
        # A = (Q - I) / dt
        self.generator = (self.Q - np.eye(self.n_regimes)) / self.dt
        
    def simulate_path(
        self,
        n_steps: int,
        initial_regime: int = 0,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Simulate a regime path.
        
        Parameters
        ----------
        n_steps : int
            Number of time steps
        initial_regime : int
            Starting regime (0-indexed)
        seed : int, optional
            Random seed for reproducibility
            
        Returns
        -------
        np.ndarray
            Array of regime indices, shape (n_steps+1,)
        """
        if seed is not None:
            np.random.seed(seed)
        
        regime_path = np.zeros(n_steps + 1, dtype=int)
        regime_path[0] = initial_regime
        
        for t in range(n_steps):
            current_regime = regime_path[t]
            # Sample next regime from transition probabilities
            regime_path[t + 1] = np.random.choice(
                self.n_regimes,
                p=self.Q[current_regime]
            )
        
        return regime_path
    
    def simulate_multiple_paths(
        self,
        n_paths: int,
        n_steps: int,
        initial_regime: int = 0,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Simulate multiple regime paths.
        
        Parameters
        ----------
        n_paths : int
            Number of paths to simulate
        n_steps : int
            Number of time steps per path
        initial_regime : int
            Starting regime for all paths
        seed : int, optional
            Random seed
            
        Returns
        -------
        np.ndarray
            Array of regime paths, shape (n_paths, n_steps+1)
        """
        if seed is not None:
            np.random.seed(seed)
        
        regime_paths = np.zeros((n_paths, n_steps + 1), dtype=int)
        regime_paths[:, 0] = initial_regime
        
        for t in range(n_steps):
            for path_idx in range(n_paths):
                current_regime = regime_paths[path_idx, t]
                regime_paths[path_idx, t + 1] = np.random.choice(
                    self.n_regimes,
                    p=self.Q[current_regime]
                )
        
        return regime_paths
    
    def stationary_distribution(self) -> np.ndarray:
        """
        Compute the stationary distribution of the Markov chain.
        
        Returns
        -------
        np.ndarray
            Stationary distribution π where π Q = π
        """
        # Solve π Q = π with constraint sum(π) = 1
        # This is equivalent to finding left eigenvector with eigenvalue 1
        eigenvalues, eigenvectors = np.linalg.eig(self.Q.T)
        
        # Find eigenvector corresponding to eigenvalue 1
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        stationary = np.real(eigenvectors[:, idx])
        stationary = stationary / stationary.sum()
        
        return stationary
    
    def expected_time_in_regime(self, regime: int) -> float:
        """
        Expected time spent in a regime before transitioning.
        
        Parameters
        ----------
        regime : int
            Regime index
            
        Returns
        -------
        float
            Expected duration in the regime
        """
        # Expected time = 1 / (exit rate)
        exit_rate = 1.0 - self.Q[regime, regime]
        if exit_rate == 0:
            return np.inf
        return self.dt / exit_rate


class RegimeSwitchingModel:
    """
    Complete regime-switching model with multiple regimes and their parameters.
    
    Parameters
    ----------
    transition_matrix : np.ndarray
        Transition probability matrix
    regime_params : List[RegimeParameters]
        List of parameters for each regime
    dt : float
        Time step
    """
    
    def __init__(
        self,
        *args,
        transition_matrix: Optional[np.ndarray] = None,
        regime_params: Optional[List[RegimeParameters]] = None,
        dt: float = 1.0 / 252
    ):
        def _is_regime_params(candidate: object) -> bool:
            return (
                isinstance(candidate, list)
                and len(candidate) > 0
                and all(isinstance(item, RegimeParameters) for item in candidate)
            )

        def _is_square_numeric_matrix(candidate: object) -> bool:
            try:
                arr = np.array(candidate)
            except Exception:
                return False
            return (
                arr.ndim == 2
                and arr.shape[0] == arr.shape[1]
                and np.issubdtype(arr.dtype, np.number)
            )

        # Accept positional or keyword combinations:
        # (transition_matrix, regime_params) or (regime_params, transition_matrix).
        if len(args) > 2:
            raise TypeError("RegimeSwitchingModel accepts at most two positional arguments.")

        if len(args) == 2:
            arg_a, arg_b = args
            if _is_regime_params(arg_a) and _is_square_numeric_matrix(arg_b):
                regime_params, transition_matrix = arg_a, arg_b
            elif _is_square_numeric_matrix(arg_a) and _is_regime_params(arg_b):
                transition_matrix, regime_params = arg_a, arg_b
            else:
                raise ValueError(
                    "RegimeSwitchingModel expects (transition_matrix, regime_params) "
                    "or (regime_params, transition_matrix)."
                )
        elif len(args) == 1:
            (arg,) = args
            if _is_regime_params(arg):
                regime_params = arg
            elif _is_square_numeric_matrix(arg):
                transition_matrix = arg
            else:
                raise ValueError(
                    "RegimeSwitchingModel expects a transition matrix or regime parameters."
                )

        if transition_matrix is None or regime_params is None:
            raise ValueError(
                "RegimeSwitchingModel requires both transition_matrix and regime_params."
            )
        
        self.regime_params = regime_params
        self.n_regimes = len(regime_params)
        self.dt = dt
        
        # Initialize Markov chain
        self.markov_chain = MarkovChain(self.n_regimes, transition_matrix, dt)
        
        logger.info(f"Initialized regime-switching model with {self.n_regimes} regimes")
    
    def get_regime_parameters(self, regime: int) -> RegimeParameters:
        """Get parameters for a specific regime."""
        return self.regime_params[regime]
    
    def get_drift(self, regime: int) -> float:
        """Get drift for a specific regime."""
        return self.regime_params[regime].drift
    
    def get_volatility(self, regime: int) -> float:
        """Get volatility for a specific regime."""
        return self.regime_params[regime].volatility
    
    def simulate_regimes(
        self,
        n_paths: int,
        n_steps: int,
        initial_regime: int = 0,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Simulate regime paths.
        
        Parameters
        ----------
        n_paths : int
            Number of paths
        n_steps : int
            Number of time steps
        initial_regime : int
            Starting regime
        seed : int, optional
            Random seed
            
        Returns
        -------
        np.ndarray
            Regime paths, shape (n_paths, n_steps+1)
        """
        return self.markov_chain.simulate_multiple_paths(
            n_paths, n_steps, initial_regime, seed
        )
    
    def get_stationary_distribution(self) -> np.ndarray:
        """Get stationary distribution of regimes."""
        return self.markov_chain.stationary_distribution()
    
    def summary(self) -> str:
        """Generate summary of the model."""
        stationary = self.get_stationary_distribution()
        
        summary_lines = [
            f"Regime-Switching Model with {self.n_regimes} Regimes",
            "=" * 60,
            "\nRegime Parameters:",
        ]
        
        for i, params in enumerate(self.regime_params):
            summary_lines.extend([
                f"\nRegime {i}: {params.name}",
                f"  Drift: {params.drift:.4f}",
                f"  Volatility: {params.volatility:.4f}",
                f"  Stationary Probability: {stationary[i]:.4f}",
            ])
        
        summary_lines.extend([
            f"\nTransition Matrix:",
            str(self.markov_chain.Q),
        ])
        
        return "\n".join(summary_lines)


def create_model_from_config(config: dict) -> RegimeSwitchingModel:
    """
    Create a RegimeSwitchingModel from a configuration dictionary.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary with regime and transition matrix specs
        
    Returns
    -------
    RegimeSwitchingModel
        Configured model instance
    """
    n_regimes = config['regimes']['n_regimes']
    transition_matrix = np.array(config['transition_matrix'])
    
    regime_params = []
    for i in range(1, n_regimes + 1):
        regime_key = f'regime_{i}'
        regime_config = config['regimes'][regime_key]
        
        params = RegimeParameters(
            regime_id=i - 1,
            name=regime_config['name'],
            drift=regime_config['drift'],
            volatility=regime_config['volatility'],
            mean_reversion=regime_config.get('mean_reversion', 0.0),
            vol_of_vol=regime_config.get('vol_of_vol', 0.0),
            long_term_var=regime_config.get('long_term_var'),
            correlation=regime_config.get('correlation', 0.0),
        )
        regime_params.append(params)
    
    dt = 1.0 / config['simulation']['n_steps']
    
    return RegimeSwitchingModel(
        transition_matrix=transition_matrix,
        regime_params=regime_params,
        dt=dt,
    )
