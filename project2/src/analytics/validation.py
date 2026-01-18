"""
Martingale and Model Validation
================================

Validates risk-neutral measure and model consistency.
"""

import numpy as np
from typing import Dict
import logging
from scipy import stats

logger = logging.getLogger(__name__)


class MartingaleValidator:
    """
    Validates martingale property of risk-neutral asset price dynamics.
    
    Under the risk-neutral measure Q:
    E^Q[S_T * exp(-rT)] = S_0
    
    Parameters
    ----------
    asset_simulator : AssetSimulator
        Asset simulator to validate
    """
    
    def __init__(self, asset_simulator):
        self.simulator = asset_simulator
    
    def test_martingale_property(
        self,
        n_paths: int = 10000,
        T: float = 1.0,
        confidence_level: float = 0.95
    ) -> Dict:
        """
        Test if discounted asset prices are martingales.
        
        Parameters
        ----------
        n_paths : int
            Number of simulated paths
        T : float
            Time horizon
        confidence_level : float
            Confidence level for statistical test
            
        Returns
        -------
        Dict
            Test results and statistics
        """
        logger.info(f"Testing martingale property with {n_paths} paths...")
        
        # Simulate under risk-neutral measure
        result = self.simulator.validate_martingale_property(n_paths, int(T * 252), T)
        
        # Statistical test
        mean_discounted = result['mean_discounted_ST']
        std_discounted = result['std_discounted_ST']
        S0 = result['S0']
        
        # t-test: H0: E[discounted_ST] = S0
        t_statistic = (mean_discounted - S0) / (std_discounted / np.sqrt(n_paths))
        p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), n_paths - 1))
        
        # Decision
        alpha = 1 - confidence_level
        reject_null = p_value < alpha
        
        result.update({
            't_statistic': t_statistic,
            'p_value': p_value,
            'confidence_level': confidence_level,
            'reject_null': reject_null,
            'test_passed': not reject_null
        })
        
        if result['test_passed']:
            logger.info(f"✓ Martingale test PASSED (p={p_value:.4f})")
        else:
            logger.warning(f"✗ Martingale test FAILED (p={p_value:.4f})")
        
        return result
    
    def test_regime_stationarity(self, n_paths: int = 10000, n_steps: int = 252) -> Dict:
        """
        Test if regime distribution converges to stationary distribution.
        
        Parameters
        ----------
        n_paths : int
            Number of paths
        n_steps : int
            Number of time steps
            
        Returns
        -------
        Dict
            Test results
        """
        logger.info("Testing regime stationarity...")
        
        # Simulate regimes
        T = n_steps / 252
        _, regimes, _ = self.simulator.simulate_paths(
            n_paths, n_steps, T, risk_neutral=True, show_progress=False
        )
        
        # Empirical distribution at terminal time
        n_regimes = self.simulator.regime_model.n_regimes
        terminal_regimes = regimes[:, -1]
        empirical_dist = np.array([
            np.mean(terminal_regimes == i) for i in range(n_regimes)
        ])
        
        # Theoretical stationary distribution
        stationary_dist = self.simulator.regime_model.get_stationary_distribution()
        
        # Chi-square test
        chi2_stat = np.sum((empirical_dist * n_paths - stationary_dist * n_paths)**2 / 
                           (stationary_dist * n_paths + 1e-10))
        p_value = 1 - stats.chi2.cdf(chi2_stat, n_regimes - 1)
        
        return {
            'empirical_distribution': empirical_dist,
            'stationary_distribution': stationary_dist,
            'chi2_statistic': chi2_stat,
            'p_value': p_value,
            'test_passed': p_value > 0.05,
            'max_difference': np.max(np.abs(empirical_dist - stationary_dist))
        }
    
    def convergence_test(
        self,
        option,
        path_counts: list = [1000, 5000, 10000, 50000],
        initial_regime: int = 0
    ) -> Dict:
        """
        Test Monte Carlo convergence rate.
        
        Parameters
        ----------
        option : ExoticOption
            Option to price
        path_counts : list
            List of path counts to test
        initial_regime : int
            Starting regime
            
        Returns
        -------
        Dict
            Convergence test results
        """
        from ..pricing.monte_carlo import MonteCarloEngine
        
        logger.info("Testing Monte Carlo convergence...")
        
        prices = []
        std_errors = []
        
        for n_paths in path_counts:
            engine = MonteCarloEngine(self.simulator, n_simulations=n_paths)
            result = engine.price_option(option, initial_regime, show_progress=False)
            prices.append(result['price'])
            std_errors.append(result['std_error'])
        
        # Check if std_error ~ 1/sqrt(N)
        log_n = np.log(path_counts)
        log_std = np.log(std_errors)
        
        # Linear regression
        slope, intercept = np.polyfit(log_n, log_std, 1)
        
        # Theoretical slope should be -0.5
        convergence_rate = -slope
        
        return {
            'path_counts': path_counts,
            'prices': prices,
            'std_errors': std_errors,
            'convergence_rate': convergence_rate,
            'expected_rate': 0.5,
            'rate_error': abs(convergence_rate - 0.5)
        }
