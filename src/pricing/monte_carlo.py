"""
Monte Carlo Pricing Engine
===========================

Monte Carlo simulation-based pricing for exotic options under regime-switching.
"""

import numpy as np
from typing import Optional, Dict, List, Tuple
import logging
try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal installs
    def tqdm(iterable, **kwargs):
        return iterable

from ..models.asset_dynamics import AssetSimulator
from .exotic_options import ExoticOption

logger = logging.getLogger(__name__)


class MonteCarloEngine:
    """
    Monte Carlo engine for pricing exotic options.
    
    Parameters
    ----------
    asset_simulator : AssetSimulator
        Asset simulator with regime-switching dynamics
    n_simulations : int
        Number of Monte Carlo paths
    seed : int, optional
        Random seed for reproducibility
    """
    
    def __init__(
        self,
        asset_simulator: AssetSimulator,
        n_simulations: int = 10000,
        seed: Optional[int] = None,
        n_paths: Optional[int] = None
    ):
        if n_paths is not None:
            if n_simulations != 10000 and n_simulations != n_paths:
                raise ValueError("Provide either n_simulations or n_paths, not conflicting values.")
            n_simulations = n_paths

        self.simulator = asset_simulator
        self.n_simulations = n_simulations
        self.seed = seed
        
        logger.info(f"Initialized Monte Carlo engine with {n_simulations} simulations")
    
    def price_option(
        self,
        option: ExoticOption,
        initial_regime: int = 0,
        antithetic: bool = True,
        control_variate: bool = False,
        show_progress: bool = True
    ) -> Dict:
        """
        Price an exotic option using Monte Carlo simulation.
        
        Parameters
        ----------
        option : ExoticOption
            Option to price
        initial_regime : int
            Starting regime
        antithetic : bool
            Use antithetic variates for variance reduction
        control_variate : bool
            Use control variate technique
        show_progress : bool
            Show progress bar
            
        Returns
        -------
        dict
            Pricing results including price, standard error, and confidence interval
        """
        logger.info(f"Pricing {option.name()} using Monte Carlo...")
        
        # Determine number of time steps (at least daily for path-dependent options)
        n_steps = max(int(option.maturity * 252), 1)
        
        # Simulate asset paths
        prices, regimes, variances = self.simulator.simulate_paths(
            self.n_simulations,
            n_steps,
            option.maturity,
            initial_regime=initial_regime,
            seed=self.seed,
            risk_neutral=True,
            antithetic=antithetic,
            show_progress=show_progress
        )
        
        # Compute payoffs for each path
        payoffs = np.array([
            option.payoff(prices[i, :])
            for i in (tqdm(range(len(prices)), desc="Computing payoffs") if show_progress else range(len(prices)))
        ])
        
        # Discount payoffs
        discount_factor = np.exp(-self.simulator.r * option.maturity)
        discounted_payoffs = payoffs * discount_factor
        
        # Compute price and statistics
        price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs) / np.sqrt(self.n_simulations)
        
        # 95% confidence interval
        ci_lower = price - 1.96 * std_error
        ci_upper = price + 1.96 * std_error
        
        results = {
            'price': price,
            'std_error': std_error,
            'ci_95_lower': ci_lower,
            'ci_95_upper': ci_upper,
            'min_payoff': np.min(payoffs),
            'max_payoff': np.max(payoffs),
            'mean_payoff': np.mean(payoffs),
            'std_payoff': np.std(payoffs),
            'n_simulations': self.n_simulations,
            'option': str(option)
        }
        
        logger.info(f"Option price: {price:.4f} ± {std_error:.4f}")
        logger.info(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        
        return results
    
    def price_with_greeks(
        self,
        option: ExoticOption,
        initial_regime: int = 0,
        bump_size: float = 0.01
    ) -> Dict:
        """
        Price option and compute Greeks using finite differences.
        
        Parameters
        ----------
        option : ExoticOption
            Option to price
        initial_regime : int
            Starting regime
        bump_size : float
            Bump size for finite difference
            
        Returns
        -------
        dict
            Pricing results with Greeks
        """
        # Base price
        base_results = self.price_option(option, initial_regime, show_progress=False)
        base_price = base_results['price']
        
        # Delta: ∂V/∂S
        S0_original = self.simulator.S0
        self.simulator.S0 = S0_original * (1 + bump_size)
        price_up = self.price_option(option, initial_regime, show_progress=False)['price']
        
        self.simulator.S0 = S0_original * (1 - bump_size)
        price_down = self.price_option(option, initial_regime, show_progress=False)['price']
        
        delta = (price_up - price_down) / (2 * S0_original * bump_size)
        gamma = (price_up - 2 * base_price + price_down) / ((S0_original * bump_size) ** 2)
        
        # Restore original spot
        self.simulator.S0 = S0_original
        
        # Vega: ∂V/∂σ (bump volatility in all regimes)
        vol_bumped_results = self._price_with_vol_bump(option, initial_regime, bump_size)
        vega = (vol_bumped_results - base_price) / bump_size
        
        # Theta: ∂V/∂T (using finite difference in time)
        T_original = option.maturity
        option.maturity = T_original - 1.0 / 252  # One day forward
        price_forward = self.price_option(option, initial_regime, show_progress=False)['price']
        theta = (price_forward - base_price) / (1.0 / 252)
        option.maturity = T_original  # Restore
        
        # Rho: ∂V/∂r
        r_original = self.simulator.r
        self.simulator.r = r_original + bump_size
        price_r_up = self.price_option(option, initial_regime, show_progress=False)['price']
        rho = (price_r_up - base_price) / bump_size
        self.simulator.r = r_original  # Restore
        
        results = base_results.copy()
        results.update({
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta,
            'rho': rho
        })
        
        logger.info(f"Greeks computed: Delta={delta:.4f}, Gamma={gamma:.4f}, Vega={vega:.4f}")
        
        return results
    
    def _price_with_vol_bump(
        self,
        option: ExoticOption,
        initial_regime: int,
        bump_size: float
    ) -> float:
        """Helper to price with bumped volatility."""
        # Save original volatilities
        original_vols = []
        for params in self.simulator.regime_model.regime_params:
            original_vols.append(params.volatility)
            params.volatility *= (1 + bump_size)
        
        # Recreate volatility models
        self.simulator.vol_models = self.simulator._create_volatility_models()
        
        # Price with bumped volatility
        price = self.price_option(option, initial_regime, show_progress=False)['price']
        
        # Restore original volatilities
        for i, params in enumerate(self.simulator.regime_model.regime_params):
            params.volatility = original_vols[i]
        self.simulator.vol_models = self.simulator._create_volatility_models()
        
        return price
    
    def convergence_analysis(
        self,
        option: ExoticOption,
        path_sizes: List[int],
        initial_regime: int = 0
    ) -> Dict:
        """
        Analyze convergence of Monte Carlo estimate with increasing paths.
        
        Parameters
        ----------
        option : ExoticOption
            Option to price
        path_sizes : List[int]
            List of path counts to test
        initial_regime : int
            Starting regime
            
        Returns
        -------
        dict
            Convergence analysis results
        """
        logger.info("Running convergence analysis...")
        
        prices = []
        std_errors = []
        
        original_n_sims = self.n_simulations
        
        for n_paths in tqdm(path_sizes, desc="Convergence test"):
            self.n_simulations = n_paths
            result = self.price_option(option, initial_regime, show_progress=False)
            prices.append(result['price'])
            std_errors.append(result['std_error'])
        
        # Restore original
        self.n_simulations = original_n_sims
        
        return {
            'path_sizes': path_sizes,
            'prices': prices,
            'std_errors': std_errors,
            'final_price': prices[-1],
            'final_std_error': std_errors[-1]
        }
    
    def price_portfolio(
        self,
        options: List[ExoticOption],
        weights: np.ndarray,
        initial_regime: int = 0
    ) -> Dict:
        """
        Price a portfolio of options.
        
        Parameters
        ----------
        options : List[ExoticOption]
            List of options in portfolio
        weights : np.ndarray
            Portfolio weights
        initial_regime : int
            Starting regime
            
        Returns
        -------
        dict
            Portfolio pricing results
        """
        if len(options) != len(weights):
            raise ValueError("Number of options must match number of weights")
        
        portfolio_prices = []
        for opt, weight in zip(options, weights):
            result = self.price_option(opt, initial_regime, show_progress=False)
            portfolio_prices.append(weight * result['price'])
        
        total_price = np.sum(portfolio_prices)
        
        return {
            'portfolio_price': total_price,
            'individual_prices': portfolio_prices,
            'weights': weights.tolist(),
            'options': [str(opt) for opt in options]
        }
    
    def compare_with_black_scholes(
        self,
        option: ExoticOption,
        initial_regime: int = 0
    ) -> Dict:
        """
        Compare regime-switching price with Black-Scholes benchmark.
        
        Parameters
        ----------
        option : ExoticOption
            Option to price (must be vanilla for BS comparison)
        initial_regime : int
            Starting regime
            
        Returns
        -------
        dict
            Comparison results
        """
        from .exotic_options import VanillaOption
        
        if not isinstance(option, VanillaOption):
            logger.warning("Black-Scholes comparison only valid for vanilla options")
        
        # Regime-switching price
        rs_result = self.price_option(option, initial_regime, show_progress=False)
        rs_price = rs_result['price']
        
        # Black-Scholes price (using first regime's volatility)
        regime_vol = self.simulator.regime_model.get_volatility(initial_regime)
        
        if isinstance(option, VanillaOption):
            bs_price = option.black_scholes_price(
                self.simulator.S0,
                regime_vol,
                self.simulator.r,
                self.simulator.q
            )
        else:
            bs_price = None
        
        return {
            'regime_switching_price': rs_price,
            'black_scholes_price': bs_price,
            'difference': rs_price - bs_price if bs_price else None,
            'relative_difference': (rs_price - bs_price) / bs_price if bs_price else None,
            'std_error': rs_result['std_error']
        }
