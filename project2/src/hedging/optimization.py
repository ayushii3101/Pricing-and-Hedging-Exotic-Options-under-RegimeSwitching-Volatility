"""
Mean-Variance Optimal Hedging
=============================

Implements mean-variance optimization for hedging portfolios.
"""

import numpy as np
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


class MeanVarianceHedger:
    """
    Mean-variance optimal hedging strategy.
    
    Minimizes: Var[Π_T - H_T]
    Subject to: E[Π_T - H_T] = 0
    
    Where Π_T is hedged portfolio value and H_T is target option payoff.
    
    Parameters
    ----------
    asset_simulator : AssetSimulator
        Asset simulator for scenario generation
    risk_aversion : float
        Risk aversion parameter λ
    """
    
    def __init__(self, asset_simulator, risk_aversion: float = 0.5):
        self.simulator = asset_simulator
        self.asset_simulator = asset_simulator
        self.risk_aversion = risk_aversion
        
        logger.info(f"Initialized mean-variance hedger with lambda={risk_aversion}")

    
    def compute_optimal_weights(
        self,
        option,
        hedging_instruments: Optional[List] = None,
        n_scenarios: int = 10000,
        initial_regime: int = 0,
        n_paths: Optional[int] = None
    ):
        """
        Compute optimal hedge weights using quadratic programming.
        
        Parameters
        ----------
        option : ExoticOption
            Target option to hedge
        hedging_instruments : List[HedgingInstrument]
            Available hedging instruments
        n_scenarios : int
            Number of scenarios for optimization
        initial_regime : int
            Starting regime
            
        Returns
        -------
        np.ndarray
            Optimal weights for hedging instruments
        """
        logger.info("Computing optimal hedge weights...")

        if n_paths is not None:
            n_scenarios = n_paths

        # Support passing a HedgingPortfolio directly.
        if hedging_instruments is None and hasattr(option, "instruments") and hasattr(option, "target_option"):
            portfolio = option
            option = portfolio.target_option
            hedging_instruments = portfolio.instruments
            return_as_dict = True
        else:
            return_as_dict = False

        if not hedging_instruments:
            raise ValueError("hedging_instruments must be provided and non-empty.")
        
        # Generate scenarios
        T = option.maturity
        n_steps = max(int(T * 252), 1)
        
        prices, regimes, _ = self.simulator.simulate_paths(
            n_scenarios,
            n_steps,
            T,
            initial_regime=initial_regime,
            risk_neutral=True,
            show_progress=False
        )
        
        # Compute option payoffs
        payoffs = np.array([option.payoff(prices[i, :]) for i in range(n_scenarios)])
        
        # Compute hedging instrument payoffs
        n_instruments = len(hedging_instruments)
        instrument_payoffs = np.zeros((n_scenarios, n_instruments))
        
        for i in range(n_scenarios):
            terminal_spot = prices[i, -1]
            for j, instrument in enumerate(hedging_instruments):
                instrument_payoffs[i, j] = self._compute_instrument_terminal_value(
                    instrument, terminal_spot
                )
        
        # Solve quadratic program
        # min w^T Σ w - λ μ^T w
        # where Σ is covariance matrix, μ is mean vector
        
        # Mean vector (expected payoff per unit)
        mu = np.mean(instrument_payoffs, axis=0)
        
        # Covariance matrix
        Sigma = np.cov(instrument_payoffs.T)
        
        # Target: match option payoff
        target_mean = np.mean(payoffs)
        
        # Solve using simple approach (can use cvxpy for constraints)
        try:
            # Add small regularization for numerical stability
            reg = 1e-6 * np.eye(n_instruments)
            weights = np.linalg.solve(Sigma + reg, mu) * (target_mean / (mu @ mu + 1e-10))
        except np.linalg.LinAlgError:
            logger.warning("Failed to solve for optimal weights, using equal weights")
            weights = np.ones(n_instruments) / n_instruments
        
        logger.info(f"Optimal weights computed: {weights}")
        
        if return_as_dict:
            return {instrument.name: weights[i] for i, instrument in enumerate(hedging_instruments)}

        return weights
    
    def _compute_instrument_terminal_value(self, instrument, terminal_spot: float) -> float:
        """Compute terminal value of a hedging instrument."""
        from .portfolio import Stock, VanillaHedge
        
        if isinstance(instrument, Stock):
            return terminal_spot
        elif isinstance(instrument, VanillaHedge):
            if instrument.option_type == 'call':
                return max(terminal_spot - instrument.strike, 0)
            else:
                return max(instrument.strike - terminal_spot, 0)
        else:
            return 0.0
    
    def dynamic_hedge(
        self,
        option,
        hedging_portfolio,
        n_rebalances: int = 50,
        n_scenarios: int = 1000,
        transaction_cost: float = 0.001
    ) -> Dict:
        """
        Execute dynamic hedging strategy with rebalancing.
        
        Parameters
        ----------
        option : ExoticOption
            Target option
        hedging_portfolio : HedgingPortfolio
            Portfolio to manage
        n_rebalances : int
            Number of rebalancing points
        n_scenarios : int
            Scenarios per rebalancing
        transaction_cost : float
            Transaction cost rate
            
        Returns
        -------
        Dict
            Hedging performance results
        """
        logger.info(f"Executing dynamic hedge with {n_rebalances} rebalances...")
        
        T = option.maturity
        dt = T / n_rebalances
        
        # Simulate one main path
        n_steps = n_rebalances
        prices, regimes, _ = self.simulator.simulate_paths(
            1, n_steps, T, risk_neutral=False, show_progress=False
        )
        main_path = prices[0, :]
        
        # Tracking
        portfolio_values = []
        hedge_errors = []
        transaction_costs = []
        
        spot = main_path[0]
        
        for t_idx in range(n_rebalances):
            spot = main_path[t_idx]
            
            # Compute Greeks at current point
            # Simplified: use Monte Carlo from current spot
            from ..pricing.monte_carlo import MonteCarloEngine
            temp_engine = MonteCarloEngine(self.simulator, n_simulations=500)
            greeks = temp_engine.price_with_greeks(option)
            
            # Rebalance portfolio
            new_ratios = hedging_portfolio.rebalance(spot, greeks, transaction_cost)
            hedging_portfolio.update_positions(new_ratios, spot)
            
            # Track portfolio value
            portfolio_value = hedging_portfolio.value(spot, volatility=0.25, risk_free_rate=self.simulator.r)
            portfolio_values.append(portfolio_value)
        
        # Terminal hedge error
        terminal_spot = main_path[-1]
        terminal_payoff = option.payoff(main_path)
        terminal_portfolio = hedging_portfolio.value(
            terminal_spot, volatility=0.25, risk_free_rate=self.simulator.r
        )
        
        hedge_error_stats = hedging_portfolio.compute_hedging_error(
            terminal_payoff, terminal_portfolio
        )
        
        return {
            'portfolio_values': portfolio_values,
            'price_path': main_path,
            'terminal_hedge_error': hedge_error_stats,
            'n_rebalances': n_rebalances,
            'avg_portfolio_value': np.mean(portfolio_values)
        }
    
    def compute_hedging_pnl(
        self,
        option,
        n_paths: int = 1000,
        n_rebalances: int = 50
    ) -> Dict:
        """
        Compute P&L distribution from hedging strategy.
        
        Parameters
        ----------
        option : ExoticOption
            Target option
        n_paths : int
            Number of paths to simulate
        n_rebalances : int
            Rebalancing frequency
            
        Returns
        -------
        Dict
            P&L statistics
        """
        logger.info(f"Computing hedging P&L over {n_paths} paths...")
        
        pnl_values = []
        
        T = option.maturity
        n_steps = n_rebalances
        
        prices, _, _ = self.simulator.simulate_paths(
            n_paths, n_steps, T, risk_neutral=False, show_progress=False
        )
        
        for path_idx in range(n_paths):
            path = prices[path_idx, :]
            
            # Simple delta hedge P&L
            payoff = option.payoff(path)
            
            # Approximate hedged P&L (simplified)
            delta_hedge_cost = np.sum(np.diff(path))  # Simplified
            pnl = payoff - delta_hedge_cost
            pnl_values.append(pnl)
        
        pnl_array = np.array(pnl_values)
        
        return {
            'mean_pnl': np.mean(pnl_array),
            'std_pnl': np.std(pnl_array),
            'min_pnl': np.min(pnl_array),
            'max_pnl': np.max(pnl_array),
            'sharpe_ratio': np.mean(pnl_array) / (np.std(pnl_array) + 1e-10),
            'var_95': np.percentile(pnl_array, 5),
            'cvar_95': np.mean(pnl_array[pnl_array <= np.percentile(pnl_array, 5)]),
            'pnl_values': pnl_array
        }
