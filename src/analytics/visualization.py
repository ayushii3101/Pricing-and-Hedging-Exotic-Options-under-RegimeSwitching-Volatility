"""
Visualization Tools
===================

Comprehensive visualization for results and analytics.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List, Dict
import logging

logger = logging.getLogger(__name__)

# Set style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class ResultsVisualizer:
    """
    Visualization tools for pricing and hedging results.
    
    Parameters
    ----------
    output_dir : str
        Directory to save plots
    dpi : int
        Plot resolution
    """
    
    def __init__(self, output_dir: str = 'results', dpi: int = 300):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        
        logger.info(f"Initialized visualizer with output dir: {output_dir}")
    
    def plot_price_paths(
        self,
        prices: np.ndarray,
        regimes: np.ndarray,
        n_paths_to_plot: int = 10,
        title: str = "Simulated Asset Price Paths",
        filename: Optional[str] = None
    ):
        """
        Plot simulated asset price paths with regime coloring.
        
        Parameters
        ----------
        prices : np.ndarray
            Price paths, shape (n_paths, n_steps+1)
        regimes : np.ndarray
            Regime paths, shape (n_paths, n_steps+1)
        n_paths_to_plot : int
            Number of paths to display
        title : str
            Plot title
        filename : str, optional
            Output filename
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        n_steps = prices.shape[1] - 1
        time_grid = np.linspace(0, 1, n_steps + 1)
        
        # Plot prices
        for i in range(min(n_paths_to_plot, len(prices))):
            ax1.plot(time_grid, prices[i, :], alpha=0.6, linewidth=1)
        
        ax1.set_xlabel('Time (years)')
        ax1.set_ylabel('Asset Price')
        ax1.set_title(title)
        ax1.grid(True, alpha=0.3)
        
        # Plot regimes for first path
        colors = ['green', 'orange', 'red']
        for i in range(min(3, len(regimes))):
            regime_path = regimes[i, :]
            ax2.plot(time_grid, regime_path, alpha=0.7, color=colors[i % 3], 
                    marker='o', markersize=2, label=f'Path {i+1}')
        
        ax2.set_xlabel('Time (years)')
        ax2.set_ylabel('Regime')
        ax2.set_title('Regime Evolution')
        ax2.set_yticks(range(regimes.max() + 1))
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(self.output_dir / filename, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved plot to {filename}")
        plt.show()
    
    def plot_payoff_distribution(
        self,
        payoffs: np.ndarray,
        option_name: str = "Option",
        filename: Optional[str] = None
    ):
        """
        Plot distribution of option payoffs.
        
        Parameters
        ----------
        payoffs : np.ndarray
            Array of payoffs
        option_name : str
            Option name for title
        filename : str, optional
            Output filename
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Histogram
        ax1.hist(payoffs, bins=50, alpha=0.7, edgecolor='black', density=True)
        ax1.axvline(np.mean(payoffs), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(payoffs):.2f}')
        ax1.axvline(np.median(payoffs), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(payoffs):.2f}')
        ax1.set_xlabel('Payoff')
        ax1.set_ylabel('Density')
        ax1.set_title(f'{option_name} Payoff Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Q-Q plot
        stats_probplot = np.sort(payoffs)
        theoretical_quantiles = np.linspace(0, 1, len(payoffs))
        ax2.scatter(theoretical_quantiles, stats_probplot, alpha=0.5)
        ax2.set_xlabel('Theoretical Quantiles')
        ax2.set_ylabel('Sample Quantiles')
        ax2.set_title('Q-Q Plot')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(self.output_dir / filename, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved plot to {filename}")
        plt.show()
    
    def plot_greeks(
        self,
        spot_range: np.ndarray,
        greeks_dict: Dict[str, np.ndarray],
        filename: Optional[str] = None
    ):
        """
        Plot option Greeks as functions of spot price.
        
        Parameters
        ----------
        spot_range : np.ndarray
            Range of spot prices
        greeks_dict : Dict[str, np.ndarray]
            Dictionary of Greeks arrays
        filename : str, optional
            Output filename
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        greek_names = ['delta', 'gamma', 'vega', 'theta', 'rho']
        
        for idx, greek in enumerate(greek_names):
            if greek in greeks_dict:
                axes[idx].plot(spot_range, greeks_dict[greek], linewidth=2)
                axes[idx].set_xlabel('Spot Price')
                axes[idx].set_ylabel(greek.capitalize())
                axes[idx].set_title(f'{greek.capitalize()} Profile')
                axes[idx].grid(True, alpha=0.3)
                axes[idx].axhline(0, color='black', linestyle='-', linewidth=0.5)
        
        # Remove unused subplot
        fig.delaxes(axes[5])
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(self.output_dir / filename, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved plot to {filename}")
        plt.show()
    
    def plot_convergence(
        self,
        path_sizes: List[int],
        prices: List[float],
        std_errors: List[float],
        filename: Optional[str] = None
    ):
        """
        Plot Monte Carlo convergence.
        
        Parameters
        ----------
        path_sizes : List[int]
            Number of paths
        prices : List[float]
            Estimated prices
        std_errors : List[float]
            Standard errors
        filename : str, optional
            Output filename
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Price convergence
        ax1.plot(path_sizes, prices, marker='o', linewidth=2)
        ax1.axhline(prices[-1], color='red', linestyle='--', label='Final estimate')
        ax1.fill_between(path_sizes, 
                         np.array(prices) - 1.96*np.array(std_errors),
                         np.array(prices) + 1.96*np.array(std_errors),
                         alpha=0.3, label='95% CI')
        ax1.set_xlabel('Number of Paths')
        ax1.set_ylabel('Option Price')
        ax1.set_title('Price Convergence')
        ax1.set_xscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Error convergence
        ax2.loglog(path_sizes, std_errors, marker='o', linewidth=2, label='Std Error')
        ax2.loglog(path_sizes, np.array(std_errors[0]) * np.sqrt(path_sizes[0] / np.array(path_sizes)),
                  linestyle='--', label='Theoretical O(1/√N)')
        ax2.set_xlabel('Number of Paths')
        ax2.set_ylabel('Standard Error')
        ax2.set_title('Error Convergence')
        ax2.legend()
        ax2.grid(True, alpha=0.3, which='both')
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(self.output_dir / filename, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved plot to {filename}")
        plt.show()
    
    def plot_hedging_performance(
        self,
        time_grid: np.ndarray,
        portfolio_values: np.ndarray,
        price_path: np.ndarray,
        filename: Optional[str] = None
    ):
        """
        Plot hedging portfolio performance.
        
        Parameters
        ----------
        time_grid : np.ndarray
            Time points
        portfolio_values : np.ndarray
            Portfolio values over time
        price_path : np.ndarray
            Underlying price path
        filename : str, optional
            Output filename
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Portfolio value
        ax1.plot(time_grid, portfolio_values, linewidth=2, label='Portfolio Value')
        ax1.set_xlabel('Time (years)')
        ax1.set_ylabel('Value')
        ax1.set_title('Hedging Portfolio Value Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Underlying price
        time_grid_price = np.linspace(time_grid[0], time_grid[-1], len(price_path))
        ax2.plot(time_grid_price, price_path, linewidth=2, color='orange', label='Spot Price')
        ax2.set_xlabel('Time (years)')
        ax2.set_ylabel('Price')
        ax2.set_title('Underlying Asset Price')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(self.output_dir / filename, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved plot to {filename}")
        plt.show()
    
    def plot_regime_statistics(
        self,
        regime_stats: Dict,
        filename: Optional[str] = None
    ):
        """
        Plot regime transition statistics.
        
        Parameters
        ----------
        regime_stats : Dict
            Regime statistics dictionary
        filename : str, optional
            Output filename
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Time in each regime
        time_in_regime = regime_stats['time_in_regime']
        stationary = regime_stats['stationary_distribution']
        
        x = np.arange(len(time_in_regime))
        width = 0.35
        
        axes[0].bar(x - width/2, time_in_regime, width, label='Empirical', alpha=0.7)
        axes[0].bar(x + width/2, stationary, width, label='Stationary', alpha=0.7)
        axes[0].set_xlabel('Regime')
        axes[0].set_ylabel('Probability')
        axes[0].set_title('Time Spent in Each Regime')
        axes[0].set_xticks(x)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Transition statistics
        stats_data = [
            regime_stats['avg_transitions_per_path'],
            regime_stats['avg_time_between_transitions']
        ]
        stats_labels = ['Avg Transitions', 'Avg Time Between']
        
        axes[1].bar(stats_labels, stats_data, alpha=0.7, color=['skyblue', 'lightcoral'])
        axes[1].set_ylabel('Value')
        axes[1].set_title('Regime Transition Statistics')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(self.output_dir / filename, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved plot to {filename}")
        plt.show()
