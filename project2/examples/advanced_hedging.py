"""
Advanced Hedging Example
========================

Demonstrates dynamic hedging strategies with mean-variance optimization.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.regime_switching import create_model_from_config
from src.models.asset_dynamics import AssetSimulator
from src.pricing.exotic_options import BarrierOption
from src.pricing.monte_carlo import MonteCarloEngine
from src.hedging.portfolio import HedgingPortfolio, Stock, VanillaHedge
from src.hedging.optimization import MeanVarianceHedger
from src.hedging.greeks import GreeksCalculator
from src.analytics.visualization import ResultsVisualizer
from src.utils.data_utils import load_config, setup_logging

def main():
    """Run advanced hedging example."""
    
    setup_logging(level=20)
    
    print("\n" + "="*70)
    print("ADVANCED HEDGING WITH MEAN-VARIANCE OPTIMIZATION")
    print("="*70 + "\n")
    
    # Load configuration
    config_path = Path(__file__).parent.parent / "config" / "model_config.yaml"
    config = load_config(str(config_path))
    
    # Create model and simulator
    print("Setting up regime-switching model...")
    regime_model = create_model_from_config(config)
    
    market_config = config['market']
    simulator = AssetSimulator(
        regime_model,
        spot_price=market_config['spot_price'],
        risk_free_rate=market_config['risk_free_rate'],
        dividend_yield=market_config['dividend_yield']
    )
    
    # Create option to hedge
    barrier_config = config['options']['barrier']
    target_option = BarrierOption(
        strike=barrier_config['strike'],
        barrier=barrier_config['barrier'],
        maturity=barrier_config['maturity'],
        option_type=barrier_config['option_type'],
        barrier_type=barrier_config['type'],
        rebate=barrier_config['rebate']
    )
    
    print(f"\nTarget Option: {target_option}")
    
    # Create Monte Carlo engine
    mc_engine = MonteCarloEngine(simulator, n_simulations=10000)
    
    # Price the option
    print("\nPricing target option...")
    option_price = mc_engine.price_option(target_option, show_progress=True)
    print(f"Option Price: ${option_price['price']:.4f}")
    
    # Calculate Greeks
    print("\nCalculating Greeks...")
    greeks_calc = GreeksCalculator(mc_engine)
    greeks = greeks_calc.calculate_all(target_option)
    
    print(f"Delta: {greeks['delta']:.4f}")
    print(f"Gamma: {greeks['gamma']:.4f}")
    print(f"Vega:  {greeks['vega']:.4f}")
    
    # Create hedging portfolio
    print("\n" + "-"*70)
    print("CONSTRUCTING HEDGING PORTFOLIO")
    print("-"*70)
    
    hedging_portfolio = HedgingPortfolio(target_option)
    
    # Add hedging instruments
    stock = Stock(quantity=0.0)
    hedging_portfolio.add_instrument(stock)
    
    vanilla_call = VanillaHedge(
        strike=barrier_config['strike'],
        maturity=barrier_config['maturity'],
        option_type='call',
        quantity=0.0
    )
    hedging_portfolio.add_instrument(vanilla_call)
    
    print(hedging_portfolio.summary())
    
    # Initialize mean-variance hedger
    print("\n" + "-"*70)
    print("MEAN-VARIANCE OPTIMAL HEDGING")
    print("-"*70)
    
    hedger = MeanVarianceHedger(simulator, risk_aversion=0.5)
    
    # Compute optimal weights
    print("\nComputing optimal hedge weights...")
    optimal_weights = hedger.compute_optimal_weights(
        target_option,
        hedging_portfolio.instruments,
        n_scenarios=5000
    )
    
    print(f"\nOptimal Weights:")
    for i, instrument in enumerate(hedging_portfolio.instruments):
        print(f"  {instrument.name}: {optimal_weights[i]:.4f}")
    
    # Execute dynamic hedging
    print("\n" + "-"*70)
    print("DYNAMIC HEDGING SIMULATION")
    print("-"*70)
    
    print("\nExecuting dynamic hedge...")
    hedging_results = hedger.dynamic_hedge(
        target_option,
        hedging_portfolio,
        n_rebalances=50,
        n_scenarios=1000,
        transaction_cost=0.001
    )
    
    print(f"\nHedging Results:")
    print(f"  Terminal Hedge Error: ${hedging_results['terminal_hedge_error']['absolute_error']:.4f}")
    print(f"  Relative Error:       {hedging_results['terminal_hedge_error']['relative_error']*100:.2f}%")
    print(f"  RMSE:                ${hedging_results['terminal_hedge_error']['rmse']:.4f}")
    
    # Compute P&L distribution
    print("\n" + "-"*70)
    print("HEDGING P&L ANALYSIS")
    print("-"*70)
    
    print("\nComputing P&L distribution...")
    pnl_results = hedger.compute_hedging_pnl(
        target_option,
        n_paths=1000,
        n_rebalances=50
    )
    
    print(f"\nP&L Statistics:")
    print(f"  Mean P&L:     ${pnl_results['mean_pnl']:.4f}")
    print(f"  Std P&L:      ${pnl_results['std_pnl']:.4f}")
    print(f"  Sharpe Ratio:  {pnl_results['sharpe_ratio']:.4f}")
    print(f"  VaR (95%):    ${pnl_results['var_95']:.4f}")
    print(f"  CVaR (95%):   ${pnl_results['cvar_95']:.4f}")
    
    # Visualize results
    print("\n" + "-"*70)
    print("GENERATING VISUALIZATIONS")
    print("-"*70)
    
    visualizer = ResultsVisualizer(output_dir='results')
    
    # Plot hedging performance
    time_grid = np.linspace(0, target_option.maturity, len(hedging_results['portfolio_values']))
    visualizer.plot_hedging_performance(
        time_grid,
        hedging_results['portfolio_values'],
        hedging_results['price_path'],
        filename='hedging_performance.png'
    )
    
    print("\n" + "="*70)
    print("ADVANCED HEDGING EXAMPLE COMPLETED")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
