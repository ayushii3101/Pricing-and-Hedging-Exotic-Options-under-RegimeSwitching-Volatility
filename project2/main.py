import numpy as np
import sys
from pathlib import Path
import time
import yfinance as yf  # Added for real data integration
import logging

sys.path.insert(0, str(Path(__file__).parent))

from src.models.regime_switching import create_model_from_config
from src.models.asset_dynamics import AssetSimulator
from src.pricing.exotic_options import BarrierOption, AsianOption, VanillaOption, create_option_from_config
from src.pricing.monte_carlo import MonteCarloEngine
from src.hedging.portfolio import HedgingPortfolio, Stock, Cash, VanillaHedge
from src.hedging.optimization import MeanVarianceHedger
from src.analytics.validation import MartingaleValidator
from src.analytics.visualization import ResultsVisualizer
from src.utils.data_utils import load_config, save_results, setup_logging, print_results_summary, create_output_directory

logger = logging.getLogger(__name__)

def fetch_real_market_data(ticker: str) -> float:
    """Fetch real-time spot price from Yahoo Finance."""
    print(f"\n[Data] Fetching real market data for {ticker}...")
    try:
        # Fetch 1 day of data with 1 minute interval to get the absolute latest price
        data = yf.download(ticker, period="1d", interval="1m", progress=False)
        if data.empty:
            # Fallback to daily if minute data fails
            data = yf.download(ticker, period="1d", progress=False)
            
        if data.empty:
            logger.warning(f"No data found for {ticker}. Using configuration fallback.")
            return None
        
        # Get latest Close (handle MultiIndex columns if necessary)
        if 'Close' in data.columns:
            close_data = data['Close']
            
            # --- FIX: Safe scalar extraction to prevent FutureWarning ---
            if hasattr(close_data, 'iloc'):
                # Extract the last element as a Python scalar
                real_spot = float(close_data.iloc[-1].item())
            else:
                real_spot = float(close_data.item())
            # ----------------------------------------------------------
        else:
            return None

        print(f"✓ Retrieved Live Spot Price: ${real_spot:.2f}")
        return real_spot
    except Exception as e:
        logger.error(f"Failed to fetch market data: {e}")
        return None

def main():
    """Run comprehensive analysis."""
    
    start_time = time.time()
    
    # Setup
    setup_logging(log_file='results/analysis.log', level=20)
    output_dir = create_output_directory('results', 'comprehensive_analysis')
    
    print("\n" + "="*70)
    print("REGIME-SWITCHING EXOTIC OPTIONS: COMPREHENSIVE ANALYSIS")
    print("="*70 + "\n")
    
    # Load configuration
    print("Loading configuration...")
    config_path = Path(__file__).parent / "config" / "model_config.yaml"
    config = load_config(str(config_path))
    
    # --- REAL DATA INTEGRATION ---
    # We fetch data BEFORE creating the model to update spot prices
    market_ticker = config['market'].get('ticker', 'SPY')
    real_spot = fetch_real_market_data(market_ticker)
    
    if real_spot:
        config['market']['spot_price'] = real_spot
        # Dynamically update Option parameters to match new Spot
        # e.g., Set barrier to 120% of spot, Strike to ATM
        config['options']['barrier']['strike'] = real_spot
        config['options']['barrier']['barrier'] = real_spot * 1.2
        config['options']['asian']['strike'] = real_spot
        config['options']['lookback']['strike'] = real_spot
    # -----------------------------

    # Create model
    print("\n[1/10] Creating regime-switching model...")
    regime_model = create_model_from_config(config)
    print(regime_model.summary())
    
    # Create simulator
    print("\n[2/10] Initializing asset simulator...")
    market_config = config['market']
    simulator = AssetSimulator(
        regime_model,
        spot_price=market_config['spot_price'],
        risk_free_rate=market_config['risk_free_rate'],
        dividend_yield=market_config['dividend_yield']
    )
    
    # Validate martingale property
    print("\n[3/10] Validating risk-neutral measure (martingale property)...")
    validator = MartingaleValidator(simulator)
    martingale_test = validator.test_martingale_property(n_paths=10000, T=1.0)
    
    print(f"  Mean discounted S_T: {martingale_test['mean_discounted_ST']:.4f}")
    print(f"  S_0:                 {martingale_test['S0']:.4f}")
    print(f"  Relative Error:      {martingale_test['relative_error']*100:.2f}%")
    print(f"  Test Result:         {'PASSED' if martingale_test['test_passed'] else 'FAILED'}")
    
    # Test regime stationarity
    print("\n[4/10] Testing regime stationarity...")
    regime_test = validator.test_regime_stationarity(n_paths=5000, n_steps=252)
    
    print("  Empirical vs Stationary Distribution:")
    for i in range(len(regime_test['empirical_distribution'])):
        print(f"    Regime {i}: {regime_test['empirical_distribution'][i]:.4f} vs "
              f"{regime_test['stationary_distribution'][i]:.4f}")
    
    # Create options
    print("\n[5/10] Creating exotic options...")
    barrier_option = create_option_from_config(config)
    
    # Ensure Strikes match the configured (potentially real) spot
    strike_price = config['options']['barrier']['strike']
    
    vanilla_option = VanillaOption(
        strike=strike_price,
        maturity=config['options']['barrier']['maturity'],
        option_type='call'
    )
    asian_option = AsianOption(
        strike=config['options']['asian']['strike'],
        maturity=config['options']['asian']['maturity'],
        option_type='call',
        averaging_type='arithmetic'
    )
    
    print(f"  Barrier Option: {barrier_option}")
    print(f"  Vanilla Option: {vanilla_option}")
    print(f"  Asian Option:   {asian_option}")
    
    # Create MC engine
    print("\n[6/10] Pricing options with Monte Carlo...")
    mc_engine = MonteCarloEngine(
        simulator,
        n_simulations=config['simulation']['n_paths'],
        seed=config['simulation']['seed']
    )
    
    # Price all options
    barrier_price = mc_engine.price_option(barrier_option, antithetic=True, show_progress=True)
    vanilla_price = mc_engine.price_option(vanilla_option, antithetic=True, show_progress=True)
    asian_price = mc_engine.price_option(asian_option, antithetic=True, show_progress=True)
    
    print("\n  PRICING RESULTS:")
    print(f"  Barrier Option: ${barrier_price['price']:.4f} ± ${barrier_price['std_error']:.4f}")
    print(f"  Vanilla Option: ${vanilla_price['price']:.4f} ± ${vanilla_price['std_error']:.4f}")
    print(f"  Asian Option:   ${asian_price['price']:.4f} ± ${asian_price['std_error']:.4f}")
    
    # Compare with Black-Scholes
    print("\n[7/10] Comparing with Black-Scholes benchmark...")
    bs_comparison = mc_engine.compare_with_black_scholes(vanilla_option)
    
    print(f"  Regime-Switching: ${bs_comparison['regime_switching_price']:.4f}")
    print(f"  Black-Scholes:    ${bs_comparison['black_scholes_price']:.4f}")
    print(f"  Difference:       ${bs_comparison['difference']:.4f} "
          f"({bs_comparison['relative_difference']*100:.2f}%)")
    
    # Convergence analysis
    print("\n[8/10] Analyzing Monte Carlo convergence...")
    path_sizes = [1000, 5000, 10000, 50000]
    convergence_results = mc_engine.convergence_analysis(barrier_option, path_sizes)
    
    print("  Convergence Rate Analysis:")
    for n, price, std_err in zip(convergence_results['path_sizes'],
                                   convergence_results['prices'],
                                   convergence_results['std_errors']):
        print(f"    N={n:6d}: Price=${price:.4f}, Std Err=${std_err:.4f}")
    
    # Compute Greeks
    print("\n[9/10] Computing Greeks...")
    barrier_greeks = mc_engine.price_with_greeks(barrier_option, bump_size=0.01)
    
    print("  GREEKS for Barrier Option:")
    print(f"    Delta:  {barrier_greeks['delta']:.4f}")
    print(f"    Gamma:  {barrier_greeks['gamma']:.4f}")
    print(f"    Vega:   {barrier_greeks['vega']:.4f}")
    print(f"    Theta:  {barrier_greeks['theta']:.4f}")
    print(f"    Rho:    {barrier_greeks['rho']:.4f}")
    
    # Hedging analysis
    print("\n[10/10] Performing hedging analysis...")
    hedger = MeanVarianceHedger(simulator, risk_aversion=0.5)
    
    # Create hedging portfolio
    hedge_qty = config.get('hedging', {}).get('position_size', 1.0)
    hedging_portfolio = HedgingPortfolio(barrier_option, quantity=hedge_qty)
    hedging_portfolio.add_instrument(Stock(quantity=0.0))
    initial_capital = barrier_price['price'] * hedge_qty
    hedging_portfolio.add_instrument(Cash(amount=initial_capital))
    print(f"  Initialized Portfolio with Cash: ${initial_capital:.2f}")
    # We add a vanilla option with the REAL strike price
    hedging_portfolio.add_instrument(VanillaHedge(strike_price, 1.0, 'call', quantity=0.0))
    
    # Dynamic hedging
    hedging_results = hedger.dynamic_hedge(
        barrier_option,
        hedging_portfolio,
        n_rebalances=104,
        n_scenarios=2500
    )
    
    print("  HEDGING RESULTS:")
    print(f"    Position Size:     {hedge_qty:.0f} option(s)")
    print(f"    Terminal Error:    ${hedging_results['terminal_hedge_error']['absolute_error']:.4f}")
    if hedge_qty > 0:
        print(f"    Error per Option:  ${hedging_results['terminal_hedge_error']['absolute_error'] / hedge_qty:.4f}")
    print(f"    Relative Error:    {hedging_results['terminal_hedge_error']['relative_error']*100:.2f}%")
    print(f"    RMSE:             ${hedging_results['terminal_hedge_error']['rmse']:.4f}")
    
    # Visualizations
    print("\nGenerating visualizations...")
    visualizer = ResultsVisualizer(output_dir=str(output_dir))
    
    # Simulate paths for visualization
    prices, regimes, variances = simulator.simulate_paths(
        100, 252, 1.0, seed=42, show_progress=False
    )
    
    # Plot price paths
    visualizer.plot_price_paths(
        prices, regimes, n_paths_to_plot=10,
        filename='asset_price_paths.png'
    )
    
    # Plot convergence
    visualizer.plot_convergence(
        convergence_results['path_sizes'],
        convergence_results['prices'],
        convergence_results['std_errors'],
        filename='convergence_analysis.png'
    )
    
    # Plot hedging performance
    time_grid = np.linspace(0, 1.0, len(hedging_results['portfolio_values']))
    visualizer.plot_hedging_performance(
        time_grid,
        hedging_results['portfolio_values'],
        hedging_results['price_path'],
        filename='hedging_performance.png'
    )
    
    # Get regime statistics
    regime_stats = simulator.get_regime_statistics(regimes)
    visualizer.plot_regime_statistics(
        regime_stats,
        filename='regime_statistics.png'
    )
    
    # Save results
    print("\nSaving results...")
    all_results = {
        'martingale_test': martingale_test,
        'regime_stationarity_test': regime_test,
        'barrier_option_price': barrier_price,
        'vanilla_option_price': vanilla_price,
        'asian_option_price': asian_price,
        'black_scholes_comparison': bs_comparison,
        'convergence_analysis': convergence_results,
        'greeks': barrier_greeks,
        'hedging_results': hedging_results,
        'regime_statistics': regime_stats
    }
    
    save_results(all_results, str(output_dir / 'comprehensive_results.json'), format='json')
    save_results(all_results, str(output_dir / 'comprehensive_results.pkl'), format='pickle')
    
    # Final summary
    elapsed_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nTotal execution time: {elapsed_time:.2f} seconds")
    print(f"Results saved to: {output_dir}")
    print("\nKey Findings:")
    print(f"  • Martingale test: {'PASSED' if martingale_test['test_passed'] else 'FAILED'}")
    print(f"  • Barrier option price: ${barrier_price['price']:.4f}")
    print(f"  • RS vs BS difference: {bs_comparison['relative_difference']*100:.2f}%")
    print(f"  • Hedging error: {hedging_results['terminal_hedge_error']['relative_error']*100:.2f}%")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":

    main()
