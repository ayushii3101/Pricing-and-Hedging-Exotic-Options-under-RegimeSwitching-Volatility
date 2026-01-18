"""
Basic Pricing Example
=====================

Simple example demonstrating option pricing under regime-switching.
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.regime_switching import RegimeSwitchingModel, RegimeParameters
from src.models.asset_dynamics import AssetSimulator
from src.pricing.exotic_options import BarrierOption, VanillaOption
from src.pricing.monte_carlo import MonteCarloEngine
from src.utils.data_utils import load_config, print_results_summary, setup_logging

def main():
    """Run basic pricing example."""
    
    # Setup logging
    setup_logging(level=20)  # INFO level
    
    print("\n" + "="*70)
    print("REGIME-SWITCHING EXOTIC OPTIONS PRICING")
    print("Basic Pricing Example")
    print("="*70 + "\n")
    
    # Load configuration
    config_path = Path(__file__).parent.parent / "config" / "model_config.yaml"
    config = load_config(str(config_path))
    
    # Create regime-switching model
    print("Setting up regime-switching model...")
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
    regime_model = RegimeSwitchingModel(regime_params, transition_matrix, dt)
    
    print(regime_model.summary())
    
    # Create asset simulator
    market_config = config['market']
    simulator = AssetSimulator(
        regime_model,
        spot_price=market_config['spot_price'],
        risk_free_rate=market_config['risk_free_rate'],
        dividend_yield=market_config['dividend_yield']
    )
    
    # Create Monte Carlo engine
    print(f"\nInitializing Monte Carlo engine with {config['simulation']['n_paths']} paths...")
    mc_engine = MonteCarloEngine(
        simulator,
        n_simulations=config['simulation']['n_paths'],
        seed=config['simulation']['seed']
    )
    
    # Price barrier option
    print("\n" + "-"*70)
    print("PRICING UP-AND-OUT BARRIER CALL OPTION")
    print("-"*70)
    
    barrier_config = config['options']['barrier']
    barrier_option = BarrierOption(
        strike=barrier_config['strike'],
        barrier=barrier_config['barrier'],
        maturity=barrier_config['maturity'],
        option_type=barrier_config['option_type'],
        barrier_type=barrier_config['type'],
        rebate=barrier_config['rebate']
    )
    
    print(f"\nOption: {barrier_option}")
    print(f"Spot Price: {market_config['spot_price']}")
    print(f"Risk-Free Rate: {market_config['risk_free_rate']}")
    
    # Price the option
    print("\nPricing option...")
    results = mc_engine.price_option(
        barrier_option,
        initial_regime=0,
        antithetic=config['simulation']['antithetic'],
        control_variate=config['simulation']['control_variate']
    )
    
    print_results_summary(results)
    
    # Price vanilla option for comparison
    print("\n" + "-"*70)
    print("PRICING VANILLA EUROPEAN CALL OPTION (Benchmark)")
    print("-"*70)
    
    vanilla_option = VanillaOption(
        strike=barrier_config['strike'],
        maturity=barrier_config['maturity'],
        option_type=barrier_config['option_type']
    )
    
    print(f"\nOption: {vanilla_option}")
    
    vanilla_results = mc_engine.price_option(
        vanilla_option,
        initial_regime=0,
        antithetic=config['simulation']['antithetic']
    )
    
    print_results_summary(vanilla_results)
    
    # Compare with Black-Scholes
    print("\n" + "-"*70)
    print("COMPARISON WITH BLACK-SCHOLES")
    print("-"*70)
    
    comparison = mc_engine.compare_with_black_scholes(vanilla_option, initial_regime=0)
    
    print(f"\nRegime-Switching Price: ${comparison['regime_switching_price']:.4f}")
    print(f"Black-Scholes Price:    ${comparison['black_scholes_price']:.4f}")
    print(f"Difference:             ${comparison['difference']:.4f}")
    print(f"Relative Difference:    {comparison['relative_difference']*100:.2f}%")
    
    # Price with Greeks
    print("\n" + "-"*70)
    print("COMPUTING GREEKS")
    print("-"*70)
    
    print("\nComputing Greeks for barrier option...")
    greeks_results = mc_engine.price_with_greeks(barrier_option, initial_regime=0)
    
    print(f"\nPrice:  ${greeks_results['price']:.4f}")
    print(f"Delta:  {greeks_results['delta']:.4f}")
    print(f"Gamma:  {greeks_results['gamma']:.4f}")
    print(f"Vega:   {greeks_results['vega']:.4f}")
    print(f"Theta:  {greeks_results['theta']:.4f}")
    print(f"Rho:    {greeks_results['rho']:.4f}")
    
    print("\n" + "="*70)
    print("EXAMPLE COMPLETED SUCCESSFULLY")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
