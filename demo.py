"""
Quick Demo Script
=================

Fast demonstration of all major features (runs in ~1 minute).
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.models.regime_switching import RegimeSwitchingModel, RegimeParameters
from src.models.asset_dynamics import AssetSimulator
from src.pricing.exotic_options import BarrierOption, VanillaOption
from src.pricing.monte_carlo import MonteCarloEngine
from src.analytics.validation import MartingaleValidator

def print_section(title):
    """Print formatted section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def main():
    print("\n" + "🚀 "*20)
    print("REGIME-SWITCHING OPTIONS: QUICK DEMO")
    print("🚀 "*20)
    
    # 1. Create Model
    print_section("1. Creating Regime-Switching Model")
    
    regimes = [
        RegimeParameters(0, "Low Vol", 0.08, 0.15),
        RegimeParameters(1, "Med Vol", 0.10, 0.25),
        RegimeParameters(2, "High Vol", 0.12, 0.45)
    ]
    
    Q = np.array([
        [0.85, 0.12, 0.03],
        [0.10, 0.80, 0.10],
        [0.05, 0.25, 0.70]
    ])
    
    model = RegimeSwitchingModel(regimes, Q)
    
    for i, r in enumerate(regimes):
        print(f"  Regime {i}: {r.name:12s} | μ={r.drift:.2f} | σ={r.volatility:.2f}")
    
    stationary = model.get_stationary_distribution()
    print(f"\n  Stationary distribution: {[f'{p:.3f}' for p in stationary]}")
    
    # 2. Create Simulator
    print_section("2. Initializing Asset Simulator")
    
    simulator = AssetSimulator(model, spot_price=100, risk_free_rate=0.03)
    print(f"  Spot Price: $100")
    print(f"  Risk-Free Rate: 3%")
    
    # 3. Validate Martingale
    print_section("3. Validating Risk-Neutral Measure")
    
    validator = MartingaleValidator(simulator)
    test = validator.test_martingale_property(n_paths=5000, T=1.0)
    
    print(f"  E[S_T × e^(-rT)]: ${test['mean_discounted_ST']:.4f}")
    print(f"  S_0:             ${test['S0']:.4f}")
    print(f"  Error:           {test['relative_error']*100:.2f}%")
    print(f"  Result:          {'✓ PASSED' if test['test_passed'] else '✗ FAILED'}")
    
    # 4. Price Barrier Option
    print_section("4. Pricing Up-and-Out Barrier Call")
    
    barrier_option = BarrierOption(
        strike=100, barrier=120, maturity=1.0,
        option_type='call', barrier_type='up-and-out'
    )
    
    print(f"  Strike: $100 | Barrier: $120 | Maturity: 1 year")
    
    mc_engine = MonteCarloEngine(simulator, n_simulations=20000, seed=42)
    result = mc_engine.price_option(barrier_option, show_progress=False)
    
    print(f"\n  💰 Option Price: ${result['price']:.4f} ± ${result['std_error']:.4f}")
    print(f"  95% CI: [${result['ci_95_lower']:.4f}, ${result['ci_95_upper']:.4f}]")
    
    # 5. Price Vanilla for Comparison
    print_section("5. Comparing with Vanilla Call")
    
    vanilla = VanillaOption(strike=100, maturity=1.0, option_type='call')
    vanilla_result = mc_engine.price_option(vanilla, show_progress=False)
    
    print(f"  Vanilla Call Price: ${vanilla_result['price']:.4f}")
    print(f"  Barrier Call Price: ${result['price']:.4f}")
    
    discount = (1 - result['price']/vanilla_result['price']) * 100
    print(f"  Barrier Discount:   {discount:.1f}%")
    
    # 6. Compute Greeks
    print_section("6. Computing Greeks")
    
    print("  Computing Delta, Gamma, Vega...")
    
    # Quick Greeks (smaller bump for speed)
    S0 = simulator.S0
    bump = 0.02
    
    simulator.S0 = S0 * (1 + bump)
    price_up = mc_engine.price_option(barrier_option, show_progress=False)['price']
    
    simulator.S0 = S0 * (1 - bump)
    price_down = mc_engine.price_option(barrier_option, show_progress=False)['price']
    
    simulator.S0 = S0
    
    delta = (price_up - price_down) / (2 * S0 * bump)
    gamma = (price_up - 2*result['price'] + price_down) / ((S0 * bump)**2)
    
    print(f"\n  Delta: {delta:7.4f} (hedge ratio)")
    print(f"  Gamma: {gamma:7.4f} (convexity)")
    print(f"\n  → To hedge 100 options, hold {delta*100:.0f} shares")
    
    # 7. Black-Scholes Comparison
    print_section("7. Comparing with Black-Scholes")
    
    bs_price = vanilla.black_scholes_price(100, 0.25, 0.03)
    
    print(f"  Black-Scholes Price:     ${bs_price:.4f}")
    print(f"  Regime-Switching Price:  ${vanilla_result['price']:.4f}")
    
    diff = vanilla_result['price'] - bs_price
    print(f"  Difference:              ${diff:.4f} ({diff/bs_price*100:+.1f}%)")
    
    # 8. Simulate Sample Path
    print_section("8. Sample Price Path Simulation")
    
    prices, regimes, _ = simulator.simulate_paths(
        1, 252, 1.0, seed=42, show_progress=False
    )
    
    path = prices[0, :]
    regime_path = regimes[0, :]
    
    print(f"  Initial Price:  ${path[0]:.2f}")
    print(f"  Final Price:    ${path[-1]:.2f}")
    print(f"  Max Price:      ${path.max():.2f}")
    print(f"  Min Price:      ${path.min():.2f}")
    
    # Count regime visits
    unique, counts = np.unique(regime_path, return_counts=True)
    print(f"\n  Time in regimes:")
    for regime, count in zip(unique, counts):
        print(f"    {regimes[regime].name:12s}: {count:3d} days ({count/253*100:.1f}%)")
    
    # Final Summary
    print_section("✨ DEMO COMPLETE")
    
    print("\n  Key Results:")
    print(f"    • Martingale test: {'✓ PASSED' if test['test_passed'] else '✗ FAILED'}")
    print(f"    • Barrier option:  ${result['price']:.4f}")
    print(f"    • Delta:           {delta:.4f}")
    print(f"    • RS vs BS diff:   {diff/bs_price*100:+.1f}%")
    
    print("\n  Next Steps:")
    print("    📊 Run:  python examples/basic_pricing.py")
    print("    📈 Run:  python examples/advanced_hedging.py")
    print("    🔬 Run:  python main.py")
    print("    📓 Open: jupyter notebook notebooks/01_model_exploration.ipynb")
    
    print("\n" + "="*70)
    print("  All systems operational! Ready for analysis. 🎯")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
