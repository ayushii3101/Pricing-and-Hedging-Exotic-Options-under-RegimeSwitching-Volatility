"""
Unit Tests for Hedging Strategies
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.regime_switching import RegimeSwitchingModel, RegimeParameters
from src.models.asset_dynamics import AssetSimulator
from src.pricing.exotic_options import VanillaOption
from src.pricing.monte_carlo import MonteCarloEngine
from src.hedging.greeks import GreeksCalculator
from src.hedging.portfolio import HedgingPortfolio, Stock, Cash, VanillaHedge
from src.hedging.optimization import MeanVarianceHedger


class TestGreeksCalculator:
    """Test Greeks calculation."""
    
    @pytest.fixture
    def setup_model(self):
        """Setup basic model for testing."""
        Q = np.array([[0.9, 0.1], [0.2, 0.8]])
        regime_params = [
            RegimeParameters(0, "Low Vol", 0.05, 0.15),
            RegimeParameters(1, "High Vol", 0.05, 0.30)
        ]
        
        regime_model = RegimeSwitchingModel(Q, regime_params, dt=1.0/252)
        simulator = AssetSimulator(regime_model, spot_price=100, risk_free_rate=0.03)
        mc_engine = MonteCarloEngine(simulator, n_paths=5000)
        greeks_calc = GreeksCalculator(mc_engine)
        
        return greeks_calc
    
    def test_delta_calculation(self, setup_model):
        """Test delta calculation."""
        greeks_calc = setup_model
        option = VanillaOption(strike=100, maturity=1.0, option_type='call')
        
        delta = greeks_calc.delta(option)
        
        # ATM call delta should be around 0.5
        assert 0.3 < delta < 0.7
    
    def test_gamma_calculation(self, setup_model):
        """Test gamma calculation."""
        greeks_calc = setup_model
        option = VanillaOption(strike=100, maturity=1.0, option_type='call')
        
        gamma = greeks_calc.gamma(option)
        
        # Gamma should be positive
        assert gamma > 0
    
    def test_vega_calculation(self, setup_model):
        """Test vega calculation."""
        greeks_calc = setup_model
        option = VanillaOption(strike=100, maturity=1.0, option_type='call')
        
        vega = greeks_calc.vega(option)
        
        # Vega should be positive for long options
        assert vega > 0
    
    def test_calculate_all(self, setup_model):
        """Test calculating all Greeks at once."""
        greeks_calc = setup_model
        option = VanillaOption(strike=100, maturity=1.0, option_type='call')
        
        greeks = greeks_calc.calculate_all(option)
        
        assert 'delta' in greeks
        assert 'gamma' in greeks
        assert 'vega' in greeks
        assert 'theta' in greeks
        assert 'rho' in greeks


class TestHedgingPortfolio:
    """Test hedging portfolio functionality."""
    
    def test_portfolio_initialization(self):
        """Test portfolio initialization."""
        target_option = VanillaOption(strike=100, maturity=1.0, option_type='call')
        portfolio = HedgingPortfolio(target_option)
        
        assert portfolio.target_option == target_option
        assert len(portfolio.instruments) == 0
    
    def test_add_instrument(self):
        """Test adding instruments to portfolio."""
        target_option = VanillaOption(strike=100, maturity=1.0, option_type='call')
        portfolio = HedgingPortfolio(target_option)
        
        stock = Stock(quantity=0.5)
        portfolio.add_instrument(stock)
        
        assert len(portfolio.instruments) == 1
        assert portfolio.instruments[0] == stock
    
    def test_portfolio_value(self):
        """Test portfolio value calculation."""
        target_option = VanillaOption(strike=100, maturity=1.0, option_type='call')
        portfolio = HedgingPortfolio(target_option)
        
        stock = Stock(quantity=0.5)
        cash = Cash(amount=50.0)
        portfolio.add_instrument(stock)
        portfolio.add_instrument(cash)
        
        spot = 105.0
        value = portfolio.value(spot, r=0.03, sigma=0.2, T=1.0)
        
        # Stock: 0.5 * 105 = 52.5, Cash: 50, minus option value
        assert value != 0
    
    def test_update_positions(self):
        """Test updating portfolio positions."""
        target_option = VanillaOption(strike=100, maturity=1.0, option_type='call')
        portfolio = HedgingPortfolio(target_option)
        
        stock = Stock(quantity=0.0)
        portfolio.add_instrument(stock)
        
        hedge_ratios = {'Stock': 0.6}
        portfolio.update_positions(hedge_ratios, spot=100.0)
        
        assert np.isclose(stock.quantity, 0.6)


class TestStock:
    """Test Stock instrument."""
    
    def test_stock_value(self):
        """Test stock value calculation."""
        stock = Stock(quantity=2.0)
        value = stock.value(spot=100.0)
        
        assert value == 200.0
    
    def test_stock_delta(self):
        """Test stock delta."""
        stock = Stock(quantity=1.0)
        delta = stock.delta()
        
        assert delta == 1.0


class TestCash:
    """Test Cash instrument."""
    
    def test_cash_value(self):
        """Test cash value calculation."""
        cash = Cash(amount=100.0)
        value = cash.value(spot=105.0)
        
        assert value == 100.0


class TestVanillaHedge:
    """Test VanillaHedge instrument."""
    
    def test_vanilla_hedge_initialization(self):
        """Test vanilla hedge initialization."""
        hedge = VanillaHedge(strike=100, maturity=1.0, option_type='call', quantity=1.0)
        
        assert hedge.strike == 100
        assert hedge.maturity == 1.0
        assert hedge.option_type == 'call'
        assert hedge.quantity == 1.0
    
    def test_vanilla_hedge_value(self):
        """Test vanilla hedge value calculation."""
        hedge = VanillaHedge(strike=100, maturity=1.0, option_type='call', quantity=1.0)
        
        value = hedge.value(spot=110.0, r=0.03, sigma=0.2, T=1.0)
        
        # Should be positive for ITM call
        assert value > 0


class TestMeanVarianceHedger:
    """Test mean-variance hedging optimization."""
    
    @pytest.fixture
    def setup_hedger(self):
        """Setup hedger for testing."""
        Q = np.array([[0.9, 0.1], [0.2, 0.8]])
        regime_params = [
            RegimeParameters(0, "Low Vol", 0.05, 0.15),
            RegimeParameters(1, "High Vol", 0.05, 0.30)
        ]
        
        regime_model = RegimeSwitchingModel(Q, regime_params, dt=1.0/252)
        simulator = AssetSimulator(regime_model, spot_price=100, risk_free_rate=0.03)
        hedger = MeanVarianceHedger(simulator, risk_aversion=0.5)
        
        return hedger, simulator
    
    def test_hedger_initialization(self, setup_hedger):
        """Test hedger initialization."""
        hedger, _ = setup_hedger
        
        assert hedger.risk_aversion == 0.5
        assert hedger.asset_simulator is not None
    
    def test_compute_optimal_weights(self, setup_hedger):
        """Test optimal weights calculation."""
        hedger, simulator = setup_hedger
        
        target_option = VanillaOption(strike=100, maturity=1.0, option_type='call')
        stock = Stock()
        
        portfolio = HedgingPortfolio(target_option, [stock])
        
        # This is computationally expensive, use small sample
        weights = hedger.compute_optimal_weights(
            portfolio,
            n_paths=1000,
            initial_regime=0
        )
        
        assert 'Stock' in weights
        # Weight should be reasonable (between -2 and 2 for delta hedging)
        assert -2.0 <= weights['Stock'] <= 2.0


class TestHedgingIntegration:
    """Integration tests for hedging strategies."""
    
    def test_complete_hedging_workflow(self):
        """Test complete hedging workflow."""
        # Setup model
        Q = np.array([[0.9, 0.1], [0.2, 0.8]])
        regime_params = [
            RegimeParameters(0, "Low Vol", 0.05, 0.15),
            RegimeParameters(1, "High Vol", 0.05, 0.30)
        ]
        
        regime_model = RegimeSwitchingModel(Q, regime_params, dt=1.0/252)
        simulator = AssetSimulator(regime_model, spot_price=100, risk_free_rate=0.03)
        mc_engine = MonteCarloEngine(simulator, n_paths=5000)
        
        # Create option and portfolio
        option = VanillaOption(strike=100, maturity=1.0, option_type='call')
        portfolio = HedgingPortfolio(option, [Stock()])
        
        # Calculate Greeks
        greeks_calc = GreeksCalculator(mc_engine)
        delta = greeks_calc.delta(option)
        
        # Update hedge
        portfolio.update_positions({'Stock': delta}, spot=100.0)
        
        # Check that hedge is applied
        assert np.isclose(portfolio.instruments[0].quantity, delta, atol=0.01)
    
    def test_hedging_error_calculation(self):
        """Test hedging error calculation."""
        Q = np.array([[0.9, 0.1], [0.2, 0.8]])
        regime_params = [
            RegimeParameters(0, "Low Vol", 0.05, 0.15),
            RegimeParameters(1, "High Vol", 0.05, 0.30)
        ]
        
        regime_model = RegimeSwitchingModel(Q, regime_params, dt=1.0/252)
        simulator = AssetSimulator(regime_model, spot_price=100, risk_free_rate=0.03)
        
        option = VanillaOption(strike=100, maturity=1.0, option_type='call')
        portfolio = HedgingPortfolio(option, [Stock(quantity=0.5)])
        
        # Compute hedging error (small sample for speed)
        error_stats = portfolio.compute_hedging_error(
            simulator,
            n_paths=1000,
            initial_regime=0
        )
        
        assert 'mean_error' in error_stats
        assert 'std_error' in error_stats
        assert 'rmse' in error_stats


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
