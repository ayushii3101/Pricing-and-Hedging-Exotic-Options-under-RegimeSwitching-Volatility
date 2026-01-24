"""
Unit Tests for Option Pricing
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pricing.exotic_options import BarrierOption, VanillaOption, AsianOption


class TestBarrierOption:
    """Test barrier option payoff."""
    
    def test_up_and_out_call(self):
        """Test up-and-out call payoff."""
        option = BarrierOption(
            strike=100,
            barrier=120,
            maturity=1.0,
            option_type='call',
            barrier_type='up-and-out'
        )
        
        # Path that doesn't hit barrier
        path1 = np.array([100, 105, 110, 115, 118])
        payoff1 = option.payoff(path1)
        assert payoff1 == max(118 - 100, 0)
        
        # Path that hits barrier
        path2 = np.array([100, 110, 121, 115, 110])
        payoff2 = option.payoff(path2)
        assert payoff2 == 0.0  # Knocked out
    
    def test_down_and_out_put(self):
        """Test down-and-out put payoff."""
        option = BarrierOption(
            strike=100,
            barrier=80,
            maturity=1.0,
            option_type='put',
            barrier_type='down-and-out'
        )
        
        # Path that doesn't hit barrier
        path1 = np.array([100, 95, 90, 85, 88])
        payoff1 = option.payoff(path1)
        assert payoff1 == max(100 - 88, 0)
        
        # Path that hits barrier
        path2 = np.array([100, 90, 79, 85, 90])
        payoff2 = option.payoff(path2)
        assert payoff2 == 0.0  # Knocked out


class TestVanillaOption:
    """Test vanilla option."""
    
    def test_call_payoff(self):
        """Test call payoff."""
        option = VanillaOption(strike=100, maturity=1.0, option_type='call')
        
        path = np.array([100, 105, 110])
        payoff = option.payoff(path)
        
        assert payoff == 10.0
    
    def test_put_payoff(self):
        """Test put payoff."""
        option = VanillaOption(strike=100, maturity=1.0, option_type='put')
        
        path = np.array([100, 95, 90])
        payoff = option.payoff(path)
        
        assert payoff == 10.0
    
    def test_black_scholes_put_call_parity(self):
        """Test put-call parity."""
        S = 100
        K = 100
        T = 1.0
        r = 0.05
        sigma = 0.2
        
        call = VanillaOption(strike=K, maturity=T, option_type='call')
        put = VanillaOption(strike=K, maturity=T, option_type='put')
        
        call_price = call.black_scholes_price(S, sigma, r)
        put_price = put.black_scholes_price(S, sigma, r)
        
        # Put-call parity: C - P = S - K*exp(-rT)
        lhs = call_price - put_price
        rhs = S - K * np.exp(-r * T)
        
        assert np.isclose(lhs, rhs, atol=1e-6)


class TestAsianOption:
    """Test Asian option."""
    
    def test_arithmetic_asian_call(self):
        """Test arithmetic Asian call."""
        option = AsianOption(
            strike=100,
            maturity=1.0,
            option_type='call',
            averaging_type='arithmetic'
        )
        
        path = np.array([100, 105, 110, 115, 120])
        payoff = option.payoff(path)
        
        avg = np.mean(path)
        expected = max(avg - 100, 0)
        
        assert np.isclose(payoff, expected)
    
    def test_geometric_asian_call(self):
        """Test geometric Asian call."""
        option = AsianOption(
            strike=100,
            maturity=1.0,
            option_type='call',
            averaging_type='geometric'
        )
        
        path = np.array([100, 105, 110, 115, 120])
        payoff = option.payoff(path)
        
        geo_avg = np.exp(np.mean(np.log(path)))
        expected = max(geo_avg - 100, 0)
        
        assert np.isclose(payoff, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
