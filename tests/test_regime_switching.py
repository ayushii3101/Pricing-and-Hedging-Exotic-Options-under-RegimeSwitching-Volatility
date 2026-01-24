"""
Unit Tests for Regime-Switching Models
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.regime_switching import MarkovChain, RegimeParameters, RegimeSwitchingModel


class TestMarkovChain:
    """Test Markov chain implementation."""
    
    def test_initialization(self):
        """Test Markov chain initialization."""
        Q = np.array([[0.8, 0.2], [0.3, 0.7]])
        chain = MarkovChain(2, Q)
        
        assert chain.n_regimes == 2
        assert np.allclose(chain.Q, Q)
    
    def test_transition_matrix_validation(self):
        """Test transition matrix validation."""
        # Invalid: rows don't sum to 1
        Q_invalid = np.array([[0.5, 0.3], [0.2, 0.7]])
        
        with pytest.raises(ValueError):
            MarkovChain(2, Q_invalid)
    
    def test_stationary_distribution(self):
        """Test stationary distribution computation."""
        Q = np.array([[0.8, 0.2], [0.3, 0.7]])
        chain = MarkovChain(2, Q)
        
        pi = chain.stationary_distribution()
        
        # Check pi * Q = pi
        assert np.allclose(pi @ Q, pi)
        # Check sum to 1
        assert np.isclose(pi.sum(), 1.0)
    
    def test_simulate_path(self):
        """Test regime path simulation."""
        Q = np.array([[0.8, 0.2], [0.3, 0.7]])
        chain = MarkovChain(2, Q, dt=1.0/252)
        
        n_steps = 100
        path = chain.simulate_path(n_steps, initial_regime=0, seed=42)
        
        assert len(path) == n_steps + 1
        assert path[0] == 0
        assert all(r in [0, 1] for r in path)


class TestRegimeSwitchingModel:
    """Test regime-switching model."""
    
    def test_model_creation(self):
        """Test model creation."""
        params = [
            RegimeParameters(0, "Low Vol", 0.08, 0.15),
            RegimeParameters(1, "High Vol", 0.12, 0.30)
        ]
        Q = np.array([[0.9, 0.1], [0.2, 0.8]])
        
        model = RegimeSwitchingModel(params, Q)
        
        assert model.n_regimes == 2
        assert model.get_drift(0) == 0.08
        assert model.get_volatility(1) == 0.30
    
    def test_simulate_regimes(self):
        """Test regime simulation."""
        params = [
            RegimeParameters(0, "Low Vol", 0.08, 0.15),
            RegimeParameters(1, "High Vol", 0.12, 0.30)
        ]
        Q = np.array([[0.9, 0.1], [0.2, 0.8]])
        
        model = RegimeSwitchingModel(params, Q)
        
        regimes = model.simulate_regimes(100, 50, seed=42)
        
        assert regimes.shape == (100, 51)
        assert all(r in [0, 1] for r in regimes.flatten())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
