"""
Exotic Option Definitions
==========================

Defines payoff functions and properties for various exotic options.
"""

import numpy as np
from typing import Optional, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExoticOption(ABC):
    """Abstract base class for exotic options."""
    
    strike: float
    maturity: float
    option_type: str  # 'call' or 'put'
    
    @abstractmethod
    def payoff(self, price_path: np.ndarray) -> float:
        """
        Compute option payoff given a price path.
        
        Parameters
        ----------
        price_path : np.ndarray
            Array of prices along the path
            
        Returns
        -------
        float
            Option payoff
        """
        pass
    
    @abstractmethod
    def name(self) -> str:
        """Return option name."""
        pass


@dataclass
class BarrierOption(ExoticOption):
    """
    Barrier option (up-and-out, down-and-out, up-and-in, down-and-in).
    
    Parameters
    ----------
    strike : float
        Strike price K
    barrier : float
        Barrier level H
    maturity : float
        Time to maturity T
    option_type : str
        'call' or 'put'
    barrier_type : str
        'up-and-out', 'down-and-out', 'up-and-in', 'down-and-in'
    rebate : float
        Rebate paid if barrier is hit (for out options)
    """
    
    barrier: float = 120.0
    barrier_type: str = 'up-and-out'
    rebate: float = 0.0
    
    def payoff(self, price_path: np.ndarray) -> float:
        """Compute barrier option payoff."""
        terminal_price = price_path[-1]
        max_price = np.max(price_path)
        min_price = np.min(price_path)
        
        # Check if barrier was hit
        if self.barrier_type == 'up-and-out':
            barrier_hit = max_price >= self.barrier
        elif self.barrier_type == 'down-and-out':
            barrier_hit = min_price <= self.barrier
        elif self.barrier_type == 'up-and-in':
            barrier_hit = max_price < self.barrier
        elif self.barrier_type == 'down-and-in':
            barrier_hit = min_price > self.barrier
        else:
            raise ValueError(f"Unknown barrier type: {self.barrier_type}")
        
        # Compute vanilla payoff
        if self.option_type == 'call':
            vanilla_payoff = max(terminal_price - self.strike, 0)
        else:
            vanilla_payoff = max(self.strike - terminal_price, 0)
        
        # Apply barrier logic
        if 'out' in self.barrier_type:
            # Knocked out
            return self.rebate if barrier_hit else vanilla_payoff
        else:
            # Knock-in
            return vanilla_payoff if barrier_hit else self.rebate
    
    def name(self) -> str:
        return f"{self.barrier_type.title()} {self.option_type.title()} Barrier Option"
    
    def __str__(self) -> str:
        return (f"{self.name()}: K={self.strike}, H={self.barrier}, "
                f"T={self.maturity}, Rebate={self.rebate}")


@dataclass
class AsianOption(ExoticOption):
    """
    Asian option with arithmetic or geometric averaging.
    
    Parameters
    ----------
    strike : float
        Strike price K
    maturity : float
        Time to maturity T
    option_type : str
        'call' or 'put'
    averaging_type : str
        'arithmetic' or 'geometric'
    """
    
    averaging_type: str = 'arithmetic'
    
    def payoff(self, price_path: np.ndarray) -> float:
        """Compute Asian option payoff."""
        if self.averaging_type == 'arithmetic':
            average_price = np.mean(price_path)
        elif self.averaging_type == 'geometric':
            average_price = np.exp(np.mean(np.log(price_path)))
        else:
            raise ValueError(f"Unknown averaging type: {self.averaging_type}")
        
        if self.option_type == 'call':
            return max(average_price - self.strike, 0)
        else:
            return max(self.strike - average_price, 0)
    
    def name(self) -> str:
        return f"{self.averaging_type.title()} Asian {self.option_type.title()}"
    
    def __str__(self) -> str:
        return f"{self.name()}: K={self.strike}, T={self.maturity}"


@dataclass
class LookbackOption(ExoticOption):
    """
    Lookback option (fixed or floating strike).
    
    Parameters
    ----------
    strike : float
        Strike price K (for fixed strike)
    maturity : float
        Time to maturity T
    option_type : str
        'call' or 'put'
    lookback_type : str
        'fixed' or 'floating'
    """
    
    lookback_type: str = 'fixed'
    
    def payoff(self, price_path: np.ndarray) -> float:
        """Compute lookback option payoff."""
        terminal_price = price_path[-1]
        max_price = np.max(price_path)
        min_price = np.min(price_path)
        
        if self.lookback_type == 'fixed':
            # Fixed strike
            if self.option_type == 'call':
                return max(max_price - self.strike, 0)
            else:
                return max(self.strike - min_price, 0)
        else:
            # Floating strike
            if self.option_type == 'call':
                return max(terminal_price - min_price, 0)
            else:
                return max(max_price - terminal_price, 0)
    
    def name(self) -> str:
        return f"{self.lookback_type.title()} Strike Lookback {self.option_type.title()}"
    
    def __str__(self) -> str:
        return f"{self.name()}: K={self.strike}, T={self.maturity}"


@dataclass
class VanillaOption(ExoticOption):
    """
    Standard European vanilla option (for benchmarking).
    
    Parameters
    ----------
    strike : float
        Strike price K
    maturity : float
        Time to maturity T
    option_type : str
        'call' or 'put'
    """
    
    def payoff(self, price_path: np.ndarray) -> float:
        """Compute vanilla option payoff."""
        terminal_price = price_path[-1]
        
        if self.option_type == 'call':
            return max(terminal_price - self.strike, 0)
        else:
            return max(self.strike - terminal_price, 0)
    
    def black_scholes_price(
        self,
        spot: float,
        volatility: float,
        risk_free_rate: float,
        dividend_yield: float = 0.0
    ) -> float:
        """Compute Black-Scholes price for comparison."""
        from scipy.stats import norm
        
        d1 = (np.log(spot / self.strike) + 
              (risk_free_rate - dividend_yield + 0.5 * volatility**2) * self.maturity) / \
             (volatility * np.sqrt(self.maturity))
        d2 = d1 - volatility * np.sqrt(self.maturity)
        
        if self.option_type == 'call':
            price = (spot * np.exp(-dividend_yield * self.maturity) * norm.cdf(d1) -
                    self.strike * np.exp(-risk_free_rate * self.maturity) * norm.cdf(d2))
        else:
            price = (self.strike * np.exp(-risk_free_rate * self.maturity) * norm.cdf(-d2) -
                    spot * np.exp(-dividend_yield * self.maturity) * norm.cdf(-d1))
        
        return price
    
    def name(self) -> str:
        return f"European {self.option_type.title()}"
    
    def __str__(self) -> str:
        return f"{self.name()}: K={self.strike}, T={self.maturity}"


class DigitalOption(ExoticOption):
    """
    Digital (binary) option that pays fixed amount if condition is met.
    
    Parameters
    ----------
    strike : float
        Strike price K
    maturity : float
        Time to maturity T
    option_type : str
        'call' or 'put'
    payout : float
        Fixed payout amount
    """
    
    def __init__(self, strike: float, maturity: float, option_type: str, payout: float = 1.0):
        self.strike = strike
        self.maturity = maturity
        self.option_type = option_type
        self.payout = payout
    
    def payoff(self, price_path: np.ndarray) -> float:
        """Compute digital option payoff."""
        terminal_price = price_path[-1]
        
        if self.option_type == 'call':
            return self.payout if terminal_price > self.strike else 0.0
        else:
            return self.payout if terminal_price < self.strike else 0.0
    
    def name(self) -> str:
        return f"Digital {self.option_type.title()}"
    
    def __str__(self) -> str:
        return f"{self.name()}: K={self.strike}, T={self.maturity}, Payout={self.payout}"


def create_option_from_config(config: dict) -> ExoticOption:
    """
    Create an exotic option from configuration.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary with option specifications
        
    Returns
    -------
    ExoticOption
        Instantiated option object
    """
    option_configs = config['options']
    
    # Determine which option type to create (default to barrier)
    if 'barrier' in option_configs:
        barrier_config = option_configs['barrier']
        return BarrierOption(
            strike=barrier_config['strike'],
            barrier=barrier_config['barrier'],
            maturity=barrier_config['maturity'],
            option_type=barrier_config['option_type'],
            barrier_type=barrier_config.get('type', 'up-and-out'),
            rebate=barrier_config.get('rebate', 0.0)
        )
    
    elif 'asian' in option_configs:
        asian_config = option_configs['asian']
        return AsianOption(
            strike=asian_config['strike'],
            maturity=asian_config['maturity'],
            option_type=asian_config['option_type'],
            averaging_type=asian_config.get('type', 'arithmetic')
        )
    
    elif 'lookback' in option_configs:
        lookback_config = option_configs['lookback']
        return LookbackOption(
            strike=lookback_config['strike'],
            maturity=lookback_config['maturity'],
            option_type=lookback_config['option_type'],
            lookback_type=lookback_config.get('type', 'fixed')
        )
    
    else:
        raise ValueError("No valid option configuration found")
