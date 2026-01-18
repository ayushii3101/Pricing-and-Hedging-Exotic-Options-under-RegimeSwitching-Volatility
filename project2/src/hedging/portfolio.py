"""
Hedging Portfolio Construction
==============================

Constructs and manages dynamic hedging portfolios.
"""

import numpy as np
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class HedgingInstrument:
    """Base class for hedging instruments."""
    
    def __init__(self, name: str, quantity: float = 0.0):
        self.name = name
        self.quantity = quantity
    
    def value(self, spot: float, **kwargs) -> float:
        """Calculate instrument value."""
        raise NotImplementedError


class Stock(HedgingInstrument):
    """Stock/underlying asset."""
    
    def __init__(self, quantity: float = 0.0):
        super().__init__("Stock", quantity)
    
    def value(self, spot: float, **kwargs) -> float:
        return self.quantity * spot
    
    def delta(self) -> float:
        return self.quantity


class Cash(HedgingInstrument):
    """Cash position."""
    
    def __init__(self, amount: float = 0.0):
        super().__init__("Cash", amount)
        self.amount = amount
    
    def value(self, spot: float, **kwargs) -> float:
        r = kwargs.get('risk_free_rate', 0.0)
        t = kwargs.get('time', 0.0)
        return self.amount * np.exp(r * t)


class VanillaHedge(HedgingInstrument):
    """Vanilla option used for hedging."""
    
    def __init__(self, strike: float, maturity: float, option_type: str, quantity: float = 0.0):
        super().__init__(f"{option_type.title()} K={strike}", quantity)
        self.strike = strike
        self.maturity = maturity
        self.option_type = option_type
    
    def value(self, spot: float, **kwargs) -> float:
        """Simplified value calculation."""
        from ..pricing.exotic_options import VanillaOption
        
        vol = kwargs.get('volatility', 0.2)
        r = kwargs.get('risk_free_rate', 0.03)
        q = kwargs.get('dividend_yield', 0.0)
        
        option = VanillaOption(self.strike, self.maturity, self.option_type)
        price = option.black_scholes_price(spot, vol, r, q)
        
        return self.quantity * price


class HedgingPortfolio:
    """
    Dynamic hedging portfolio that replicates exotic option payoff.
    
    Parameters
    ----------
    target_option : ExoticOption
        Option to hedge
    instruments : List[HedgingInstrument]
        Available hedging instruments
    """
    
    def __init__(self, target_option, instruments: Optional[List[HedgingInstrument]] = None):
        self.target_option = target_option
        self.instruments = instruments or []
        self.history = []
        
        logger.info(f"Created hedging portfolio for {target_option.name()}")
    
    def add_instrument(self, instrument: HedgingInstrument):
        """Add an instrument to the portfolio."""
        self.instruments.append(instrument)
        logger.info(f"Added {instrument.name} to portfolio")
    
    def value(self, spot: float, **kwargs) -> float:
        """Calculate total portfolio value."""
        return sum(inst.value(spot, **kwargs) for inst in self.instruments)
    
    def update_positions(self, hedge_ratios: Dict[str, float], spot: float):
        """
        Update portfolio positions based on hedge ratios.
        
        Parameters
        ----------
        hedge_ratios : Dict[str, float]
            Dictionary mapping instrument names to quantities
        spot : float
            Current spot price
        """
        for instrument in self.instruments:
            if instrument.name in hedge_ratios:
                old_quantity = instrument.quantity
                instrument.quantity = hedge_ratios[instrument.name]
                
                logger.debug(f"{instrument.name}: {old_quantity:.4f} -> {instrument.quantity:.4f}")
        
        portfolio_value = self.value(spot)
        self.history.append({
            'spot': spot,
            'portfolio_value': portfolio_value,
            'positions': {inst.name: inst.quantity for inst in self.instruments}
        })
    
    def rebalance(
        self,
        spot: float,
        greeks: Dict[str, float],
        transaction_cost: float = 0.001
    ) -> Dict[str, float]:
        """
        Rebalance portfolio to match target Greeks.
        
        Parameters
        ----------
        spot : float
            Current spot price
        greeks : Dict[str, float]
            Target Greeks to match
        transaction_cost : float
            Transaction cost as fraction of trade value
            
        Returns
        -------
        Dict[str, float]
            New hedge ratios
        """
        # Simple delta hedge for now
        target_delta = greeks.get('delta', 0.0)
        
        # Find stock instrument
        stock = next((inst for inst in self.instruments if isinstance(inst, Stock)), None)
        if stock:
            new_quantity = target_delta
            
            # Apply transaction costs
            trade_size = abs(new_quantity - stock.quantity)
            cost = trade_size * spot * transaction_cost
            
            return {stock.name: new_quantity}
        
        return {}
    
    def compute_hedging_error(
        self,
        terminal_payoff: float,
        terminal_value: float
    ) -> Dict[str, float]:
        """
        Compute hedging error metrics.
        
        Parameters
        ----------
        terminal_payoff : float
            True option payoff at maturity
        terminal_value : float
            Hedging portfolio value at maturity
            
        Returns
        -------
        Dict[str, float]
            Hedging error metrics
        """
        error = terminal_value - terminal_payoff
        relative_error = error / max(abs(terminal_payoff), 1e-6)
        
        return {
            'absolute_error': error,
            'relative_error': relative_error,
            'rmse': np.sqrt(error ** 2),
            'terminal_payoff': terminal_payoff,
            'terminal_value': terminal_value
        }
    
    def summary(self) -> str:
        """Generate portfolio summary."""
        lines = [
            f"Hedging Portfolio for {self.target_option.name()}",
            "=" * 60,
            f"\nInstruments ({len(self.instruments)}):"
        ]
        
        for inst in self.instruments:
            lines.append(f"  {inst.name}: {inst.quantity:.4f}")
        
        if self.history:
            lines.append(f"\nRebalancing History: {len(self.history)} updates")
        
        return "\n".join(lines)
