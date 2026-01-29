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
    
    def __init__(
        self,
        target_option,
        instruments: Optional[List[HedgingInstrument]] = None,
        quantity: float = 1.0
    ):
        self.target_option = target_option
        self.instruments = instruments or []
        self.quantity = quantity
        self.history = []
        
        logger.info(
            f"Created hedging portfolio for {target_option.name()} (qty={quantity})"
        )
    
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
                if isinstance(instrument, Cash):
                    instrument.amount = instrument.quantity
                
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
        # Self-financing delta hedge
        target_delta = greeks.get('delta', 0.0) * self.quantity
        updates: Dict[str, float] = {}

        # 1) Stock rebalancing
        stock = next((inst for inst in self.instruments if isinstance(inst, Stock)), None)
        cost_of_trades = 0.0
        if stock:
            old_quantity = stock.quantity
            new_quantity = target_delta

            trade_size = new_quantity - old_quantity
            cost_of_trades = trade_size * spot

            trading_fees = abs(trade_size) * spot * transaction_cost
            cost_of_trades += trading_fees

            updates[stock.name] = new_quantity

        # 2) Cash adjustment (self-financing)
        cash = next((inst for inst in self.instruments if isinstance(inst, Cash)), None)
        if cash:
            updates[cash.name] = cash.amount - cost_of_trades
        elif cost_of_trades != 0.0:
            logger.warning("Rebalancing required cash flow but no Cash instrument found!")

        return updates
    
    def compute_hedging_error(
        self,
        terminal_payoff,
        terminal_value: Optional[float] = None,
        n_paths: int = 1000,
        initial_regime: int = 0,
        show_progress: bool = False
    ) -> Dict[str, float]:
        """
        Compute hedging error metrics.
        
        Parameters
        ----------
        terminal_payoff : float or AssetSimulator
            True option payoff at maturity, or an AssetSimulator to run a Monte Carlo
        terminal_value : float, optional
            Hedging portfolio value at maturity
        n_paths : int
            Number of Monte Carlo paths (when passing AssetSimulator)
        initial_regime : int
            Starting regime (when passing AssetSimulator)
        show_progress : bool
            Show progress bar for simulations
            
        Returns
        -------
        Dict[str, float]
            Hedging error metrics
        """
        # Path-based Monte Carlo error estimation.
        if terminal_value is None and hasattr(terminal_payoff, "simulate_paths"):
            simulator = terminal_payoff
            option = self.target_option
            T = option.maturity
            n_steps = max(int(T * 252), 1)

            prices, regimes, _ = simulator.simulate_paths(
                n_paths,
                n_steps,
                T,
                initial_regime=initial_regime,
                risk_neutral=True,
                show_progress=show_progress
            )

            payoffs = np.array([option.payoff(prices[i, :]) for i in range(n_paths)])
            payoffs = payoffs * self.quantity
            terminal_spots = prices[:, -1]

            implied_vol = float(
                np.mean([params.volatility for params in simulator.regime_model.regime_params])
            )

            terminal_values = np.array([
                self.value(
                    spot,
                    risk_free_rate=getattr(simulator, "r", 0.0),
                    dividend_yield=getattr(simulator, "q", 0.0),
                    time=T,
                    volatility=implied_vol
                )
                for spot in terminal_spots
            ])

            errors = terminal_values - payoffs
            return {
                'mean_error': float(np.mean(errors)),
                'std_error': float(np.std(errors)),
                'rmse': float(np.sqrt(np.mean(errors ** 2))),
                'n_paths': int(n_paths)
            }

        # Terminal error for a single realization.
        scaled_payoff = terminal_payoff * self.quantity
        error = terminal_value - scaled_payoff
        
        # Robust relative error calculation (Safe against divide-by-zero)
        denominator = abs(scaled_payoff)
        if denominator < 1e-4:
            # Option expired OTM/ATM, relative error is not meaningful
            relative_error = 0.0 
        else:
            relative_error = error / denominator

        return {
            'absolute_error': error,
            'relative_error': relative_error,
            'rmse': np.sqrt(error ** 2),
            'terminal_payoff': scaled_payoff,
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
