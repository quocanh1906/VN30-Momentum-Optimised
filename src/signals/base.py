"""
base.py — Abstract base class for all signals.

Every signal must implement:
    compute(prices) → DataFrame of scores (dates × tickers)

This standardised interface means vol_scaling.py, portfolio.py,
and backtest.py never need to know which signal they're running.
Signals are fully interchangeable and combinable.
"""

from abc import ABC, abstractmethod
import pandas as pd


class BaseSignal(ABC):
    """
    Abstract base class for all momentum signals.

    Subclasses must implement compute().
    All other infrastructure (vol scaling, portfolio
    construction, backtesting) consumes only the
    output of compute() — never the signal internals.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Compute signal scores for all stocks and dates.

        Parameters
        ----------
        prices : DataFrame (dates × tickers)
                 weekly close prices, point-in-time

        Returns
        -------
        DataFrame (dates × tickers)
            scores in [-inf, +inf]
            higher = stronger long signal
            NaN = no signal (insufficient data, not in universe)
        """
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name})"