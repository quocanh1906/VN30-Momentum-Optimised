"""
base.py — Abstract base class for exposure filters.

An ExposureFilter converts market/strategy context into a per-date
scalar in [0, 1] that multiplies total gross exposure. All filters
share a single compute() interface so v2 / v3 filters can be swapped
without touching the strategy driver.
"""

from abc import ABC, abstractmethod
import pandas as pd


class ExposureFilter(ABC):

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def compute(self, **context) -> pd.Series:
        """
        Parameters
        ----------
        context : keyword arguments providing strategy/market data.
                  Accepted keys depend on the concrete filter —
                  typical keys: scores, closes_weekly, market_returns,
                  holding_weeks. Unknown keys must be ignored via
                  **_ so filters remain interchangeable.

        Returns
        -------
        Series of exposure scalars in [0, 1] indexed by date,
        already shifted so it is usable without lookahead.
        """
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name})"
