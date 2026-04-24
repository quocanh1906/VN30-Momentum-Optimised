"""
bollinger.py — Bollinger %B cross-sectional signal.

For each stock i and week t with N-week rolling SMA (mid) and std:

    upper = mid + k * std
    lower = mid − k * std
    %B    = (price − lower) / (upper − lower)

%B ≥ 1 ⇒ price above upper band (breakout / strong trend).
%B ≤ 0 ⇒ price below lower band (sharp decline).

Cross-sectionally z-scored per date. Interprets band position as
**momentum confirmation** (high %B = bullish), not mean reversion —
matches the long-only, cross-sectional momentum framework.

Defaults: window=20w, k=2 (canonical Bollinger setting).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
from signals.base import BaseSignal


class BollingerBandSignal(BaseSignal):

    def __init__(self, window=20, k=2.0, min_history=None):
        super().__init__(name=f"Bollinger_{window}_{k}")
        self.window      = window
        self.k           = k
        self.min_history = min_history or window

    def compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        min_p = max(int(self.window * 0.8), 10)
        mid   = prices.rolling(self.window, min_periods=min_p).mean()
        std   = prices.rolling(self.window, min_periods=min_p).std()

        band_width = 2 * self.k * std
        pct_b      = (prices - (mid - self.k * std)) / \
                     band_width.replace(0, np.nan)

        data_count = prices.notna().cumsum()
        pct_b      = pct_b.where(data_count >= self.min_history)

        scores = pct_b.apply(
            lambda row: (row - row.mean()) / row.std()
            if row.notna().sum() > 2 else row * np.nan,
            axis=1
        )
        return scores


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

    closes = pd.read_csv(
        "data/processed/closes_weekly.csv",
        index_col=0, parse_dates=True
    )
    sig    = BollingerBandSignal()
    scores = sig.compute(closes)

    print(f"Signal: {sig.name}")
    print(f"Non-NaN scores: {scores.notna().sum().sum()}")
    latest = scores.dropna(how='all').iloc[-1].dropna()\
                   .sort_values(ascending=False)
    print(f"\nTop 10 %B scores (latest date):")
    print(latest.head(10).round(3))
