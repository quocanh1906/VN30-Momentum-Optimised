"""
rsi.py — Relative Strength Index cross-sectional signal.

For each stock i:

    RSI_i,t = 100 − 100 / (1 + avg_gain / avg_loss)

Where avg_gain / avg_loss are Wilder-smoothed (EWM, α = 1/N) over
N weeks of gains / absolute losses. High RSI = consistent recent
gains; low RSI = consistent recent losses.

Cross-sectionally z-scored per date. Interprets RSI as **momentum-
confirming** (high RSI = bullish, long candidate), aligned with the
long-only, medium-horizon strategy. For a mean-reversion variant,
subclass and negate the score in compute().

Default: window=14w (~3.5 months) — too long to be noise, short
enough to be responsive to regime changes.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
from signals.base import BaseSignal


class RSISignal(BaseSignal):

    def __init__(self, window=14, min_history=None):
        super().__init__(name=f"RSI_{window}")
        self.window      = window
        self.min_history = min_history or (window * 2)

    def compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        delta = prices.diff()
        gain  = delta.clip(lower=0)
        loss  = (-delta).clip(lower=0)

        alpha    = 1.0 / self.window
        avg_gain = gain.ewm(alpha=alpha, adjust=False,
                            min_periods=self.window).mean()
        avg_loss = loss.ewm(alpha=alpha, adjust=False,
                            min_periods=self.window).mean()

        rs  = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - 100 / (1 + rs)

        data_count = prices.notna().cumsum()
        rsi        = rsi.where(data_count >= self.min_history)

        scores = rsi.apply(
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
    sig    = RSISignal(window=14)
    scores = sig.compute(closes)

    print(f"Signal: {sig.name}")
    print(f"Non-NaN scores: {scores.notna().sum().sum()}")
    latest = scores.dropna(how='all').iloc[-1].dropna()\
                   .sort_values(ascending=False)
    print(f"\nTop 10 RSI scores (latest date):")
    print(latest.head(10).round(3))
