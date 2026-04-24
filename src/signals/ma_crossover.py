"""
ma_crossover.py — Moving-average crossover trend signal.

For each stock i and week t:

    raw_i,t = SMA_fast_i,t / SMA_slow_i,t − 1

Then cross-sectionally z-scored per date. Positive score = short-term
price above long-term trend. Uses SMA (not EMA) for robustness to
single-week data gaps (e.g. Tết). Rolling windows tolerate up to 20%
missing observations so a holiday NaN does not poison the signal.

Defaults: fast=10w (~2.5mo), slow=40w (~10mo) — intermediate trend
horizon aligned with the 26w momentum formation period.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
from signals.base import BaseSignal


class MACrossoverSignal(BaseSignal):

    def __init__(self, fast_weeks=10, slow_weeks=40, min_history=None):
        if fast_weeks >= slow_weeks:
            raise ValueError("fast_weeks must be < slow_weeks")
        super().__init__(
            name=f"MACross_{fast_weeks}_{slow_weeks}"
        )
        self.fast_weeks  = fast_weeks
        self.slow_weeks  = slow_weeks
        self.min_history = min_history or slow_weeks

    def compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        fast_min = max(int(self.fast_weeks * 0.8), 5)
        slow_min = max(int(self.slow_weeks * 0.8), 10)

        fast = prices.rolling(self.fast_weeks,
                              min_periods=fast_min).mean()
        slow = prices.rolling(self.slow_weeks,
                              min_periods=slow_min).mean()

        raw = (fast / slow) - 1

        data_count = prices.notna().cumsum()
        raw        = raw.where(data_count >= self.min_history)

        scores = raw.apply(
            lambda row: (row - row.mean()) / row.std()
            if row.notna().sum() > 2 else row * np.nan,
            axis=1
        )
        return scores


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

    print("="*60)
    print("  MA Crossover Signal — Test")
    print("="*60)

    closes = pd.read_csv(
        "data/processed/closes_weekly.csv",
        index_col=0, parse_dates=True
    )
    print(f"\nLoaded: {closes.shape} (weeks × tickers)")

    sig    = MACrossoverSignal(fast_weeks=10, slow_weeks=40)
    scores = sig.compute(closes)

    print(f"\nSignal: {sig.name}")
    print(f"Scores shape     : {scores.shape}")
    print(f"Non-NaN scores   : {scores.notna().sum().sum()}")

    latest = scores.dropna(how='all').iloc[-1].dropna()\
                   .sort_values(ascending=False)
    print(f"\nTop 10 trend scores (latest date):")
    print(latest.head(10).round(3))
