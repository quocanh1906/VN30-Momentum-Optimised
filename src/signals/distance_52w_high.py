"""
distance_52w_high.py — George & Hwang (2004) 52-week high signal.

For each stock i and week t:

    score_i,t = price_i,t / rolling_max(price_i, 52w) − 1 ∈ [−1, 0]

Cross-sectionally z-scored per date. Closer to 0 (price at 52w high)
= strong anchor-based momentum. Deeper negative = well below 52w high.

Reference:
    George, T. J., & Hwang, C. Y. (2004). The 52-week high and
    momentum investing. Journal of Finance, 59(5).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
from signals.base import BaseSignal


class Distance52WHighSignal(BaseSignal):

    def __init__(self, window_weeks=52, min_history=None):
        super().__init__(name=f"Dist52W_{window_weeks}")
        self.window_weeks = window_weeks
        self.min_history  = min_history or window_weeks // 2

    def compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        min_p       = max(int(self.window_weeks * 0.8), 20)
        rolling_max = prices.rolling(
            self.window_weeks, min_periods=min_p
        ).max()

        dist = prices / rolling_max - 1  # ∈ [-1, 0]

        data_count = prices.notna().cumsum()
        dist       = dist.where(data_count >= self.min_history)

        scores = dist.apply(
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
    sig    = Distance52WHighSignal()
    scores = sig.compute(closes)

    print(f"Signal: {sig.name}")
    print(f"Non-NaN scores: {scores.notna().sum().sum()}")
    latest = scores.dropna(how='all').iloc[-1].dropna()\
                   .sort_values(ascending=False)
    print(f"\nTop 10 (closest to 52w high):")
    print(latest.head(10).round(3))
