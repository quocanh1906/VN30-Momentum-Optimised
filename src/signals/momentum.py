import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
from signals.base import BaseSignal


class MomentumSignal(BaseSignal):

    def __init__(self, formation_weeks=26, skip_weeks=1, min_history=13):
        super().__init__(
            name=f"Momentum_J{formation_weeks}_K{skip_weeks}"
        )
        self.formation_weeks = formation_weeks
        self.skip_weeks      = skip_weeks
        self.min_history     = min_history

    def compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Cross-sectional momentum scores.
        Returns z-scored formation returns.
        """
        lagged  = prices.shift(self.skip_weeks)
        raw_ret = lagged / lagged.shift(self.formation_weeks) - 1

        data_count = prices.notna().cumsum()
        raw_ret    = raw_ret.where(data_count >= self.min_history)

        scores = raw_ret.apply(
            lambda row: (row - row.mean()) / row.std()
            if row.notna().sum() > 2 else row * np.nan,
            axis=1
        )
        return scores


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

    print("="*60)
    print("  Momentum Signal — Test")
    print("="*60)

    closes = pd.read_csv(
        "data/processed/closes_weekly.csv",
        index_col=0, parse_dates=True
    )
    print(f"\nLoaded: {closes.shape} (weeks × tickers)")

    signal = MomentumSignal(formation_weeks=26, skip_weeks=1)
    scores = signal.compute(closes)

    print(f"\nScores shape: {scores.shape}")
    print(f"Non-NaN scores: {scores.notna().sum().sum()}")

    print(f"\nSample scores (latest date):")
    latest = scores.dropna(how='all').iloc[-1].dropna()\
                   .sort_values(ascending=False)
    print(latest.head(10).round(3))
    print(f"\nSignal name: {signal.name}")