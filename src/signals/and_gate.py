"""
and_gate.py — AND-gate signal composition.

Long a stock only when every constituent signal is simultaneously
positive. The output score at permitted tickers is the mean of the
constituents' cross-sectional z-scores; elsewhere it is NaN.

More restrictive than additive CombinedSignal: fewer trades, higher
per-trade conviction. Composable like any other BaseSignal —
    AndGateSignal([momentum, ma_crossover])
returns a BaseSignal whose `compute()` yields the gated scores.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
from signals.base import BaseSignal


class AndGateSignal(BaseSignal):

    def __init__(self, signals: list):
        if len(signals) < 2:
            raise ValueError("AndGateSignal requires at least 2 signals")
        name = " & ".join(s.name for s in signals)
        super().__init__(name=f"AndGate({name})")
        self.signals = signals

    def compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        z_list = []
        for s in self.signals:
            raw = s.compute(prices)
            z = raw.apply(
                lambda row: (row - row.mean()) / row.std()
                if row.notna().sum() > 2 else row * np.nan,
                axis=1
            )
            z_list.append(z)

        # Align on intersection of columns and union of dates
        common_cols = sorted(set.intersection(
            *(set(z.columns) for z in z_list)
        ))
        idx = z_list[0].index
        for z in z_list[1:]:
            idx = idx.union(z.index)

        z_list = [z.reindex(index=idx, columns=common_cols) for z in z_list]

        # Mean z across signals — ignores NaN positions per cell only
        # after the all-positive check below, so NaN → NaN propagates.
        stacked = pd.concat(z_list, axis=0, keys=range(len(z_list)))
        mean_z  = stacked.groupby(level=1).mean()

        # Require all signals strictly positive (NaN fails the check).
        all_positive = z_list[0] > 0
        for z in z_list[1:]:
            all_positive = all_positive & (z > 0)

        return mean_z.where(all_positive)


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

    from signals.momentum import MomentumSignal
    from signals.ma_crossover import MACrossoverSignal

    closes = pd.read_csv(
        "data/processed/closes_weekly.csv",
        index_col=0, parse_dates=True
    )

    mom = MomentumSignal(formation_weeks=26, skip_weeks=1)
    ma  = MACrossoverSignal(fast_weeks=10, slow_weeks=40)

    gated  = AndGateSignal([mom, ma])
    scores = gated.compute(closes)

    print(f"Signal: {gated.name}")
    print(f"Non-NaN scores: {scores.notna().sum().sum()}")
    print(f"Coverage ratio: "
          f"{scores.notna().sum().sum() / scores.size:.1%}")

    latest = scores.dropna(how='all').iloc[-1].dropna()\
                   .sort_values(ascending=False)
    print(f"\nTop 10 gated scores (latest date):")
    print(latest.head(10).round(3))
