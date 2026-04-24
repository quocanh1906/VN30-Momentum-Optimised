"""
combined.py — Combine multiple signals into one score.

Example usage:
    mom  = MomentumSignal(formation_weeks=26)
    rev  = ReversalSignal(formation_weeks=4)
    
    combined = CombinedSignal([
        (mom, 0.7),   # 70% momentum
        (rev, 0.3),   # 30% reversal
    ])
    
    scores = combined.compute(prices)
"""

from signals.base import BaseSignal
import pandas as pd
import numpy as np


class CombinedSignal(BaseSignal):
    """
    Linearly combine multiple signals with given weights.
    Each signal is z-scored before combining to ensure
    equal scale regardless of signal magnitude.
    """

    def __init__(self, signals_and_weights: list):
        """
        Parameters
        ----------
        signals_and_weights : list of (BaseSignal, float) tuples
            e.g. [(momentum_signal, 0.7), (reversal_signal, 0.3)]
        """
        names = "+".join(
            f"{w:.0%}{s.name}" for s, w in signals_and_weights
        )
        super().__init__(name=f"Combined({names})")
        self.signals_and_weights = signals_and_weights

    def compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        combined = None

        for signal, weight in self.signals_and_weights:
            scores = signal.compute(prices)

            # Z-score each signal before combining
            scores_z = scores.apply(
                lambda row: (row - row.mean()) / row.std()
                if row.notna().sum() > 2 else row * np.nan,
                axis=1
            )

            if combined is None:
                combined = scores_z * weight
            else:
                combined = combined.add(scores_z * weight,
                                        fill_value=0)

        return combined