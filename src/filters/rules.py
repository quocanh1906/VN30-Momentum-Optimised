"""
rules.py — Rules-based exposure filter.

Three independent conditions, each halves exposure when true:

    1. Market drawdown worse than -10%                  (Daniel-Moskowitz)
    2. Momentum portfolio realised vol above 75th pct   (Barroso-Santa-Clara)
    3. Trailing 12-week realised IC negative            (signal quality gate)

Exposure = 0.5 ** n_flags ∈ {1.0, 0.5, 0.25, 0.125}.

No training surface — cannot overfit. Recommended baseline against
which any ML-based filter must prove value on walk-forward OOS.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
from filters.base import ExposureFilter
from signals.strength_filter import _momentum_long_short_returns


class RulesBasedFilter(ExposureFilter):

    def __init__(self, dd_threshold=-0.10,
                 mom_vol_percentile=0.75,
                 mom_vol_window=26,
                 ic_window=12,
                 halve_factor=0.5):
        super().__init__(name="RulesBased_DM_BSC")
        self.dd_threshold       = dd_threshold
        self.mom_vol_percentile = mom_vol_percentile
        self.mom_vol_window     = mom_vol_window
        self.ic_window          = ic_window
        self.halve_factor       = halve_factor

    def compute(self, scores, closes_weekly, market_returns,
                holding_weeks=4, **_):
        weekly_returns = closes_weekly.pct_change()

        # --- Condition 1: market drawdown ---
        mkt_cum  = (1 + market_returns).cumprod()
        mkt_peak = mkt_cum.cummax()
        mkt_dd   = (mkt_cum - mkt_peak) / mkt_peak

        # --- Condition 2: momentum portfolio realised vol ---
        mom_ret = _momentum_long_short_returns(scores, weekly_returns)
        mom_vol = mom_ret.rolling(
            self.mom_vol_window,
            min_periods=max(int(self.mom_vol_window * 0.8), 10)
        ).std() * np.sqrt(52)
        mom_vol_rank = mom_vol.expanding(min_periods=52).rank(pct=True)

        # --- Condition 3: trailing realised IC ---
        # Cumulative h-week forward return; ic_raw[t] uses [t, t+h] so
        # it is only observable at t+h — shift by holding_weeks.
        fwd_cum = closes_weekly.shift(-holding_weeks) / closes_weekly - 1
        from scipy.stats import spearmanr

        ic_list = []
        for date in scores.index:
            if date not in fwd_cum.index:
                ic_list.append(np.nan)
                continue
            s = scores.loc[date].dropna()
            r = fwd_cum.loc[date].reindex(s.index).dropna()
            common = s.index.intersection(r.index)
            if len(common) < 5:
                ic_list.append(np.nan)
                continue
            ic, _p = spearmanr(s[common], r[common])
            ic_list.append(ic)
        ic_raw      = pd.Series(ic_list, index=scores.index)
        ic_realised = ic_raw.shift(holding_weeks)
        ic_roll     = ic_realised.rolling(self.ic_window).mean()

        # --- Combine ---
        idx = scores.index.union(market_returns.index)
        dd_al  = mkt_dd.reindex(idx).ffill()
        mv_al  = mom_vol_rank.reindex(idx).ffill()
        ic_al  = ic_roll.reindex(idx).ffill()

        flag_dd = (dd_al < self.dd_threshold).astype(int)
        flag_mv = (mv_al > self.mom_vol_percentile).astype(int)
        flag_ic = (ic_al < 0).astype(int)

        n_flags  = flag_dd + flag_mv + flag_ic
        exposure = pd.Series(self.halve_factor, index=idx).pow(n_flags)
        exposure.name = f"exposure_{self.name}"

        # Use previous week's context to decide this week's exposure.
        return exposure.shift(1)


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

    from signals.momentum import MomentumSignal

    closes_weekly = pd.read_csv(
        "data/processed/closes_weekly.csv",
        index_col=0, parse_dates=True
    )
    market_ret = pd.read_csv(
        "data/processed/market_returns_weekly.csv",
        index_col=0, parse_dates=True
    ).squeeze()

    scores = MomentumSignal().compute(closes_weekly)
    filt   = RulesBasedFilter()
    exp    = filt.compute(
        scores=scores, closes_weekly=closes_weekly,
        market_returns=market_ret,
    )

    print(f"Filter: {filt.name}")
    print(f"\nExposure distribution:")
    print(exp.value_counts(dropna=False).sort_index())
    print(f"\nMean exposure     : {exp.mean():.3f}")
    print(f"Fraction < 1.0    : {(exp < 1.0).mean():.1%}")
    print(f"Fraction ≤ 0.25   : {(exp <= 0.25).mean():.1%}")
    print(f"\nLast 8 weeks:")
    print(exp.dropna().tail(8).round(3))
