"""
vol_scaling.py — Volatility scaling and position sizing.

Implements the full Citadel-style position sizing pipeline:

Step 1 — Cross-sectional sizing (within portfolio):
    A. Equal weight        — baseline
    B. Signal weight       — proportional to momentum z-score
    C. Vol-adjusted signal — score / stock_vol (Citadel-style)
    D. Mean-variance       — optimised via covariance matrix

Step 2 — Portfolio-level vol scaling (time-series):
    Scale total exposure by market volatility.
    When vol is high → reduce exposure automatically.
    When vol is low  → increase exposure (up to max_leverage).
    Keeps portfolio risk constant across regimes.

All weights shifted 1 period before use — no lookahead bias.

References:
    Barroso & Santa-Clara (2015) — Momentum has its moments
    Daniel & Moskowitz (2016)    — Momentum crashes
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# ── Individual stock volatility ──────────────────────────────────────────────────

def compute_stock_vol(closes_weekly, window=26, freq=52):
    """
    Compute annualised individual stock volatility from weekly returns.
    Shifted 1 week — no lookahead bias.

    Parameters
    ----------
    closes_weekly : DataFrame of weekly close prices (dates × tickers)
    window        : rolling window in weeks (default 26 = 6 months)
    freq          : annualisation factor (default 52)

    Returns
    -------
    DataFrame of annualised vol (dates × tickers), shifted 1 week
    """
    returns = closes_weekly.pct_change()
    # Tolerate occasional missing weeks (e.g. Tết) — otherwise one
    # NaN pollutes the full `window` weeks of rolling vol.
    min_obs = max(int(window * 0.8), 13)
    vol     = returns.rolling(window, min_periods=min_obs).std() * np.sqrt(freq)
    return vol.shift(1)


# ── Portfolio-level vol scale factor ─────────────────────────────────────────────

def vol_scale_factor(market_vol_weekly, target_vol=0.15,
                      max_leverage=1.0):
    """
    Compute portfolio-level volatility scaling factor.

        scale_t = min(target_vol / market_vol_t, max_leverage)

    Market vol is already shifted 1 day in data.py.
    No additional shift needed here.

    Parameters
    ----------
    market_vol_weekly : Series of weekly market vol (pre-shifted)
    target_vol        : annualised target portfolio vol (default 15%)
    max_leverage      : cap on scale factor (default 1.0 = no leverage)

    Returns
    -------
    Series of scale factors clipped to [0, max_leverage]
    """
    scale      = (target_vol / market_vol_weekly).clip(0, max_leverage)
    scale.name = "vol_scale"
    return scale


# ── Cross-sectional sizing methods ───────────────────────────────────────────────

def _select_top_tercile(scores_row, eligible, top_pct=0.33):
    """
    Select top tercile of eligible stocks by momentum score.
    Returns Series of scores for long stocks only.
    Internal helper used by all sizing methods.
    """
    valid = scores_row[
        [t for t in eligible if t in scores_row.index]
    ].dropna()

    if len(valid) < 5:
        return pd.Series(dtype=float)

    threshold = valid.quantile(1 - top_pct)
    long      = valid[valid >= threshold]
    return long


def equal_weights(scores_row, eligible, top_pct=0.33):
    """
    Method A — Equal weight across top tercile.
    Baseline for comparison against all other methods.

    Returns
    -------
    Series of weights summing to 1.0
    """
    long = _select_top_tercile(scores_row, eligible, top_pct)
    if len(long) == 0:
        return pd.Series(dtype=float)
    return pd.Series(1 / len(long), index=long.index)


def signal_weights(scores_row, eligible, top_pct=0.33):
    """
    Method B — Weight proportional to momentum z-score.
    Higher score = larger position.
    Only positive scores receive weight — if all negative,
    falls back to equal weight.

    Returns
    -------
    Series of weights summing to 1.0
    """
    long = _select_top_tercile(scores_row, eligible, top_pct)
    if len(long) == 0:
        return pd.Series(dtype=float)

    pos = long.clip(lower=0)
    if pos.sum() <= 0:
        return equal_weights(scores_row, eligible, top_pct)

    return pos / pos.sum()


def vol_adjusted_signal_weights(scores_row, stock_vols_row,
                                 eligible, top_pct=0.33):
    """
    Method C — Vol-adjusted signal weight (Citadel-style).

        weight_i = score_i / vol_i,  then normalise

    Key insight: the same momentum signal in a high-vol stock
    represents more risk than in a low-vol stock. Dividing by
    vol ensures equal risk contribution per unit of signal.

    Falls back to signal_weights if vol data unavailable.

    Returns
    -------
    Series of weights summing to 1.0
    """
    long = _select_top_tercile(scores_row, eligible, top_pct)
    if len(long) == 0:
        return pd.Series(dtype=float)

    # Get vol for long stocks — only those with valid vol data
    vols = stock_vols_row.reindex(long.index).dropna()
    vols = vols.clip(lower=0.01)  # floor to avoid division by zero

    if len(vols) == 0:
        return signal_weights(scores_row, eligible, top_pct)

    common  = long.index.intersection(vols.index)
    raw     = long[common].clip(lower=0) / vols[common]

    if raw.sum() <= 0:
        return signal_weights(scores_row, eligible, top_pct)

    return raw / raw.sum()


def mean_variance_weights(scores_row, weekly_returns, eligible,
                           top_pct=0.33,
                           risk_aversion=1.0):
    """
    Method D — Mean-variance optimised weights.

    Solves: w* = (1/λ) × Σ⁻¹ × μ

    Where:
        μ = momentum scores (expected return proxy)
        Σ = covariance matrix (Ledoit-Wolf shrinkage)
        λ = risk aversion

    Requires ≥80% non-NaN observations per stock in the return
    window (LedoitWolf does not accept NaN). Falls back to
    signal_weights if insufficient data or the optimisation fails.

    Note: only reliable from 2018 onwards when most VN30
    stocks have sufficient history for stable covariance.

    Returns
    -------
    Series of weights summing to 1.0
    """
    long = _select_top_tercile(scores_row, eligible, top_pct)
    if len(long) == 0:
        return pd.Series(dtype=float)

    # Filter to stocks available in return window
    available = [t for t in long.index
                 if t in weekly_returns.columns]
    if len(available) < 3:
        return signal_weights(scores_row, eligible, top_pct)

    # Filter to stocks with sufficient history
    ret_sub = weekly_returns[available]
    # LedoitWolf requires no NaN — keep stocks with ≥80% obs,
    # then drop any rows that still contain NaN.
    min_obs = int(len(ret_sub) * 0.8)
    valid   = ret_sub.columns[ret_sub.notna().sum() >= min_obs].tolist()

    if len(valid) < 3:
        return signal_weights(scores_row, eligible, top_pct)

    ret_clean = ret_sub[valid].dropna()

    if len(ret_clean) < 13:  # ~3 months needed for stable cov
        return signal_weights(scores_row, eligible, top_pct)

    # Covariance matrix with Ledoit-Wolf shrinkage
    try:
        from sklearn.covariance import LedoitWolf
        lw    = LedoitWolf().fit(ret_clean.values)
        Sigma = lw.covariance_ * 52  # annualise
    except Exception:
        Sigma = ret_clean.cov().values * 52

    mu = long[valid].clip(lower=0).values

    if mu.sum() <= 0:
        return signal_weights(scores_row, eligible, top_pct)

    # Solve: w = (1/λ) × Σ⁻¹ × μ
    try:
        Sigma_inv = np.linalg.pinv(Sigma)
        w_raw     = (1 / risk_aversion) * Sigma_inv @ mu
        w         = pd.Series(w_raw, index=valid).clip(lower=0)

        if w.sum() <= 0:
            return signal_weights(scores_row, eligible, top_pct)

        return w / w.sum()

    except Exception:
        return signal_weights(scores_row, eligible, top_pct)


# ── Main pipeline ────────────────────────────────────────────────────────────────

def compute_weights(scores, closes_weekly, market_vol_weekly,
                    sizing="vol_adjusted",
                    target_vol=0.15,
                    max_leverage=1.0,
                    top_pct=0.33,
                    start_date=None,
                    end_date=None):
    """
    Full position sizing pipeline.

    For each week:
    1. Get point-in-time VN30 eligible universe
    2. Apply cross-sectional sizing method (A/B/C/D)
    3. Apply portfolio-level vol scaling
    4. Return both raw (unscaled) and scaled weights

    Parameters
    ----------
    scores            : DataFrame of signal scores (dates × tickers)
    closes_weekly     : DataFrame of weekly close prices
    market_vol_weekly : Series of weekly market vol (pre-shifted)
    sizing            : 'equal', 'signal', 'vol_adjusted', 'mean_variance'
    target_vol        : portfolio vol target (default 15%)
    max_leverage      : max total exposure (default 1.0 = no leverage)
    top_pct           : top fraction to go long (default 33%)
    start_date        : optional backtest start date
    end_date          : optional backtest end date

    Returns
    -------
    raw_weights    : DataFrame — within-portfolio weights (sum to 1)
    scaled_weights : DataFrame — vol-scaled weights (sum to scale)
    scale_factors  : Series — vol scale factor per week
    """
    from data import get_constituents, EXCLUDED

    # Precompute stock vols and weekly returns
    stock_vols     = compute_stock_vol(closes_weekly)
    weekly_returns = closes_weekly.pct_change()

    # Portfolio-level scale factors
    scale_factors = vol_scale_factor(
        market_vol_weekly, target_vol, max_leverage
    )

    # Filter date range
    dates = scores.index
    if start_date:
        dates = dates[dates >= pd.Timestamp(start_date)]
    if end_date:
        dates = dates[dates <= pd.Timestamp(end_date)]

    raw_dict    = {}
    scaled_dict = {}

    for date in dates:
        if date not in scores.index:
            continue

        # Point-in-time eligible universe (excludes ROS, VPL)
        eligible = [t for t in get_constituents(date)
                    if t in scores.columns
                    and t not in EXCLUDED]

        if len(eligible) < 5:
            continue

        scores_row = scores.loc[date]

        # Stock vols for this date
        vols_row = stock_vols.loc[date] \
                   if date in stock_vols.index \
                   else pd.Series(dtype=float)

        # Cross-sectional weights
        if sizing == "equal":
            w = equal_weights(scores_row, eligible, top_pct)

        elif sizing == "signal":
            w = signal_weights(scores_row, eligible, top_pct)

        elif sizing == "vol_adjusted":
            w = vol_adjusted_signal_weights(
                scores_row, vols_row, eligible, top_pct
            )

        elif sizing == "mean_variance":
            if date in weekly_returns.index:
                idx     = weekly_returns.index.get_loc(date)
                ret_win = weekly_returns.iloc[max(0, idx-52):idx]
                w = mean_variance_weights(
                    scores_row, ret_win, eligible, top_pct
                )
            else:
                w = pd.Series(dtype=float)

        else:
            raise ValueError(f"Unknown sizing method: {sizing}")

        if len(w) == 0:
            continue

        # Portfolio-level vol scale
        scale = scale_factors.loc[date] \
                if date in scale_factors.index else 1.0
        if pd.isna(scale):
            scale = 1.0

        raw_dict[date]    = w
        scaled_dict[date] = w * scale

    raw_weights    = pd.DataFrame(raw_dict).T
    scaled_weights = pd.DataFrame(scaled_dict).T

    raw_weights.index.name    = "date"
    scaled_weights.index.name = "date"

    return raw_weights, scaled_weights, scale_factors


def compare_sizing_methods(scores, closes_weekly,
                            market_vol_weekly,
                            start_date=None,
                            end_date=None):
    """
    Run all four sizing methods and return results for comparison.

    Mean-variance starts from 2018 to ensure sufficient covariance
    history — documented as a data requirement, not a limitation.

    Returns
    -------
    dict of {method: (raw_weights, scaled_weights, scale_factors)}
    """
    results = {}
    methods = {
        "equal"        : start_date,
        "signal"       : start_date,
        "vol_adjusted" : start_date,
        "mean_variance": "2018-01-01",  # needs covariance history
    }

    for method, s_date in methods.items():
        print(f"  Computing {method:<20}", end=" ", flush=True)
        try:
            raw, scaled, scale_f = compute_weights(
                scores, closes_weekly, market_vol_weekly,
                sizing     = method,
                start_date = s_date,
                end_date   = end_date,
            )
            results[method] = (raw, scaled, scale_f)
            n_weeks = scaled.notna().any(axis=1).sum()
            print(f"✓  {n_weeks} weeks")
        except Exception as e:
            print(f"✗  {e}")

    return results


# ── Diagnostics ──────────────────────────────────────────────────────────────────

def scale_factor_summary(scale_factors, regimes):
    """
    Summarise vol scale factors by regime.
    Shows how vol scaling automatically reduces exposure
    in high-vol environments.
    """
    df = pd.DataFrame({
        "scale" : scale_factors,
        "regime": regimes,
    }).dropna()

    labels = {0.0: "Low vol ", 1.0: "Mid vol ", 2.0: "High vol"}

    print(f"\n  Vol scale factor summary (target=15%):")
    print(f"  Overall: mean={scale_factors.mean():.3f}  "
          f"min={scale_factors.min():.3f}  "
          f"max={scale_factors.max():.3f}")
    print(f"\n  By vol regime:")
    for r, label in labels.items():
        subset = df[df["regime"] == r]["scale"]
        if len(subset) > 0:
            bar = "█" * int(subset.mean() * 20)
            print(f"    {label}: {bar:<20} "
                  f"mean={subset.mean():.3f}  "
                  f"min={subset.min():.3f}")


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    print("="*60)
    print("  Vol Scaling — Test")
    print("="*60)

    # Load data
    closes_weekly = pd.read_csv(
        "data/processed/closes_weekly.csv",
        index_col=0, parse_dates=True
    )
    vol_ewma = pd.read_csv(
        "data/processed/vol_ewma_weekly.csv",
        index_col=0, parse_dates=True
    ).squeeze()
    regimes = pd.read_csv(
        "data/processed/vol_regimes.csv",
        index_col=0, parse_dates=True
    ).squeeze()

    print(f"\nLoaded: {closes_weekly.shape} (weeks × tickers)")

    # Compute momentum signal
    from signals.momentum import MomentumSignal
    signal = MomentumSignal(formation_weeks=26, skip_weeks=1)
    scores = signal.compute(closes_weekly)
    print(f"Scores: {scores.notna().sum().sum()} non-NaN values")

    # Diagnostic — verify mean_variance is working
    from data import get_constituents
    weekly_returns = closes_weekly.pct_change()

    test_dates = [
        pd.Timestamp("2016-06-03"),
        pd.Timestamp("2019-01-04"),
        pd.Timestamp("2022-06-03"),
    ]

    for test_date in test_dates:
        eligible = get_constituents(test_date)
        eligible = [t for t in eligible if t in scores.columns]

        # Get nearest date in index
        nearest = closes_weekly.index[
            closes_weekly.index >= test_date
        ]
        if len(nearest) == 0:
            continue
        nearest = nearest[0]

        idx     = weekly_returns.index.get_loc(nearest)
        ret_win = weekly_returns.iloc[max(0, idx-52):idx]

        in_returns  = [t for t in eligible if t in ret_win.columns]
        missing     = [t for t in eligible if t not in ret_win.columns]
        has_data    = [t for t in in_returns
                    if ret_win[t].notna().sum() >= 26]
        insufficient= [t for t in in_returns
                    if ret_win[t].notna().sum() < 26]

        print(f"\n{test_date.date()}:")
        print(f"  Eligible         : {len(eligible)} stocks")
        print(f"  In returns       : {len(in_returns)} stocks")
        print(f"  Missing entirely : {missing}")
        print(f"  Sufficient data  : {len(has_data)} stocks")
        print(f"  Insufficient (<26): {insufficient}")
        print(f"  Cov matrix size  : {len(has_data)}×{len(has_data)}")
        print(f"  Will use MV?     : {len(has_data) >= 3}")

    # Vol scale factor summary
    scale = vol_scale_factor(vol_ewma, target_vol=0.15)
    scale_factor_summary(scale, regimes)

    # Compare all four sizing methods
    print(f"\nComparing all four sizing methods...")
    results = compare_sizing_methods(
        scores, closes_weekly, vol_ewma,
        start_date="2016-01-01",
    )

    # Show sample weights for latest date
    print(f"\nSample scaled weights (latest date):")
    for method in ["equal", "signal", "vol_adjusted", "mean_variance"]:
        if method not in results:
            continue
        _, scaled, _ = results[method]
        if scaled.empty:
            continue
        latest_date = scaled.dropna(how='all').index[-1]
        latest      = scaled.loc[latest_date].dropna()\
                            .sort_values(ascending=False)
        total_exp   = latest.sum()
        print(f"\n  [{method}] {latest_date.date()} "
              f"(exposure={total_exp:.3f}):")
        for ticker, w in latest.head(5).items():
            print(f"    {ticker}: {w:.4f}")

    # Save scale factors
    os.makedirs("output", exist_ok=True)
    scale.to_csv("output/vol_scale_factors.csv")
    print(f"\nSaved to output/vol_scale_factors.csv")