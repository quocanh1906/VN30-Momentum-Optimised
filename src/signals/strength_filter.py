"""
strength_filter.py — Signal strength classification.

Trains a logistic classifier to predict whether the momentum
signal will be profitable in the next holding period.

Used as a trade gate — only execute when classifier
predicts momentum will work (probability > threshold).

Features:
    - Cross-sectional signal spread (winner-loser gap)
    - Top tercile average signal strength
    - Rolling IC over past 12 weeks (realised signal quality)
    - IC positive fraction over past 12 weeks
    - Market trend (4-week market return)
    - Vol regime (0/1/2)
    - Drawdown from peak
    - Cross-sectional return dispersion
    - Vol trend (4-week change in EWMA vol)
    - Bear-market dummy           — Daniel & Moskowitz (2016)
    - Momentum portfolio vol      — Barroso & Santa-Clara (2015)
    - Bear × high-vol interaction — classical momentum-crash regime

All features constructed from data available at signal date.
No lookahead bias.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


def _momentum_long_short_returns(scores, weekly_returns, top_pct=0.33):
    """
    Historical long-short momentum portfolio returns (equal-weighted).

    At each date t, long the top tercile and short the bottom tercile
    using scores known at t-1; held for one week. The resulting series
    is used to estimate the momentum portfolio's own realised volatility
    (Barroso & Santa-Clara 2015).
    """
    scores_lag = scores.shift(1)
    mom_ret    = pd.Series(np.nan, index=weekly_returns.index,
                           name="mom_ls_return")

    for date in weekly_returns.index:
        if date not in scores_lag.index:
            continue
        s = scores_lag.loc[date].dropna()
        if len(s) < 5:
            continue

        top_thresh = s.quantile(1 - top_pct)
        bot_thresh = s.quantile(top_pct)
        top_idx    = s[s >= top_thresh].index
        bot_idx    = s[s <= bot_thresh].index

        r       = weekly_returns.loc[date]
        top_ret = r.reindex(top_idx).dropna()
        bot_ret = r.reindex(bot_idx).dropna()
        if len(top_ret) == 0 or len(bot_ret) == 0:
            continue

        mom_ret.loc[date] = top_ret.mean() - bot_ret.mean()

    return mom_ret


def build_features(scores, closes_weekly, market_returns,
                   regimes, holding_weeks=4):
    """
    Build feature matrix for signal strength classifier.

    All features are constructed from data available at
    signal date — no lookahead bias.

    Parameters
    ----------
    scores         : DataFrame of momentum z-scores
    closes_weekly  : DataFrame of weekly close prices
    market_returns : Series of weekly market returns
    regimes        : Series of vol regime labels
    holding_weeks  : forward return horizon for IC computation

    Returns
    -------
    features : DataFrame (dates × features)
    """
    features = pd.DataFrame(index=scores.index)

    # 1. Cross-sectional signal spread
    # Wide spread = clear momentum environment
    # Narrow spread = noisy, hard to distinguish winners/losers
    features["signal_spread"] = scores.apply(
        lambda row: row.dropna().quantile(0.67) -
                    row.dropna().quantile(0.33)
        if row.notna().sum() > 5 else np.nan,
        axis=1
    )

    # 2. Top tercile average signal strength
    features["top_avg_score"] = scores.apply(
        lambda row: row.dropna()[
            row.dropna() >= row.dropna().quantile(0.67)
        ].mean() if row.notna().sum() > 5 else np.nan,
        axis=1
    )

    # 3. Rolling IC — recent signal quality
    # How well has momentum predicted returns over past 12 weeks?
    weekly_returns = closes_weekly.pct_change()
    # Cumulative h-week forward return (not 1-week return h weeks out).
    fwd_cum_returns = closes_weekly.shift(-holding_weeks) / closes_weekly - 1

    ic_series = []
    for date in scores.index:
        if date not in fwd_cum_returns.index:
            ic_series.append(np.nan)
            continue
        s = scores.loc[date].dropna()
        r = fwd_cum_returns.loc[date].reindex(s.index).dropna()
        common = s.index.intersection(r.index)
        if len(common) < 5:
            ic_series.append(np.nan)
            continue
        from scipy.stats import spearmanr
        ic, _ = spearmanr(s[common], r[common])
        ic_series.append(ic)

    ic_raw = pd.Series(ic_series, index=scores.index)
    # ic_raw[t] correlates signal at t with returns over [t, t+h] — only
    # observable at t+h. Shift by holding_weeks so the rolling window
    # contains ICs whose forward returns have already realised.
    ic_realised = ic_raw.shift(holding_weeks)
    features["rolling_ic_12w"]  = ic_realised.rolling(12).mean()
    features["ic_positive_pct"] = (ic_realised > 0).rolling(12).mean()

    # 4. Market trend — 4-week market return
    features["market_trend_4w"] = market_returns.rolling(4).sum()

    # 5. Market vol level — already in vol regime
    features["vol_regime"] = regimes

    # 6. Drawdown from peak
    mkt_cum   = (1 + market_returns).cumprod()
    mkt_peak  = mkt_cum.cummax()
    mkt_dd    = (mkt_cum - mkt_peak) / mkt_peak
    features["market_drawdown"] = mkt_dd

    # 7. Cross-sectional return dispersion
    # High dispersion = stock-specific moves dominate
    # = good environment for stock selection
    features["cs_dispersion"] = weekly_returns.std(axis=1)

    # 8. Vol trend — is vol rising or falling?
    from data import compute_market_vol
    vol_ewma = compute_market_vol(market_returns, "ewma")
    vol_4w_ago = vol_ewma.shift(4)
    features["vol_trend"] = (vol_ewma - vol_4w_ago) / vol_4w_ago

    # 9. Bear-market dummy (Daniel-Moskowitz crash predictor)
    # Drawdown > 10% OR trailing 24-month cumulative return negative.
    mkt_24m_cum = mkt_cum / mkt_cum.shift(104) - 1
    bear        = ((mkt_dd < -0.10) | (mkt_24m_cum < 0)).astype(float)
    features["bear_dummy"] = bear

    # 10. Realised vol of the momentum portfolio itself
    # (Barroso & Santa-Clara 2015) — long top tercile, short bottom
    # tercile, using signal known at t-1. Momentum-specific, not market.
    mom_ret = _momentum_long_short_returns(scores, weekly_returns)
    features["mom_portfolio_vol"] = (
        mom_ret.rolling(26, min_periods=20).std() * np.sqrt(52)
    )

    # 11. Interaction: bear market × high-vol regime
    # This is the classical momentum-crash regime (Daniel-Moskowitz).
    features["bear_x_highvol"] = bear * (regimes >= 2).astype(float)

    return features.shift(1)  # shift 1 week — no lookahead


def build_target(scores, closes_weekly, holding_weeks=4,
                  min_return=0.0):
    """
    Build binary target: did momentum have cross-sectional skill?

    Target = 1 if (top-tercile h-week return − bottom-tercile h-week
                   return) > min_return
             0 otherwise

    Long-short removes the market-beta component — target=1 genuinely
    means the signal ranked stocks correctly, not just that the market
    rallied. Uses cumulative h-week returns (not 1-week return h weeks
    ahead).

    Parameters
    ----------
    scores        : DataFrame of momentum z-scores
    closes_weekly : DataFrame of weekly close prices
    holding_weeks : forward return horizon
    min_return    : minimum long-short return to classify as success

    Returns
    -------
    Series of binary labels (1 = signal worked, 0 = didn't)
    """
    # Cumulative h-week forward return per stock.
    fwd_cum = closes_weekly.shift(-holding_weeks) / closes_weekly - 1

    targets = []
    for date in scores.index:
        if date not in fwd_cum.index:
            targets.append(np.nan)
            continue

        row = scores.loc[date].dropna()
        if len(row) < 5:
            targets.append(np.nan)
            continue

        # Top and bottom terciles
        top_thresh = row.quantile(0.67)
        bot_thresh = row.quantile(0.33)
        top_idx    = row[row >= top_thresh].index
        bot_idx    = row[row <= bot_thresh].index

        fwd     = fwd_cum.loc[date]
        top_ret = fwd.reindex(top_idx).dropna()
        bot_ret = fwd.reindex(bot_idx).dropna()
        if len(top_ret) == 0 or len(bot_ret) == 0:
            targets.append(np.nan)
            continue

        long_short = top_ret.mean() - bot_ret.mean()
        targets.append(1 if long_short > min_return else 0)

    return pd.Series(targets, index=scores.index,
                     name="signal_worked")


class SignalStrengthClassifier:
    """
    Logistic classifier predicting whether momentum signal
    will be profitable in the next holding period.

    Trained on IS data, applied to OOS.
    Outputs probability in [0,1] — continuous trade gate.

    High probability → strong signal → full position
    Low probability  → weak signal  → reduce/skip position
    """

    def __init__(self, C=0.1, threshold=0.5):
        """
        Parameters
        ----------
        C         : regularisation strength (lower = more regularised)
        threshold : probability threshold for trade decision
        """
        self.C          = C
        self.threshold  = threshold
        self.model      = LogisticRegression(
            C=C, random_state=42, max_iter=1000
        )
        self.scaler     = StandardScaler()
        self.feature_names = None
        self.is_fitted  = False

    def fit(self, features, target):
        """
        Train classifier on IS data.

        Parameters
        ----------
        features : DataFrame of features (IS period only)
        target   : Series of binary labels (IS period only)
        """
        # Align and drop NaN
        df = pd.concat([features, target], axis=1).dropna()

        if len(df) < 52:
            print(f"  Warning: only {len(df)} training samples")
            return self

        X = df[features.columns].values
        y = df[target.name].values

        # Check class balance
        pos_rate = y.mean()
        if pos_rate < 0.1 or pos_rate > 0.9:
            print(f"  Warning: imbalanced classes "
                  f"({pos_rate:.1%} positive)")

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Fit classifier
        self.model.fit(X_scaled, y)
        self.feature_names = features.columns.tolist()
        self.is_fitted     = True

        # Feature importance
        coefs = pd.Series(
            self.model.coef_[0],
            index=self.feature_names
        ).sort_values(key=abs, ascending=False)

        train_acc = self.model.score(X_scaled, y)
        print(f"  Training accuracy: {train_acc:.3f}")
        print(f"  Class balance: {pos_rate:.1%} positive")
        print(f"  Top features:")
        for feat, coef in coefs.head(4).items():
            print(f"    {feat:<25}: {coef:+.3f}")

        return self

    def predict_proba(self, features):
        """
        Predict probability that signal will work.

        Parameters
        ----------
        features : DataFrame of features (OOS period)

        Returns
        -------
        Series of trade probabilities in [0,1]
        """
        if not self.is_fitted:
            # Return 1.0 (always trade) if not fitted
            return pd.Series(1.0, index=features.index)

        # Handle NaN — fill with median from training
        feat_clean = features[self.feature_names].copy()
        feat_clean = feat_clean.fillna(feat_clean.median())

        X_scaled = self.scaler.transform(feat_clean.values)
        proba    = self.model.predict_proba(X_scaled)[:, 1]

        return pd.Series(proba, index=features.index,
                         name="trade_probability")

    def evaluate(self, features, target):
        """
        Evaluate classifier on OOS data.

        Returns dict with accuracy, precision, recall, AUC.
        """
        from sklearn.metrics import (accuracy_score,
                                      precision_score,
                                      recall_score,
                                      roc_auc_score)

        df = pd.concat([features, target], axis=1).dropna()
        if len(df) < 10:
            return {}

        X = self.scaler.transform(
            df[self.feature_names].fillna(0).values
        )
        y = df[target.name].values
        y_pred = self.model.predict(X)
        y_prob = self.model.predict_proba(X)[:, 1]

        return {
            "accuracy" : round(accuracy_score(y, y_pred), 3),
            "precision": round(precision_score(y, y_pred,
                                               zero_division=0), 3),
            "recall"   : round(recall_score(y, y_pred,
                                            zero_division=0), 3),
            "auc"      : round(roc_auc_score(y, y_prob), 3),
            "n_samples": len(df),
        }


def apply_strength_filter(scaled_weights, trade_proba,
                           mode="scale"):
    """
    Apply signal strength filter to portfolio weights.

    Two modes:
    - 'scale'     : multiply weights by trade probability
                    continuous — partial positions when uncertain
    - 'threshold' : binary gate — trade if prob > 0.5, else cash

    Parameters
    ----------
    scaled_weights : DataFrame of vol-scaled weights
    trade_proba    : Series of trade probabilities
    mode           : 'scale' or 'threshold'

    Returns
    -------
    DataFrame of strength-filtered weights
    """
    filtered = scaled_weights.copy()

    for date in filtered.index:
        if date not in trade_proba.index:
            continue

        prob = trade_proba.loc[date]

        if pd.isna(prob):
            continue

        if mode == "scale":
            # Scale all weights by trade probability
            filtered.loc[date] *= prob

        elif mode == "threshold":
            # Binary: full position or zero
            if prob < 0.5:
                filtered.loc[date] = 0.0

    return filtered


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

    print("="*60)
    print("  Signal Strength Filter — Test")
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
    market_ret = pd.read_csv(
        "data/processed/market_returns_weekly.csv",
        index_col=0, parse_dates=True
    ).squeeze()

    # Compute signal
    from signals.momentum import MomentumSignal
    signal = MomentumSignal(formation_weeks=26, skip_weeks=1)
    scores = signal.compute(closes_weekly)

    # Build features and target
    print("\nBuilding features...")
    features = build_features(
        scores, closes_weekly, market_ret, regimes
    )
    target = build_target(scores, closes_weekly)

    print(f"  Features shape: {features.shape}")
    print(f"  Target shape  : {len(target.dropna())}")
    print(f"  Signal worked : "
          f"{target.mean():.1%} of weeks")

    # IS/OOS split
    is_end   = pd.Timestamp("2021-12-31")
    oos_start = pd.Timestamp("2022-01-01")

    feat_is  = features.loc[:is_end]
    feat_oos = features.loc[oos_start:]
    tgt_is   = target.loc[:is_end]
    tgt_oos  = target.loc[oos_start:]

    # Train classifier
    print(f"\nTraining classifier on IS data...")
    clf = SignalStrengthClassifier(C=0.1, threshold=0.5)
    clf.fit(feat_is, tgt_is)

    # OOS evaluation
    print(f"\nOOS classifier evaluation:")
    eval_metrics = clf.evaluate(feat_oos, tgt_oos)
    for k, v in eval_metrics.items():
        print(f"  {k:<15}: {v}")

    # Trade probabilities
    trade_prob = clf.predict_proba(features)

    print(f"\nTrade probability distribution:")
    print(f"  Mean   : {trade_prob.mean():.3f}")
    print(f"  Std    : {trade_prob.std():.3f}")
    print(f"  >0.6   : {(trade_prob > 0.6).mean():.1%} of weeks")
    print(f"  <0.4   : {(trade_prob < 0.4).mean():.1%} of weeks")

    # Show recent probabilities
    print(f"\nRecent trade probabilities:")
    recent = trade_prob.dropna().tail(8)
    for date, prob in recent.items():
        regime = regimes.get(date, np.nan)
        regime_label = {0:"Low",1:"Mid",2:"High"}.get(regime,"?")
        bar = "█" * int(prob * 20)
        print(f"  {date.date()} [{regime_label} vol]: "
              f"{bar:<20} {prob:.3f}")

    # Save
    os.makedirs("output", exist_ok=True)
    trade_prob.to_csv("output/trade_probabilities.csv")
    features.to_csv("output/signal_features.csv")
    print(f"\nSaved to output/")