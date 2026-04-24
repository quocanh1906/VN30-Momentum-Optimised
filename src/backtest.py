"""
backtest.py — Walk-forward validation engine.

Implements expanding window walk-forward validation:
    IS: all data from start to year Y-1
    OOS: year Y only
    Repeat for Y = 2019, 2020, 2021, 2022, 2023, 2024

For each IS window:
    1. Optimise formation period (J) on IS data
    2. Select best sizing method on IS data
    3. Apply to OOS year with fixed parameters

This eliminates lookahead bias in parameter selection —
parameters are chosen only from data available at that point.

Final result: stitched OOS equity curve 2019-2024
plus 2025-present as live forward test.

Lookahead bias prevention:
    - IS optimisation uses only data up to IS end date
    - OOS parameters fixed from IS — no re-optimisation
    - Vol scaling uses pre-shifted market vol (no lookahead)
    - Universe selection uses point-in-time constituents
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# ── IS parameter optimisation ────────────────────────────────────────────────────

def optimise_is(scores_dict, closes_weekly, market_vol_weekly,
                is_end_date, sizing="vol_adjusted"):
    """
    Select best formation period on IS data by Sharpe ratio.

    Tests formation periods J = [4, 8, 13, 26, 39, 52] weeks.
    Uses vol_adjusted sizing throughout for consistency.

    Parameters
    ----------
    scores_dict       : dict of {J: scores_DataFrame}
                        pre-computed scores for each J
    closes_weekly     : DataFrame of weekly close prices
    market_vol_weekly : Series of weekly market vol
    is_end_date       : pd.Timestamp — end of IS period
    sizing            : sizing method for IS optimisation

    Returns
    -------
    dict with best_J, best_sharpe, all_results
    """
    from vol_scaling import compute_weights
    from portfolio import compute_portfolio_returns, compute_equity_curve

    results = []

    for J, scores in scores_dict.items():
        try:
            _, scaled_w, _ = compute_weights(
                scores, closes_weekly, market_vol_weekly,
                sizing   = sizing,
                end_date = is_end_date,
            )

            net_ret, _, _ = compute_portfolio_returns(
                scaled_w, closes_weekly
            )

            # IS performance
            is_ret = net_ret.loc[:is_end_date].dropna()
            if len(is_ret) < 52:
                continue

            ann_ret = is_ret.mean() * 52
            ann_vol = is_ret.std() * np.sqrt(52)
            sharpe  = ann_ret / ann_vol if ann_vol > 0 else np.nan

            results.append({
                "J"      : J,
                "sharpe" : sharpe,
                "ann_ret": ann_ret,
                "ann_vol": ann_vol,
                "n_weeks": len(is_ret),
            })

        except Exception as e:
            continue

    if not results:
        return {"best_J": 26, "best_sharpe": np.nan, "all_results": []}

    results_df = pd.DataFrame(results).sort_values(
        "sharpe", ascending=False
    )
    best = results_df.iloc[0]

    return {
        "best_J"      : int(best["J"]),
        "best_sharpe" : round(best["sharpe"], 3),
        "all_results" : results_df,
    }


# ── Walk-forward engine ──────────────────────────────────────────────────────────

def walk_forward(closes_weekly, market_vol_weekly, regimes,
                 opens_weekly=None,
                 formation_periods=[4, 8, 13, 26, 39, 52],
                 oos_years=range(2019, 2025),
                 sizing_methods=["equal", "signal",
                                  "vol_adjusted", "mean_variance"],
                 optimise_J=True):
    """
    Expanding window walk-forward validation.

    For each OOS year:
        IS  = all data from 2015 to end of previous year
        OOS = that year only

    Parameters
    ----------
    closes_weekly      : DataFrame of weekly close prices
    market_vol_weekly  : Series of weekly market vol
    regimes            : Series of vol regime labels
    opens_weekly       : DataFrame of weekly open prices (optional)
    formation_periods  : list of J values to test in IS
    oos_years          : years to use as OOS test periods
    sizing_methods     : list of sizing methods to compare
    optimise_J         : if True, optimise J each IS window
                         if False, use J=26 throughout

    Returns
    -------
    oos_results : dict of {method: {year: metrics}}
    oos_equity  : dict of {method: full OOS equity curve}
    is_params   : dict of {year: best IS parameters}
    """
    from signals.momentum import MomentumSignal
    from vol_scaling import compute_weights
    from portfolio import (compute_portfolio_returns,
                            compute_equity_curve,
                            compute_drawdown,
                            regime_returns)

    print(f"\n{'='*60}")
    print(f"  Walk-Forward Validation")
    print(f"  OOS years: {list(oos_years)}")
    print(f"  Methods  : {sizing_methods}")
    print(f"{'='*60}\n")

    # Pre-compute scores for all formation periods
    print("Pre-computing momentum scores for all J values...")
    scores_dict = {}
    for J in formation_periods:
        signal = MomentumSignal(formation_weeks=J, skip_weeks=1)
        scores_dict[J] = signal.compute(closes_weekly)
        print(f"  J={J:>3} ✓")

    # Storage
    oos_returns_all = {m: [] for m in sizing_methods}
    is_params       = {}

    for oos_year in oos_years:
        is_end  = pd.Timestamp(f"{oos_year-1}-12-31")
        oos_start = pd.Timestamp(f"{oos_year}-01-01")
        oos_end   = pd.Timestamp(f"{oos_year}-12-31")

        print(f"\nIS: 2015–{oos_year-1}  →  OOS: {oos_year}")
        print(f"  IS end: {is_end.date()}  "
              f"OOS: {oos_start.date()} to {oos_end.date()}")

        # Optimise J on IS data
        if optimise_J:
            print(f"  Optimising J on IS data...")
            opt = optimise_is(
                scores_dict, closes_weekly, market_vol_weekly,
                is_end_date = is_end,
                sizing      = "vol_adjusted",
            )
            best_J = opt["best_J"]
            print(f"  Best J={best_J} "
                  f"(IS Sharpe={opt['best_sharpe']:.3f})")
        else:
            best_J = 26
            print(f"  Using fixed J={best_J}")

        is_params[oos_year] = {"J": best_J}

        # Run OOS for each sizing method
        oos_scores = scores_dict[best_J]

        for method in sizing_methods:
            try:
                _, scaled_w, _ = compute_weights(
                    oos_scores, closes_weekly, market_vol_weekly,
                    sizing     = method,
                    start_date = oos_start,
                    end_date   = oos_end,
                )

                net_ret, gross_ret, turnover = compute_portfolio_returns(
                    scaled_w, closes_weekly, opens_weekly
                )

                oos_ret = net_ret.loc[oos_start:oos_end].dropna()
                oos_returns_all[method].append(oos_ret)

            except Exception as e:
                print(f"  ✗ {method}: {e}")

    # Stitch OOS returns
    print(f"\nStitching OOS equity curves...")
    oos_equity  = {}
    oos_metrics = {}

    for method in sizing_methods:
        if not oos_returns_all[method]:
            continue

        # Concatenate all OOS years
        full_oos = pd.concat(oos_returns_all[method]).sort_index()
        full_oos = full_oos[~full_oos.index.duplicated(keep='first')]

        equity   = compute_equity_curve(full_oos)
        drawdown = compute_drawdown(equity)

        ann_ret  = full_oos.mean() * 52
        ann_vol  = full_oos.std() * np.sqrt(52)
        sharpe   = ann_ret / ann_vol if ann_vol > 0 else np.nan
        max_dd   = drawdown.min()
        calmar   = ann_ret / abs(max_dd) if max_dd != 0 else np.nan

        # Regime breakdown
        reg_ret  = regime_returns(full_oos, regimes)

        oos_equity[method]  = equity
        oos_metrics[method] = {
            "ann_return"  : round(ann_ret * 100, 2),
            "ann_vol"     : round(ann_vol * 100, 2),
            "sharpe"      : round(sharpe, 3),
            "max_dd"      : round(max_dd * 100, 2),
            "calmar"      : round(calmar, 3),
            "n_weeks"     : len(full_oos),
            "regime_ret"  : {
                k: round(v.mean() * 52 * 100, 2)
                for k, v in reg_ret.items()
            },
        }

        print(f"  {method:<20}: "
              f"Sharpe={sharpe:.3f}  "
              f"Ann.Ret={ann_ret*100:.1f}%  "
              f"MaxDD={max_dd*100:.1f}%")

    return oos_equity, oos_metrics, is_params


# ── Live forward test ────────────────────────────────────────────────────────────

def live_forward_test(closes_weekly, market_vol_weekly,
                       opens_weekly=None,
                       best_J=26,
                       sizing="vol_adjusted",
                       start_date="2025-01-01"):
    """
    Apply strategy to 2025-present as live forward test.

    Parameters fixed from full IS (2015-2024) optimisation.
    This is genuinely unseen data — no parameter touching.

    Parameters
    ----------
    closes_weekly     : DataFrame of weekly close prices
    market_vol_weekly : Series of weekly market vol
    opens_weekly      : DataFrame of weekly open prices (optional)
    best_J            : formation period from IS optimisation
    sizing            : sizing method from IS optimisation
    start_date        : start of live forward period

    Returns
    -------
    equity   : Series of cumulative returns
    net_ret  : Series of weekly returns
    metrics  : dict of performance metrics
    """
    from signals.momentum import MomentumSignal
    from vol_scaling import compute_weights
    from portfolio import (compute_portfolio_returns,
                            compute_equity_curve,
                            compute_drawdown)

    print(f"\n{'='*60}")
    print(f"  Live Forward Test ({start_date} → present)")
    print(f"  J={best_J}, sizing={sizing}")
    print(f"{'='*60}")

    signal = MomentumSignal(formation_weeks=best_J, skip_weeks=1)
    scores = signal.compute(closes_weekly)

    _, scaled_w, _ = compute_weights(
        scores, closes_weekly, market_vol_weekly,
        sizing     = sizing,
        start_date = start_date,
    )

    net_ret, gross_ret, turnover = compute_portfolio_returns(
        scaled_w, closes_weekly, opens_weekly
    )

    live_ret = net_ret.loc[start_date:].dropna()

    if len(live_ret) == 0:
        print(f"  No data available from {start_date}")
        return None, None, {}

    equity   = compute_equity_curve(live_ret)
    drawdown = compute_drawdown(equity)

    ann_ret  = live_ret.mean() * 52
    ann_vol  = live_ret.std() * np.sqrt(52)
    sharpe   = ann_ret / ann_vol if ann_vol > 0 else np.nan
    max_dd   = drawdown.min()
    total    = equity.iloc[-1] - 1

    metrics = {
        "period"      : f"{start_date} → {live_ret.index[-1].date()}",
        "n_weeks"     : len(live_ret),
        "total_return": round(total * 100, 2),
        "ann_return"  : round(ann_ret * 100, 2),
        "ann_vol"     : round(ann_vol * 100, 2),
        "sharpe"      : round(sharpe, 3),
        "max_dd"      : round(max_dd * 100, 2),
    }

    print(f"\n  Period    : {metrics['period']}")
    print(f"  N weeks   : {metrics['n_weeks']}")
    print(f"  Total ret : {metrics['total_return']}%")
    print(f"  Ann.Ret   : {metrics['ann_return']}%")
    print(f"  Sharpe    : {metrics['sharpe']}")
    print(f"  Max DD    : {metrics['max_dd']}%")

    return equity, live_ret, metrics


# ── IS full-sample benchmark ─────────────────────────────────────────────────────

def full_sample_backtest(closes_weekly, market_vol_weekly,
                          opens_weekly=None,
                          formation_weeks=26,
                          sizing_methods=["equal", "signal",
                                           "vol_adjusted",
                                           "mean_variance"],
                          start_date="2016-01-01",
                          end_date="2024-12-31"):
    """
    Full-sample backtest with fixed parameters.
    Used as IS benchmark before walk-forward.

    Returns
    -------
    results : dict from portfolio.run_all_methods
    """
    from signals.momentum import MomentumSignal
    from portfolio import run_all_methods

    signal = MomentumSignal(
        formation_weeks=formation_weeks, skip_weeks=1
    )
    scores = signal.compute(closes_weekly)

    print(f"\nFull-sample backtest "
          f"(J={formation_weeks}, {start_date}–{end_date}):")

    return run_all_methods(
        scores, closes_weekly, market_vol_weekly,
        opens_weekly = opens_weekly,
        start_date   = start_date,
        end_date     = end_date,
    )


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    print("="*60)
    print("  Backtest — Walk-Forward Validation")
    print("="*60)

    # Load data
    closes_weekly = pd.read_csv(
        "data/processed/closes_weekly.csv",
        index_col=0, parse_dates=True
    )
    opens_weekly = pd.read_csv(
        "data/processed/opens_weekly.csv",
        index_col=0, parse_dates=True
    ) if os.path.exists("data/processed/opens_weekly.csv") else None

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

    print(f"\nLoaded: {closes_weekly.shape} (weeks × tickers)")

    # ── Step 1: Full IS backtest (2016-2024) ────────────────────────
    print(f"\n{'─'*60}")
    print(f"Step 1 — Full-sample IS backtest (J=26, 2016-2024)")
    print(f"{'─'*60}")

    is_results = full_sample_backtest(
        closes_weekly, vol_ewma,
        opens_weekly    = opens_weekly,
        formation_weeks = 26,
        start_date      = "2016-01-01",
        end_date        = "2024-12-31",
    )

    # ── Step 2: Walk-forward OOS (2019-2024) ────────────────────────
    print(f"\n{'─'*60}")
    print(f"Step 2 — Walk-forward OOS validation (2019-2024)")
    print(f"{'─'*60}")

    oos_equity, oos_metrics, is_params = walk_forward(
        closes_weekly, vol_ewma, regimes,
        opens_weekly      = opens_weekly,
        formation_periods = [4, 8, 13, 26, 39, 52],
        oos_years         = range(2019, 2025),
        sizing_methods    = ["equal", "signal",
                              "vol_adjusted", "mean_variance"],
        optimise_J        = True,
    )

    # Print IS parameter choices
    print(f"\nIS parameter choices per OOS year:")
    for year, params in is_params.items():
        print(f"  {year}: J={params['J']}")

    # Print full OOS comparison
    print(f"\nFull OOS results (2019-2024 stitched):")
    print(f"  {'Method':<20} {'Sharpe':>8} {'Ann.Ret':>9} "
          f"{'MaxDD':>8} {'Calmar':>8}")
    print(f"  {'─'*55}")
    for method, m in oos_metrics.items():
        print(f"  {method:<20} {m['sharpe']:>8.3f} "
              f"{m['ann_return']:>8.1f}% "
              f"{m['max_dd']:>7.1f}% "
              f"{m['calmar']:>8.3f}")

    # Benchmark OOS
    mkt_oos  = market_ret.loc["2019-01-01":"2024-12-31"].dropna()
    mkt_ann  = mkt_oos.mean() * 52
    mkt_vol  = mkt_oos.std() * np.sqrt(52)
    mkt_sr   = mkt_ann / mkt_vol
    from portfolio import compute_equity_curve, compute_drawdown
    mkt_eq   = compute_equity_curve(mkt_oos)
    mkt_dd   = compute_drawdown(mkt_eq).min()
    print(f"  {'E1VFVN30':<20} {mkt_sr:>8.3f} "
          f"{mkt_ann*100:>8.1f}% "
          f"{mkt_dd*100:>7.1f}%")

    # OOS regime breakdown
    print(f"\nOOS regime-conditional returns:")
    for method, m in oos_metrics.items():
        print(f"\n  [{method}]")
        for regime, ret in m["regime_ret"].items():
            print(f"    {regime}: {ret:.1f}% ann.")

    # ── Step 3: Live forward test (2025-present) ────────────────────
    print(f"\n{'─'*60}")
    print(f"Step 3 — Live forward test (2025–present)")
    print(f"{'─'*60}")

    # Use best method from OOS
    best_method = max(
        oos_metrics, key=lambda m: oos_metrics[m]["sharpe"]
    )
    best_J = is_params.get(2024, {}).get("J", 26)

    print(f"  Best OOS method: {best_method} "
          f"(Sharpe={oos_metrics[best_method]['sharpe']:.3f})")

    live_equity, live_ret, live_metrics = live_forward_test(
        closes_weekly, vol_ewma,
        opens_weekly = opens_weekly,
        best_J       = best_J,
        sizing       = best_method,
        start_date   = "2025-01-01",
    )

    # ── Save outputs ────────────────────────────────────────────────
    os.makedirs("output", exist_ok=True)

    # OOS equity curves
    oos_eq_df = pd.DataFrame(oos_equity)
    oos_eq_df["benchmark"] = compute_equity_curve(
        market_ret.loc["2019-01-01":"2024-12-31"]
    )
    oos_eq_df.to_csv("output/equity_oos.csv")

    # OOS metrics
    metrics_rows = []
    for method, m in oos_metrics.items():
        row = {"method": method}
        row.update({k: v for k, v in m.items()
                    if k != "regime_ret"})
        metrics_rows.append(row)
    pd.DataFrame(metrics_rows).to_csv(
        "output/oos_metrics.csv", index=False
    )

    # Live forward
    if live_equity is not None:
        live_equity.to_csv("output/equity_live.csv")

    # IS params
    pd.DataFrame(is_params).T.to_csv(
        "output/is_params.csv"
    )

    print(f"\nAll outputs saved to output/")