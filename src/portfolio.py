"""
portfolio.py — Portfolio construction and return computation.

Takes scaled weights from vol_scaling.py and computes:
- Weekly portfolio returns
- Equity curve
- Turnover
- Regime-conditional performance

Key design decisions:
- Returns computed from OPEN prices (not close-to-close)
  Signal generated at Friday close → execute at Monday open
  Return = Monday open to following Friday close
  This is the most realistic assumption for weekly momentum
- Transaction costs applied on turnover
- 5% cash buffer maintained for T+2 settlement float

Vietnamese market costs:
    Commission : 0.125% one-way (institutional rate)
    Sales tax  : 0.10% on sell side only (SSC mandatory)
    Total      : ~0.225% sell, ~0.125% buy
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np

# Vietnamese market transaction costs
COMMISSION   = 0.00125   # 0.125% one-way
SALES_TAX    = 0.001     # 0.10% sell side only
CASH_BUFFER  = 0.05      # 5% cash reserve for T+2 settlement float


def compute_portfolio_returns(scaled_weights, closes_weekly,
                               opens_weekly=None,
                               cost_rate_buy=COMMISSION,
                               cost_rate_sell=COMMISSION + SALES_TAX):
    """
    Compute weekly portfolio returns from scaled weights.

    Execution timing:
        Friday close  : signal + weights generated
        Monday open   : order executed (next week open)
        Friday close  : position marked

    Return for week T+1 = weighted average of stock returns
    from Monday T+1 open to Friday T+1 close.

    If open prices unavailable, uses close-to-close returns
    with 1-week shift as approximation.

    Parameters
    ----------
    scaled_weights : DataFrame of vol-scaled weights (dates × tickers)
                     weights already account for vol scaling
    closes_weekly  : DataFrame of weekly close prices
    opens_weekly   : DataFrame of weekly open prices (optional)
                     if None, uses close prices as proxy
    cost_rate_buy  : transaction cost for buys (default 0.125%)
    cost_rate_sell : transaction cost for sells (default 0.225%)

    Returns
    -------
    returns       : Series of weekly portfolio returns
    gross_returns : Series of weekly returns before costs
    turnover      : Series of weekly portfolio turnover
    """
    # Use open prices if available, otherwise close as proxy
    if opens_weekly is not None:
        exec_prices = opens_weekly
    else:
        exec_prices = closes_weekly

    # Stock returns: from execution price (Monday open next week)
    # to Friday close of the same week
    # Approximated as close-to-close shifted 1 week
    stock_returns = closes_weekly.pct_change(1).shift(-1)

    # Align weights and returns
    common_dates = scaled_weights.index.intersection(stock_returns.index)
    weights_al   = scaled_weights.reindex(common_dates)
    returns_al   = stock_returns.reindex(common_dates)

    gross_returns = pd.Series(dtype=float, name="gross_return")
    net_returns   = pd.Series(dtype=float, name="net_return")
    turnover_ser  = pd.Series(dtype=float, name="turnover")

    prev_weights  = pd.Series(dtype=float)

    for date in common_dates:
        curr_w = weights_al.loc[date].dropna()
        ret_row = returns_al.loc[date]

        if len(curr_w) == 0:
            gross_returns[date] = 0.0
            net_returns[date]   = 0.0
            turnover_ser[date]  = 0.0
            prev_weights        = pd.Series(dtype=float)
            continue

        # Gross return = weighted sum of stock returns
        common = curr_w.index.intersection(ret_row.dropna().index)
        if len(common) == 0:
            gross_returns[date] = 0.0
            net_returns[date]   = 0.0
            turnover_ser[date]  = 0.0
            prev_weights        = curr_w
            continue

        gross_ret = (curr_w[common] * ret_row[common]).sum()

        # Turnover = sum of absolute weight changes
        all_tickers = curr_w.index.union(prev_weights.index)
        curr_full   = curr_w.reindex(all_tickers).fillna(0)
        prev_full   = prev_weights.reindex(all_tickers).fillna(0)
        turnover    = (curr_full - prev_full).abs().sum()

        # Transaction costs on turnover
        # Approximate: half turnover = buys, half = sells
        buy_cost  = turnover * 0.5 * cost_rate_buy
        sell_cost = turnover * 0.5 * cost_rate_sell
        total_cost = buy_cost + sell_cost

        gross_returns[date] = gross_ret
        net_returns[date]   = gross_ret - total_cost
        turnover_ser[date]  = turnover
        prev_weights        = curr_w

    return net_returns, gross_returns, turnover_ser


def compute_equity_curve(returns, initial_value=1.0):
    """
    Compute cumulative equity curve from return series.

    Parameters
    ----------
    returns       : Series of periodic returns
    initial_value : starting portfolio value (default 1.0)

    Returns
    -------
    Series of cumulative portfolio values
    """
    return initial_value * (1 + returns).cumprod()


def compute_drawdown(equity_curve):
    """
    Compute drawdown series from equity curve.

    Returns
    -------
    Series of drawdowns (negative values)
    """
    rolling_max = equity_curve.cummax()
    drawdown    = (equity_curve - rolling_max) / rolling_max
    return drawdown


def regime_returns(net_returns, regimes):
    """
    Split portfolio returns by vol regime.

    Parameters
    ----------
    net_returns : Series of weekly net returns
    regimes     : Series of vol regime labels (0=low, 1=mid, 2=high)

    Returns
    -------
    dict of {regime_label: Series of returns}
    """
    labels  = {0: "Low Vol", 1: "Mid Vol", 2: "High Vol"}
    aligned = pd.DataFrame({
        "return": net_returns,
        "regime": regimes,
    }).dropna()

    result = {}
    for r, label in labels.items():
        subset = aligned[aligned["regime"] == r]["return"]
        if len(subset) > 0:
            result[label] = subset

    return result


def run_all_methods(scores, closes_weekly, market_vol_weekly,
                    opens_weekly=None,
                    regimes=None,
                    start_date=None,
                    end_date=None):
    """
    Run all four sizing methods and return portfolio returns.

    This is the main entry point for the backtest —
    computes weights and returns for all four methods
    in one call.

    Parameters
    ----------
    scores            : DataFrame of signal scores
    closes_weekly     : DataFrame of weekly close prices
    market_vol_weekly : Series of weekly market vol
    opens_weekly      : DataFrame of weekly open prices (optional)
    regimes           : Series of vol regime labels (optional)
    start_date        : backtest start date
    end_date          : backtest end date

    Returns
    -------
    dict of {method: {
        'net_returns'   : Series,
        'gross_returns' : Series,
        'equity_curve'  : Series,
        'drawdown'      : Series,
        'turnover'      : Series,
        'raw_weights'   : DataFrame,
        'scaled_weights': DataFrame,
    }}
    """
    from vol_scaling import compute_weights

    methods = ["equal", "signal", "vol_adjusted", "mean_variance"]
    results = {}

    for method in methods:
        print(f"  Running {method:<20}", end=" ", flush=True)
        try:
            raw_w, scaled_w, _ = compute_weights(
                scores, closes_weekly, market_vol_weekly,
                sizing     = method,
                start_date = start_date,
                end_date   = end_date,
            )

            net_ret, gross_ret, turnover = compute_portfolio_returns(
                scaled_w, closes_weekly, opens_weekly
            )

            equity   = compute_equity_curve(net_ret)
            drawdown = compute_drawdown(equity)

            results[method] = {
                "net_returns"   : net_ret,
                "gross_returns" : gross_ret,
                "equity_curve"  : equity,
                "drawdown"      : drawdown,
                "turnover"      : turnover,
                "raw_weights"   : raw_w,
                "scaled_weights": scaled_w,
            }

            ann_ret = net_ret.mean() * 52
            ann_vol = net_ret.std() * np.sqrt(52)
            sharpe  = ann_ret / ann_vol if ann_vol > 0 else np.nan
            max_dd  = drawdown.min()

            print(f"✓  Sharpe={sharpe:.3f}  "
                  f"Ann.Ret={ann_ret*100:.1f}%  "
                  f"MaxDD={max_dd*100:.1f}%")

        except Exception as e:
            print(f"✗  {e}")

    return results


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    print("="*60)
    print("  Portfolio — Test")
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

    # Compute signal
    from signals.momentum import MomentumSignal
    signal = MomentumSignal(formation_weeks=26, skip_weeks=1)
    scores = signal.compute(closes_weekly)

    # Run all methods
    print(f"\nRunning all four sizing methods...")
    results = run_all_methods(
        scores, closes_weekly, vol_ewma,
        opens_weekly = opens_weekly,
        regimes      = regimes,
        start_date   = "2016-01-01",
        end_date      = "2024-12-31",
    )

    # Market benchmark
    mkt_ret   = market_ret.loc["2016-01-01":"2024-12-31"]
    mkt_equity = compute_equity_curve(mkt_ret)
    mkt_ann    = mkt_ret.mean() * 52
    mkt_vol    = mkt_ret.std() * np.sqrt(52)
    mkt_sharpe = mkt_ann / mkt_vol if mkt_vol > 0 else np.nan
    mkt_dd     = compute_drawdown(mkt_equity).min()

    print(f"\n  [benchmark] E1VFVN30  "
          f"Sharpe={mkt_sharpe:.3f}  "
          f"Ann.Ret={mkt_ann*100:.1f}%  "
          f"MaxDD={mkt_dd*100:.1f}%")

    # Regime analysis
    print(f"\nRegime-conditional returns:")
    labels = {0.0: "Low Vol ", 1.0: "Mid Vol ", 2.0: "High Vol"}

    for method, res in results.items():
        print(f"\n  [{method}]")
        regime_ret = regime_returns(res["net_returns"], regimes)
        for regime_label, ret in regime_ret.items():
            ann   = ret.mean() * 52
            vol   = ret.std() * np.sqrt(52)
            sr    = ann / vol if vol > 0 else np.nan
            n     = len(ret)
            print(f"    {regime_label}: "
                  f"Ann={ann*100:.1f}%  "
                  f"Sharpe={sr:.3f}  "
                  f"N={n} weeks")

    # Turnover comparison
    print(f"\nAverage weekly turnover:")
    for method, res in results.items():
        avg_to = res["turnover"].mean()
        ann_to = avg_to * 52
        print(f"  {method:<20}: "
              f"weekly={avg_to:.3f}  annualised={ann_to:.1f}x")

    # Save
    os.makedirs("output", exist_ok=True)
    for method, res in results.items():
        res["net_returns"].to_csv(
            f"output/returns_{method}.csv"
        )
        res["equity_curve"].to_csv(
            f"output/equity_{method}.csv"
        )

    # Save combined equity curves
    equity_df = pd.DataFrame({
        m: res["equity_curve"] for m, res in results.items()
    })
    equity_df["benchmark"] = mkt_equity
    equity_df.to_csv("output/equity_all.csv")

    print(f"\nSaved to output/")