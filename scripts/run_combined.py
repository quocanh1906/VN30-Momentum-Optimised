"""
run_combined.py — End-to-end comparison of signal strategies.

Architecture (3 pluggable layers):

    Signal          (src/signals/*.py)        — BaseSignal subclass
    Combination     (Combined / AndGate / …)  — also a BaseSignal
    Exposure filter (src/filters/*.py)        — ExposureFilter subclass

Each backtest is described by a StrategyConfig. To add a new variant,
append a row to the `strategies` list at the bottom of main() — no
other file changes required.

Outputs:
    output/combined_strategy_equity.png  — daily equity vs VN30
    output/combined_strategy_stats.csv   — headline metrics per strategy
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional

from signals.base            import BaseSignal
from signals.momentum        import MomentumSignal
from signals.ma_crossover    import MACrossoverSignal
from signals.bollinger       import BollingerBandSignal
from signals.rsi             import RSISignal
from signals.distance_52w_high import Distance52WHighSignal
from signals.combined        import CombinedSignal
from signals.and_gate        import AndGateSignal

from filters.base  import ExposureFilter
from filters.rules import RulesBasedFilter

from vol_scaling  import compute_weights
from portfolio    import compute_portfolio_returns
from performance  import plot_equity_curves, summary_stats
from data         import MARKET_ETF  # single source of truth for benchmark


# ── Global assumptions (edit here if cost model changes) ──────────
TXN_COST_RATE = 0.0015   # 0.15% flat, symmetric buy/sell


@dataclass
class StrategyConfig:
    """Declarative description of a single backtest run."""
    name: str
    signal: BaseSignal
    sizing: str = "vol_adjusted"
    top_pct: float = 0.33
    filter: Optional[ExposureFilter] = None
    color: str = "C0"
    linestyle: str = "-"


def load_data():
    closes_weekly = pd.read_csv(
        "data/processed/closes_weekly.csv",
        index_col=0, parse_dates=True
    )
    try:
        opens_weekly = pd.read_csv(
            "data/processed/opens_weekly.csv",
            index_col=0, parse_dates=True
        )
    except FileNotFoundError:
        opens_weekly = None

    closes_daily = pd.read_csv(
        "data/processed/closes_daily.csv",
        index_col=0, parse_dates=True
    )
    market_vol_weekly = pd.read_csv(
        "data/processed/vol_ewma_weekly.csv",
        index_col=0, parse_dates=True
    ).squeeze()
    market_returns_weekly = pd.read_csv(
        "data/processed/market_returns_weekly.csv",
        index_col=0, parse_dates=True
    ).squeeze()

    return {
        "closes_weekly"        : closes_weekly,
        "opens_weekly"         : opens_weekly,
        "closes_daily"         : closes_daily,
        "market_vol_weekly"    : market_vol_weekly,
        "market_returns_weekly": market_returns_weekly,
    }


def weekly_weights_to_daily(weekly_weights, daily_index):
    """Forward-fill weekly weights onto daily index; shift 1 day so
    weights are known strictly before the daily return is realised."""
    daily = weekly_weights.reindex(daily_index, method="ffill")
    return daily.shift(1).fillna(0)


def weekly_costs_to_daily(scaled_w, daily_index, cost_rate):
    """
    Convert weekly turnover into a daily cost series.

    At each weekly rebalance (Friday t), the full weekly cost
    (turnover × cost_rate, applied symmetrically) is charged to
    the first daily bar strictly after t — i.e. Monday t+1 open,
    matching portfolio.py's execution model.
    """
    w        = scaled_w.fillna(0)
    turnover = (w - w.shift(1).fillna(0)).abs().sum(axis=1)
    weekly_c = turnover * cost_rate

    daily_cost = pd.Series(0.0, index=daily_index)
    daily_arr  = daily_index.values
    for t, c in weekly_c.items():
        if c == 0:
            continue
        mask = daily_arr > np.datetime64(t)
        if mask.any():
            first_next = daily_index[mask][0]
            daily_cost.loc[first_next] += c
    return daily_cost


def run_strategy(cfg, bundle):
    # 1. signal
    scores = cfg.signal.compute(bundle["closes_weekly"])

    # 2. cross-sectional sizing + portfolio-level vol scale
    _, scaled_w, _ = compute_weights(
        scores, bundle["closes_weekly"], bundle["market_vol_weekly"],
        sizing  = cfg.sizing,
        top_pct = cfg.top_pct,
    )

    # 3. exposure filter (multiplicative on top of vol scale)
    exposure = None
    if cfg.filter is not None:
        exposure = cfg.filter.compute(
            scores         = scores,
            closes_weekly  = bundle["closes_weekly"],
            market_returns = bundle["market_returns_weekly"],
        )
        exp_al   = exposure.reindex(scaled_w.index).fillna(1.0)
        scaled_w = scaled_w.mul(exp_al, axis=0)

    # 4. weekly net returns (incl. transaction costs)
    weekly_net, _, _ = compute_portfolio_returns(
        scaled_w,
        bundle["closes_weekly"],
        bundle.get("opens_weekly"),
        cost_rate_buy  = TXN_COST_RATE,
        cost_rate_sell = TXN_COST_RATE,
    )

    # 5. daily PnL via forward-filled weights (gross)
    daily_ret     = bundle["closes_daily"].pct_change()
    weights_daily = weekly_weights_to_daily(
        scaled_w.fillna(0), daily_ret.index
    )
    common_cols   = weights_daily.columns.intersection(daily_ret.columns)
    pnl_gross     = (weights_daily[common_cols] *
                     daily_ret[common_cols]).sum(axis=1)

    # Deduct weekly turnover cost on each rebalance's Monday open.
    daily_cost    = weekly_costs_to_daily(
        scaled_w, daily_ret.index, TXN_COST_RATE
    )
    pnl_daily     = pnl_gross - daily_cost
    equity_daily  = (1 + pnl_daily).cumprod()

    return {
        "name"         : cfg.name,
        "weekly_net"   : weekly_net,
        "daily_pnl"    : pnl_daily,
        "equity_daily" : equity_daily,
        "exposure"     : exposure,
        "color"        : cfg.color,
        "linestyle"    : cfg.linestyle,
    }


def benchmark_daily_equity(closes_daily, market_etf=MARKET_ETF):
    """
    Default benchmark is the VN30 tracker ETF defined by MARKET_ETF
    in src/data.py. Falls back to equal-weighted cross-section only
    if the ETF is missing from the dataset.
    """
    if market_etf in closes_daily.columns:
        px = closes_daily[market_etf].ffill()
    else:
        print(f"  ⚠ {market_etf} not found — "
              f"falling back to equal-weighted cross-section")
        px = closes_daily.mean(axis=1)
    ret = px.pct_change().fillna(0)
    return (1 + ret).cumprod()


def main():
    bundle = load_data()

    # --- base signals (instantiate once, reuse in combinations) ---
    mom  = MomentumSignal(formation_weeks=26, skip_weeks=1)
    ma   = MACrossoverSignal(fast_weeks=10, slow_weeks=40)
    boll = BollingerBandSignal(window=20, k=2.0)
    rsi  = RSISignal(window=14)
    d52  = Distance52WHighSignal(window_weeks=52)

    # --- exposure filter ---
    rules = RulesBasedFilter()

    # --- strategy catalogue (add rows below to extend) ---
    strategies = [
        StrategyConfig(
            "Momentum only",
            signal    = mom,
            filter    = None,
            color     = "gray",
            linestyle = "--",
        ),
        StrategyConfig(
            "MA crossover only",
            signal    = ma,
            filter    = None,
            color     = "C3",
            linestyle = ":",
        ),
        StrategyConfig(
            "Combined: Mom + MA (additive 0.6/0.4) + rules",
            signal = CombinedSignal([(mom, 0.6), (ma, 0.4)]),
            filter = rules,
            color  = "C0",
        ),
        StrategyConfig(
            "Combined: Mom + MA (AND-gate) + rules",
            signal = AndGateSignal([mom, ma]),
            filter = rules,
            color  = "C2",
        ),
        StrategyConfig(
            "4-way additive (Mom/MA/Bollinger/RSI) + rules",
            signal = CombinedSignal([
                (mom, 0.4), (ma, 0.2),
                (boll, 0.2), (rsi, 0.2),
            ]),
            filter = rules,
            color  = "C4",
        ),
        StrategyConfig(
            "5-way additive (+52w high) + rules",
            signal = CombinedSignal([
                (mom, 0.35), (ma, 0.2), (boll, 0.15),
                (rsi, 0.15), (d52, 0.15),
            ]),
            filter = rules,
            color  = "C1",
        ),
    ]

    results    = {}
    stats_rows = []
    for cfg in strategies:
        print(f"Running: {cfg.name}")
        r = run_strategy(cfg, bundle)
        results[cfg.name] = r

        stats = summary_stats(r["daily_pnl"])
        stats_rows.append({"name": cfg.name, **stats})

        final_eq = r["equity_daily"].dropna().iloc[-1]
        print(f"  final_equity={final_eq:.3f}  "
              f"sharpe={stats.get('sharpe')}  "
              f"ann_ret={stats.get('ann_return')}  "
              f"max_dd={stats.get('max_dd')}")

    # Benchmark (default = MARKET_ETF from src/data.py)
    bench_equity = benchmark_daily_equity(bundle["closes_daily"])
    bench_ret    = bench_equity.pct_change()
    bench_stats  = summary_stats(bench_ret)
    bench_label  = f"Benchmark ({MARKET_ETF})"
    stats_rows.append({"name": bench_label, **bench_stats})

    # Save outputs
    os.makedirs("output", exist_ok=True)
    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv("output/combined_strategy_stats.csv", index=False)

    save_path = "output/combined_strategy_equity.png"
    plot_equity_curves(
        results,
        benchmark       = bench_equity,
        benchmark_label = bench_label,
        save_path       = save_path,
        title           = f"VN30 Momentum Strategies — Daily Equity vs {MARKET_ETF}",
    )
    print(f"\nChart  : {save_path}")
    print(f"Stats  : output/combined_strategy_stats.csv\n")
    print(stats_df.to_string(index=False))


if __name__ == "__main__":
    main()
