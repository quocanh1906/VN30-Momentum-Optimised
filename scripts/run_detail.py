"""
run_detail.py — Detailed per-trade / per-day output for one strategy.

Inspection artefacts for the strategy configured in `make_strategy()`:

    output/detail/<slug>_positions_weekly.csv      — holdings per rebalance (long format)
    output/detail/<slug>_transactions.csv          — trade blotter (deltas + est. cost bps)
    output/detail/<slug>_pnl_daily.csv             — daily PnL vs E1VFVN30
    output/detail/<slug>_pnl_per_position_daily.csv — daily PnL contribution per ticker
    output/detail/<slug>_detail_chart.png          — equity + drawdown + exposure

Edit `make_strategy()` to inspect a different strategy. All output
files are scoped by slug so running detail on multiple strategies
does not overwrite previous runs.

Benchmark = MARKET_ETF (E1VFVN30) from src/data.py — single source
of truth.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import re

from signals.momentum        import MomentumSignal
from signals.ma_crossover    import MACrossoverSignal
from signals.bollinger       import BollingerBandSignal
from signals.rsi             import RSISignal
from signals.combined        import CombinedSignal
from filters.rules           import RulesBasedFilter

from vol_scaling   import compute_weights
from portfolio     import compute_portfolio_returns
from performance   import (summary_stats,
                           plot_return_distribution,
                           plot_rolling_sharpe)
from data          import MARKET_ETF


# ── Global assumptions (edit here if cost model changes) ──────────
TXN_COST_RATE   = 0.0015         # 0.15% flat, symmetric buy/sell
INITIAL_CAPITAL = 1_000_000_000  # 1 bn VND notional for absolute amounts


# ── Configure the strategy to inspect ─────────────────────────────

def make_strategy():
    """Return (name, signal, filter, sizing, top_pct)."""
    mom  = MomentumSignal(formation_weeks=26, skip_weeks=1)
    ma   = MACrossoverSignal(fast_weeks=10, slow_weeks=40)
    boll = BollingerBandSignal(window=20, k=2.0)
    rsi  = RSISignal(window=14)

    signal = CombinedSignal([
        (mom,  0.4),
        (ma,   0.2),
        (boll, 0.2),
        (rsi,  0.2),
    ])
    return {
        "name"    : "4way_additive_mom_ma_bollinger_rsi_rules",
        "signal"  : signal,
        "filter"  : RulesBasedFilter(),
        "sizing"  : "vol_adjusted",
        "top_pct" : 0.33,
    }


# ── Helpers ───────────────────────────────────────────────────────

def slugify(name):
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_").lower()


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


def positions_long_format(scaled_w):
    """Wide weights DataFrame → long (date, ticker, weight) excluding zero weights."""
    stk = scaled_w.stack().dropna()
    stk = stk[stk.abs() > 1e-9]  # drop zero weights
    df  = stk.rename("weight").reset_index()
    df.columns = ["date", "ticker", "weight"]
    return df.sort_values(["date", "weight"], ascending=[True, False])


def trade_blotter(scaled_w, cost_rate=TXN_COST_RATE):
    """
    Per-rebalance deltas + action + estimated cost in bps.
    Flat symmetric cost rate (buy = sell) by assumption.
    """
    w     = scaled_w.fillna(0)
    prev  = w.shift(1).fillna(0)
    delta = w - prev

    rows = []
    cost_bps = cost_rate * 1e4
    for date, row in delta.iterrows():
        for tkr, d in row.items():
            if abs(d) < 1e-9:
                continue
            action = "BUY" if d > 0 else "SELL"
            rows.append({
                "date"       : date,
                "ticker"     : tkr,
                "action"     : action,
                "weight_prev": round(prev.loc[date, tkr], 6),
                "weight_new" : round(w.loc[date, tkr], 6),
                "delta"      : round(d, 6),
                "cost_bps"   : round(cost_bps, 2),
                "cost_abs"   : round(abs(d) * cost_rate, 6),
            })
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    return df.sort_values(["date", "action", "ticker"]).reset_index(drop=True)


def weekly_to_daily_positions(scaled_w, daily_index):
    """Forward-fill weekly weights onto daily index, shift 1 day."""
    daily = scaled_w.reindex(daily_index, method="ffill")
    return daily.shift(1).fillna(0)


def weekly_costs_to_daily(scaled_w, daily_index, cost_rate):
    """
    Map weekly turnover cost onto the first daily bar after each
    rebalance (Monday t+1 open) — matches portfolio.py execution.
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


def benchmark_daily(closes_daily, etf=MARKET_ETF):
    if etf not in closes_daily.columns:
        raise RuntimeError(f"{etf} not in closes_daily — cannot build benchmark")
    px  = closes_daily[etf].ffill()
    ret = px.pct_change().fillna(0)
    eq  = (1 + ret).cumprod()
    return ret.rename("benchmark_pnl"), eq.rename("benchmark_equity")


def plot_holdings_stacked_area(scaled_w, save_path, title, top_n=15):
    """
    Stacked-area chart of portfolio composition over time.
    Top-N tickers by total cumulative weight shown individually;
    remaining tickers aggregated into an 'Others' band.
    """
    w      = scaled_w.fillna(0)
    totals = w.sum(axis=0).sort_values(ascending=False)
    top    = totals.head(top_n).index.tolist()
    rest   = [c for c in w.columns if c not in top]

    data = w[top].copy()
    if rest:
        data["Others"] = w[rest].sum(axis=1)

    # Use a deterministic palette so re-runs produce identical charts
    cmap   = plt.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(len(data.columns))]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.stackplot(data.index, data.T.values,
                 labels=data.columns, colors=colors,
                 alpha=0.9, edgecolor="white", linewidth=0.25)
    ax.set_ylabel("Portfolio weight")
    ax.set_xlabel("Date")
    ax.set_title(title)
    ax.set_ylim(0, max(1.05, data.sum(axis=1).max() * 1.05))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12),
              ncol=6, fontsize=8, frameon=False)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def compute_ticker_summary(scaled_w, closes_weekly, initial_capital):
    """
    Per-ticker lifetime statistics across the backtest.

    contrib[t, i] = w[t, i] × return_from_t_to_t+1 — PnL realised from
    holding stock i across week t (forward-shifted convention, matches
    portfolio.py). Lifetime sum gives gross % contribution to NAV;
    absolute amount uses actual NAV trajectory at each week.
    """
    w        = scaled_w.fillna(0)
    ret_fwd  = closes_weekly.pct_change(1).shift(-1).reindex(w.index)
    contrib  = w * ret_fwd
    port_ret = contrib.sum(axis=1).fillna(0)
    nav_end  = initial_capital * (1 + port_ret).cumprod()
    nav_at_t = nav_end.shift(1).fillna(initial_capital)

    rows = []
    for ticker in w.columns:
        weights = w[ticker]
        held    = weights > 1e-9
        if not held.any() or ticker not in contrib.columns:
            continue
        c        = contrib[ticker].fillna(0)
        c_abs    = c * nav_at_t
        dates    = weights.index
        idx_held = np.where(held.values)[0]

        rows.append({
            "ticker"               : ticker,
            "first_held"           : dates[idx_held[0]].date(),
            "last_held"            : dates[idx_held[-1]].date(),
            "weeks_held"           : int(held.sum()),
            "avg_weight_when_held" : round(float(weights[held].mean()), 4),
            "max_weight"           : round(float(weights.max()), 4),
            "pnl_pct"              : round(float(c.sum()), 5),
            "pnl_abs"              : round(float(c_abs.sum()), 0),
            "best_weekly_pct"      : round(float(c.max()), 5),
            "worst_weekly_pct"     : round(float(c.min()), 5),
            "hit_rate_when_held"   : round(
                float((c > 0).sum() / max((c != 0).sum(), 1)), 3),
        })
    return (pd.DataFrame(rows)
            .sort_values("pnl_abs", ascending=False)
            .reset_index(drop=True))


def compute_position_events(scaled_w, closes_weekly, initial_capital):
    """
    One row per contiguous holding episode (weight > 0).

    Forward-shifted return convention so PnL realised across week t
    is attributed to the weight at week t. Episode = weeks
    [start..end]; first week of realised PnL is week start.
    """
    w        = scaled_w.fillna(0)
    ret_fwd  = closes_weekly.pct_change(1).shift(-1).reindex(w.index)
    contrib  = w * ret_fwd
    port_ret = contrib.sum(axis=1).fillna(0)
    nav_end  = initial_capital * (1 + port_ret).cumprod()
    nav_at_t = nav_end.shift(1).fillna(initial_capital)

    rows = []
    for ticker in w.columns:
        held  = (w[ticker] > 1e-9).values
        if not held.any() or ticker not in contrib.columns:
            continue
        dates = w.index
        n     = len(held)
        i = 0
        while i < n:
            if not held[i]:
                i += 1
                continue
            start = i
            while i < n and held[i]:
                i += 1
            end = i - 1

            ep_w   = w[ticker].iloc[start:end+1]
            ep_c   = contrib[ticker].iloc[start:end+1].fillna(0)
            ep_nav = nav_at_t.iloc[start:end+1]

            entry_date = dates[start]
            exit_date  = dates[end]

            # Stock's own return entry-close → next-week-close (price
            # at which the position is unwound at the next rebalance).
            try:
                p_entry  = closes_weekly[ticker].loc[entry_date]
                idx_exit = closes_weekly.index.get_loc(exit_date)
                p_exit   = closes_weekly[ticker].iloc[
                    min(idx_exit + 1, len(closes_weekly) - 1)
                ]
                stock_ret = p_exit / p_entry - 1
            except (KeyError, IndexError):
                stock_ret = float("nan")

            rows.append({
                "ticker"            : ticker,
                "entry_date"        : entry_date.date(),
                "exit_date"         : exit_date.date(),
                "weeks_held"        : end - start + 1,
                "avg_weight"        : round(float(ep_w.mean()), 4),
                "max_weight"        : round(float(ep_w.max()), 4),
                "stock_return_pct"  : (round(float(stock_ret), 4)
                                       if stock_ret == stock_ret else None),
                "pnl_pct"           : round(float(ep_c.sum()), 5),
                "pnl_abs"           : round(float((ep_c * ep_nav).sum()), 0),
            })
    return (pd.DataFrame(rows)
            .sort_values(["entry_date", "ticker"])
            .reset_index(drop=True))


def compute_weekly_pnl_table(scaled_w, closes_weekly, opens_weekly,
                              cost_rate, initial_capital):
    """
    Weekly portfolio PnL with %, absolute, and cost columns.

    Uses portfolio.py's compute_portfolio_returns for gross/net/turnover
    (Friday signal → Monday open execution model). Absolute amounts are
    based on the actual net-of-cost NAV trajectory.
    """
    weekly_net, weekly_gross, turnover = compute_portfolio_returns(
        scaled_w, closes_weekly, opens_weekly,
        cost_rate_buy  = cost_rate,
        cost_rate_sell = cost_rate,
    )
    weekly_gross = weekly_gross.fillna(0)
    weekly_net   = weekly_net.fillna(0)
    cost_pct     = weekly_gross - weekly_net

    nav_end   = initial_capital * (1 + weekly_net).cumprod()
    nav_start = nav_end.shift(1).fillna(initial_capital)

    df = pd.DataFrame({
        "gross_return_pct"     : weekly_gross.round(6),
        "transaction_cost_pct" : cost_pct.round(6),
        "net_return_pct"       : weekly_net.round(6),
        "turnover"             : turnover.round(4),
        "nav_start"            : nav_start.round(0),
        "gross_pnl_abs"        : (nav_start * weekly_gross).round(0),
        "transaction_cost_abs" : (nav_start * cost_pct).round(0),
        "net_pnl_abs"          : (nav_start * weekly_net).round(0),
        "nav_end"              : nav_end.round(0),
    })
    df.index.name = "date"
    return df


def aggregate_pnl(weekly_df, freq):
    """
    Aggregate the weekly PnL table to monthly ('ME') or yearly ('YE').

    Percent columns compound: (1+r).prod() − 1.
    Absolute columns sum (each week's amount is on actual NAV at the time).
    NAV start/end = first/last on the actual net trajectory in the period.
    """
    g            = weekly_df.resample(freq)
    gross_pct    = g["gross_return_pct"].apply(lambda x: (1 + x).prod() - 1)
    net_pct      = g["net_return_pct"  ].apply(lambda x: (1 + x).prod() - 1)
    cost_pct     = gross_pct - net_pct
    nav_start    = g["nav_start"].first()
    nav_end      = g["nav_end"  ].last()

    df = pd.DataFrame({
        "gross_return_pct"     : gross_pct.round(6),
        "transaction_cost_pct" : cost_pct.round(6),
        "net_return_pct"       : net_pct.round(6),
        "turnover_total"       : g["turnover"].sum().round(4),
        "nav_start"            : nav_start.round(0),
        "gross_pnl_abs"        : g["gross_pnl_abs"       ].sum().round(0),
        "transaction_cost_abs" : g["transaction_cost_abs"].sum().round(0),
        "net_pnl_abs"          : (nav_end - nav_start).round(0),
        "nav_end"              : nav_end.round(0),
    })
    df.index.name = "period"
    return df


def make_cost_impact_chart(gross_pnl, net_pnl, bench_eq,
                           save_path, title):
    """
    Two-panel chart isolating the impact of transaction cost:
        top    — strategy GROSS (no cost), strategy NET, benchmark
        bottom — cumulative cost drag, expressed as 1 − net_eq/gross_eq
    """
    gross_eq = (1 + gross_pnl).cumprod()
    net_eq   = (1 + net_pnl).cumprod()
    drag     = 1 - (net_eq / gross_eq)   # fractional cumulative drag

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]}
    )

    ax1.plot(gross_eq.index, gross_eq.values,
             color="C0", linestyle="--", linewidth=1.2,
             label="Strategy GROSS (no cost)")
    ax1.plot(net_eq.index, net_eq.values,
             color="C0", linewidth=1.6,
             label=f"Strategy NET ({TXN_COST_RATE*1e4:.0f} bps cost)")
    ax1.plot(bench_eq.index, bench_eq.values,
             color="black", linestyle=":", linewidth=1.2,
             label=f"Benchmark ({MARKET_ETF})")
    ax1.set_yscale("log")
    ax1.set_ylabel("Equity (log)")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(title)

    ax2.fill_between(drag.index, drag.values * 100, 0,
                     color="C3", alpha=0.35)
    ax2.plot(drag.index, drag.values * 100,
             color="C3", linewidth=1.2)
    ax2.set_ylabel("Cost drag (%)")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel("Date")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_detail_chart(equity, benchmark_eq, exposure, save_path, title):
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(12, 10), sharex=True,
        gridspec_kw={"height_ratios": [3, 1, 1]}
    )

    # Panel 1: equity (log)
    ax1.plot(equity.index,     equity.values,
             color="C0", linewidth=1.4, label="Strategy")
    ax1.plot(benchmark_eq.index, benchmark_eq.values,
             color="black", linewidth=1.2, linestyle=":",
             label=f"Benchmark ({MARKET_ETF})")
    ax1.set_yscale("log")
    ax1.set_ylabel("Equity (log)")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(title)

    # Panel 2: drawdown
    dd_strat = (equity / equity.cummax()) - 1
    dd_bench = (benchmark_eq / benchmark_eq.cummax()) - 1
    ax2.fill_between(dd_strat.index, dd_strat.values, 0,
                     color="C0", alpha=0.4, label="Strategy")
    ax2.plot(dd_bench.index, dd_bench.values,
             color="black", linewidth=0.9, linestyle=":",
             label="Benchmark")
    ax2.set_ylabel("Drawdown")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower left", fontsize=8)

    # Panel 3: filter exposure
    if exposure is not None:
        e = exposure.dropna()
        ax3.plot(e.index, e.values, color="C3", linewidth=0.9)
        ax3.set_ylabel("Filter exposure")
        ax3.set_ylim(-0.05, 1.1)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlabel("Date")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────

def main():
    cfg    = make_strategy()
    slug   = slugify(cfg["name"])
    bundle = load_data()

    out_dir = "output/detail"
    os.makedirs(out_dir, exist_ok=True)

    # 1. Scores
    print(f"Signal: {cfg['signal'].name}")
    scores = cfg["signal"].compute(bundle["closes_weekly"])

    # 2. Cross-sectional sizing + vol scale
    raw_w, scaled_w, scale_f = compute_weights(
        scores,
        bundle["closes_weekly"], bundle["market_vol_weekly"],
        sizing  = cfg["sizing"],
        top_pct = cfg["top_pct"],
    )

    # 3. Apply filter
    filt     = cfg["filter"]
    exposure = filt.compute(
        scores         = scores,
        closes_weekly  = bundle["closes_weekly"],
        market_returns = bundle["market_returns_weekly"],
    )
    exp_al   = exposure.reindex(scaled_w.index).fillna(1.0)
    scaled_w = scaled_w.mul(exp_al, axis=0)

    # 4. Weekly PnL (with transaction costs at flat TXN_COST_RATE)
    weekly_net, weekly_gross, turnover = compute_portfolio_returns(
        scaled_w, bundle["closes_weekly"], bundle.get("opens_weekly"),
        cost_rate_buy  = TXN_COST_RATE,
        cost_rate_sell = TXN_COST_RATE,
    )

    # 5. Daily PnL (strategy) — gross per-position minus weekly cost
    #    charged on each rebalance's Monday open
    daily_ret = bundle["closes_daily"].pct_change().fillna(0)
    w_daily   = weekly_to_daily_positions(
        scaled_w.fillna(0), daily_ret.index
    )
    common       = w_daily.columns.intersection(daily_ret.columns)
    per_pos_pnl  = w_daily[common] * daily_ret[common]
    pnl_gross    = per_pos_pnl.sum(axis=1)
    daily_cost   = weekly_costs_to_daily(
        scaled_w, daily_ret.index, TXN_COST_RATE
    )
    strat_pnl    = (pnl_gross - daily_cost).rename("strategy_pnl")
    strat_eq     = (1 + strat_pnl).cumprod().rename("strategy_equity")

    # 6. Benchmark daily PnL
    bench_pnl, bench_eq = benchmark_daily(bundle["closes_daily"])

    # 7. Save artefacts
    # --- weekly positions (long) ---
    positions = positions_long_format(scaled_w)
    positions_path = f"{out_dir}/{slug}_positions_weekly.csv"
    positions.to_csv(positions_path, index=False)

    # --- transactions ---
    blotter = trade_blotter(scaled_w)
    blotter_path = f"{out_dir}/{slug}_transactions.csv"
    blotter.to_csv(blotter_path, index=False)

    # --- daily PnL ---
    pnl_daily = pd.DataFrame({
        "strategy_pnl_gross" : pnl_gross,
        "cost"               : daily_cost,
        "strategy_pnl"       : strat_pnl,
        "strategy_equity"    : strat_eq,
        "benchmark_pnl"      : bench_pnl,
        "benchmark_equity"   : bench_eq,
    })
    pnl_daily["excess_pnl"]    = pnl_daily["strategy_pnl"] - pnl_daily["benchmark_pnl"]
    pnl_daily["excess_equity"] = (1 + pnl_daily["excess_pnl"]).cumprod()
    pnl_daily.index.name = "date"
    pnl_daily_path = f"{out_dir}/{slug}_pnl_daily.csv"
    pnl_daily.to_csv(pnl_daily_path)

    # --- per-position daily PnL (wide) ---
    pp = per_pos_pnl.copy()
    pp["strategy_total"]  = strat_pnl
    pp[f"benchmark_{MARKET_ETF}"] = bench_pnl
    pp.index.name = "date"
    # Drop tickers with zero activity across the whole history
    active_cols = [c for c in pp.columns
                   if c in ("strategy_total", f"benchmark_{MARKET_ETF}")
                   or pp[c].abs().sum() > 1e-9]
    pp = pp[active_cols]
    pp_path = f"{out_dir}/{slug}_pnl_per_position_daily.csv"
    pp.to_csv(pp_path)

    # --- equity chart ---
    chart_path = f"{out_dir}/{slug}_detail_chart.png"
    make_detail_chart(
        strat_eq, bench_eq, exposure,
        save_path=chart_path,
        title=f"{cfg['name']} — Daily Equity vs {MARKET_ETF}",
    )

    # --- return distribution chart ---
    dist_path = f"{out_dir}/{slug}_return_distribution.png"
    plot_return_distribution(
        returns_dict = {
            "Strategy"              : {"series": strat_pnl, "color": "C0"},
            f"Benchmark ({MARKET_ETF})": {"series": bench_pnl, "color": "black"},
        },
        save_path = dist_path,
        title     = f"{cfg['name']} vs {MARKET_ETF}",
    )

    # --- gross vs net cost-impact chart ---
    cost_chart_path = f"{out_dir}/{slug}_cost_impact_chart.png"
    make_cost_impact_chart(
        gross_pnl    = pnl_gross,
        net_pnl      = strat_pnl,
        bench_eq     = bench_eq,
        save_path    = cost_chart_path,
        title        = f"{cfg['name']} — cost impact (gross vs net vs {MARKET_ETF})",
    )

    # --- holdings composition over time (stacked area) ---
    holdings_chart_path = f"{out_dir}/{slug}_holdings_stacked.png"
    plot_holdings_stacked_area(
        scaled_w,
        save_path = holdings_chart_path,
        title     = f"{cfg['name']} — portfolio composition over time",
    )

    # --- per-ticker lifetime summary ---
    ticker_summary = compute_ticker_summary(
        scaled_w, bundle["closes_weekly"], INITIAL_CAPITAL
    )
    ticker_summary_path = f"{out_dir}/{slug}_ticker_summary.csv"
    ticker_summary.to_csv(ticker_summary_path, index=False)

    # --- position-episode events (per ticker, per holding period) ---
    position_events = compute_position_events(
        scaled_w, bundle["closes_weekly"], INITIAL_CAPITAL
    )
    position_events_path = f"{out_dir}/{slug}_pnl_per_position.csv"
    position_events.to_csv(position_events_path, index=False)

    # --- weekly portfolio PnL table (gross/cost/net %, abs, NAV) ---
    weekly_pnl = compute_weekly_pnl_table(
        scaled_w, bundle["closes_weekly"], bundle.get("opens_weekly"),
        cost_rate       = TXN_COST_RATE,
        initial_capital = INITIAL_CAPITAL,
    )
    weekly_pnl_path = f"{out_dir}/{slug}_pnl_weekly_table.csv"
    weekly_pnl.to_csv(weekly_pnl_path)

    # --- monthly + yearly aggregates with same column structure ---
    monthly_pnl = aggregate_pnl(weekly_pnl, "ME")
    yearly_pnl  = aggregate_pnl(weekly_pnl, "YE")
    monthly_pnl_path = f"{out_dir}/{slug}_pnl_monthly_table.csv"
    yearly_pnl_path  = f"{out_dir}/{slug}_pnl_yearly_table.csv"
    monthly_pnl.to_csv(monthly_pnl_path)
    yearly_pnl.to_csv(yearly_pnl_path)

    # --- rolling Sharpe (1-year window) ---
    rolling_sharpe_path = f"{out_dir}/{slug}_rolling_sharpe.png"
    plot_rolling_sharpe(
        returns_dict = {
            "Strategy (net)"           : {"series": strat_pnl, "color": "C0"},
            f"Benchmark ({MARKET_ETF})": {"series": bench_pnl,
                                           "color": "black",
                                           "linestyle": ":"},
        },
        window    = 252,
        freq      = 252,
        save_path = rolling_sharpe_path,
        title     = f"{cfg['name']} — 1-yr rolling Sharpe vs {MARKET_ETF}",
    )

    # ── Summary printout ──
    print(f"\n{'='*60}")
    print(f"Strategy   : {cfg['name']}")
    print(f"Slug       : {slug}")
    print(f"Benchmark  : {MARKET_ETF}")
    print(f"Txn cost   : {TXN_COST_RATE*1e4:.1f} bps flat (buy = sell)")
    print(f"{'='*60}")

    def fmt(s): return s if isinstance(s, str) else f"{s:.3f}"
    strat_stats_gross = summary_stats(pnl_gross)
    strat_stats_net   = summary_stats(strat_pnl)
    bench_stats       = summary_stats(bench_pnl)
    print(f"\nHeadline metrics (daily):")
    print(f"  {'metric':<13}  {'gross':>10}  {'net':>10}  {'benchmark':>10}")
    for k in ("ann_return", "ann_vol", "sharpe", "max_dd", "calmar"):
        print(f"  {k:<13}  "
              f"{fmt(strat_stats_gross.get(k)):>10}  "
              f"{fmt(strat_stats_net.get(k)):>10}  "
              f"{fmt(bench_stats.get(k)):>10}")

    total_cost = daily_cost.sum()
    print(f"\nTotal transaction cost over backtest: "
          f"{total_cost*1e2:.2f}% (equity drag: "
          f"{(1 - (1 - total_cost)):.2%})")

    print(f"\nWeekly turnover : mean={turnover.mean():.3f}  "
          f"median={turnover.median():.3f}  max={turnover.max():.3f}")

    n_rebalances = (turnover > 0.01).sum()
    total_trades = len(blotter)
    print(f"Rebalances      : {n_rebalances}  "
          f"(weeks with turnover > 1%)")
    print(f"Trades total    : {total_trades}  "
          f"(BUY: {(blotter['action']=='BUY').sum()}, "
          f"SELL: {(blotter['action']=='SELL').sum()})")

    print(f"\nFiles written:")
    print(f"  {positions_path}")
    print(f"  {blotter_path}")
    print(f"  {pnl_daily_path}")
    print(f"  {pp_path}")
    print(f"  {chart_path}")
    print(f"  {dist_path}")
    print(f"  {cost_chart_path}")
    print(f"  {holdings_chart_path}")
    print(f"  {ticker_summary_path}")
    print(f"  {position_events_path}")
    print(f"  {weekly_pnl_path}")
    print(f"  {monthly_pnl_path}")
    print(f"  {yearly_pnl_path}")
    print(f"  {rolling_sharpe_path}")

    print(f"\nNotional capital: {INITIAL_CAPITAL:,.0f} VND")

    print(f"\nTop 10 tickers by absolute PnL contribution (gross of cost):")
    cols = ["ticker", "weeks_held", "avg_weight_when_held",
            "pnl_pct", "pnl_abs", "hit_rate_when_held"]
    print(ticker_summary.head(10)[cols].to_string(index=False))

    print(f"\nPosition episodes: {len(position_events)} total "
          f"(avg {position_events['weeks_held'].mean():.1f} weeks, "
          f"median {position_events['weeks_held'].median():.0f} weeks)")
    print(f"\nTop 5 episodes by absolute PnL:")
    print(position_events.nlargest(5, "pnl_abs").to_string(index=False))

    print(f"\nYearly PnL summary:")
    yr_show = yearly_pnl.copy()
    yr_show.index = yr_show.index.year
    print(yr_show[["gross_return_pct", "transaction_cost_pct",
                   "net_return_pct", "gross_pnl_abs",
                   "transaction_cost_abs", "net_pnl_abs",
                   "nav_end"]].to_string())

    # Sneak preview of recent holdings
    print(f"\nLatest portfolio ({positions['date'].max().date()}):")
    latest = positions[positions["date"] == positions["date"].max()]
    print(latest.to_string(index=False))


if __name__ == "__main__":
    main()
