"""
performance.py — Performance reporting and charting helpers.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as scistats


def summary_stats(returns, freq=252, rf=0.0):
    """
    Compute headline performance metrics from a daily return series.
    """
    r = returns.dropna()
    if len(r) == 0:
        return {}

    ann_ret = (1 + r).prod() ** (freq / len(r)) - 1
    ann_vol = r.std() * np.sqrt(freq)
    sharpe  = (ann_ret - rf) / ann_vol if ann_vol > 0 else np.nan
    eq      = (1 + r).cumprod()
    max_dd  = ((eq / eq.cummax()) - 1).min()
    calmar  = ann_ret / abs(max_dd) if max_dd < 0 else np.nan

    return {
        "ann_return": round(ann_ret, 4),
        "ann_vol"   : round(ann_vol, 4),
        "sharpe"    : round(sharpe, 3),
        "max_dd"    : round(max_dd, 4),
        "calmar"    : round(calmar, 3),
    }


def plot_equity_curves(results, benchmark=None, benchmark_label="Benchmark",
                       save_path=None,
                       title="Strategy vs Benchmark — Daily Equity",
                       log_scale=True):
    """
    Plot daily equity curves and (if any strategy has a filter) a
    filter-exposure subpanel.

    Parameters
    ----------
    results   : dict {name: {equity_daily, exposure, color, linestyle}}
    benchmark : optional Series of daily equity
    save_path : optional file path (.png)
    title     : figure title
    log_scale : log-y equity panel (default True)
    """
    has_exposure = any(
        r.get("exposure") is not None for r in results.values()
    )

    if has_exposure:
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(12, 8), sharex=True,
            gridspec_kw={"height_ratios": [3, 1]}
        )
    else:
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax2 = None

    if benchmark is not None:
        b = benchmark.dropna()
        ax1.plot(b.index, b.values, color="black",
                 linewidth=1.2, linestyle=":",
                 label=benchmark_label)

    for name, r in results.items():
        eq = r["equity_daily"].dropna()
        ax1.plot(eq.index, eq.values,
                 color=r.get("color", "C0"),
                 linestyle=r.get("linestyle", "-"),
                 linewidth=1.4, label=name)

    if log_scale:
        ax1.set_yscale("log")
    ax1.set_ylabel("Equity" + (" (log)" if log_scale else ""))
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(title)

    if ax2 is not None:
        for name, r in results.items():
            exp = r.get("exposure")
            if exp is None:
                continue
            e = exp.dropna()
            ax2.plot(e.index, e.values,
                     color=r.get("color", "C0"),
                     linewidth=0.8, label=name)
        ax2.set_ylabel("Filter exposure")
        ax2.set_ylim(-0.05, 1.1)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="lower left", fontsize=8)
        ax2.set_xlabel("Date")
    else:
        ax1.set_xlabel("Date")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_rolling_sharpe(returns_dict, window=252, freq=252,
                        save_path=None, title="Rolling Sharpe"):
    """
    Rolling Sharpe (annualised) for each series in `returns_dict`.

        sharpe_t = mean(r[t-W:t]) / std(r[t-W:t]) * sqrt(freq)

    Defaults assume daily returns with a 1-year (252-day) window.
    For weekly data pass window=52, freq=52.

    Parameters
    ----------
    returns_dict : dict {label: {"series": returns, "color": str,
                                  "linestyle": str (optional)}}
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    min_p = max(int(window * 0.8), 20)
    for label, meta in returns_dict.items():
        r      = meta["series"].dropna()
        mu     = r.rolling(window, min_periods=min_p).mean()
        sigma  = r.rolling(window, min_periods=min_p).std()
        sharpe = (mu / sigma) * np.sqrt(freq)
        ax.plot(sharpe.index, sharpe.values,
                color=meta.get("color", "C0"),
                linestyle=meta.get("linestyle", "-"),
                linewidth=1.4, label=label)

    ax.axhline(0,  color="black", linewidth=0.8, linestyle="--")
    ax.axhline(1,  color="gray",  linewidth=0.6, linestyle=":", alpha=0.7)
    ax.axhline(-1, color="gray",  linewidth=0.6, linestyle=":", alpha=0.4)
    ax.set_ylabel(f"{window}-period rolling Sharpe (annualised)")
    ax.set_xlabel("Date")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_return_distribution(returns_dict, save_path=None,
                             title="Return distribution",
                             bins=60):
    """
    Two-panel distribution chart:
        top    — overlaid histogram of daily returns
        bottom — monthly-return bar chart for the FIRST series,
                 with benchmark (if present) overlaid as a line

    Parameters
    ----------
    returns_dict : dict {label: {"series": daily_return_Series, "color": str}}
                   First key is treated as the "main" strategy for the
                   bottom bar panel; later keys are overlaid for comparison.
    save_path    : PNG file path
    """
    labels = list(returns_dict.keys())
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 8),
        gridspec_kw={"height_ratios": [1, 1]}
    )

    # ── Panel 1: daily-return histogram ──
    all_vals = pd.concat([
        returns_dict[k]["series"].dropna() for k in labels
    ])
    xmin, xmax = np.percentile(all_vals, [0.5, 99.5])
    bin_edges = np.linspace(xmin, xmax, bins)

    stats_rows = []
    for k in labels:
        meta = returns_dict[k]
        r    = meta["series"].dropna()
        ax1.hist(r.values, bins=bin_edges, alpha=0.45,
                 color=meta.get("color", "C0"),
                 label=k, edgecolor="none")

        stats_rows.append({
            "name"  : k,
            "mean"  : r.mean(),
            "std"   : r.std(),
            "skew"  : scistats.skew(r),
            "kurt"  : scistats.kurtosis(r),
            "var5"  : r.quantile(0.05),
            "best"  : r.max(),
            "worst" : r.min(),
        })

    ax1.axvline(0, color="black", linewidth=0.8, linestyle="--")

    # Normal-distribution overlay for the primary series (red line).
    # Scale PDF → expected counts per bin: pdf * N * bin_width.
    main_key  = labels[0]
    main_r    = returns_dict[main_key]["series"].dropna()
    mu, sigma = main_r.mean(), main_r.std()
    bin_width = bin_edges[1] - bin_edges[0]
    x_norm    = np.linspace(xmin, xmax, 400)
    y_norm    = scistats.norm.pdf(x_norm, loc=mu, scale=sigma) \
                * len(main_r) * bin_width
    ax1.plot(x_norm, y_norm, color="red", linewidth=1.6,
             label=f"Normal fit ({main_key}: "
                   f"μ={mu*1e4:.1f}bp, σ={sigma*1e2:.2f}%)")

    ax1.set_xlabel("Daily return")
    ax1.set_ylabel("Frequency")
    ax1.set_title(title + " — daily returns")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Annotate stats in upper-right corner
    lines = [f"{'series':<22} {'mean':>7} {'std':>7} {'skew':>6} "
             f"{'kurt':>6} {'VaR5':>7} {'best':>7} {'worst':>7}"]
    for row in stats_rows:
        lines.append(
            f"{row['name'][:22]:<22} "
            f"{row['mean']*1e4:>6.1f}b {row['std']*1e2:>6.2f}% "
            f"{row['skew']:>6.2f} {row['kurt']:>6.2f} "
            f"{row['var5']*1e2:>6.2f}% "
            f"{row['best']*1e2:>6.2f}% {row['worst']*1e2:>6.2f}%"
        )
    ax1.text(
        0.99, 0.97, "\n".join(lines),
        transform=ax1.transAxes, va="top", ha="right",
        fontfamily="monospace", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.4",
                  facecolor="white", alpha=0.9, edgecolor="gray")
    )

    # ── Panel 2: monthly returns as a bar chart ──
    main_key   = labels[0]
    main_daily = returns_dict[main_key]["series"].dropna()
    monthly    = (1 + main_daily).resample("ME").prod() - 1
    colors     = ["#2ca02c" if v >= 0 else "#d62728" for v in monthly.values]
    ax2.bar(monthly.index, monthly.values * 100,
            color=colors, width=22, alpha=0.85, label=main_key)

    if len(labels) > 1:
        bench_key   = labels[1]
        bench_daily = returns_dict[bench_key]["series"].dropna()
        bench_month = (1 + bench_daily).resample("ME").prod() - 1
        ax2.plot(bench_month.index, bench_month.values * 100,
                 color="black", linewidth=1.1, linestyle=":",
                 label=bench_key)

    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_ylabel("Monthly return (%)")
    ax2.set_xlabel("Month")
    ax2.set_title(title + " — monthly P&L")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path
