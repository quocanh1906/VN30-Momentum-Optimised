"""
data.py — Data pipeline for VN30 Momentum Optimised.

Downloads daily OHLCV data for all VN30 master tickers from vnstock.
Computes market volatility estimates and vol regimes for vol scaling.
Performs data coverage analysis before proceeding.

Key design decisions:
- ROS excluded: suspended 2022 due to FLC fraud, data unreliable
- Start date 2014-01-01: maximises history for regime detection
- End date: today (latest available)
- Fetch open, high, low, close, volume for all tickers
- Rate limiting: 3s between requests, 60s pause every 15 requests

Rebalancing note:
    VN30 constituent changes are announced by HOSE 2-3 weeks before
    effective date. We rebalance on announcement date — reflecting
    real institutional practice of gradually transitioning positions
    upon announcement rather than waiting for the effective date.
    In practice, large positions are unwound over 3-5 days to minimise
    market impact (VWAP/TWAP execution). This is left as a future
    extension connecting to an Almgren-Chriss optimal execution model.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import time
from datetime import date

# ── Configuration ───────────────────────────────────────────────────────────────
START_DATE   = "2014-01-01"
END_DATE     = date.today().strftime("%Y-%m-%d")
MARKET_ETF   = "E1VFVN30"

# ROS excluded — suspended 2022 due to FLC fraud scandal
# vnstock data for ROS is unreliable and contains gaps
EXCLUDED     = {"ROS"}

# ── Point-in-time VN30 constituents ─────────────────────────────────────────────
# Source: HOSE official announcements
# Note: rebalance on announcement date, not effective date
VN30_CONSTITUENTS = {
    "2018-01": ["BID","BMP","BVH","CII","CTD","CTG","DHG","DPM","FPT","GAS","GMD","HPG","HSG","KDC","MBB","MSN","MWG","NT2","NVL","PLX","REE","ROS","SAB","SBT","SSI","STB","VCB","VIC","VJC","VNM"],
    "2018-07": ["BMP","CII","CTD","CTG","DHG","DPM","FPT","GAS","GMD","HPG","HSG","KDC","MBB","MSN","MWG","NVL","PLX","PNJ","REE","ROS","SAB","SBT","SSI","STB","VCB","VIC","VJC","VNM","VPB","VRE"],
    "2019-01": ["CII","CTD","CTG","DHG","DPM","EIB","FPT","GAS","GMD","HDB","HPG","MBB","MSN","MWG","NVL","PNJ","REE","ROS","SAB","SBT","SSI","STB","TCB","VCB","VHM","VIC","VJC","VNM","VPB","VRE"],
    "2019-07": ["BID","BVH","CTD","CTG","DPM","EIB","FPT","GAS","GMD","HDB","HPG","MBB","MSN","MWG","NVL","PNJ","REE","ROS","SAB","SBT","SSI","STB","TCB","VCB","VHM","VIC","VJC","VNM","VPB","VRE"],
    "2020-01": ["BID","BVH","CTD","CTG","EIB","FPT","GAS","HDB","HPG","MBB","MSN","MWG","NVL","PLX","PNJ","POW","REE","ROS","SAB","SBT","SSI","STB","TCB","VCB","VHM","VIC","VJC","VNM","VPB","VRE"],
    "2020-07": ["BID","CTG","EIB","FPT","GAS","HDB","HPG","KDH","MBB","MSN","MWG","NVL","PLX","PNJ","POW","REE","ROS","SAB","SBT","SSI","STB","TCB","TCH","VCB","VHM","VIC","VJC","VNM","VPB","VRE"],
    "2021-01": ["BID","BVH","CTG","FPT","GAS","HDB","HPG","KDH","MBB","MSN","MWG","NVL","PDR","PLX","PNJ","POW","REE","SBT","SSI","STB","TCB","TCH","TPB","VCB","VHM","VIC","VJC","VNM","VPB","VRE"],
    "2021-07": ["ACB","BID","BVH","CTG","FPT","GAS","GVR","HDB","HPG","KDH","MBB","MSN","MWG","NVL","PDR","PLX","PNJ","POW","SAB","SSI","STB","TCB","TPB","VCB","VHM","VIC","VJC","VNM","VPB","VRE"],
    "2022-01": ["ACB","BID","BVH","CTG","FPT","GAS","GVR","HDB","HPG","KDH","MBB","MSN","MWG","NVL","PDR","PLX","PNJ","POW","SAB","SSI","STB","TCB","TPB","VCB","VHM","VIC","VJC","VNM","VPB","VRE"],
    "2022-07": ["ACB","BID","BVH","CTG","FPT","GAS","GVR","HDB","HPG","KDH","MBB","MSN","MWG","NVL","PDR","PLX","POW","SAB","SSI","STB","TCB","TPB","VCB","VHM","VIB","VIC","VJC","VNM","VPB","VRE"],
    "2023-01": ["ACB","BCM","BID","BVH","CTG","FPT","GAS","GVR","HDB","HPG","MBB","MSN","MWG","NVL","PDR","PLX","POW","SAB","SSI","STB","TCB","TPB","VCB","VHM","VIB","VIC","VJC","VNM","VPB","VRE"],
    "2023-07": ["ACB","BCM","BID","BVH","CTG","FPT","GAS","GVR","HDB","HPG","MBB","MSN","MWG","PLX","POW","SAB","SHB","SSB","SSI","STB","TCB","TPB","VCB","VHM","VIB","VIC","VJC","VNM","VPB","VRE"],
    "2024-01": ["ACB","BCM","BID","BVH","CTG","FPT","GAS","GVR","HDB","HPG","MBB","MSN","MWG","PLX","POW","SAB","SHB","SSB","SSI","STB","TCB","TPB","VCB","VHM","VIB","VIC","VJC","VNM","VPB","VRE"],
    "2024-07": ["ACB","BCM","BID","BVH","CTG","FPT","GAS","GVR","HDB","HPG","MBB","MSN","MWG","PLX","POW","SAB","SHB","SSB","SSI","STB","TCB","TPB","VCB","VHM","VIB","VIC","VJC","VNM","VPB","VRE"],
    "2025-01": ["ACB","BCM","BID","BVH","CTG","FPT","GAS","GVR","HDB","HPG","LPB","MBB","MSN","MWG","PLX","SAB","SHB","SSB","SSI","STB","TCB","TPB","VCB","VHM","VIB","VIC","VJC","VNM","VPB","VRE"],
    "2025-07": ["ACB","BCM","BID","CTG","DGC","FPT","GAS","GVR","HDB","HPG","LPB","MBB","MSN","MWG","PLX","SAB","SHB","SSB","SSI","STB","TCB","TPB","VCB","VHM","VIB","VIC","VJC","VNM","VPB","VRE"],
    "2026-01": ["ACB","BID","CTG","DGC","FPT","GAS","GVR","HDB","HPG","LPB","MBB","MSN","MWG","PLX","SAB","SHB","SSB","SSI","STB","TCB","TPB","VCB","VHM","VIB","VIC","VJC","VNM","VPB","VPL","VRE"],
}

# All unique tickers ever in VN30 — excluding ROS
VN30_MASTER = sorted(
    set(t for tickers in VN30_CONSTITUENTS.values()
        for t in tickers) - EXCLUDED
)


# ── Universe helpers ─────────────────────────────────────────────────────────────

def get_constituents(date):
    """
    Return point-in-time VN30 constituents for a given date.
    Excludes ROS regardless of period.

    Parameters
    ----------
    date : str or pd.Timestamp

    Returns
    -------
    list of tickers
    """
    date    = pd.Timestamp(date)
    key     = f"{date.year}-01" if date.month < 7 else f"{date.year}-07"
    periods = sorted(VN30_CONSTITUENTS.keys())

    if key not in VN30_CONSTITUENTS:
        available = [p for p in periods if p <= key]
        key = available[-1] if available else periods[0]

    constituents = VN30_CONSTITUENTS[key]
    return [t for t in constituents if t not in EXCLUDED]


# ── Download ─────────────────────────────────────────────────────────────────────

def download_single(symbol, start=START_DATE, end=END_DATE,
                    retries=3, delay=5):
    """
    Download daily OHLCV from vnstock (KBS source) with retry.
    Returns DataFrame with columns: open, high, low, close, volume
    Returns None if download fails after all retries.
    """
    from vnstock import Quote

    for attempt in range(retries):
        try:
            quote = Quote(symbol=symbol, source='KBS')
            df    = quote.history(start=start, end=end, interval='1D')

            if df is None or df.empty:
                return None

            df['time'] = pd.to_datetime(df['time'])
            df = df.set_index('time')
            df = df[['open', 'high', 'low', 'close', 'volume']]
            df = df.sort_index()
            df = df[~df.index.duplicated(keep='last')]
            df.index.name = 'date'
            return df

        except Exception as e:
            if attempt < retries - 1:
                print(f"  ↻ {symbol} attempt {attempt+1} failed "
                      f"— retrying in {delay}s")
                time.sleep(delay)
            else:
                print(f"  ✗ {symbol}: {e}")
                return None


def download_all(symbols, start=START_DATE, end=END_DATE,
                 delay=3, batch_pause=60, batch_size=15):
    """
    Download OHLCV for all symbols with rate limiting.

    Returns
    -------
    opens, highs, lows, closes, volumes : DataFrames (date × ticker)
    failed : list of tickers that failed
    """
    opens   = {}
    highs   = {}
    lows    = {}
    closes  = {}
    volumes = {}
    failed  = []
    total   = len(symbols)

    est_mins = (total * delay + (total // batch_size) * batch_pause) // 60 + 1
    print(f"\nDownloading {total} tickers | "
          f"Est. time: ~{est_mins} mins\n")

    for i, symbol in enumerate(symbols):
        print(f"[{i+1:>3}/{total}] {symbol:<8}", end=" ", flush=True)
        df = download_single(symbol, start, end)

        if df is not None and not df.empty:
            opens[symbol]   = df['open']
            highs[symbol]   = df['high']
            lows[symbol]    = df['low']
            closes[symbol]  = df['close']
            volumes[symbol] = df['volume']
            print(f"✓  {len(df):>4} days  "
                  f"{df.index[0].date()} → {df.index[-1].date()}")
        else:
            failed.append(symbol)
            print("✗  no data")

        # Rate limiting
        if (i + 1) % batch_size == 0 and i + 1 < total:
            print(f"\n  ─── Pausing {batch_pause}s (rate limit) ───\n")
            time.sleep(batch_pause)
        else:
            time.sleep(delay)

    print(f"\nDownload complete: "
          f"{total - len(failed)}/{total} succeeded, "
          f"{len(failed)} failed")
    if failed:
        print(f"Failed: {failed}")

    return (pd.DataFrame(opens),
            pd.DataFrame(highs),
            pd.DataFrame(lows),
            pd.DataFrame(closes),
            pd.DataFrame(volumes),
            failed)


# ── Coverage analysis ────────────────────────────────────────────────────────────

def coverage_report(closes, volumes, opens,
                    min_coverage=0.8, min_start="2018-01-01"):
    """
    Analyse data coverage across all tickers.

    Reports:
    - Date range per ticker
    - % of trading days with valid data
    - Tickers with insufficient coverage
    - Coverage heatmap by year

    Parameters
    ----------
    min_coverage : minimum fraction of days required (default 80%)
    min_start    : latest acceptable start date for full coverage

    Returns
    -------
    DataFrame with coverage statistics per ticker
    """
    print(f"\n{'='*65}")
    print(f"  Data Coverage Report")
    print(f"{'='*65}")

    total_days = len(closes)
    min_start  = pd.Timestamp(min_start)

    rows = []
    for ticker in closes.columns:
        col        = closes[ticker].dropna()
        vol_col    = volumes[ticker].dropna() \
                     if ticker in volumes.columns else pd.Series()
        open_col   = opens[ticker].dropna() \
                     if ticker in opens.columns else pd.Series()

        if len(col) == 0:
            rows.append({
                "ticker"    : ticker,
                "start"     : None,
                "end"       : None,
                "n_days"    : 0,
                "coverage"  : 0.0,
                "zero_vol"  : 0,
                "status"    : "NO DATA",
            })
            continue

        start_date = col.index[0]
        end_date   = col.index[-1]
        n_days     = len(col)
        coverage   = n_days / total_days

        # Count zero-volume days (possible trading halts)
        zero_vol = (vol_col == 0).sum() if len(vol_col) > 0 else 0

        # Determine status
        if coverage < min_coverage:
            status = "LOW COVERAGE"
        elif start_date > min_start:
            status = "LATE START"
        elif zero_vol > 20:
            status = "SUSPICIOUS"
        else:
            status = "OK"

        rows.append({
            "ticker"    : ticker,
            "start"     : start_date.date(),
            "end"       : end_date.date(),
            "n_days"    : n_days,
            "coverage"  : round(coverage * 100, 1),
            "zero_vol"  : int(zero_vol),
            "status"    : status,
        })

    df = pd.DataFrame(rows).sort_values("coverage", ascending=False)

    # Summary
    ok      = (df["status"] == "OK").sum()
    low     = (df["status"] == "LOW COVERAGE").sum()
    late    = (df["status"] == "LATE START").sum()
    susp    = (df["status"] == "SUSPICIOUS").sum()
    no_data = (df["status"] == "NO DATA").sum()

    print(f"\n  Total tickers  : {len(df)}")
    print(f"  Total days     : {total_days}")
    print(f"  Date range     : {closes.index[0].date()} → "
          f"{closes.index[-1].date()}")
    print(f"\n  Status summary:")
    print(f"    ✓ OK            : {ok}")
    print(f"    ⚠ Low coverage  : {low}")
    print(f"    ⚠ Late start    : {late}")
    print(f"    ⚠ Suspicious    : {susp}")
    print(f"    ✗ No data       : {no_data}")

    # Print full table
    print(f"\n  {'Ticker':<8} {'Start':>12} {'End':>12} "
          f"{'Days':>6} {'Cover%':>8} {'ZeroVol':>8} {'Status':<15}")
    print(f"  {'─'*65}")
    for _, row in df.iterrows():
        flag = "  " if row["status"] == "OK" else "⚠ "
        print(f"  {flag}{row['ticker']:<6} "
              f"{str(row['start']):>12} "
              f"{str(row['end']):>12} "
              f"{row['n_days']:>6} "
              f"{row['coverage']:>7.1f}% "
              f"{row['zero_vol']:>8} "
              f"{row['status']:<15}")

    # Coverage by year
    print(f"\n  Coverage by year (% tickers with data):")
    years = range(2014, pd.Timestamp(END_DATE).year + 1)
    for year in years:
        year_closes = closes.loc[str(year)]
        if len(year_closes) == 0:
            continue
        pct = (year_closes.notna().mean() * 100).mean()
        bar = "█" * int(pct / 5)
        print(f"    {year}: {bar:<20} {pct:5.1f}%")

    print(f"{'='*65}")
    return df


# ── Volatility ───────────────────────────────────────────────────────────────────

def compute_market_vol(market_returns, estimator="ewma",
                        span=14, window=20, freq=252):
    """
    Compute annualised market volatility shifted 1 day (no lookahead).

    Parameters
    ----------
    estimator : 'rolling', 'ewma', or 'garch'
    span      : EWMA span (default 14 ≈ RiskMetrics λ=0.94)
    window    : rolling window in days
    freq      : annualisation factor (252 for daily)

    Returns
    -------
    Series of annualised vol, shifted 1 day
    """
    r = market_returns.dropna()

    if estimator == "rolling":
        vol = r.rolling(window).std() * np.sqrt(freq)

    elif estimator == "ewma":
        vol = r.ewm(span=span).std() * np.sqrt(freq)

    elif estimator == "garch":
        try:
            from arch import arch_model
            model  = arch_model(r * 100, vol='Garch',
                                p=1, q=1, dist='normal')
            result = model.fit(disp='off')
            cond   = result.conditional_volatility / 100
            vol    = pd.Series(cond.values * np.sqrt(freq),
                               index=r.index, name="garch_vol")
        except Exception as e:
            print(f"  GARCH failed ({e}) — falling back to EWMA")
            vol = r.ewm(span=span).std() * np.sqrt(freq)
    else:
        raise ValueError(f"Unknown estimator: {estimator}")

    # Shift 1 day — use yesterday's vol for today's position
    vol = vol.shift(1)
    vol.name = f"vol_{estimator}"
    return vol


def get_vol_regime(market_vol, n_terciles=3):
    """
    Classify each period into vol regime using expanding window terciles.

    Expanding window prevents lookahead — boundaries computed only
    from data available at each point in time.

    Returns
    -------
    Series: 0 = low vol, 1 = mid vol, 2 = high vol
    """
    regimes = pd.Series(np.nan, index=market_vol.index,
                        name="vol_regime")

    for i in range(20, len(market_vol)):
        hist = market_vol.iloc[:i+1].dropna()
        if len(hist) < 20:
            continue

        bounds  = np.percentile(hist, np.linspace(0, 100, n_terciles+1))
        vol_now = market_vol.iloc[i]

        if pd.isna(vol_now):
            continue

        for k in range(n_terciles):
            if vol_now <= bounds[k+1]:
                regimes.iloc[i] = k
                break
        else:
            regimes.iloc[i] = n_terciles - 1

    return regimes


# ── Save / Load ──────────────────────────────────────────────────────────────────

def save_data(opens, highs, lows, closes, volumes,
              path="data/processed"):
    """Save all OHLCV DataFrames to CSV."""
    os.makedirs(path, exist_ok=True)
    opens.to_csv(f"{path}/opens_daily.csv")
    highs.to_csv(f"{path}/highs_daily.csv")
    lows.to_csv(f"{path}/lows_daily.csv")
    closes.to_csv(f"{path}/closes_daily.csv")
    volumes.to_csv(f"{path}/volumes_daily.csv")
    print(f"\nSaved OHLCV to {path}/")
    print(f"  Shape: {closes.shape} (days × tickers)")


def load_data(path="data/processed"):
    """Load saved OHLCV data from disk."""
    opens   = pd.read_csv(f"{path}/opens_daily.csv",
                           index_col=0, parse_dates=True)
    highs   = pd.read_csv(f"{path}/highs_daily.csv",
                           index_col=0, parse_dates=True)
    lows    = pd.read_csv(f"{path}/lows_daily.csv",
                           index_col=0, parse_dates=True)
    closes  = pd.read_csv(f"{path}/closes_daily.csv",
                           index_col=0, parse_dates=True)
    volumes = pd.read_csv(f"{path}/volumes_daily.csv",
                           index_col=0, parse_dates=True)
    return opens, highs, lows, closes, volumes


# ── Main pipeline ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("="*65)
    print("  VN30 Momentum Optimised — Data Pipeline")
    print(f"  Period: {START_DATE} → {END_DATE}")
    print(f"  Universe: {len(VN30_MASTER)} tickers "
          f"(ROS excluded)")
    print("="*65)

    os.makedirs("data/processed", exist_ok=True)

    # ── Load or download ────────────────────────────────────────────
    if os.path.exists("data/processed/closes_daily.csv"):
        print("\nExisting data found — loading from disk...")
        opens, highs, lows, closes, volumes = load_data()
        print(f"  Loaded: {closes.shape} (days × tickers)")
        print(f"  Period: {closes.index[0].date()} → "
              f"{closes.index[-1].date()}")
        print("\nTo re-download: delete data/processed/ and run again.")
    else:
        print("\nNo existing data — downloading all tickers...")
        all_symbols = VN30_MASTER + [MARKET_ETF]
        opens, highs, lows, closes, volumes, failed = download_all(
            all_symbols,
            start=START_DATE,
            end=END_DATE,
        )
        save_data(opens, highs, lows, closes, volumes)

    # ── Coverage analysis ───────────────────────────────────────────
    print("\nRunning coverage analysis...")
    # Separate market ETF for analysis
    stock_closes  = closes.drop(columns=[MARKET_ETF], errors='ignore')
    stock_opens   = opens.drop(columns=[MARKET_ETF], errors='ignore')
    stock_volumes = volumes.drop(columns=[MARKET_ETF], errors='ignore')

    coverage_df = coverage_report(
        stock_closes, stock_volumes, stock_opens,
        min_coverage=0.7,
        min_start="2016-01-01",
    )
    coverage_df.to_csv("data/processed/coverage_report.csv", index=False)
    print("\nCoverage report saved to data/processed/coverage_report.csv")

    # ── Market vol and regimes ──────────────────────────────────────
    if MARKET_ETF in closes.columns:
        market_prices  = closes[MARKET_ETF]
    else:
        print(f"\n  ⚠ {MARKET_ETF} not found — "
              f"using equal-weighted VN30 as market proxy")
        market_prices  = stock_closes.mean(axis=1)

    market_returns = market_prices.pct_change().dropna()

    print("\nComputing volatility estimates...")
    vol_ewma    = compute_market_vol(market_returns, "ewma")
    vol_rolling = compute_market_vol(market_returns, "rolling")

    try:
        vol_garch = compute_market_vol(market_returns, "garch")
        print(f"  ✓ GARCH vol computed")
    except Exception as e:
        print(f"  ⚠ GARCH failed: {e}")
        vol_garch = vol_ewma.copy()
        vol_garch.name = "vol_garch"

    # Weekly resampling (Friday close)
    def to_weekly(s):
        return s.resample('W-FRI').last()

    closes_weekly  = to_weekly(stock_closes)
    opens_weekly   = to_weekly(stock_opens)
    vol_ewma_w     = to_weekly(vol_ewma)
    vol_rolling_w  = to_weekly(vol_rolling)
    vol_garch_w    = to_weekly(vol_garch)
    market_ret_w   = to_weekly(market_returns)

    # Vol regimes
    print("\nComputing vol regimes (expanding window)...")
    regimes = get_vol_regime(vol_ewma_w)

    regime_labels = {0: "Low vol ", 1: "Mid vol ", 2: "High vol"}
    print(f"\n  Vol regime distribution (weekly):")
    for r, count in regimes.value_counts().sort_index().items():
        pct = count / len(regimes.dropna()) * 100
        bar = "█" * int(pct / 2)
        print(f"    Regime {int(r)} ({regime_labels[int(r)]}): "
              f"{bar:<25} {count:>4} weeks ({pct:.1f}%)")

    # Save processed data
    closes_weekly.to_csv("data/processed/closes_weekly.csv")
    opens_weekly.to_csv("data/processed/opens_weekly.csv")
    market_returns.to_csv("data/processed/market_returns_daily.csv")
    market_ret_w.to_csv("data/processed/market_returns_weekly.csv")
    vol_ewma_w.to_csv("data/processed/vol_ewma_weekly.csv")
    vol_rolling_w.to_csv("data/processed/vol_rolling_weekly.csv")
    vol_garch_w.to_csv("data/processed/vol_garch_weekly.csv")
    regimes.to_csv("data/processed/vol_regimes.csv")
    market_prices.to_csv("data/processed/market_prices_daily.csv")

    print(f"\n{'='*65}")
    print(f"  Pipeline complete")
    print(f"{'='*65}")
    print(f"  Daily  : {stock_closes.shape[0]} days × "
          f"{stock_closes.shape[1]} tickers")
    print(f"  Weekly : {closes_weekly.shape[0]} weeks × "
          f"{closes_weekly.shape[1]} tickers")
    print(f"  Vol regimes computed: {regimes.notna().sum()} weeks")
    print(f"\n  Proceed to src/signals.py once coverage looks clean.")