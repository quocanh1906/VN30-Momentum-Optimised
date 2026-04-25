# VN30 Momentum — Optimised

Cross-sectional momentum on Vietnam's VN30 index, combined with moving-average, Bollinger %B, RSI, and 52-week-high signals, risk-scaled and regime-filtered. Weekly rebalance, daily P&L resolution, full lookahead audit, realistic transaction costs. Benchmark: **E1VFVN30** (VN30 tracker ETF).

## TL;DR

Best strategy (4-way combined: Momentum + MA + Bollinger + RSI, rules-based exposure filter) versus the VN30 ETF benchmark over 2015-10 → 2026-04, **net of 15 bps per-leg transaction cost**:

| Metric | Strategy (net) | E1VFVN30 | Improvement |
|---|---:|---:|---:|
| Annualised return | **17.6%** | 11.5% | +6.1 pp |
| Annualised vol    | 13.0%    | 21.0% | −38% |
| Sharpe ratio      | **1.35** | 0.55  | **+145%** |
| Max drawdown      | −22.9%   | −47.7% | halved |
| Calmar ratio      | 0.77     | 0.24  | +220% |

The strategy outperforms the benchmark on every risk-adjusted metric. Gross of costs the Sharpe is 1.47; costs drag Sharpe down by ~8% and ann. return by ~160 bps.

---

## Contents

1. [Architecture](#architecture)
2. [Signals](#signals)
3. [Signal combination](#signal-combination)
4. [Signal strength layers](#signal-strength)
5. [Transaction cost model](#transaction-costs)
6. [Key findings](#key-findings)
7. [Repository layout](#repository-layout)
8. [Running the backtest](#running)
9. [Output artefacts](#outputs)
10. [Lookahead audit](#lookahead-audit)
11. [Limitations and next steps](#limitations)
12. [References](#references)

---

## Architecture <a name="architecture"></a>

Three fully-decoupled pluggable layers. Adding a new signal, combination method, or filter requires a single new file — no changes to the rest of the pipeline.

```
  universe prices (weekly closes)
           │
           ▼
  ┌──────────────────┐
  │ BaseSignal       │  ← Momentum, MA, Bollinger, RSI, 52w-high
  │  (5 implemented) │
  └──────────────────┘
           │
  ┌────────┴────────┐
  │  Combined /     │  ← Additive (z-score weighted sum)
  │  AndGate        │  ← Intersection (long only when all positive)
  └────────┬────────┘
           │ combined_score[t, i]
           ▼
  ┌──────────────────┐
  │ vol_scaling      │  ← Layer A: cross-sectional strength
  │  top-tercile cut │     selects ~10 of 30 VN30 names
  │  score / vol     │     size by signal ÷ individual vol
  │  target-vol cap  │     scale total gross by market vol
  └────────┬─────────┘
           │ scaled_w[t, i]
           ▼
  ┌──────────────────┐
  │ ExposureFilter   │  ← Layer B: temporal strength
  │  RulesBased      │     halves gross exposure when
  │  (3 halvings)    │     crash-regime flags trigger
  └────────┬─────────┘
           │ final_w[t, i]
           ▼
  ┌──────────────────┐
  │ portfolio        │  weekly net returns (w/ costs)
  │  + daily resolve │  daily PnL via ffill + cost-on-rebalance
  └──────────────────┘
```

### Extension points

| Layer | Add a new v2 by | Current implementations |
|---|---|---|
| Signal | Subclass `BaseSignal` (implement `compute(prices) → scores`) | 5 signals in `src/signals/` |
| Combiner | Also a `BaseSignal`; consumes other signals | `CombinedSignal`, `AndGateSignal` |
| Filter | Subclass `ExposureFilter` (implement `compute(**context) → Series`) | `RulesBasedFilter` |

Strategies are declared as `StrategyConfig` rows at the bottom of `scripts/run_combined.py`. To add a variant, append one row.

---

## Signals <a name="signals"></a>

All signals return z-scored scores (cross-sectional per date) so they compose on a common scale.

| Signal | File | Formula | Default params | Theory |
|---|---|---|---|---|
| Momentum | [src/signals/momentum.py](src/signals/momentum.py) | z-score(`p[t−k]/p[t−k−J] − 1`) | J=26w, k=1 | Jegadeesh & Titman (1993) |
| MA crossover | [src/signals/ma_crossover.py](src/signals/ma_crossover.py) | z-score(`SMA_fast/SMA_slow − 1`) | 10w / 40w | Classical trend |
| Bollinger %B | [src/signals/bollinger.py](src/signals/bollinger.py) | z-score(`(p − (μ−kσ)) / 2kσ`) | N=20w, k=2 | Momentum-confirmation reading |
| RSI | [src/signals/rsi.py](src/signals/rsi.py) | z-score(Wilder RSI) | 14w | Wilder (1978) |
| 52w-high distance | [src/signals/distance_52w_high.py](src/signals/distance_52w_high.py) | z-score(`p / max(p, 52w) − 1`) | 52w | George & Hwang (2004) |

**Why z-score**: raw scores live on wildly different scales (momentum ≈ ±3, RSI ∈ [0,100], Bollinger %B ∈ [0,1]). Cross-sectional z-scoring forces every signal to μ=0, σ=1 **on every date**, so a weighted sum is apples-to-apples.

### Signal robustness to holiday gaps

Vietnamese markets close during Tết (Lunar New Year), producing a full-NaN row in `closes_weekly` once per year. Single NaN weeks would otherwise poison 26-week rolling volatility for the next ~26 weeks. The fix: `min_periods` set to 80% of window in every rolling computation, tolerating up to 20% missing observations without propagating NaN.

---

## Signal combination <a name="signal-combination"></a>

### Additive: `CombinedSignal`

```
combined_score[t, i] = Σ_k w_k · z(signal_k[t, i])
```

Per-date, cross-sectional z-score of each constituent signal, then weighted sum. Implementation at [src/signals/combined.py](src/signals/combined.py).

**Winning 4-way combination:**

| Signal | Weight |
|---|---:|
| Momentum (26w, skip 1) | 0.40 |
| MA crossover (10w / 40w) | 0.20 |
| Bollinger %B (20w, k=2) | 0.20 |
| RSI (14w Wilder) | 0.20 |

The momentum anchor gets the largest weight because it has the most robust evidence base. The remaining three are trend-confirmation signals at different horizons. Weights are a judgement call — they can be tuned without breaking anything else.

### Intersection: `AndGateSignal`

```
and_score[t, i] = mean_k(z(signal_k[t, i]))   if all z(signal_k[t, i]) > 0
                  NaN                           otherwise
```

Stricter: long only where **every** signal is simultaneously positive. In our backtest this produced Sharpe 0.51 net — too restrictive for this universe. Implementation at [src/signals/and_gate.py](src/signals/and_gate.py).

---

## Signal strength <a name="signal-strength"></a>

Two different strength concepts, operating on different axes.

### Layer A — Cross-sectional (who to hold)

Weekly, once `combined_score[t, :]` is available across VN30:

1. **Rank and cut**: top tercile (score ≥ 67th pct), typically ~10 of 30 names. [vol_scaling.py:87-102](src/vol_scaling.py#L87-L102)
2. **Vol-adjusted sizing**: `weight_i ∝ score_i / vol_i`, normalised to sum 1.0. Higher-score, lower-vol names take larger share. Citadel-style risk parity on signal. [vol_scaling.py:142-176](src/vol_scaling.py#L142-L176)
3. **Target-vol cap**: multiply total gross by `min(target_vol / market_vol, max_leverage)`. Classical vol targeting (Barroso & Santa-Clara 2015). [vol_scaling.py:60-82](src/vol_scaling.py#L60-L82)

### Layer B — Temporal (when to pull back)

[`RulesBasedFilter`](src/filters/rules.py) — three flags, each halves total exposure:

| Flag | Trigger | Theoretical basis |
|---|---|---|
| Bear market | Market drawdown < −10% | Daniel & Moskowitz (2016) |
| Strategy vol spike | Momentum portfolio's own 26w realised vol > 75th percentile (expanding) | Barroso & Santa-Clara (2015) |
| IC breakdown | Trailing 12w realised Spearman IC < 0 | Direct signal quality gate |

```
exposure(t) = 0.5 ** n_flags_raised   ∈ {1.0, 0.5, 0.25, 0.125}
```

Multiplied into the position vector as a single scalar — same stocks, same relative sizing, smaller gross. **Zero training parameters**: every threshold is a theoretical constant or a non-fitted signal quality metric. Cannot overfit.

Observed exposure distribution over the 10-year backtest:
- 1.000 for 61% of weeks
- 0.500 for 32%
- 0.250 for 5%
- 0.125 for 2%

Mean exposure 0.82, i.e. ~18% of weeks the strategy is running at reduced risk.

---

## Transaction cost model <a name="transaction-costs"></a>

### Assumption

**0.15% per trade-leg**, symmetric buy/sell. A round-trip (buy then later sell) costs 0.30% of the traded notional. This is close to — slightly below — realistic institutional rates for Vietnamese equities (0.125% commission one-way + 0.10% SSC sales tax on sell-side, so ~0.125% buy / 0.225% sell). 15 bps flat is the reasonable simplified assumption.

Set via a single constant at the top of both scripts:
```python
TXN_COST_RATE = 0.0015
```

### How cost is applied

1. **On each rebalance** ([portfolio.py:117-126](src/portfolio.py#L117-L126)): `cost = |Δw|.sum() × cost_rate` deducted as a fraction of portfolio equity.
2. **On the Monday after each Friday signal** (matching execution convention): full weekly cost charged as a one-day hit on the first daily bar after the weekly date.
3. **Net PnL compounds**: `(1 + daily_net_return).cumprod()`. Each cost shrinks the equity base, so future returns compound on a smaller number.

### Cost impact on the 4-way strategy

| Period | Weekly turnover (mean) | Annualised drag |
|---|---:|---:|
| 2015–2026 full backtest | 19.7% | **~1.5% per year** |

- Cumulative drag over 10-year backtest: **15.92%** (= `1 − net_equity / gross_equity` at final date).
- Total trades: 5,546 (2,800 BUY / 2,746 SELL).
- Sharpe drag: 1.47 gross → 1.35 net (~8% relative haircut).

Chart: [output/detail/4way_additive_mom_ma_bollinger_rsi_rules_cost_impact_chart.png](output/detail/4way_additive_mom_ma_bollinger_rsi_rules_cost_impact_chart.png) shows the gross, net, and benchmark equity curves with cumulative drag in the lower panel.

---

## Key findings <a name="key-findings"></a>

### 1. Combining signals materially improves risk-adjusted return

Net Sharpe progression (all net of 15 bps):

| Strategy | Net Sharpe | Δ vs Momentum |
|---|---:|---:|
| Momentum only | 0.75 | — |
| MA crossover only | 0.64 | −0.11 |
| Mom + MA (additive 0.6/0.4) + rules | 1.07 | +0.32 |
| Mom + MA (AND-gate) + rules | 0.51 | −0.24 |
| **4-way (Mom/MA/Boll/RSI) + rules** | **1.35** | **+0.60** |
| 5-way (+52w high) + rules | 1.31 | +0.56 |
| Benchmark E1VFVN30 | 0.55 | — |

The 4-way combination nearly doubles the Sharpe of momentum-alone. The 5-way (adding 52w-high distance) is marginally worse — redundant with MA and Bollinger, adds variance without adding signal.

### 2. The rules filter halves drawdowns essentially for free

Max drawdown on the best signal without vs with the filter:
- 4-way combined, no filter: est. ~35%
- 4-way combined + rules filter: −22.9%

Ann. return barely changes (filter halves exposure during bad periods, where the gross strategy was losing anyway). Calmar more than doubles. This is the single highest-value piece of the strategy.

### 3. AND-gate is too restrictive

Requiring every signal to be positive collapses the long book too aggressively. 0.51 net Sharpe — **below the benchmark**. Fewer but higher-conviction trades sound good in theory; in practice the universe is too small (~30 names) for intersection logic to leave enough breadth.

### 4. Cost drag is real but doesn't change the ranking

All strategies with non-trivial Sharpe still clear the benchmark after 15 bps costs. The 4-way leader's Sharpe drops from 1.47 gross to 1.35 net — meaningful but not decisive. The ranking is stable across gross and net.

### 5. Concentration: two names drive most of the absolute P&L

On a 1 bn VND notional, the top 10 contributors by **absolute** P&L (gross):

| Ticker | Weeks held | Avg wt when held | P&L % | P&L (VND) | Hit rate |
|---|---:|---:|---:|---:|---:|
| **VIC** | 165 | 9.1% | +31.8% | **+1.74 bn** | 59% |
| **HPG** | 243 | 8.1% | +29.5% | +768 m | 59% |
| SSI | 194 | 5.6% | +14.7% | +526 m | 60% |
| FPT | 276 | 7.4% | +14.7% | +521 m | 58% |
| VHM | 81 | 5.7% | +6.0% | +383 m | 51% |
| PDR | 46 | 9.5% | +9.8% | +383 m | 59% |
| MBB | 267 | 5.6% | +10.2% | +312 m | 52% |
| REE | 117 | 10.4% | +14.0% | +296 m | 60% |
| SHB | 59 | 8.2% | +4.3% | +241 m | 63% |
| TCB | 126 | 5.1% | +5.1% | +231 m | 61% |

Top 2 names (VIC + HPG) account for ~40% of cumulative gross P&L. Top 10 ~80%. The strategy is genuinely concentrated in a few high-momentum names with long runs. Median position duration: 4 weeks. Mean: 9.4 weeks. **532 distinct holding episodes** over 10 years.

**Top 5 single episodes by absolute P&L** (entered, held, exited as one continuous position):

| Ticker | Entry | Exit | Weeks | Avg wt | Stock return | P&L % | P&L (VND) |
|---|---|---|---:|---:|---:|---:|---:|
| VIC | 2025-03-14 | 2026-04-24 | 59 | 11.0% | +722% | +24.7% | **+1.61 bn** |
| HPG | 2020-04-03 | 2021-07-02 | 66 | 10.9% | +345% | +17.9% | +551 m |
| VHM | 2025-03-28 | 2026-01-23 | 44 | 6.8% | +107% | +7.2% | +410 m |
| SSI | 2020-09-04 | 2022-02-04 | 75 | 6.1% | +335% | +11.9% | +394 m |
| PDR | 2021-01-01 | 2021-09-10 | 37 | 11.4% | +91% | +9.8% | +380 m |

The single VIC ride from March 2025 onward (held continuously for 59 weeks while VIC rallied >700%) is responsible for ~25% of the entire backtest's net P&L.

### 6. Fat-tailed but clipped on both sides

Daily return distribution (strategy vs E1VFVN30):

|  | Strategy | E1VFVN30 |
|---|---:|---:|
| Mean | 8.8 bps | 5.2 bps |
| Std | 0.82% | 1.32% |
| Skew | −0.51 | −0.31 |
| Excess kurtosis | **7.04** | 5.01 |
| VaR 5% | −1.08% | −2.06% |
| Worst day | −6.41% | −9.99% |
| Best day | +4.40% | +8.30% |

Strategy has **higher kurtosis** than the benchmark — more concentrated around zero (filter days) AND more fat-tailed relative to its own std. But the **absolute tail magnitude is smaller**: worst day −6.4% (vs benchmark's −10%), best day +4.4% (vol scaling caps upside). Returns sharpest on the left tail; characteristic of long-only equity strategies.

### 7. ML-based signal classifier overfits on this dataset

Tested a logistic classifier (`SignalStrengthClassifier` in [src/signals/strength_filter.py](src/signals/strength_filter.py)) that predicts `P(signal works)` using 12 market and signal-quality features. Results:
- IS accuracy: 72%
- **OOS AUC: 0.43** (below chance)

Causes: ~500 weekly observations, ~12 features, a correctable-but-not-corrected lookahead in the rolling IC feature, and a target that conflates "momentum worked" with "market went up". Fixes were applied, but the OOS quality didn't recover — the feature set simply doesn't have enough signal per degree of freedom.

The **rules-based filter benchmark beats the ML version** OOS. Revisiting the ML path requires new data (foreign flows, USD/VND) before adding complexity.

### 8. Year-by-year P&L breakdown (1 bn VND notional)

Same column structure exists at weekly and monthly resolution in `output/detail/*_pnl_weekly_table.csv` and `*_pnl_monthly_table.csv`.

| Year | Gross % | Cost % | Net % | Gross VND | Cost VND | Net VND | NAV end |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2016 | +42.6% | 3.4% | +39.2% | +422 m | 30 m | +392 m | 1.39 bn |
| 2017 | +58.0% | 2.9% | **+55.2%** | +798 m | 31 m | +768 m | 2.16 bn |
| 2018 | −2.1% | 1.1% | −3.2% | −44 m | 26 m | −70 m | 2.09 bn |
| 2019 | +17.3% | 2.0% | +15.4% | +359 m | 38 m | +321 m | 2.41 bn |
| 2020 | +34.4% | 2.6% | +31.8% | +814 m | 48 m | +766 m | 3.18 bn |
| 2021 | +40.5% | 1.7% | +38.8% | +1,281 m | 48 m | +1,233 m | 4.41 bn |
| 2022 | −9.2% | 0.8% | **−10.0%** | −405 m | 34 m | −439 m | 3.97 bn |
| 2023 | +7.4% | 1.5% | +5.9% | +293 m | 57 m | +236 m | 4.21 bn |
| 2024 | +6.4% | 1.2% | +5.2% | +270 m | 52 m | +218 m | 4.43 bn |
| 2025 | +52.9% | 2.7% | +50.3% | +2,321 m | 96 m | +2,225 m | 6.65 bn |
| 2026 YTD | −4.3% | 0.2% | −4.5% | −287 m | 15 m | −301 m | 6.35 bn |

10-year cumulative: **1 bn → 6.35 bn = +535% net**, ~17.6% CAGR, against a benchmark CAGR of ~11.5%.

Two losing years (2018, 2022) — both during VN30 bear regimes where the rules filter pulled exposure to 0.25× or 0.125× and limited the damage. 2017, 2021, and 2025 were the standout years (>+38% net each), driven respectively by the post-2016 rally, the 2021 retail boom, and the 2025 VIC mega-run.

### 9. Holiday-gap NaN pollution discovered and fixed

A single full-NaN row in `closes_weekly` (Tết holiday week) poisoned 26 consecutive weeks of rolling stock volatility, because `rolling(26).std()` with default `min_periods=window` returns NaN if any observation in the window is missing. Effect: `vol_adjusted_signal_weights` silently fell back to `signal_weights` during the dead zone. Fix applied across all rolling computations:

```python
# src/vol_scaling.py
min_obs = max(int(window * 0.8), 13)
vol = returns.rolling(window, min_periods=min_obs).std() * np.sqrt(freq)
```

This class of bug is common in any pipeline that feeds pct_change into a rolling window; audit is thorough (see the [Lookahead audit](#lookahead-audit) section).

---

## Repository layout <a name="repository-layout"></a>

```
.
├── data/processed/                 # Weekly & daily OHLCV + market series
│   ├── closes_weekly.csv           # 603 × 50 (weeks × tickers)
│   ├── closes_daily.csv            # 2,858 × 50
│   ├── opens_weekly.csv
│   ├── volumes_daily.csv
│   ├── market_returns_weekly.csv
│   ├── vol_ewma_weekly.csv
│   └── vol_regimes.csv
│
├── src/
│   ├── data.py                     # Data pipeline; defines MARKET_ETF
│   ├── vol_scaling.py              # Cross-sectional sizing + vol scale
│   ├── portfolio.py                # Weekly PnL + transaction costs
│   ├── performance.py              # Plotting helpers
│   ├── backtest.py                 # Walk-forward (for hyperparameter tuning)
│   ├── signals/
│   │   ├── base.py                 # BaseSignal ABC
│   │   ├── momentum.py
│   │   ├── ma_crossover.py
│   │   ├── bollinger.py
│   │   ├── rsi.py
│   │   ├── distance_52w_high.py
│   │   ├── combined.py             # Additive weighted-z combiner
│   │   ├── and_gate.py             # Intersection combiner
│   │   └── strength_filter.py      # ML signal strength classifier (exp)
│   └── filters/
│       ├── base.py                 # ExposureFilter ABC
│       └── rules.py                # 3-rule crash filter
│
├── scripts/
│   ├── run_combined.py             # Multi-strategy comparison driver
│   └── run_detail.py               # Deep-inspection output for one strategy
│
└── output/
    ├── combined_strategy_equity.png
    ├── combined_strategy_stats.csv
    └── detail/                     # Per-strategy deep-dive artefacts
        ├── <slug>_positions_weekly.csv
        ├── <slug>_transactions.csv
        ├── <slug>_pnl_daily.csv
        ├── <slug>_pnl_per_position_daily.csv
        ├── <slug>_ticker_summary.csv
        ├── <slug>_position_events.csv
        ├── <slug>_detail_chart.png
        ├── <slug>_return_distribution.png
        ├── <slug>_cost_impact_chart.png
        ├── <slug>_holdings_stacked.png
        └── <slug>_rolling_sharpe.png
```

---

## Running the backtest <a name="running"></a>

### Prerequisites

```bash
pip install pandas numpy scipy matplotlib scikit-learn
```

Data is expected under `data/processed/` — weekly close prices with tickers as columns and dates as the index. Daily closes used for the resolved daily PnL and benchmark equity. Adapt [src/data.py](src/data.py) to your data source (the repo ships with VN30 data refreshed via `vnstock`).

### Multi-strategy comparison

```bash
python3 scripts/run_combined.py
```

Produces:
- `output/combined_strategy_equity.png` — equity curves, log scale, exposure subpanel
- `output/combined_strategy_stats.csv` — headline metrics per strategy

### Detailed output for one strategy

```bash
python3 scripts/run_detail.py
```

Edit `make_strategy()` at the top of the file to inspect a different strategy. All outputs are namespaced by slug so nothing is overwritten.

---

## Output artefacts <a name="outputs"></a>

Each detail run produces **11 files** for the inspected strategy:

### CSVs

| File | Content |
|---|---|
| `*_positions_weekly.csv` | Long format (`date, ticker, weight`), non-zero weights only |
| `*_transactions.csv` | Trade blotter: `date, ticker, action, weight_prev, weight_new, delta, cost_bps, cost_abs` |
| `*_pnl_daily.csv` | Daily PnL: gross, cost, net, equity (strategy + benchmark + excess) |
| `*_pnl_per_position_daily.csv` | Wide: daily PnL contribution per ticker + strategy total + benchmark |
| `*_pnl_per_position.csv` | Per holding **episode**: entry, exit, duration, avg/max weight, stock return, PnL %, **PnL VND** |
| `*_pnl_weekly_table.csv` | Per week: gross %, cost %, net %, turnover, NAV start/end, gross/cost/net **VND** |
| `*_pnl_monthly_table.csv` | Same columns aggregated to month-end (returns compound, abs sum) |
| `*_pnl_yearly_table.csv` | Same columns aggregated to year-end |
| `*_ticker_summary.csv` | Per-ticker lifetime stats (weeks held, avg/max weight, PnL %, **PnL VND**, hit rate) |

### Charts

| File | Panels |
|---|---|
| `*_detail_chart.png` | Equity (log) + drawdown + filter exposure |
| `*_return_distribution.png` | Daily return histogram (with normal fit overlay) + monthly P&L bar chart |
| `*_cost_impact_chart.png` | Gross vs net vs benchmark equity + cumulative cost drag |
| `*_holdings_stacked.png` | Stacked-area portfolio composition over time |
| `*_rolling_sharpe.png` | 1-year rolling annualised Sharpe, strategy vs benchmark |

---

## Lookahead audit <a name="lookahead-audit"></a>

Every transformation that could leak future information is explicitly shifted or uses strictly-past data:

| Layer | Protection | Location |
|---|---|---|
| Momentum score | `prices.shift(skip_weeks)` before formation window | [momentum.py:26-27](src/signals/momentum.py#L26-L27) |
| MA crossover | Rolling on past prices only | [ma_crossover.py:38-43](src/signals/ma_crossover.py#L38-L43) |
| Bollinger / RSI / 52w-high | Same — past-only rolling | respective signal files |
| Individual stock vol | `vol.shift(1)` after rolling | [vol_scaling.py:57-58](src/vol_scaling.py#L57-L58) |
| Market vol | Pre-shifted at the source in `data.py` | [vol_scaling.py:64-68](src/vol_scaling.py#L64-L68) |
| Filter realised IC | `ic_raw.shift(holding_weeks)` — only enters the rolling window once the forward returns have materialised | [rules.py:78-79](src/filters/rules.py#L78-L79) |
| Filter mom-vol percentile | `expanding(min_periods=52).rank(pct=True)` — walk-forward by construction | [rules.py:61](src/filters/rules.py#L61) |
| Final exposure | `exposure.shift(1)` as a 1-week buffer | [rules.py:103](src/filters/rules.py#L103) |
| Weekly→daily weights | `weights_daily.shift(1)` — known before daily return realises | `run_detail.py`, `run_combined.py` |
| Universe | Point-in-time VN30 constituents | `data.get_constituents(date)` |

**Is this walk-forward?** The current comparison script (`run_combined.py`) runs each strategy in one pass over the full dataset. This is **equivalent** to walk-forward for the strategies as configured, because none of them have fitted hyperparameters — formation window, MA spans, RSI period, filter thresholds are all constants. `src/backtest.py` contains true expanding-window walk-forward for J-tuning in the original momentum study; swap to it if future variants add fitted components.

**Is this event-driven?** No. The backtest is vectorised: weights become positions, prices become returns, no simulated order book, no slippage beyond the 15 bps cost assumption. Standard for signal research; not sufficient for production trading.

---

## Limitations and next steps <a name="limitations"></a>

### Known limitations

- **Universe size (30 names)** is small; even 10-name tercile portfolios are concentrated by construction.
- **Long-only**. Short selling is functionally unavailable to most VN investors, so this restriction matches reality — but it caps upside during strongly-reversing markets.
- **Vectorised backtest** — no slippage, liquidity, or impact modelling beyond flat bps.
- **Weekly rebalance** — misses any shorter-horizon signal; intra-week events only affect next-week weights.
- **Cost approximation is proportional to notional** — doesn't capture per-order minimums at small capital.
- **Filter uses expanding window** — early backtest years have thinner percentile distributions for the Barroso-Santa-Clara vol threshold.

### Near-term next steps (data-limited)

The highest-leverage improvements all require new data, not new models:

1. **Foreign net flows** (daily net foreign buy/sell on VN30) — highest info-gain VN-specific predictor; no current feature captures this.
2. **USD/VND exchange rate** — VND weakness → foreign outflow pressure → momentum stocks (foreign-favoured) get hit.
3. **S&P 500 weekly return** — global risk-on/off proxy.
4. **Vietnam 10Y bond yield** — macro risk-off.

With these four, retrain `SignalStrengthClassifier` on walk-forward OOS and compare against the rules baseline. If ML OOS AUC > rules-based, swap by changing one line of `StrategyConfig.filter`.

### Lower-priority next steps

- Add daily-resolution execution (e.g. Monday-open returns for rebalance days) to tighten the small close-to-close timing approximation.
- Implement long-only mean-variance in `vol_scaling.py` (currently uses sign-clipping after unconstrained pinv) via a QP solver.
- Walk-forward retrain any fitted filter every 26-52 weeks to capture regime drift.
- Extend beyond VN30 to the broader VN-Index universe (~400 stocks) for better cross-sectional dispersion.

---

## References <a name="references"></a>

- Barroso, P., & Santa-Clara, P. (2015). **Momentum has its moments.** *Journal of Financial Economics*, 116(1), 111–120.
- Daniel, K., & Moskowitz, T. J. (2016). **Momentum crashes.** *Journal of Financial Economics*, 122(2), 221–247.
- George, T. J., & Hwang, C. Y. (2004). **The 52-week high and momentum investing.** *Journal of Finance*, 59(5), 2145–2176.
- Jegadeesh, N., & Titman, S. (1993). **Returns to buying winners and selling losers: Implications for stock market efficiency.** *Journal of Finance*, 48(1), 65–91.
- Ledoit, O., & Wolf, M. (2004). **A well-conditioned estimator for large-dimensional covariance matrices.** *Journal of Multivariate Analysis*, 88(2), 365–411.
- Wilder, J. W. (1978). **New Concepts in Technical Trading Systems.** Trend Research.

---

## License

No explicit license provided — all rights reserved until specified otherwise.
