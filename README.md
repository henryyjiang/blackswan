# Mandelbrotian Vol-Clustering Options Strategy

> **Strategy Link:** [QuantConnect](https://www.quantconnect.com/strategies/315/Mandelbrotian-Vol-Clustering-Options-Strat)

---

## Overview

The strategy seeks to monetize short-term swing opportunities in large-cap US technology equities by conditioning classic technical entry signals on a fractal and tail-risk regime model. Daily data are used to compute multi-horizon Hurst exponents, extreme-return tail index, multifractal width, realized volatility, and an implied-over-realized volatility ratio (from near-ATM options), alongside RSI, Bollinger %B, MACD, moving-average trend, volume, and VWAP deviation.

These features classify each underlying into one of several regimes — **trending up**, **trending down**, **mean-reverting**, **elevated risk**, or **crisis** — and signals are only acted on outside crisis regimes. Signal strength is strongest when the detected regime aligns with either trend-pullback-continuation or mean-reversion setups, and is scaled by fractal confidence and volatility-mispricing proxies.

### Universe

AAPL · AMZN · GOOGL · META · MSFT · NVDA · TSLA

---

## Backtest Results

**Period:** Backtest ending 2025-01-01 · **Starting Capital:** $1,000,000 (approx.)

| Metric | Value |
|---|---|
| **Final Portfolio** | **$13,764,131** |
| **Total Return** | **≈ +1,276%** |
| Total Trades | 105 |
| Win Rate | 31.7% (33 / 104) |
| Avg Winning Trade | +207.3% |
| Avg Losing Trade | −43.3% |
| Avg Trade PnL | +36.2% |

### Performance by Signal Type

| Signal | Trades | Win Rate | Avg PnL |
|---|---|---|---|
| Trend Continuation | 70 | 33% | +45.1% |
| Mean Reversion Buy | 31 | 29% | +21.8% |
| Downtrend Continuation | 2 | 50% | −11.4% |
| Mean Reversion Sell | 1 | 0% | −48.7% |

### Performance by Exit Reason

| Exit Type | Trades | Avg PnL |
|---|---|---|
| Take Profit | 4 | +566.8% |
| Expiry Near | 30 | +150.7% |
| Signal Reversal | 1 | +22.3% |
| Elevated Stop | 13 | −24.5% |
| Crisis Exit | 1 | −32.2% |
| Stop Loss | 55 | −49.0% |

### Performance by Ticker

| Ticker | Trades | Win Rate | Avg PnL |
|---|---|---|---|
| TSLA | 11 | 27% | +84.9% |
| NVDA | 11 | 27% | +57.9% |
| MSFT | 15 | 40% | +52.8% |
| GOOGL | 15 | 47% | +39.0% |
| AMZN | 15 | 33% | +26.7% |
| META | 22 | 23% | +22.4% |
| AAPL | 15 | 27% | −5.3% |

---

## How It Works

### 1 · Feature Computation (`utils.py`)
Each day, 260 bars of daily OHLCV data per ticker feed a feature pipeline that produces both fractal and classical technical indicators:

- **Multi-horizon Hurst exponents** — DFA-based estimates over 20-, 60-, and 120-day windows of log returns, plus a short−long divergence term. Values above ≈0.52 suggest persistence; below ≈0.45, anti-persistence.
- **Tail index** — a blended Hill / Dekkers-Einmahl-de Haan estimator swept across multiple order-statistic thresholds (5–25% of the distribution). Low values (< 2) flag heavy tails and elevated crash risk.
- **Multifractal width** — the spread between generalized Hurst exponents at q = 0.5 and q = 3 from a multifractal DFA. Wider spectra indicate richer volatility clustering.
- **Realized vol & vol-of-vol** — 20-day annualized standard deviation of log returns and the dispersion of rolling sub-window volatilities.
- **IV / RV ratio** — median implied volatility from near-ATM options (within 3% moneyness, 14–45 DTE) divided by realized vol, serving as a volatility mispricing proxy.
- **Technicals** — 14-period RSI, 20-period Bollinger %B, normalized ATR, MACD (12/26/9), 20- and 50-day SMA trend, volume ratio, and VWAP deviation.

### 2 · Regime Classification (`classify_regime`)
A composite **danger score** is computed as a weighted sum of normalized realized-vol, tail-index, multifractal-width, and 20-day drawdown scores (weights 0.35 / 0.15 / 0.20 / 0.25), with hard overrides when the tail index drops below 1.8 alongside wide multifractal spectra or extreme volatility. The regime mapping is:

| Regime | Code | Condition |
|---|---|---|
| **Crisis** | 4 | danger ≥ 0.7, or tail < 1.8 with MF width > 0.4 |
| **Elevated risk** | 3 | danger ≥ 0.5 |
| **Mean-reverting** | 2 | H_med < 0.48 or H_short < 0.45 (anti-persistent fractal signature) |
| **Trending up** | 0 | Persistent Hurst + positive 20-day return and/or price above SMA-50 |
| **Trending down** | 1 | Persistent Hurst + negative 20-day return and/or price below SMA-50 |

### 3 · Signal Generation (`generate_signal`)
No signals fire in crisis (regime 4). In other regimes, the strategy looks for specific technical setups whose strength is multiplicatively scaled by three modifiers — **fractal confidence** (distance of H_med from the persistence/anti-persistence threshold), **tail mispricing** (inverse tail index, capped), and **IV/RV adjustment** (bonus when implied vol is cheap relative to realized):

- **Regime 0 — Trend pullback buy:** RSI < 40, 5-day return < −2%, BB %B < 0.5, MACD histogram not deeply negative, price above SMA-50.
- **Regime 0 — Trend continuation:** 5-day return > +2%, RSI 52–70, MACD positive, BB %B in the 0.5–0.9 band, price above SMA-50.
- **Regime 1 — Downtrend continuation:** 5-day return < −2%, RSI 30–48, MACD negative, BB %B 0.1–0.5, price below SMA-50, plus tail < 2.8 and MF width > 0.18.
- **Regime 2 — Mean-reversion buy:** RSI < 35, BB %B < 0.25, drawdown not extreme, no large single-day crash.
- **Regime 2 — Mean-reversion sell:** RSI > 65, BB %B > 0.95, price below SMA-50.

Candidates are ranked by composite strength and the top two per day are sent to execution, subject to a five-position cap.

### 4 · Execution & Risk Management
Trades are expressed through **near-ATM options** (calls for long signals, puts for short) with 21–45 DTE and moneyness within ±5%. Position size is scaled by signal strength and further adjusted by tail index, ATR, and danger score, capping at 20% of portfolio per position. Idle capital is parked in QQQ.

Exits follow a layered rule set evaluated daily:

| Exit | Trigger |
|---|---|
| **Stop loss** | Option PnL < −40% |
| **Take profit** | Option PnL > +500% |
| **Expiry near** | ≤ 5 DTE remaining |
| **Time exit** | ≥ 20 days held and PnL ≤ +10% |
| **Crisis exit** | Regime jumps to 4, held ≥ 2 days, PnL negative |
| **Elevated stop** | Regime ≥ 3, PnL < −15%, held ≥ 3 days |
| **Signal reversal** | Current signal direction flips, held ≥ 5 days |

### 5 · Optional ML Trade Filter (`ml_filter.py`)
An XGBoost classifier can be toggled on (`use_ml = True`) to gate entries. It trains on the same 27-dimensional feature vector used for regime detection, labelling each closed trade as win (PnL > 0) or loss. The model retrains every 20 closed trades once 30+ samples are available, using cross-validated accuracy to monitor fit. Trades with predicted win probability below 0.45 are blocked. In the current backtest the ML filter is **disabled** — all signals that pass the regime and strength filters are executed.

---

## Key Takeaways

- The strategy wins on only ~32% of trades but generates outsized returns on winners (avg +207%), producing a positive expected value of +36% per trade.
- **Trend continuation** is the dominant signal (70 of 104 trades) and the strongest performer at +45.1% average PnL.
- **Expiry-near** exits capture the most value on average (+150.7%), suggesting the algorithm is effective at letting winners run into expiration.
- All tickers are net profitable except AAPL (−5.3% avg), with TSLA and NVDA contributing the highest average returns.

---

## Disclaimer

This repository is provided for **educational and research purposes only**. Past backtest performance does not guarantee future results. Options trading involves substantial risk of loss. Always do your own due diligence before deploying capital.
