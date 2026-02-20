# Mandelbrotian Vol-Clustering Options Strategy

A QuantConnect algorithm that trades short-term swing opportunities in large-cap US tech equities by conditioning classic technical signals on a fractal and tail-risk regime model.

> **Platform:** [QuantConnect](https://www.quantconnect.com/strategies/315/Mandelbrotian-Vol-Clustering-Options-Strat) · **Language:** Python · **Asset Class:** US Equity Options

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

### 1 · Regime Detection
Each day the algorithm computes fractal and statistical features for every underlying:
- **Hurst exponent** (multi-horizon) — distinguishes trending vs mean-reverting dynamics
- **Tail index** — captures extreme-return risk
- **Multifractal width** — measures complexity of the return process
- **Realized volatility** vs **implied volatility** — identifies volatility mispricing
- Standard technicals: RSI, Bollinger %B, MACD, MA trend, volume profile, VWAP deviation

These features map each ticker to a regime: trending up (0), trending down (1), mean-reverting (2), elevated risk, or crisis.

### 2 · Signal Generation
Signals are generated only when the regime is **not** in crisis. The strategy looks for:
- **Trend continuation** — pullback entries in a confirmed uptrend
- **Downtrend continuation** — short-biased entries in a confirmed downtrend
- **Mean-reversion buy/sell** — entries when a mean-reverting regime shows overextension

Signal strength is scaled by fractal confidence and the implied/realized volatility ratio.

### 3 · Execution & Risk Management
Trades are expressed through near-ATM options. Exits are governed by:
- **Take-profit** and **stop-loss** thresholds
- **Expiry-near** management to avoid theta decay
- **Elevated-risk stops** when the regime shifts toward stress
- **Crisis exits** for rapid de-risking

---

## Key Takeaways

- The strategy wins on only ~32% of trades but generates outsized returns on winners (avg +207%), producing a positive expected value of +36% per trade.
- **Trend continuation** is the dominant signal (70 of 104 trades) and the strongest performer at +45.1% average PnL.
- **Expiry-near** exits capture the most value on average (+150.7%), suggesting the algorithm is effective at letting winners run into expiration.
- All tickers are net profitable except AAPL (−5.3% avg), with TSLA and NVDA contributing the highest average returns.

---

## Disclaimer

This repository is provided for **educational and research purposes only**. Past backtest performance does not guarantee future results. Options trading involves substantial risk of loss. Always do your own due diligence before deploying capital.
