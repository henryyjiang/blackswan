import numpy as np
from scipy import stats

def estimate_hurst(series, min_window=8, max_window=None):
    series = np.asarray(series, dtype=np.float64)
    n = len(series)
    if n < 20:
        return 0.5
    if max_window is None:
        max_window = max(n // 4, min_window + 1)
    y = np.cumsum(series - np.mean(series))
    window_sizes = []
    fluctuations = []
    for window in range(min_window, max_window + 1):
        n_segments = n // window
        if n_segments < 2:
            continue
        f2 = []
        for i in range(n_segments):
            segment = y[i * window:(i + 1) * window]
            x = np.arange(window)
            trend = np.polyval(np.polyfit(x, segment, 1), x)
            f2.append(np.mean((segment - trend) ** 2))
        window_sizes.append(window)
        fluctuations.append(np.sqrt(np.mean(f2)))
    if len(window_sizes) < 3:
        return 0.5
    slope, _, _, _, _ = stats.linregress(np.log(window_sizes), np.log(fluctuations))
    return float(np.clip(slope, 0.0, 1.0))

def hill_estimator(sorted_abs_returns, k):
    if k < 2 or k >= len(sorted_abs_returns):
        return np.nan
    tail = sorted_abs_returns[:k]
    threshold = sorted_abs_returns[k]
    if threshold <= 0:
        return np.nan
    log_ratios = np.log(tail / threshold)
    log_ratios = log_ratios[log_ratios > 0]
    if len(log_ratios) < 2:
        return np.nan
    return len(log_ratios) / np.sum(log_ratios)

def dekkers_einmahl_dehaan_estimator(sorted_abs_returns, k):
    n = len(sorted_abs_returns)
    if k < 5 or k >= n:
        return np.nan
    tail = sorted_abs_returns[:k]
    threshold = sorted_abs_returns[k]
    if threshold <= 0:
        return np.nan
    log_excess = np.log(tail / threshold)
    log_excess = log_excess[log_excess > 0]
    if len(log_excess) < 5:
        return np.nan
    m1 = np.mean(log_excess)
    m2 = np.mean(log_excess ** 2)
    if m2 <= m1 ** 2:
        return 1 / m1 if m1 > 0 else np.nan
    gamma = m1 + 1 - 0.5 * (1 / (1 - m1 ** 2 / m2))
    if gamma <= 0:
        return np.nan
    return float(np.clip(1 / gamma, 0.5, 6.0))

def estimate_tail_index(returns, min_data_for_tail):
    returns = np.asarray(returns, dtype=np.float64)
    abs_ret = np.abs(returns)
    abs_ret = abs_ret[~np.isnan(abs_ret)]
    abs_ret = abs_ret[abs_ret > 1e-10]
    n = len(abs_ret)
    if n < min_data_for_tail:
        return 2.0
    sorted_ret = np.sort(abs_ret)[::-1]
    k_min = max(10, int(n * 0.05))
    k_max = min(int(n * 0.25), n // 3)
    k_values = np.linspace(k_min, k_max, 10, dtype=int)
    all_estimates = []
    for k in k_values:
        h = hill_estimator(sorted_ret, k)
        if not np.isnan(h) and 0.5 < h < 6:
            all_estimates.append(('hill', h))
        d = dekkers_einmahl_dehaan_estimator(sorted_ret, k)
        if not np.isnan(d) and 0.5 < d < 6:
            all_estimates.append(('deh', d))
    if len(all_estimates) < 3:
        return 2.0
    hill_only = [e[1] for e in all_estimates if e[0] == 'hill']
    hill_std = np.std(hill_only) if len(hill_only) > 2 else np.inf
    weights = {'hill': 0.6, 'deh': 0.4} if hill_std < 0.3 else {'hill': 0.5, 'deh': 0.5}
    weighted_sum = sum(weights[m] * a for m, a in all_estimates)
    weight_total = sum(weights[m] for m, _ in all_estimates)
    return float(np.clip(weighted_sum / weight_total, 0.5, 5.0)) if weight_total > 0 else 2.0

def compute_mf_width(returns_arr):
    returns_arr = np.asarray(returns_arr, dtype=np.float64)
    returns_arr = returns_arr[~np.isnan(returns_arr)]
    n = len(returns_arr)
    if n < 50:
        return 0.0
    profile = np.cumsum(returns_arr - np.mean(returns_arr))
    max_scale = n // 4
    scales = np.unique(np.logspace(np.log10(4), np.log10(max_scale), 12).astype(int))
    def get_hurst_at_q(q):
        log_scales, log_Fq = [], []
        for scale in scales:
            n_seg = n // scale
            if n_seg < 2:
                continue
            variances = []
            for i in range(n_seg):
                seg = profile[i * scale:(i + 1) * scale]
                x = np.arange(len(seg))
                var = np.mean((seg - np.polyval(np.polyfit(x, seg, 1), x)) ** 2)
                if var > 0:
                    variances.append(var)
            if len(variances) < 2:
                continue
            variances = np.array(variances)
            if q < 0:
                variances = np.clip(variances, 1e-20, None)
            Fq = np.mean(variances ** (q / 2)) ** (1 / q)
            if Fq > 0 and np.isfinite(Fq):
                log_scales.append(np.log(scale))
                log_Fq.append(np.log(Fq))
        if len(log_scales) >= 3:
            slope, _, r_value, _, _ = stats.linregress(log_scales, log_Fq)
            if r_value ** 2 > 0.5:
                return slope
        return np.nan
    h_low = get_hurst_at_q(0.5)
    h_high = get_hurst_at_q(3)
    if np.isfinite(h_low) and np.isfinite(h_high):
        return abs(h_low - h_high)
    return 0.0


def compute_sma(prices, period):
    if len(prices) < period:
        return prices[-1] if len(prices) > 0 else 0.0
    return float(np.mean(prices[-period:]))

def compute_ema(prices, period):
    prices = np.asarray(prices, dtype=np.float64)
    if len(prices) < period:
        return prices[-1] if len(prices) > 0 else 0.0
    multiplier = 2.0 / (period + 1)
    ema = np.mean(prices[:period])
    for p in prices[period:]:
        ema = (p - ema) * multiplier + ema
    return float(ema)

def compute_macd(prices, fast=12, slow=26, signal=9):
    prices = np.asarray(prices, dtype=np.float64)
    if len(prices) < slow + signal:
        return 0.0, 0.0, 0.0
    ema_fast = compute_ema(prices, fast)
    ema_slow = compute_ema(prices, slow)
    macd_line = ema_fast - ema_slow
    macd_series = []
    mult_f = 2.0 / (fast + 1)
    mult_s = 2.0 / (slow + 1)
    ef = np.mean(prices[:fast])
    es = np.mean(prices[:slow])
    for p in prices[max(fast, slow):]:
        ef = (p - ef) * mult_f + ef
        es = (p - es) * mult_s + es
        macd_series.append(ef - es)
    if len(macd_series) < signal:
        return macd_line, 0.0, macd_line
    mult_sig = 2.0 / (signal + 1)
    sig_ema = np.mean(macd_series[:signal])
    for m in macd_series[signal:]:
        sig_ema = (m - sig_ema) * mult_sig + sig_ema
    histogram = macd_series[-1] - sig_ema
    return macd_series[-1], sig_ema, histogram

def compute_rsi(prices, period=14):
    prices = np.asarray(prices)
    if len(prices) < period + 1:
        return 50.0
    delta = np.diff(prices)
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss < 1e-10:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

def compute_bollinger_pctb(prices, period=20, std_dev=2):
    prices = np.asarray(prices)
    if len(prices) < period:
        return 0.5
    window = prices[-period:]
    ma = np.mean(window)
    std = np.std(window, ddof=1)
    upper = ma + std_dev * std
    lower = ma - std_dev * std
    if upper - lower < 1e-10:
        return 0.5
    return (prices[-1] - lower) / (upper - lower)

def compute_atr_normalized(highs, lows, closes, period=14):
    highs, lows, closes = np.asarray(highs), np.asarray(lows), np.asarray(closes)
    if len(closes) < period + 1:
        return 0.015
    trs = []
    for i in range(-period, 0):
        trs.append(max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1])))
    return np.mean(trs) / closes[-1] if closes[-1] > 0 else 0.015

def compute_volume_ratio(volumes, period=20):
    volumes = np.asarray(volumes)
    if len(volumes) < period:
        return 1.0
    avg = np.mean(volumes[-period:])
    return volumes[-1] / avg if avg > 0 else 1.0

def compute_vwap_deviation(closes, highs, lows, volumes, period=20):
    closes, highs, lows, volumes = [np.asarray(a) for a in [closes, highs, lows, volumes]]
    if len(closes) < period:
        return 0.0
    tp = (highs[-period:] + lows[-period:] + closes[-period:]) / 3.0
    vol = volumes[-period:]
    total_vol = np.sum(vol)
    if total_vol <= 0:
        return 0.0
    vwap = np.sum(tp * vol) / total_vol
    return (closes[-1] - vwap) / closes[-1] if closes[-1] > 0 else 0.0

def compute_realized_vol(returns_arr, window=20):
    if len(returns_arr) < window:
        window = len(returns_arr)
    if window < 5:
        return 0.15
    return float(np.std(returns_arr[-window:]) * np.sqrt(252))

def compute_vol_of_vol(returns_arr, sub_window=5, n_subs=4):
    if len(returns_arr) < sub_window * n_subs:
        return 0.0
    sub_vols = []
    for i in range(n_subs):
        end = len(returns_arr) - i * sub_window
        start = end - sub_window
        if start < 0:
            break
        sub_vols.append(np.std(returns_arr[start:end]))
    return float(np.std(sub_vols)) if len(sub_vols) > 1 else 0.0

def compute_iv_rv_ratio(ticker, realized_vol, option_iv_cache):
    if realized_vol < 0.01:
        return 1.0
    recent_keys = [(d, t) for (d, t) in option_iv_cache.keys() if t == ticker]
    if not recent_keys:
        return 1.0
    recent_keys.sort(reverse=True)
    iv = option_iv_cache.get(recent_keys[0])
    if iv is None:
        return 1.0
    return iv / realized_vol

def compute_features(ticker, price_data, min_data_for_trading, min_data_for_tail, option_iv_cache):
    pd = price_data[ticker]
    closes = pd['close']
    n = len(closes)
    if n < min_data_for_trading:
        return None
    closes_arr = np.array(closes)
    highs_arr = np.array(pd['high'])
    lows_arr = np.array(pd['low'])
    volumes_arr = np.array(pd['volume'])
    log_returns = np.diff(np.log(closes_arr))
    if len(log_returns) < min_data_for_trading - 1:
        return None

    hurst_short = estimate_hurst(log_returns[-min(20, len(log_returns)):])
    hurst_med = estimate_hurst(log_returns[-min(60, len(log_returns)):])
    hurst_long = estimate_hurst(log_returns[-min(120, len(log_returns)):])
    hurst = hurst_med

    tail_index = estimate_tail_index(log_returns, min_data_for_tail)
    mf_width = compute_mf_width(log_returns[-min(120, len(log_returns)):])
    realized_vol = compute_realized_vol(log_returns, window=20)
    vol_of_vol = compute_vol_of_vol(log_returns)
    iv_rv_ratio = compute_iv_rv_ratio(ticker, realized_vol, option_iv_cache)

    rsi = compute_rsi(closes_arr, period=14)
    bb_pctb = compute_bollinger_pctb(closes_arr, period=20)
    atr = compute_atr_normalized(highs_arr, lows_arr, closes_arr, period=14)
    volume_ratio = compute_volume_ratio(volumes_arr, period=20)
    vwap_dev = compute_vwap_deviation(closes_arr, highs_arr, lows_arr, volumes_arr, period=20)

    sma_50 = compute_sma(closes_arr, 50)
    sma_20 = compute_sma(closes_arr, 20)
    macd_line, macd_signal, macd_hist = compute_macd(closes_arr)
    price = closes_arr[-1]
    above_sma50 = 1.0 if price > sma_50 else 0.0
    sma_trend = (sma_20 - sma_50) / sma_50 if sma_50 > 0 else 0.0

    ret_1d = log_returns[-1]
    ret_5d = np.sum(log_returns[-min(5, len(log_returns)):])
    ret_20d = np.sum(log_returns[-min(20, len(log_returns)):])

    drawdown_20d = 0.0
    if n >= 20:
        peak = np.max(closes_arr[-20:])
        drawdown_20d = (price - peak) / peak if peak > 0 else 0.0

    hurst_divergence = hurst_short - hurst_long

    return {
        'hurst': hurst, 'hurst_short': hurst_short, 'hurst_med': hurst_med,
        'hurst_long': hurst_long, 'hurst_divergence': hurst_divergence,
        'tail_index': tail_index, 'mf_width': mf_width,
        'realized_vol': realized_vol, 'vol_of_vol': vol_of_vol,
        'iv_rv_ratio': iv_rv_ratio,
        'rsi': rsi, 'bb_pctb': bb_pctb, 'atr': atr,
        'volume_ratio': volume_ratio, 'vwap_dev': vwap_dev,
        'above_sma50': above_sma50, 'sma_trend': sma_trend,
        'macd_line': macd_line, 'macd_signal': macd_signal, 'macd_hist': macd_hist,
        'ret_1d': ret_1d, 'ret_5d': ret_5d, 'ret_20d': ret_20d,
        'drawdown_20d': drawdown_20d, 'price': price,
    }


def classify_regime(features):
    h_short = features['hurst_short']
    h_med = features['hurst_med']
    h_long = features['hurst_long']
    rv = features['realized_vol']
    tail = features['tail_index']
    mf = features['mf_width']
    drawdown = features['drawdown_20d']
    ret_20d = features['ret_20d']
    above_sma50 = features['above_sma50']
    sma_trend = features['sma_trend']
    macd_hist = features['macd_hist']

    vol_score = np.clip((rv - 0.10) / 0.35, 0, 1)
    tail_score = np.clip((2.5 - tail) / 1.5, 0, 1)
    mf_score = np.clip(mf / 0.5, 0, 1)
    dd_score = np.clip(-drawdown / 0.10, 0, 1)
    danger = 0.35 * vol_score + 0.15 * tail_score + 0.20 * mf_score + 0.25 * dd_score

    if tail < 1.8 and mf > 0.4:
        danger = max(danger, 0.7)
    if tail < 1.8 and rv > 0.4:
        danger = max(danger, 0.5)

    if danger >= 0.7:
        return 4, danger
    if danger >= 0.5:
        return 3, danger

    fractal_persistent = h_med > 0.52 and h_short > 0.48
    fractal_anti = h_med < 0.48 or h_short < 0.45

    if fractal_anti:
        return 2, danger

    if fractal_persistent:
        if ret_20d > 0 and above_sma50 > 0.5:
            return 0, danger
        if ret_20d < 0 and above_sma50 < 0.5:
            return 1, danger
        if above_sma50 > 0.5 and sma_trend > 0:
            return 0, danger
        if above_sma50 < 0.5 and sma_trend < 0:
            return 1, danger
        if ret_20d > 0.0:
            return 0, danger
        else:
            return 1, danger

    if above_sma50 > 0.5 and sma_trend > 0 and macd_hist > 0 and h_med > 0.48:
        return 0, danger
    if above_sma50 < 0.5 and sma_trend < 0 and macd_hist < 0 and h_med > 0.48:
        return 1, danger

    return 3, danger


def generate_signal(features, regime, danger, cfg):
    rsi = features['rsi']
    bb_pctb = features['bb_pctb']
    ret_5d = features['ret_5d']
    ret_1d = features['ret_1d']
    ret_20d = features['ret_20d']
    volume_ratio = features['volume_ratio']
    iv_rv_ratio = features.get('iv_rv_ratio', 1.0)
    macd_hist = features['macd_hist']
    above_sma50 = features['above_sma50']
    h_med = features['hurst_med']
    h_short = features['hurst_short']
    tail = features['tail_index']
    mf = features['mf_width']

    base = {'regime': regime, 'danger_score': danger, **features}

    if regime >= 4:
        return {'direction': 0, 'strength': 0.0, 'reason': 'crisis_regime', **base}

    regime_penalty = 1.0

    def fractal_confidence(h, threshold, side='above'):
        if side == 'above':
            return float(np.clip((h - threshold) / 0.15, 0.3, 1.0))
        return float(np.clip((threshold - h) / 0.15, 0.3, 1.0))

    def tail_mispricing(tail_idx, direction):
        if direction > 0:
            return float(np.clip(2.5 / max(tail_idx, 0.8), 0.6, 1.25))
        return float(np.clip(2.5 / max(tail_idx, 0.8), 0.6, 1.15))

    def iv_rv_adj(ratio, direction):
        if direction > 0:
            if ratio < 0.90:
                return 1.12
            if ratio < 1.0:
                return 1.05
            if ratio > 1.25:
                return 0.85
            return 1.0
        else:
            if ratio < 0.90:
                return 1.10
            if ratio > 1.25:
                return 0.88
            return 1.0

    if regime == 0:
        fc = fractal_confidence(h_med, 0.55, 'above')
        tm = tail_mispricing(tail, 1)
        iv_adj = iv_rv_adj(iv_rv_ratio, 1)

        if (rsi < cfg['rsi_oversold'] and ret_5d < -0.02 and bb_pctb < 0.5 and macd_hist > -0.5 and above_sma50 > 0.5):
                tech_q = float(np.clip((cfg['rsi_oversold'] - rsi) / 25.0 + 0.45, 0.4, 0.95))
                strength = fc * tm * tech_q * iv_adj * regime_penalty
                return {'direction': 1, 'strength': np.clip(strength, 0, 1.0), 'reason': 'trend_pullback_buy', **base}

        if (ret_5d > 0.02 and rsi > 52 and rsi < 70 and macd_hist > 0
                and bb_pctb > 0.5 and bb_pctb < 0.9 and above_sma50 > 0.5):
            tech_q = float(np.clip(ret_5d * 20 + 0.40, 0.4, 0.90))
            strength = fc * tm * tech_q * iv_adj * regime_penalty
            return {'direction': 1, 'strength': np.clip(strength, 0, 1.0), 'reason': 'trend_continuation', **base}

    if regime == 1:
        fc = fractal_confidence(h_med, 0.55, 'above')
        tm = tail_mispricing(tail, -1)
        iv_adj = iv_rv_adj(iv_rv_ratio, -1)

        if (rsi > cfg['rsi_overbought'] and ret_5d > 0.02 and bb_pctb > 0.5 and macd_hist < 0.5 and above_sma50 < 0.5):
                tech_q = float(np.clip((cfg['rsi_overbought'] - rsi) / 25.0 + 0.45, 0.4, 0.95))
                strength = fc * tm * tech_q * iv_adj * regime_penalty
                return {'direction': -1, 'strength': np.clip(strength, 0, 1.0), 'reason': 'trend_pullback_sell', **base}

        if (ret_5d < -0.02 and rsi < 48 and rsi > 30 and macd_hist < 0
                and bb_pctb < 0.5 and bb_pctb > 0.1 and above_sma50 < 0.5 
                and tail < 2.8 and mf > 0.18):
            tech_q = float(np.clip(abs(ret_5d) * 20 + 0.40, 0.4, 0.90))
            strength = fc * tm * tech_q * iv_adj * regime_penalty
            return {'direction': -1, 'strength': np.clip(strength, 0, 1.0), 'reason': 'downtrend_continuation', **base}

    if regime == 2:
        fc = fractal_confidence(h_med, 0.42, 'below')
        tm = 1.0
        iv_adj_buy = iv_rv_adj(iv_rv_ratio, 1)
        iv_adj_sell = iv_rv_adj(iv_rv_ratio, -1)

        if (rsi < cfg['rsi_mr_oversold'] and bb_pctb < 0.25
                and ret_20d > -0.08 and ret_1d > -0.04):
            tech_q = float(np.clip((cfg['rsi_mr_oversold'] - rsi) / 18.0 + 0.45, 0.4, 0.90))
            if iv_rv_ratio < 1.0:
                tech_q = min(tech_q + 0.08, 0.95)
            strength = fc * tm * tech_q * iv_adj_buy * regime_penalty
            return {'direction': 1, 'strength': np.clip(strength, 0, 1.0), 'reason': 'mean_reversion_buy', **base}
        
        if (rsi > cfg['rsi_mr_overbought'] and bb_pctb > 0.95
                and ret_20d < 0.08 and ret_1d < 0.02 and above_sma50 < 0.5):
            tech_q = float(np.clip((rsi - cfg['rsi_mr_overbought']) / 18.0 + 0.45, 0.4, 0.90))
            if iv_rv_ratio < 1.0:
                tech_q = min(tech_q + 0.08, 0.95)
            strength = fc * tm * tech_q * iv_adj_sell * regime_penalty
            return {'direction': -1, 'strength': np.clip(strength, 0, 1.0), 'reason': 'mean_reversion_sell', **base}

    return {'direction': 0, 'strength': 0.0, 'reason': 'no_signal', **base}

def compute_position_size(signal, position_size_max):
    base_size = position_size_max * signal['strength']
    tail_index = signal.get('tail_index', 2.0)
    atr = signal.get('atr', 0.015)
    danger = signal.get('danger_score', 0.0)
    tail_adj = min(tail_index / 2.5, 1.0)
    atr_adj = 0.02 / atr if atr > 0.02 else 1.0
    danger_adj = max(1.0 - danger, 0.4)
    risk_adj = min(tail_adj, atr_adj, danger_adj)
    base_size *= max(risk_adj, 0.3)
    return np.clip(base_size, 0.03, position_size_max)