"""
Deriv Synthetic Indices Backtesting Engine
Instruments:
  - Volatility 100 (1s) Index  [1HZ100V]
  - Volatility 75 (1s) Index   [1HZ75V]
  - Volatility 100 Index        [R_100]
  - Volatility 75 Index         [R_75]
  - Step Index                   [stpRNG]

Fetches real candle data from Deriv WebSocket API.
"""

import asyncio
import json
import pandas as pd
import numpy as np
import websockets
from datetime import datetime, timedelta
import time as _time


# ---------------------------------------------------------------------------
# Deriv API - Data Fetching
# ---------------------------------------------------------------------------

DERIV_WS = "wss://ws.derivws.com/websockets/v3?app_id=1089"

INSTRUMENTS = {
    "1HZ100V": {"name": "Volatility 100 (1s)", "pip": 0.01},
    "1HZ75V":  {"name": "Volatility 75 (1s)", "pip": 0.01},
    "R_100":   {"name": "Volatility 100 Index", "pip": 0.01},
    "R_75":    {"name": "Volatility 75 Index", "pip": 0.01},
    "stpRNG":  {"name": "Step Index", "pip": 0.01},
}


async def _fetch_candles(symbol, granularity=3600, count=5000):
    """
    Fetch candle data from Deriv API.
    granularity: 60=1m, 300=5m, 900=15m, 3600=1h, 14400=4h, 86400=1d
    max count per request: 5000
    """
    all_candles = []
    end_time = int(datetime.now().timestamp())

    # Fetch in batches (5000 candles per request)
    batches_needed = 2  # 10000 candles total
    for batch in range(batches_needed):
        try:
            async with websockets.connect(DERIV_WS) as ws:
                request = {
                    "ticks_history": symbol,
                    "adjust_start_time": 1,
                    "count": count,
                    "end": str(end_time),
                    "granularity": granularity,
                    "style": "candles",
                }
                await ws.send(json.dumps(request))
                response = json.loads(await ws.recv())

                if "error" in response:
                    print(f"  API Error for {symbol}: {response['error']['message']}")
                    break

                candles = response.get("candles", [])
                if not candles:
                    break

                all_candles = candles + all_candles
                # Set end time to before the earliest candle for next batch
                end_time = candles[0]["epoch"] - 1

        except Exception as e:
            print(f"  Connection error: {e}")
            break

    if not all_candles:
        return None

    df = pd.DataFrame(all_candles)
    df["time"] = pd.to_datetime(df["epoch"], unit="s")
    df.set_index("time", inplace=True)
    df = df.rename(columns={"open": "Open", "high": "High",
                             "low": "Low", "close": "Close"})
    df["Open"] = df["Open"].astype(float)
    df["High"] = df["High"].astype(float)
    df["Low"] = df["Low"].astype(float)
    df["Close"] = df["Close"].astype(float)
    df["Volume"] = 1  # Synthetic indices don't have real volume

    # Remove duplicates
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()
    return df


def fetch_candles(symbol, granularity=3600, count=5000):
    """Synchronous wrapper for async fetch."""
    return asyncio.run(_fetch_candles(symbol, granularity, count))


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def compute_atr(df, period=14):
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_stoch_rsi(series, period=14, smooth_k=3, smooth_d=3):
    rsi = compute_rsi(series, period)
    rsi_min = rsi.rolling(period).min()
    rsi_max = rsi.rolling(period).max()
    stoch = (rsi - rsi_min) / (rsi_max - rsi_min + 1e-10) * 100
    k = stoch.rolling(smooth_k).mean()
    d = k.rolling(smooth_d).mean()
    return k, d


def compute_bollinger(series, period=20, std_dev=2):
    mid = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    return mid + std_dev * std, mid, mid - std_dev * std


def add_indicators(df):
    """Add all technical indicators."""
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    # EMAs
    df["EMA_5"] = compute_ema(close, 5)
    df["EMA_9"] = compute_ema(close, 9)
    df["EMA_21"] = compute_ema(close, 21)
    df["EMA_50"] = compute_ema(close, 50)
    df["EMA_200"] = compute_ema(close, 200)

    # RSI
    df["RSI"] = compute_rsi(close, 14)

    # Stochastic RSI
    df["StochRSI_K"], df["StochRSI_D"] = compute_stoch_rsi(close)

    # MACD
    df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = compute_macd(close)

    # Bollinger Bands
    df["BB_Upper"], df["BB_Mid"], df["BB_Lower"] = compute_bollinger(close)
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Mid"]

    # ATR
    df["ATR"] = compute_atr(df, 14)
    df["ATR_Pct"] = df["ATR"] / close * 100

    # Rate of Change
    df["ROC_5"] = close.pct_change(5) * 100
    df["ROC_10"] = close.pct_change(10) * 100

    # Support/Resistance
    df["Resistance_20"] = high.rolling(20).max()
    df["Support_20"] = low.rolling(20).min()

    df.dropna(inplace=True)
    return df


# ---------------------------------------------------------------------------
# Trade Exit Logic
# ---------------------------------------------------------------------------

def _check_exit(row, pos, trailing_sl=None):
    """Check SL/TP. Returns (closed, pnl_pct, exit_price)."""
    sl = trailing_sl if trailing_sl is not None else pos["sl"]

    if pos["type"] == "BUY":
        if row["Low"] <= sl:
            pnl = (sl - pos["entry"]) / pos["entry"] * 100
            return True, pnl, sl
        if row["High"] >= pos["tp"]:
            pnl = (pos["tp"] - pos["entry"]) / pos["entry"] * 100
            return True, pnl, pos["tp"]
    else:
        if row["High"] >= sl:
            pnl = (pos["entry"] - sl) / pos["entry"] * 100
            return True, pnl, sl
        if row["Low"] <= pos["tp"]:
            pnl = (pos["entry"] - pos["tp"]) / pos["entry"] * 100
            return True, pnl, pos["tp"]
    return False, 0, 0


# ---------------------------------------------------------------------------
# STRATEGIES (tuned for synthetic indices)
# ---------------------------------------------------------------------------

def strategy_ema_crossover(df, fast_p=9, slow_p=21, sl_atr=1.5, tp_atr=2.0):
    """EMA crossover with ATR-based SL/TP."""
    trades, pos = [], None
    fast = compute_ema(df["Close"], fast_p)
    slow = compute_ema(df["Close"], slow_p)

    for i in range(1, len(df)):
        r = df.iloc[i]
        if pos is None:
            if fast.iloc[i-1] <= slow.iloc[i-1] and fast.iloc[i] > slow.iloc[i]:
                pos = {
                    "type": "BUY", "entry": r["Close"],
                    "sl": r["Close"] - r["ATR"] * sl_atr,
                    "tp": r["Close"] + r["ATR"] * tp_atr,
                    "entry_date": r.name,
                }
            elif fast.iloc[i-1] >= slow.iloc[i-1] and fast.iloc[i] < slow.iloc[i]:
                pos = {
                    "type": "SELL", "entry": r["Close"],
                    "sl": r["Close"] + r["ATR"] * sl_atr,
                    "tp": r["Close"] - r["ATR"] * tp_atr,
                    "entry_date": r.name,
                }
        else:
            closed, pnl, _ = _check_exit(r, pos)
            if closed:
                pos["exit_date"] = r.name
                pos["pnl"] = round(pnl, 2)
                pos["result"] = "WIN" if pnl > 0 else "LOSS"
                trades.append(pos)
                pos = None
    return trades


def strategy_rsi_reversal(df, oversold=30, overbought=70, sl_atr=1.5, tp_atr=2.0):
    """RSI mean reversion."""
    trades, pos = [], None
    for i in range(1, len(df)):
        r, p = df.iloc[i], df.iloc[i - 1]
        if pos is None:
            if p["RSI"] < oversold and r["RSI"] >= oversold:
                pos = {
                    "type": "BUY", "entry": r["Close"],
                    "sl": r["Close"] - r["ATR"] * sl_atr,
                    "tp": r["Close"] + r["ATR"] * tp_atr,
                    "entry_date": r.name,
                }
            elif p["RSI"] > overbought and r["RSI"] <= overbought:
                pos = {
                    "type": "SELL", "entry": r["Close"],
                    "sl": r["Close"] + r["ATR"] * sl_atr,
                    "tp": r["Close"] - r["ATR"] * tp_atr,
                    "entry_date": r.name,
                }
        else:
            closed, pnl, _ = _check_exit(r, pos)
            if closed:
                pos["exit_date"] = r.name
                pos["pnl"] = round(pnl, 2)
                pos["result"] = "WIN" if pnl > 0 else "LOSS"
                trades.append(pos)
                pos = None
    return trades


def strategy_macd_momentum(df, sl_atr=1.5, tp_atr=2.0):
    """MACD histogram crossover."""
    trades, pos = [], None
    for i in range(1, len(df)):
        r, p = df.iloc[i], df.iloc[i - 1]
        if pos is None:
            if p["MACD_Hist"] <= 0 and r["MACD_Hist"] > 0 and r["MACD"] > r["MACD_Signal"]:
                pos = {
                    "type": "BUY", "entry": r["Close"],
                    "sl": r["Close"] - r["ATR"] * sl_atr,
                    "tp": r["Close"] + r["ATR"] * tp_atr,
                    "entry_date": r.name,
                }
            elif p["MACD_Hist"] >= 0 and r["MACD_Hist"] < 0 and r["MACD"] < r["MACD_Signal"]:
                pos = {
                    "type": "SELL", "entry": r["Close"],
                    "sl": r["Close"] + r["ATR"] * sl_atr,
                    "tp": r["Close"] - r["ATR"] * tp_atr,
                    "entry_date": r.name,
                }
        else:
            closed, pnl, _ = _check_exit(r, pos)
            if closed:
                pos["exit_date"] = r.name
                pos["pnl"] = round(pnl, 2)
                pos["result"] = "WIN" if pnl > 0 else "LOSS"
                trades.append(pos)
                pos = None
    return trades


def strategy_bollinger_bounce(df, sl_atr=2.0, tp_atr=1.5):
    """Bollinger Band bounce - buy at lower, sell at upper."""
    trades, pos = [], None
    for i in range(1, len(df)):
        r, p = df.iloc[i], df.iloc[i - 1]
        if pos is None:
            # Buy: price touches/crosses below lower BB then bounces back
            if p["Close"] <= p["BB_Lower"] and r["Close"] > r["BB_Lower"] and r["RSI"] < 40:
                pos = {
                    "type": "BUY", "entry": r["Close"],
                    "sl": r["Close"] - r["ATR"] * sl_atr,
                    "tp": r["BB_Mid"],  # Target middle band
                    "entry_date": r.name,
                }
            # Sell: price touches/crosses above upper BB then drops
            elif p["Close"] >= p["BB_Upper"] and r["Close"] < r["BB_Upper"] and r["RSI"] > 60:
                pos = {
                    "type": "SELL", "entry": r["Close"],
                    "sl": r["Close"] + r["ATR"] * sl_atr,
                    "tp": r["BB_Mid"],
                    "entry_date": r.name,
                }
        else:
            closed, pnl, _ = _check_exit(r, pos)
            if closed:
                pos["exit_date"] = r.name
                pos["pnl"] = round(pnl, 2)
                pos["result"] = "WIN" if pnl > 0 else "LOSS"
                trades.append(pos)
                pos = None
    return trades


def strategy_stoch_rsi_cross(df, sl_atr=1.5, tp_atr=1.5):
    """Stochastic RSI K/D crossover in OB/OS zones."""
    trades, pos = [], None
    for i in range(1, len(df)):
        r, p = df.iloc[i], df.iloc[i - 1]
        if pos is None:
            # Buy: StochRSI K crosses above D in oversold zone
            if (p["StochRSI_K"] <= p["StochRSI_D"] and
                    r["StochRSI_K"] > r["StochRSI_D"] and
                    r["StochRSI_K"] < 30):
                pos = {
                    "type": "BUY", "entry": r["Close"],
                    "sl": r["Close"] - r["ATR"] * sl_atr,
                    "tp": r["Close"] + r["ATR"] * tp_atr,
                    "entry_date": r.name,
                }
            # Sell: StochRSI K crosses below D in overbought zone
            elif (p["StochRSI_K"] >= p["StochRSI_D"] and
                  r["StochRSI_K"] < r["StochRSI_D"] and
                  r["StochRSI_K"] > 70):
                pos = {
                    "type": "SELL", "entry": r["Close"],
                    "sl": r["Close"] + r["ATR"] * sl_atr,
                    "tp": r["Close"] - r["ATR"] * tp_atr,
                    "entry_date": r.name,
                }
        else:
            closed, pnl, _ = _check_exit(r, pos)
            if closed:
                pos["exit_date"] = r.name
                pos["pnl"] = round(pnl, 2)
                pos["result"] = "WIN" if pnl > 0 else "LOSS"
                trades.append(pos)
                pos = None
    return trades


def strategy_trend_pullback(df, sl_atr=2.0, tp_atr=1.5):
    """Trend pullback to EMA 21 in a strong trend."""
    trades, pos, trailing_sl = [], None, None
    for i in range(2, len(df)):
        r, p = df.iloc[i], df.iloc[i - 1]
        if pos is None:
            trailing_sl = None
            bull_stack = r["EMA_9"] > r["EMA_21"] > r["EMA_50"]
            bear_stack = r["EMA_9"] < r["EMA_21"] < r["EMA_50"]
            near_ema21 = abs(r["Close"] - r["EMA_21"]) < r["ATR"] * 0.6

            if bull_stack and near_ema21 and r["StochRSI_K"] < 30 and r["Close"] > r["EMA_21"]:
                pos = {
                    "type": "BUY", "entry": r["Close"],
                    "sl": r["EMA_21"] - r["ATR"] * 0.5,
                    "tp": r["Close"] + r["ATR"] * tp_atr,
                    "entry_date": r.name,
                }
            elif bear_stack and near_ema21 and r["StochRSI_K"] > 70 and r["Close"] < r["EMA_21"]:
                pos = {
                    "type": "SELL", "entry": r["Close"],
                    "sl": r["EMA_21"] + r["ATR"] * 0.5,
                    "tp": r["Close"] - r["ATR"] * tp_atr,
                    "entry_date": r.name,
                }
        else:
            # Trailing stop
            if pos["type"] == "BUY":
                profit = r["High"] - pos["entry"]
                if profit >= r["ATR"]:
                    new_trail = r["High"] - r["ATR"]
                    if trailing_sl is None or new_trail > trailing_sl:
                        trailing_sl = new_trail
            else:
                profit = pos["entry"] - r["Low"]
                if profit >= r["ATR"]:
                    new_trail = r["Low"] + r["ATR"]
                    if trailing_sl is None or new_trail < trailing_sl:
                        trailing_sl = new_trail

            closed, pnl, _ = _check_exit(r, pos, trailing_sl)
            if closed:
                pos["exit_date"] = r.name
                pos["pnl"] = round(pnl, 2)
                pos["result"] = "WIN" if pnl > 0 else "LOSS"
                trades.append(pos)
                pos = None
                trailing_sl = None
    return trades


def strategy_breakout(df, sl_atr=1.5, tp_atr=2.0):
    """Breakout of 20-period high/low with momentum confirmation."""
    trades, pos = [], None
    for i in range(2, len(df)):
        r, p = df.iloc[i], df.iloc[i - 1]
        if pos is None:
            bb_expanding = r["BB_Width"] > df.iloc[max(0,i-5):i]["BB_Width"].mean()

            if (r["Close"] > p["Resistance_20"] and bb_expanding and
                    r["RSI"] > 50 and r["RSI"] < 80 and r["MACD_Hist"] > 0):
                pos = {
                    "type": "BUY", "entry": r["Close"],
                    "sl": r["Close"] - r["ATR"] * sl_atr,
                    "tp": r["Close"] + r["ATR"] * tp_atr,
                    "entry_date": r.name,
                }
            elif (r["Close"] < p["Support_20"] and bb_expanding and
                  r["RSI"] < 50 and r["RSI"] > 20 and r["MACD_Hist"] < 0):
                pos = {
                    "type": "SELL", "entry": r["Close"],
                    "sl": r["Close"] + r["ATR"] * sl_atr,
                    "tp": r["Close"] - r["ATR"] * tp_atr,
                    "entry_date": r.name,
                }
        else:
            closed, pnl, _ = _check_exit(r, pos)
            if closed:
                pos["exit_date"] = r.name
                pos["pnl"] = round(pnl, 2)
                pos["result"] = "WIN" if pnl > 0 else "LOSS"
                trades.append(pos)
                pos = None
    return trades


def strategy_step_index_range(df, sl_atr=1.0, tp_atr=0.8):
    """
    Step Index specific: trades in tight range.
    Step Index moves in fixed steps (0.1 up or down per tick).
    Good for range-bound strategies with tight SL/TP.
    """
    trades, pos = [], None
    for i in range(2, len(df)):
        r, p = df.iloc[i], df.iloc[i - 1]
        if pos is None:
            # Buy near support with RSI oversold
            if (r["Close"] <= r["BB_Lower"] and r["RSI"] < 35 and
                    r["StochRSI_K"] < 20):
                pos = {
                    "type": "BUY", "entry": r["Close"],
                    "sl": r["Close"] - r["ATR"] * sl_atr,
                    "tp": r["Close"] + r["ATR"] * tp_atr,
                    "entry_date": r.name,
                }
            # Sell near resistance with RSI overbought
            elif (r["Close"] >= r["BB_Upper"] and r["RSI"] > 65 and
                  r["StochRSI_K"] > 80):
                pos = {
                    "type": "SELL", "entry": r["Close"],
                    "sl": r["Close"] + r["ATR"] * sl_atr,
                    "tp": r["Close"] - r["ATR"] * tp_atr,
                    "entry_date": r.name,
                }
        else:
            closed, pnl, _ = _check_exit(r, pos)
            if closed:
                pos["exit_date"] = r.name
                pos["pnl"] = round(pnl, 2)
                pos["result"] = "WIN" if pnl > 0 else "LOSS"
                trades.append(pos)
                pos = None
    return trades


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def calc_stats(trades):
    if not trades:
        return {"total": 0, "wins": 0, "losses": 0, "win_rate": 0,
                "total_return": 0, "avg_win": 0, "avg_loss": 0,
                "profit_factor": 0, "max_dd": 0, "avg_rr": 0, "expectancy": 0}

    wins = [t for t in trades if t["result"] == "WIN"]
    losses = [t for t in trades if t["result"] == "LOSS"]
    total_ret = sum(t["pnl"] for t in trades)

    gp = sum(t["pnl"] for t in wins) if wins else 0
    gl = abs(sum(t["pnl"] for t in losses)) if losses else 0
    aw = gp / len(wins) if wins else 0
    al = gl / len(losses) if losses else 0

    cumulative = np.cumsum([t["pnl"] for t in trades])
    peak = np.maximum.accumulate(cumulative)
    dd = peak - cumulative
    max_dd = float(np.max(dd)) if len(dd) > 0 else 0

    return {
        "total": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(trades) * 100, 1),
        "total_return": round(total_ret, 2),
        "avg_win": round(aw, 2),
        "avg_loss": round(-al, 2),
        "profit_factor": round(gp / gl, 2) if gl > 0 else float("inf"),
        "max_dd": round(max_dd, 2),
        "avg_rr": round(aw / al, 2) if al > 0 else 0,
        "expectancy": round(total_ret / len(trades), 2),
    }


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

def optimize_instrument(symbol, df, min_trades=15):
    """Sweep parameters for one instrument."""
    results = []

    # EMA Crossover sweep
    for fast, slow in [(5, 13), (5, 21), (9, 21), (7, 21), (8, 34), (12, 26)]:
        for sl in [1.0, 1.5, 2.0, 2.5, 3.0]:
            for tp in [0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]:
                trades = strategy_ema_crossover(df, fast, slow, sl, tp)
                if len(trades) >= min_trades:
                    s = calc_stats(trades)
                    results.append({"strategy": f"EMA({fast}/{slow})",
                                    "params": f"SL:{sl} TP:{tp}", **s})

    # RSI sweep
    for os_val in [25, 30, 35]:
        for ob_val in [65, 70, 75]:
            for sl in [1.0, 1.5, 2.0, 2.5, 3.0]:
                for tp in [0.5, 0.75, 1.0, 1.5, 2.0, 2.5]:
                    trades = strategy_rsi_reversal(df, os_val, ob_val, sl, tp)
                    if len(trades) >= min_trades:
                        s = calc_stats(trades)
                        results.append({"strategy": f"RSI({os_val}/{ob_val})",
                                        "params": f"SL:{sl} TP:{tp}", **s})

    # MACD sweep
    for sl in [1.0, 1.5, 2.0, 2.5, 3.0]:
        for tp in [0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]:
            trades = strategy_macd_momentum(df, sl, tp)
            if len(trades) >= min_trades:
                s = calc_stats(trades)
                results.append({"strategy": "MACD", "params": f"SL:{sl} TP:{tp}", **s})

    # Bollinger Bounce sweep
    for sl in [1.0, 1.5, 2.0, 2.5]:
        for tp in [1.0, 1.5, 2.0]:
            trades = strategy_bollinger_bounce(df, sl, tp)
            if len(trades) >= min_trades:
                s = calc_stats(trades)
                results.append({"strategy": "BollingerBounce",
                                "params": f"SL:{sl} TP:{tp}", **s})

    # Stochastic RSI sweep
    for sl in [1.0, 1.5, 2.0, 2.5]:
        for tp in [0.75, 1.0, 1.5, 2.0]:
            trades = strategy_stoch_rsi_cross(df, sl, tp)
            if len(trades) >= min_trades:
                s = calc_stats(trades)
                results.append({"strategy": "StochRSI",
                                "params": f"SL:{sl} TP:{tp}", **s})

    # Trend Pullback sweep
    for sl in [1.5, 2.0, 2.5, 3.0]:
        for tp in [1.0, 1.5, 2.0, 2.5]:
            trades = strategy_trend_pullback(df, sl, tp)
            if len(trades) >= min_trades:
                s = calc_stats(trades)
                results.append({"strategy": "TrendPullback",
                                "params": f"SL:{sl} TP:{tp}", **s})

    # Breakout sweep
    for sl in [1.0, 1.5, 2.0, 2.5]:
        for tp in [1.0, 1.5, 2.0, 2.5, 3.0]:
            trades = strategy_breakout(df, sl, tp)
            if len(trades) >= min_trades:
                s = calc_stats(trades)
                results.append({"strategy": "Breakout",
                                "params": f"SL:{sl} TP:{tp}", **s})

    # Step Index specific (only for stpRNG)
    if symbol == "stpRNG":
        for sl in [0.5, 0.75, 1.0, 1.5, 2.0]:
            for tp in [0.5, 0.75, 1.0, 1.5]:
                trades = strategy_step_index_range(df, sl, tp)
                if len(trades) >= min_trades:
                    s = calc_stats(trades)
                    results.append({"strategy": "StepRange",
                                    "params": f"SL:{sl} TP:{tp}", **s})

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_all():
    """Fetch data and run backtests for all Deriv synthetic indices."""
    all_best = {}

    for symbol, info in INSTRUMENTS.items():
        print(f"\n{'='*85}")
        print(f"  {info['name']} [{symbol}]")
        print(f"{'='*85}")

        # Fetch 1-hour candles (good balance of data and detail)
        print(f"  Fetching 1H candles from Deriv API...")
        df = fetch_candles(symbol, granularity=3600, count=5000)

        if df is None or len(df) < 200:
            print(f"  ERROR: Insufficient data ({0 if df is None else len(df)} bars)")
            continue

        df = add_indicators(df)
        print(f"  Data: {len(df)} bars")
        print(f"  Range: {df.index[0]} to {df.index[-1]}")
        print(f"  Price: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
        print(f"  Current: ${df['Close'].iloc[-1]:.2f} | RSI: {df['RSI'].iloc[-1]:.1f} | "
              f"ATR: {df['ATR'].iloc[-1]:.2f}")

        # Run default strategies
        print(f"\n  --- Default Strategies ---")
        strategies = {
            "EMA Crossover (9/21)": strategy_ema_crossover(df),
            "RSI Reversal (30/70)": strategy_rsi_reversal(df),
            "MACD Momentum": strategy_macd_momentum(df),
            "Bollinger Bounce": strategy_bollinger_bounce(df),
            "StochRSI Cross": strategy_stoch_rsi_cross(df),
            "Trend Pullback": strategy_trend_pullback(df),
            "Breakout": strategy_breakout(df),
        }
        if symbol == "stpRNG":
            strategies["Step Range"] = strategy_step_index_range(df)

        print(f"  {'Strategy':<25} {'#':>5} {'WR%':>6} {'Return':>8} {'PF':>6} {'R:R':>5} {'Expect':>7}")
        print(f"  {'-'*65}")
        for name, trades in sorted(strategies.items(),
                                    key=lambda x: calc_stats(x[1])["total_return"],
                                    reverse=True):
            s = calc_stats(trades)
            pf_str = f"{s['profit_factor']:>5.2f}" if s['profit_factor'] < 100 else "  inf"
            print(f"  {name:<25} {s['total']:>5} {s['win_rate']:>5.1f}% "
                  f"{s['total_return']:>7.1f}% {pf_str} "
                  f"{s['avg_rr']:>5.2f} {s['expectancy']:>6.2f}%")

        # Optimize
        print(f"\n  --- Optimizing... ---")
        opt = optimize_instrument(symbol, df)

        if not opt:
            print(f"  No profitable configs found")
            continue

        df_opt = pd.DataFrame(opt)
        profitable = df_opt[df_opt["total_return"] > 0]
        print(f"  Configs: {len(df_opt)} | Profitable: {len(profitable)}")

        # Balanced score
        df_opt["score"] = (
            (df_opt["win_rate"] / 100) *
            df_opt["profit_factor"].clip(upper=10) *
            df_opt["expectancy"].clip(lower=0) *
            np.log(df_opt["total"].clip(lower=1))
        )

        top = df_opt.sort_values("score", ascending=False).head(10)
        print(f"\n  TOP 10 BY BALANCED SCORE:")
        print(f"  {'Strategy':<22} {'Params':<14} {'#':>5} {'WR%':>6} {'Ret%':>8} {'PF':>6} {'R:R':>5} {'Exp':>7}")
        print(f"  {'-'*75}")
        for _, r in top.iterrows():
            pf_str = f"{r['profit_factor']:>5.2f}" if r['profit_factor'] < 100 else "  inf"
            print(f"  {r['strategy']:<22} {r['params']:<14} "
                  f"{r['total']:>5} {r['win_rate']:>5.1f}% "
                  f"{r['total_return']:>7.1f}% {pf_str} "
                  f"{r['avg_rr']:>5.2f} {r['expectancy']:>6.2f}%")

        # Top by Win Rate
        top_wr = df_opt[df_opt["total"] >= 20].sort_values("win_rate", ascending=False).head(5)
        if len(top_wr) > 0:
            print(f"\n  TOP 5 BY WIN RATE (20+ trades):")
            print(f"  {'Strategy':<22} {'Params':<14} {'#':>5} {'WR%':>6} {'Ret%':>8} {'PF':>6}")
            print(f"  {'-'*65}")
            for _, r in top_wr.iterrows():
                pf_str = f"{r['profit_factor']:>5.2f}" if r['profit_factor'] < 100 else "  inf"
                print(f"  {r['strategy']:<22} {r['params']:<14} "
                      f"{r['total']:>5} {r['win_rate']:>5.1f}% "
                      f"{r['total_return']:>7.1f}% {pf_str}")

        # Top by total return
        top_ret = df_opt.sort_values("total_return", ascending=False).head(5)
        print(f"\n  TOP 5 BY TOTAL RETURN:")
        print(f"  {'Strategy':<22} {'Params':<14} {'#':>5} {'WR%':>6} {'Ret%':>8} {'PF':>6}")
        print(f"  {'-'*65}")
        for _, r in top_ret.iterrows():
            pf_str = f"{r['profit_factor']:>5.2f}" if r['profit_factor'] < 100 else "  inf"
            print(f"  {r['strategy']:<22} {r['params']:<14} "
                  f"{r['total']:>5} {r['win_rate']:>5.1f}% "
                  f"{r['total_return']:>7.1f}% {pf_str}")

        best = df_opt.sort_values("score", ascending=False).iloc[0]
        all_best[symbol] = {
            "name": info["name"],
            "strategy": best["strategy"],
            "params": best["params"],
            "trades": int(best["total"]),
            "win_rate": best["win_rate"],
            "total_return": best["total_return"],
            "profit_factor": best["profit_factor"],
            "expectancy": best["expectancy"],
            "max_dd": best["max_dd"],
            "avg_rr": best["avg_rr"],
        }

    # Final Summary
    print(f"\n\n{'*'*90}")
    print(f"  BEST STRATEGY FOR EACH DERIV SYNTHETIC INDEX")
    print(f"{'*'*90}")
    print(f"{'Symbol':<10} {'Name':<22} {'Strategy':<22} {'Params':<14} {'#':>5} {'WR%':>6} {'Ret%':>8} {'PF':>6}")
    print(f"{'-'*90}")
    for sym, r in all_best.items():
        pf_str = f"{r['profit_factor']:>5.2f}" if r['profit_factor'] < 100 else "  inf"
        print(f"{sym:<10} {r['name']:<22} {r['strategy']:<22} {r['params']:<14} "
              f"{r['trades']:>5} {r['win_rate']:>5.1f}% "
              f"{r['total_return']:>7.1f}% {pf_str}")

    for sym, r in all_best.items():
        print(f"\n  {r['name']}:")
        print(f"    Strategy: {r['strategy']} | {r['params']}")
        print(f"    Trades: {r['trades']} | Win Rate: {r['win_rate']}%")
        print(f"    Total Return: {r['total_return']}% | PF: {r['profit_factor']}")
        print(f"    R:R: {r['avg_rr']} | Expectancy: {r['expectancy']}%/trade")
        print(f"    Max Drawdown: {r['max_dd']}%")

    print(f"\n{'*'*90}")
    return all_best


if __name__ == "__main__":
    run_all()
