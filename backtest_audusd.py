"""
AUD/USD Backtesting Engine
Adapted from XAU/USD engine for forex pair characteristics.
Strategies: EMA Crossover, RSI Reversal, MACD Momentum,
            Triple Confluence, Trend Pullback, Breakout Momentum,
            BB Squeeze, Optimized Trend
"""

import pandas as pd
import numpy as np
import yfinance as yf
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.volatility import BollingerBands
from datetime import datetime


def fetch_audusd_data(period="5y", interval="1d"):
    """Fetch AUD/USD historical data via yfinance."""
    ticker = yf.Ticker("AUDUSD=X")
    df = ticker.history(period=period, interval=interval)
    df = df.dropna()
    return df


def add_indicators(df):
    """Add technical indicators to dataframe."""
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    # EMAs
    df["EMA_9"] = EMAIndicator(close, window=9).ema_indicator()
    df["EMA_21"] = EMAIndicator(close, window=21).ema_indicator()
    df["EMA_50"] = EMAIndicator(close, window=50).ema_indicator()
    df["EMA_200"] = EMAIndicator(close, window=200).ema_indicator()

    # RSI
    df["RSI"] = RSIIndicator(close, window=14).rsi()

    # Stochastic RSI
    stoch_rsi = StochRSIIndicator(close, window=14, smooth1=3, smooth2=3)
    df["StochRSI_K"] = stoch_rsi.stochrsi_k() * 100
    df["StochRSI_D"] = stoch_rsi.stochrsi_d() * 100

    # MACD
    macd = MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["MACD_Hist"] = macd.macd_diff()

    # ADX (trend strength)
    adx = ADXIndicator(high, low, close, window=14)
    df["ADX"] = adx.adx()
    df["DI_Plus"] = adx.adx_pos()
    df["DI_Minus"] = adx.adx_neg()

    # Bollinger Bands
    bb = BollingerBands(close, window=20, window_dev=2)
    df["BB_Upper"] = bb.bollinger_hband()
    df["BB_Lower"] = bb.bollinger_lband()
    df["BB_Mid"] = bb.bollinger_mavg()
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Mid"]

    # ATR for stop loss sizing
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(window=14).mean()

    # Support/Resistance: rolling high/low
    df["Resistance_20"] = high.rolling(window=20).max()
    df["Support_20"] = low.rolling(window=20).min()

    # Volume MA
    if "Volume" in df.columns and df["Volume"].sum() > 0:
        df["Vol_MA"] = df["Volume"].rolling(window=20).mean()
    else:
        # Forex pairs often have 0 volume on yfinance; use synthetic
        df["Volume"] = 1
        df["Vol_MA"] = 1

    # Candle patterns
    df["Body"] = abs(close - df["Open"])
    df["Range"] = high - low
    df["Body_Ratio"] = df["Body"] / df["Range"].replace(0, np.nan)

    # Pips moved (for AUD/USD, 1 pip = 0.0001)
    df["ATR_Pips"] = df["ATR"] * 10000

    df.dropna(inplace=True)
    return df


# ---------------------------------------------------------------------------
# Helper: close trade logic (pip-based for forex)
# ---------------------------------------------------------------------------
PIP = 0.0001  # AUD/USD pip size
LOT_SIZE = 100000  # standard lot


def _check_exit(row, position, trailing_sl=None):
    """Check SL/TP exit. Returns (closed, pnl_pips, exit_price)."""
    if position["type"] == "BUY":
        sl = trailing_sl if trailing_sl is not None else position["sl"]
        if row["Low"] <= sl:
            pnl = (sl - position["entry"]) / PIP
            return True, pnl, sl
        if row["High"] >= position["tp"]:
            pnl = (position["tp"] - position["entry"]) / PIP
            return True, pnl, position["tp"]
    else:
        sl = trailing_sl if trailing_sl is not None else position["sl"]
        if row["High"] >= sl:
            pnl = (position["entry"] - sl) / PIP
            return True, pnl, sl
        if row["Low"] <= position["tp"]:
            pnl = (position["entry"] - position["tp"]) / PIP
            return True, pnl, position["tp"]
    return False, 0, 0


# ---------------------------------------------------------------------------
# STRATEGIES
# ---------------------------------------------------------------------------

def strategy_ema_crossover(df, sl_atr_mult=1.5, tp_atr_mult=2.0):
    """EMA 9/21 crossover with ATR-based SL/TP."""
    trades, position = [], None
    for i in range(1, len(df)):
        row, prev = df.iloc[i], df.iloc[i - 1]
        if position is None:
            if (prev["EMA_9"] <= prev["EMA_21"] and
                    row["EMA_9"] > row["EMA_21"] and
                    row["Close"] > row["EMA_50"]):
                position = {
                    "type": "BUY", "entry": row["Close"],
                    "sl": row["Close"] - row["ATR"] * sl_atr_mult,
                    "tp": row["Close"] + row["ATR"] * tp_atr_mult,
                    "entry_date": row.name,
                }
            elif (prev["EMA_9"] >= prev["EMA_21"] and
                  row["EMA_9"] < row["EMA_21"] and
                  row["Close"] < row["EMA_50"]):
                position = {
                    "type": "SELL", "entry": row["Close"],
                    "sl": row["Close"] + row["ATR"] * sl_atr_mult,
                    "tp": row["Close"] - row["ATR"] * tp_atr_mult,
                    "entry_date": row.name,
                }
        else:
            closed, pnl, _ = _check_exit(row, position)
            if closed:
                position["exit_date"] = row.name
                position["pnl"] = round(pnl, 1)
                position["result"] = "WIN" if pnl > 0 else "LOSS"
                trades.append(position)
                position = None
    return trades


def strategy_rsi_reversal(df, rsi_oversold=30, rsi_overbought=70,
                          sl_atr_mult=1.5, tp_atr_mult=2.5):
    """RSI mean reversion strategy."""
    trades, position = [], None
    for i in range(1, len(df)):
        row, prev = df.iloc[i], df.iloc[i - 1]
        if position is None:
            if prev["RSI"] < rsi_oversold and row["RSI"] >= rsi_oversold:
                position = {
                    "type": "BUY", "entry": row["Close"],
                    "sl": row["Close"] - row["ATR"] * sl_atr_mult,
                    "tp": row["Close"] + row["ATR"] * tp_atr_mult,
                    "entry_date": row.name,
                }
            elif prev["RSI"] > rsi_overbought and row["RSI"] <= rsi_overbought:
                position = {
                    "type": "SELL", "entry": row["Close"],
                    "sl": row["Close"] + row["ATR"] * sl_atr_mult,
                    "tp": row["Close"] - row["ATR"] * tp_atr_mult,
                    "entry_date": row.name,
                }
        else:
            closed, pnl, _ = _check_exit(row, position)
            if closed:
                position["exit_date"] = row.name
                position["pnl"] = round(pnl, 1)
                position["result"] = "WIN" if pnl > 0 else "LOSS"
                trades.append(position)
                position = None
    return trades


def strategy_macd_momentum(df, sl_atr_mult=1.5, tp_atr_mult=2.0):
    """MACD histogram momentum strategy."""
    trades, position = [], None
    for i in range(1, len(df)):
        row, prev = df.iloc[i], df.iloc[i - 1]
        if position is None:
            if (prev["MACD_Hist"] <= 0 and row["MACD_Hist"] > 0 and
                    row["MACD"] > row["MACD_Signal"]):
                position = {
                    "type": "BUY", "entry": row["Close"],
                    "sl": row["Close"] - row["ATR"] * sl_atr_mult,
                    "tp": row["Close"] + row["ATR"] * tp_atr_mult,
                    "entry_date": row.name,
                }
            elif (prev["MACD_Hist"] >= 0 and row["MACD_Hist"] < 0 and
                  row["MACD"] < row["MACD_Signal"]):
                position = {
                    "type": "SELL", "entry": row["Close"],
                    "sl": row["Close"] + row["ATR"] * sl_atr_mult,
                    "tp": row["Close"] - row["ATR"] * tp_atr_mult,
                    "entry_date": row.name,
                }
        else:
            closed, pnl, _ = _check_exit(row, position)
            if closed:
                position["exit_date"] = row.name
                position["pnl"] = round(pnl, 1)
                position["result"] = "WIN" if pnl > 0 else "LOSS"
                trades.append(position)
                position = None
    return trades


def strategy_triple_confluence(df, sl_atr_mult=2.0, tp_atr_mult=1.2):
    """Triple Confluence: EMA trend + RSI zone + MACD agree."""
    trades, position = [], None
    for i in range(2, len(df)):
        row, prev = df.iloc[i], df.iloc[i - 1]
        if position is None:
            uptrend = (row["EMA_50"] > row["EMA_200"] and
                       row["Close"] > row["EMA_50"])
            downtrend = (row["EMA_50"] < row["EMA_200"] and
                         row["Close"] < row["EMA_50"])

            if (uptrend and 35 < row["RSI"] < 60 and
                    row["MACD_Hist"] > prev["MACD_Hist"] and
                    row["MACD_Hist"] > 0 and row["ADX"] > 20):
                position = {
                    "type": "BUY", "entry": row["Close"],
                    "sl": row["Close"] - row["ATR"] * sl_atr_mult,
                    "tp": row["Close"] + row["ATR"] * tp_atr_mult,
                    "entry_date": row.name,
                }
            elif (downtrend and 40 < row["RSI"] < 65 and
                  row["MACD_Hist"] < prev["MACD_Hist"] and
                  row["MACD_Hist"] < 0 and row["ADX"] > 20):
                position = {
                    "type": "SELL", "entry": row["Close"],
                    "sl": row["Close"] + row["ATR"] * sl_atr_mult,
                    "tp": row["Close"] - row["ATR"] * tp_atr_mult,
                    "entry_date": row.name,
                }
        else:
            closed, pnl, _ = _check_exit(row, position)
            if closed:
                position["exit_date"] = row.name
                position["pnl"] = round(pnl, 1)
                position["result"] = "WIN" if pnl > 0 else "LOSS"
                trades.append(position)
                position = None
    return trades


def strategy_trend_pullback(df, sl_atr_mult=2.5, tp_atr_mult=1.5):
    """Trend Pullback: Enter on pullbacks to EMA 21 in a strong trend."""
    trades, position, trailing_sl = [], None, None
    for i in range(2, len(df)):
        row, prev = df.iloc[i], df.iloc[i - 1]
        if position is None:
            trailing_sl = None
            ema_stack_bull = (row["EMA_9"] > row["EMA_21"] > row["EMA_50"])
            ema_stack_bear = (row["EMA_9"] < row["EMA_21"] < row["EMA_50"])
            strong_trend = row["ADX"] > 25
            near_ema21 = abs(row["Close"] - row["EMA_21"]) < row["ATR"] * 0.5

            if (ema_stack_bull and strong_trend and near_ema21 and
                    row["StochRSI_K"] < 30 and row["Close"] > row["EMA_21"]):
                position = {
                    "type": "BUY", "entry": row["Close"],
                    "sl": row["EMA_21"] - row["ATR"] * 0.5,
                    "tp": row["Close"] + row["ATR"] * tp_atr_mult,
                    "entry_date": row.name,
                }
            elif (ema_stack_bear and strong_trend and near_ema21 and
                  row["StochRSI_K"] > 70 and row["Close"] < row["EMA_21"]):
                position = {
                    "type": "SELL", "entry": row["Close"],
                    "sl": row["EMA_21"] + row["ATR"] * 0.5,
                    "tp": row["Close"] - row["ATR"] * tp_atr_mult,
                    "entry_date": row.name,
                }
        else:
            if position["type"] == "BUY":
                profit = row["High"] - position["entry"]
                if profit >= row["ATR"] * 1.0:
                    new_trail = row["High"] - row["ATR"] * 1.0
                    if trailing_sl is None or new_trail > trailing_sl:
                        trailing_sl = new_trail
            else:
                profit = position["entry"] - row["Low"]
                if profit >= row["ATR"] * 1.0:
                    new_trail = row["Low"] + row["ATR"] * 1.0
                    if trailing_sl is None or new_trail < trailing_sl:
                        trailing_sl = new_trail

            closed, pnl, _ = _check_exit(row, position, trailing_sl)
            if closed:
                position["exit_date"] = row.name
                position["pnl"] = round(pnl, 1)
                position["result"] = "WIN" if pnl > 0 else "LOSS"
                trades.append(position)
                position = None
                trailing_sl = None
    return trades


def strategy_breakout_momentum(df, sl_atr_mult=1.5, tp_atr_mult=1.0):
    """Breakout of 20-period high/low with volume and ADX confirmation."""
    trades, position = [], None
    for i in range(2, len(df)):
        row, prev = df.iloc[i], df.iloc[i - 1]
        if position is None:
            vol_spike = row["Volume"] > row["Vol_MA"] * 1.3
            bb_expanding = row["BB_Width"] > df.iloc[i - 5:i]["BB_Width"].mean()
            adx_rising = row["ADX"] > prev["ADX"] and row["ADX"] > 20

            if (row["Close"] > prev["Resistance_20"] and
                    vol_spike and adx_rising and bb_expanding and
                    50 < row["RSI"] < 80):
                position = {
                    "type": "BUY", "entry": row["Close"],
                    "sl": row["Close"] - row["ATR"] * sl_atr_mult,
                    "tp": row["Close"] + row["ATR"] * tp_atr_mult,
                    "entry_date": row.name,
                }
            elif (row["Close"] < prev["Support_20"] and
                  vol_spike and adx_rising and bb_expanding and
                  20 < row["RSI"] < 50):
                position = {
                    "type": "SELL", "entry": row["Close"],
                    "sl": row["Close"] + row["ATR"] * sl_atr_mult,
                    "tp": row["Close"] - row["ATR"] * tp_atr_mult,
                    "entry_date": row.name,
                }
        else:
            closed, pnl, _ = _check_exit(row, position)
            if closed:
                position["exit_date"] = row.name
                position["pnl"] = round(pnl, 1)
                position["result"] = "WIN" if pnl > 0 else "LOSS"
                trades.append(position)
                position = None
    return trades


def strategy_bb_squeeze(df, sl_atr_mult=2.0, tp_atr_mult=1.0):
    """Bollinger Band Squeeze breakout strategy."""
    trades, position = [], None

    bb_roll_min = df["BB_Width"].rolling(100).min()
    bb_roll_max = df["BB_Width"].rolling(100).max()
    df["BB_Pct"] = (df["BB_Width"] - bb_roll_min) / (bb_roll_max - bb_roll_min + 1e-10)

    for i in range(2, len(df)):
        row, prev = df.iloc[i], df.iloc[i - 1]
        if position is None:
            was_squeezed = any(df.iloc[i - j]["BB_Pct"] < 0.25
                               for j in range(1, 4) if i - j >= 0)
            if not was_squeezed:
                continue

            if (row["Close"] > row["BB_Upper"] and
                    row["EMA_50"] > prev["EMA_50"] and row["RSI"] > 50):
                position = {
                    "type": "BUY", "entry": row["Close"],
                    "sl": row["BB_Mid"] - row["ATR"] * 0.5,
                    "tp": row["Close"] + row["ATR"] * tp_atr_mult,
                    "entry_date": row.name,
                }
            elif (row["Close"] < row["BB_Lower"] and
                  row["EMA_50"] < prev["EMA_50"] and row["RSI"] < 50):
                position = {
                    "type": "SELL", "entry": row["Close"],
                    "sl": row["BB_Mid"] + row["ATR"] * 0.5,
                    "tp": row["Close"] - row["ATR"] * tp_atr_mult,
                    "entry_date": row.name,
                }
        else:
            closed, pnl, _ = _check_exit(row, position)
            if closed:
                position["exit_date"] = row.name
                position["pnl"] = round(pnl, 1)
                position["result"] = "WIN" if pnl > 0 else "LOSS"
                trades.append(position)
                position = None

    df.drop(columns=["BB_Pct"], inplace=True, errors="ignore")
    return trades


def strategy_optimized_trend(df, ema_len=50, adx_thresh=20, rsi_low=35,
                             rsi_high=75, sl_atr_mult=3.0, tp_atr_mult=0.75):
    """Trend Follow with parameterizable settings."""
    ema = EMAIndicator(df["Close"], window=ema_len).ema_indicator()
    trades, position = [], None

    for i in range(2, len(df)):
        row, prev = df.iloc[i], df.iloc[i - 1]
        if position is None:
            uptrend = row["Close"] > ema.iloc[i] and ema.iloc[i] > ema.iloc[i - 1]
            downtrend = row["Close"] < ema.iloc[i] and ema.iloc[i] < ema.iloc[i - 1]

            if (uptrend and row["ADX"] > adx_thresh and
                    rsi_low < row["RSI"] < rsi_high):
                position = {
                    "type": "BUY", "entry": row["Close"],
                    "sl": row["Close"] - row["ATR"] * sl_atr_mult,
                    "tp": row["Close"] + row["ATR"] * tp_atr_mult,
                    "entry_date": row.name,
                }
            elif (downtrend and row["ADX"] > adx_thresh and
                  (100 - rsi_high) < row["RSI"] < (100 - rsi_low)):
                position = {
                    "type": "SELL", "entry": row["Close"],
                    "sl": row["Close"] + row["ATR"] * sl_atr_mult,
                    "tp": row["Close"] - row["ATR"] * tp_atr_mult,
                    "entry_date": row.name,
                }
        else:
            closed, pnl, _ = _check_exit(row, position)
            if closed:
                position["exit_date"] = row.name
                position["pnl"] = round(pnl, 1)
                position["result"] = "WIN" if pnl > 0 else "LOSS"
                trades.append(position)
                position = None
    return trades


# ---------------------------------------------------------------------------
# Stats & runner
# ---------------------------------------------------------------------------

def calc_stats(trades):
    """Calculate performance statistics (pips-based for forex)."""
    if not trades:
        return {"total": 0, "wins": 0, "losses": 0, "win_rate": 0,
                "total_pips": 0, "avg_win_pips": 0, "avg_loss_pips": 0,
                "profit_factor": 0, "max_drawdown_pips": 0, "avg_rr": 0,
                "expectancy_pips": 0}

    wins = [t for t in trades if t["result"] == "WIN"]
    losses = [t for t in trades if t["result"] == "LOSS"]
    total_pips = sum(t["pnl"] for t in trades)

    gross_profit = sum(t["pnl"] for t in wins) if wins else 0
    gross_loss = abs(sum(t["pnl"] for t in losses)) if losses else 0

    avg_win = gross_profit / len(wins) if wins else 0
    avg_loss = gross_loss / len(losses) if losses else 0
    avg_rr = round(avg_win / avg_loss, 2) if avg_loss > 0 else 0

    # Max drawdown in pips
    cumulative = np.cumsum([t["pnl"] for t in trades])
    peak = np.maximum.accumulate(cumulative)
    drawdown = peak - cumulative
    max_dd = float(np.max(drawdown)) if len(drawdown) > 0 else 0

    # Expectancy per trade
    expectancy = total_pips / len(trades)

    return {
        "total": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(trades) * 100, 1),
        "total_pips": round(total_pips, 1),
        "avg_win_pips": round(avg_win, 1),
        "avg_loss_pips": round(-avg_loss, 1),
        "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else float("inf"),
        "max_drawdown_pips": round(max_dd, 1),
        "avg_rr": avg_rr,
        "expectancy_pips": round(expectancy, 1),
    }


def run_all_strategies(period="5y"):
    """Run all strategies on AUD/USD and return results."""
    df = fetch_audusd_data(period=period)
    df = add_indicators(df)

    strategies = {
        "Optimized Trend": strategy_optimized_trend(df),
        "EMA Crossover (9/21)": strategy_ema_crossover(df),
        "RSI Reversal": strategy_rsi_reversal(df),
        "MACD Momentum": strategy_macd_momentum(df),
        "Triple Confluence": strategy_triple_confluence(df),
        "Trend Pullback": strategy_trend_pullback(df),
        "Breakout Momentum": strategy_breakout_momentum(df),
        "BB Squeeze": strategy_bb_squeeze(df),
    }

    results = {}
    for name, trades in strategies.items():
        stats = calc_stats(trades)
        results[name] = {"trades": trades, "stats": stats}

    return results, df


if __name__ == "__main__":
    results, df = run_all_strategies()
    print(f"\nAUD/USD Backtest Results (5y daily data, {len(df)} bars)")
    print(f"{'='*85}")
    print(f"{'Strategy':<25} {'Trades':>6} {'WR%':>6} {'Pips':>10} {'PF':>6} {'R:R':>5} {'Expect':>7}")
    print(f"{'='*85}")
    for name, data in sorted(results.items(),
                              key=lambda x: x[1]["stats"]["total_pips"], reverse=True):
        s = data["stats"]
        print(f"{name:<25} {s['total']:>6} {s['win_rate']:>5.1f}% "
              f"{s['total_pips']:>9.1f} {s['profit_factor']:>5.2f} "
              f"{s['avg_rr']:>5.2f} {s['expectancy_pips']:>6.1f}p")
    print(f"{'='*85}")

    best = max(results.items(), key=lambda x: x[1]["stats"]["total_pips"])
    print(f"\nBest by Total Pips: {best[0]}")
    s = best[1]["stats"]
    print(f"  Trades: {s['total']} | Win Rate: {s['win_rate']}%")
    print(f"  Total Pips: {s['total_pips']} | Profit Factor: {s['profit_factor']}")
    print(f"  Avg Win: {s['avg_win_pips']}p | Avg Loss: {s['avg_loss_pips']}p")
    print(f"  Max Drawdown: {s['max_drawdown_pips']}p | Expectancy: {s['expectancy_pips']}p/trade")
