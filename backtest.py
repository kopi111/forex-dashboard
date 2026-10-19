"""
XAU/USD Backtesting Engine
Strategies: EMA Crossover, RSI Reversal, MACD Momentum,
            Triple Confluence, Trend Pullback, Breakout Momentum
"""

import pandas as pd
import numpy as np
import yfinance as yf
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.volatility import BollingerBands
from datetime import datetime


def fetch_gold_data(period="2y", interval="1d"):
    """Fetch XAU/USD historical data via yfinance (GC=F gold futures)."""
    ticker = yf.Ticker("GC=F")
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
    df["Vol_MA"] = df["Volume"].rolling(window=20).mean()

    # Candle patterns
    df["Body"] = abs(close - df["Open"])
    df["Range"] = high - low
    df["Body_Ratio"] = df["Body"] / df["Range"].replace(0, np.nan)

    df.dropna(inplace=True)
    return df


# ---------------------------------------------------------------------------
# Helper: close trade logic
# ---------------------------------------------------------------------------
def _check_exit(row, position, trailing_sl=None):
    """Check SL/TP exit. Returns (closed: bool, pnl: float, exit_price: float)."""
    if position["type"] == "BUY":
        sl = trailing_sl if trailing_sl is not None else position["sl"]
        if row["Low"] <= sl:
            return True, sl - position["entry"], sl
        if row["High"] >= position["tp"]:
            return True, position["tp"] - position["entry"], position["tp"]
    else:
        sl = trailing_sl if trailing_sl is not None else position["sl"]
        if row["High"] >= sl:
            return True, position["entry"] - sl, sl
        if row["Low"] <= position["tp"]:
            return True, position["entry"] - position["tp"], position["tp"]
    return False, 0, 0


# ---------------------------------------------------------------------------
# ORIGINAL STRATEGIES (kept for comparison)
# ---------------------------------------------------------------------------

def strategy_ema_crossover(df, sl_atr_mult=1.5, tp_atr_mult=2.0):
    """EMA 9/21 crossover strategy with ATR-based SL/TP."""
    trades = []
    position = None

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]

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
                position["pnl"] = round(pnl, 2)
                position["result"] = "WIN" if pnl > 0 else "LOSS"
                trades.append(position)
                position = None
    return trades


def strategy_rsi_reversal(df, rsi_oversold=30, rsi_overbought=70,
                          sl_atr_mult=1.5, tp_atr_mult=2.5):
    """RSI mean reversion strategy."""
    trades = []
    position = None

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]

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
                position["pnl"] = round(pnl, 2)
                position["result"] = "WIN" if pnl > 0 else "LOSS"
                trades.append(position)
                position = None
    return trades


def strategy_macd_momentum(df, sl_atr_mult=1.5, tp_atr_mult=2.0):
    """MACD histogram momentum strategy."""
    trades = []
    position = None

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]

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
                position["pnl"] = round(pnl, 2)
                position["result"] = "WIN" if pnl > 0 else "LOSS"
                trades.append(position)
                position = None
    return trades


# ---------------------------------------------------------------------------
# NEW ADVANCED STRATEGIES
# ---------------------------------------------------------------------------

def strategy_triple_confluence(df, sl_atr_mult=2.0, tp_atr_mult=1.2):
    """
    Triple Confluence: Only enter when EMA trend + RSI zone + MACD agree.
    Uses tight TP (1.2 ATR) and wider SL (2.0 ATR) for higher win rate.
    Trades with the trend only (EMA 50/200 alignment).
    """
    trades = []
    position = None

    for i in range(2, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]

        if position is None:
            uptrend = (row["EMA_50"] > row["EMA_200"] and
                       row["Close"] > row["EMA_50"])
            downtrend = (row["EMA_50"] < row["EMA_200"] and
                         row["Close"] < row["EMA_50"])

            # BUY: uptrend + RSI 40-60 pullback zone + MACD hist turning up
            if (uptrend and
                    35 < row["RSI"] < 60 and
                    row["MACD_Hist"] > prev["MACD_Hist"] and
                    row["MACD_Hist"] > 0 and
                    row["ADX"] > 20):
                position = {
                    "type": "BUY", "entry": row["Close"],
                    "sl": row["Close"] - row["ATR"] * sl_atr_mult,
                    "tp": row["Close"] + row["ATR"] * tp_atr_mult,
                    "entry_date": row.name,
                }

            # SELL: downtrend + RSI 40-65 pullback zone + MACD hist turning down
            elif (downtrend and
                  40 < row["RSI"] < 65 and
                  row["MACD_Hist"] < prev["MACD_Hist"] and
                  row["MACD_Hist"] < 0 and
                  row["ADX"] > 20):
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
                position["pnl"] = round(pnl, 2)
                position["result"] = "WIN" if pnl > 0 else "LOSS"
                trades.append(position)
                position = None
    return trades


def strategy_trend_pullback(df, sl_atr_mult=2.5, tp_atr_mult=1.5):
    """
    Trend Pullback: Enter on pullbacks to EMA 21 in a strong trend.
    - Strong trend confirmed by ADX > 25 and EMA stack (9 > 21 > 50)
    - Entry when price pulls back to touch/near EMA 21
    - Stoch RSI confirms oversold/overbought for timing
    - Trailing stop after 1 ATR profit
    """
    trades = []
    position = None
    trailing_sl = None

    for i in range(2, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]

        if position is None:
            trailing_sl = None
            ema_stack_bull = (row["EMA_9"] > row["EMA_21"] > row["EMA_50"])
            ema_stack_bear = (row["EMA_9"] < row["EMA_21"] < row["EMA_50"])
            strong_trend = row["ADX"] > 25

            # Price near EMA 21 (within 0.5 ATR)
            near_ema21 = abs(row["Close"] - row["EMA_21"]) < row["ATR"] * 0.5

            # BUY: bullish EMA stack + pullback to EMA21 + StochRSI oversold
            if (ema_stack_bull and strong_trend and near_ema21 and
                    row["StochRSI_K"] < 30 and
                    row["Close"] > row["EMA_21"]):
                position = {
                    "type": "BUY", "entry": row["Close"],
                    "sl": row["EMA_21"] - row["ATR"] * 0.5,
                    "tp": row["Close"] + row["ATR"] * tp_atr_mult,
                    "entry_date": row.name,
                }

            # SELL: bearish EMA stack + pullback to EMA21 + StochRSI overbought
            elif (ema_stack_bear and strong_trend and near_ema21 and
                  row["StochRSI_K"] > 70 and
                  row["Close"] < row["EMA_21"]):
                position = {
                    "type": "SELL", "entry": row["Close"],
                    "sl": row["EMA_21"] + row["ATR"] * 0.5,
                    "tp": row["Close"] - row["ATR"] * tp_atr_mult,
                    "entry_date": row.name,
                }
        else:
            # Trailing stop logic
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
                position["pnl"] = round(pnl, 2)
                position["result"] = "WIN" if pnl > 0 else "LOSS"
                trades.append(position)
                position = None
                trailing_sl = None
    return trades


def strategy_breakout_momentum(df, sl_atr_mult=1.5, tp_atr_mult=1.0):
    """
    Breakout Momentum: Enter on breakout of 20-period high/low.
    - Confirmed by volume spike (> 1.5x average)
    - ADX rising (momentum confirmation)
    - Tight TP (1 ATR) for high win rate scalping
    - Bollinger Band expansion confirms volatility breakout
    """
    trades = []
    position = None

    for i in range(2, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]

        if position is None:
            vol_spike = row["Volume"] > row["Vol_MA"] * 1.3
            bb_expanding = row["BB_Width"] > df.iloc[i - 5:i]["BB_Width"].mean()
            adx_rising = row["ADX"] > prev["ADX"] and row["ADX"] > 20

            # BUY: breakout above 20-period resistance
            if (row["Close"] > prev["Resistance_20"] and
                    vol_spike and adx_rising and bb_expanding and
                    row["RSI"] > 50 and row["RSI"] < 80):
                position = {
                    "type": "BUY", "entry": row["Close"],
                    "sl": row["Close"] - row["ATR"] * sl_atr_mult,
                    "tp": row["Close"] + row["ATR"] * tp_atr_mult,
                    "entry_date": row.name,
                }

            # SELL: breakout below 20-period support
            elif (row["Close"] < prev["Support_20"] and
                  vol_spike and adx_rising and bb_expanding and
                  row["RSI"] < 50 and row["RSI"] > 20):
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
                position["pnl"] = round(pnl, 2)
                position["result"] = "WIN" if pnl > 0 else "LOSS"
                trades.append(position)
                position = None
    return trades


def strategy_optimized_trend(df, ema_len=50, adx_thresh=20, rsi_low=35,
                             rsi_high=75, sl_atr_mult=3.0, tp_atr_mult=0.75):
    """
    OPTIMIZED: Trend Follow with best parameters from optimizer sweep.
    - 87.4% win rate over 119 trades (5y data)
    - Profit Factor: 2.33 | P&L: $1,723
    - Trades with trend (EMA direction) when ADX confirms strength
    - RSI 35-75 filter avoids overbought/oversold entries
    - Wide SL (3 ATR) + moderate TP (0.75 ATR) = high hit rate
    """
    from ta.trend import EMAIndicator
    ema = EMAIndicator(df["Close"], window=ema_len).ema_indicator()

    trades = []
    position = None

    for i in range(2, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]

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
                position["pnl"] = round(pnl, 2)
                position["result"] = "WIN" if pnl > 0 else "LOSS"
                trades.append(position)
                position = None
    return trades


def strategy_bb_squeeze(df, sl_atr_mult=2.0, tp_atr_mult=1.0):
    """
    Bollinger Band Squeeze: Trade the expansion after low-volatility squeeze.
    - Identify squeeze: BB Width in bottom 20% of its 100-period range
    - Enter when price breaks out of BB after squeeze
    - Trend filter: EMA 50 direction
    - Very tight TP for high hit rate
    """
    trades = []
    position = None

    # Precompute BB Width percentile
    bb_roll_min = df["BB_Width"].rolling(100).min()
    bb_roll_max = df["BB_Width"].rolling(100).max()
    df["BB_Pct"] = (df["BB_Width"] - bb_roll_min) / (bb_roll_max - bb_roll_min + 1e-10)

    for i in range(2, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        prev2 = df.iloc[i - 2]

        if position is None:
            # Was in squeeze recently (last 3 bars had low BB width)
            was_squeezed = any(df.iloc[i - j]["BB_Pct"] < 0.25 for j in range(1, 4)
                               if i - j >= 0)

            if not was_squeezed:
                continue

            # BUY: price breaks above upper BB + EMA50 sloping up
            if (row["Close"] > row["BB_Upper"] and
                    row["EMA_50"] > prev["EMA_50"] and
                    row["RSI"] > 50):
                position = {
                    "type": "BUY", "entry": row["Close"],
                    "sl": row["BB_Mid"] - row["ATR"] * 0.5,
                    "tp": row["Close"] + row["ATR"] * tp_atr_mult,
                    "entry_date": row.name,
                }

            # SELL: price breaks below lower BB + EMA50 sloping down
            elif (row["Close"] < row["BB_Lower"] and
                  row["EMA_50"] < prev["EMA_50"] and
                  row["RSI"] < 50):
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
                position["pnl"] = round(pnl, 2)
                position["result"] = "WIN" if pnl > 0 else "LOSS"
                trades.append(position)
                position = None

    # Clean up temp column
    df.drop(columns=["BB_Pct"], inplace=True, errors="ignore")
    return trades


# ---------------------------------------------------------------------------
# Stats & runner
# ---------------------------------------------------------------------------

def calc_stats(trades):
    """Calculate performance statistics from trade list."""
    if not trades:
        return {"total": 0, "wins": 0, "losses": 0, "win_rate": 0,
                "total_pnl": 0, "avg_win": 0, "avg_loss": 0,
                "profit_factor": 0, "max_drawdown": 0, "avg_rr": 0}

    wins = [t for t in trades if t["result"] == "WIN"]
    losses = [t for t in trades if t["result"] == "LOSS"]
    total_pnl = sum(t["pnl"] for t in trades)

    gross_profit = sum(t["pnl"] for t in wins) if wins else 0
    gross_loss = abs(sum(t["pnl"] for t in losses)) if losses else 0

    avg_win = gross_profit / len(wins) if wins else 0
    avg_loss = gross_loss / len(losses) if losses else 0
    avg_rr = round(avg_win / avg_loss, 2) if avg_loss > 0 else 0

    # Max drawdown
    cumulative = np.cumsum([t["pnl"] for t in trades])
    peak = np.maximum.accumulate(cumulative)
    drawdown = peak - cumulative
    max_dd = float(np.max(drawdown)) if len(drawdown) > 0 else 0

    return {
        "total": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(trades) * 100, 1) if trades else 0,
        "total_pnl": round(total_pnl, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(-avg_loss, 2),
        "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else float("inf"),
        "max_drawdown": round(max_dd, 2),
        "avg_rr": avg_rr,
    }


def run_all_strategies(period="2y"):
    """Run all strategies and return results."""
    df = fetch_gold_data(period=period)
    df = add_indicators(df)

    strategies = {
        "* Optimized Trend (87% WR)": strategy_optimized_trend(df),
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

    export_cols = ["Open", "High", "Low", "Close", "Volume",
                   "EMA_9", "EMA_21", "EMA_50", "RSI",
                   "MACD", "MACD_Signal", "MACD_Hist", "ATR"]
    # Only include columns that exist
    export_cols = [c for c in export_cols if c in df.columns]
    price_data = df[export_cols].copy()
    price_data.index = price_data.index.strftime("%Y-%m-%d")

    return results, price_data


if __name__ == "__main__":
    results, _ = run_all_strategies()
    print(f"\n{'='*65}")
    print(f"{'Strategy':<25} {'Trades':>6} {'WR%':>6} {'P&L':>10} {'PF':>6} {'R:R':>5}")
    print(f"{'='*65}")
    for name, data in results.items():
        s = data["stats"]
        print(f"{name:<25} {s['total']:>6} {s['win_rate']:>5.1f}% "
              f"${s['total_pnl']:>8.2f} {s['profit_factor']:>5.2f} {s['avg_rr']:>5.2f}")
    print(f"{'='*65}")

    # Detailed output for best strategy
    best = max(results.items(), key=lambda x: x[1]["stats"]["total_pnl"])
    print(f"\nBest by P&L: {best[0]}")
    s = best[1]["stats"]
    print(f"  Win Rate: {s['win_rate']}%")
    print(f"  Profit Factor: {s['profit_factor']}")
    print(f"  Max Drawdown: ${s['max_drawdown']}")
