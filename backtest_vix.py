"""
VIX Instruments Backtesting Engine
Tests multiple strategies across VIX-related instruments:
  - ^VIX     (VIX Index - not directly tradable)
  - UVXY     (2x Long VIX ETF)
  - SVXY     (Short VIX ETF - inverse)
  - VIXY     (Long VIX ETF)
  - VXX      (VIX Short-Term Futures ETN)

Strategies adapted for VIX characteristics:
  - VIX is mean-reverting (always snaps back)
  - VIX spikes fast, decays slowly
  - VIX has contango drag (UVXY/VXX decay over time)
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime


# ---------------------------------------------------------------------------
# VIX Instruments
# ---------------------------------------------------------------------------

VIX_INSTRUMENTS = {
    "^VIX":  {"name": "VIX Index", "tradable": False, "type": "index"},
    "UVXY":  {"name": "UVXY (2x Long VIX)", "tradable": True, "type": "long_vix"},
    "SVXY":  {"name": "SVXY (Short VIX)", "tradable": True, "type": "short_vix"},
    "VIXY":  {"name": "VIXY (Long VIX)", "tradable": True, "type": "long_vix"},
}


def fetch_data(symbol, period="5y", interval="1d"):
    """Fetch historical data for a VIX instrument."""
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval)
    df = df.dropna()
    if len(df) == 0:
        return None
    return df


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


def compute_bollinger(series, period=20, std_dev=2):
    mid = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    return upper, mid, lower


def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def add_indicators(df):
    """Add all technical indicators."""
    close = df["Close"]

    # EMAs
    df["EMA_9"] = compute_ema(close, 9)
    df["EMA_21"] = compute_ema(close, 21)
    df["EMA_50"] = compute_ema(close, 50)

    # RSI
    df["RSI"] = compute_rsi(close, 14)

    # MACD
    df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = compute_macd(close)

    # Bollinger Bands
    df["BB_Upper"], df["BB_Mid"], df["BB_Lower"] = compute_bollinger(close)
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Mid"]

    # ATR
    df["ATR"] = compute_atr(df, 14)
    df["ATR_Pct"] = df["ATR"] / close * 100  # ATR as % of price

    # VIX-specific: percentile rank (where is VIX relative to its range)
    df["Pct_Rank_50"] = close.rolling(50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    df["Pct_Rank_200"] = close.rolling(200).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)

    # Rate of change
    df["ROC_5"] = close.pct_change(5) * 100
    df["ROC_10"] = close.pct_change(10) * 100

    # Volume MA
    if "Volume" in df.columns and df["Volume"].sum() > 0:
        df["Vol_MA"] = df["Volume"].rolling(window=20).mean()
    else:
        df["Volume"] = 1
        df["Vol_MA"] = 1

    df.dropna(inplace=True)
    return df


# ---------------------------------------------------------------------------
# Trade Logic
# ---------------------------------------------------------------------------

def _check_exit(row, pos, trailing_sl=None):
    """Check SL/TP exit. Returns (closed, pnl_pct, exit_price)."""
    sl = trailing_sl if trailing_sl is not None else pos["sl"]

    if pos["type"] == "BUY":
        if row["Low"] <= sl:
            pnl = (sl - pos["entry"]) / pos["entry"] * 100
            return True, pnl, sl
        if row["High"] >= pos["tp"]:
            pnl = (pos["tp"] - pos["entry"]) / pos["entry"] * 100
            return True, pnl, pos["tp"]
    else:  # SELL (short)
        if row["High"] >= sl:
            pnl = (pos["entry"] - sl) / pos["entry"] * 100
            return True, pnl, sl
        if row["Low"] <= pos["tp"]:
            pnl = (pos["entry"] - pos["tp"]) / pos["entry"] * 100
            return True, pnl, pos["tp"]
    return False, 0, 0


# ---------------------------------------------------------------------------
# VIX STRATEGIES
# ---------------------------------------------------------------------------

def strategy_rsi_mean_reversion(df, rsi_oversold=30, rsi_overbought=70,
                                 sl_pct=8, tp_pct=5):
    """
    VIX Mean Reversion via RSI.
    VIX tends to spike and revert. Buy VIX instruments when oversold,
    sell/short when overbought.
    """
    trades, pos = [], None
    for i in range(1, len(df)):
        r, p = df.iloc[i], df.iloc[i - 1]
        if pos is None:
            # BUY when RSI recovers from oversold
            if p["RSI"] < rsi_oversold and r["RSI"] >= rsi_oversold:
                pos = {
                    "type": "BUY", "entry": r["Close"],
                    "sl": r["Close"] * (1 - sl_pct / 100),
                    "tp": r["Close"] * (1 + tp_pct / 100),
                    "entry_date": r.name,
                }
            # SELL when RSI drops from overbought
            elif p["RSI"] > rsi_overbought and r["RSI"] <= rsi_overbought:
                pos = {
                    "type": "SELL", "entry": r["Close"],
                    "sl": r["Close"] * (1 + sl_pct / 100),
                    "tp": r["Close"] * (1 - tp_pct / 100),
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


def strategy_vix_spike_fade(df, spike_threshold=15, sl_pct=12, tp_pct=8):
    """
    Fade VIX spikes. When VIX/UVXY spikes up sharply (>15% in 5 days),
    short it expecting mean reversion. Vice versa for crashes.
    """
    trades, pos = [], None
    for i in range(1, len(df)):
        r, p = df.iloc[i], df.iloc[i - 1]
        if pos is None:
            # Short after spike up (VIX always comes down)
            if r["ROC_5"] > spike_threshold and r["RSI"] > 65:
                pos = {
                    "type": "SELL", "entry": r["Close"],
                    "sl": r["Close"] * (1 + sl_pct / 100),
                    "tp": r["Close"] * (1 - tp_pct / 100),
                    "entry_date": r.name,
                }
            # Buy after crash (VIX has floor)
            elif r["ROC_5"] < -spike_threshold and r["RSI"] < 35:
                pos = {
                    "type": "BUY", "entry": r["Close"],
                    "sl": r["Close"] * (1 - sl_pct / 100),
                    "tp": r["Close"] * (1 + tp_pct / 100),
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


def strategy_bollinger_reversion(df, sl_pct=10, tp_pct=6):
    """
    Bollinger Band mean reversion for VIX.
    Buy when price drops below lower BB, sell when above upper BB.
    """
    trades, pos = [], None
    for i in range(1, len(df)):
        r, p = df.iloc[i], df.iloc[i - 1]
        if pos is None:
            if p["Close"] < p["BB_Lower"] and r["Close"] >= r["BB_Lower"]:
                pos = {
                    "type": "BUY", "entry": r["Close"],
                    "sl": r["Close"] * (1 - sl_pct / 100),
                    "tp": r["Close"] * (1 + tp_pct / 100),
                    "entry_date": r.name,
                }
            elif p["Close"] > p["BB_Upper"] and r["Close"] <= r["BB_Upper"]:
                pos = {
                    "type": "SELL", "entry": r["Close"],
                    "sl": r["Close"] * (1 + sl_pct / 100),
                    "tp": r["Close"] * (1 - tp_pct / 100),
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


def strategy_ema_crossover(df, sl_pct=8, tp_pct=5):
    """EMA 9/21 crossover strategy."""
    trades, pos = [], None
    for i in range(1, len(df)):
        r, p = df.iloc[i], df.iloc[i - 1]
        if pos is None:
            if p["EMA_9"] <= p["EMA_21"] and r["EMA_9"] > r["EMA_21"]:
                pos = {
                    "type": "BUY", "entry": r["Close"],
                    "sl": r["Close"] * (1 - sl_pct / 100),
                    "tp": r["Close"] * (1 + tp_pct / 100),
                    "entry_date": r.name,
                }
            elif p["EMA_9"] >= p["EMA_21"] and r["EMA_9"] < r["EMA_21"]:
                pos = {
                    "type": "SELL", "entry": r["Close"],
                    "sl": r["Close"] * (1 + sl_pct / 100),
                    "tp": r["Close"] * (1 - tp_pct / 100),
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


def strategy_macd_momentum(df, sl_pct=10, tp_pct=6):
    """MACD histogram momentum."""
    trades, pos = [], None
    for i in range(1, len(df)):
        r, p = df.iloc[i], df.iloc[i - 1]
        if pos is None:
            if p["MACD_Hist"] <= 0 and r["MACD_Hist"] > 0:
                pos = {
                    "type": "BUY", "entry": r["Close"],
                    "sl": r["Close"] * (1 - sl_pct / 100),
                    "tp": r["Close"] * (1 + tp_pct / 100),
                    "entry_date": r.name,
                }
            elif p["MACD_Hist"] >= 0 and r["MACD_Hist"] < 0:
                pos = {
                    "type": "SELL", "entry": r["Close"],
                    "sl": r["Close"] * (1 + sl_pct / 100),
                    "tp": r["Close"] * (1 - tp_pct / 100),
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


def strategy_percentile_reversion(df, low_pct=0.15, high_pct=0.85,
                                   sl_pct=10, tp_pct=6):
    """
    Trade based on 200-day percentile rank.
    Buy when in bottom 15% of range, sell when in top 85%.
    """
    trades, pos = [], None
    for i in range(1, len(df)):
        r, p = df.iloc[i], df.iloc[i - 1]
        if pos is None:
            if (p["Pct_Rank_200"] < low_pct and r["Pct_Rank_200"] >= low_pct
                    and r["RSI"] < 40):
                pos = {
                    "type": "BUY", "entry": r["Close"],
                    "sl": r["Close"] * (1 - sl_pct / 100),
                    "tp": r["Close"] * (1 + tp_pct / 100),
                    "entry_date": r.name,
                }
            elif (p["Pct_Rank_200"] > high_pct and r["Pct_Rank_200"] <= high_pct
                  and r["RSI"] > 60):
                pos = {
                    "type": "SELL", "entry": r["Close"],
                    "sl": r["Close"] * (1 + sl_pct / 100),
                    "tp": r["Close"] * (1 - tp_pct / 100),
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


def strategy_contango_decay(df, sl_pct=15, tp_pct=4):
    """
    Exploit VIX contango decay (for UVXY/VIXY which decay over time).
    Short VIX ETFs when trend is down and RSI is not oversold.
    Uses tight TP to capture the natural decay.
    Only generates SELL signals (short bias).
    """
    trades, pos = [], None
    for i in range(2, len(df)):
        r, p = df.iloc[i], df.iloc[i - 1]
        if pos is None:
            # Short when: downtrend + RSI mid-range + not after a crash
            if (r["EMA_21"] < r["EMA_50"] and
                    r["Close"] < r["EMA_21"] and
                    35 < r["RSI"] < 65 and
                    r["ROC_5"] > -10):
                pos = {
                    "type": "SELL", "entry": r["Close"],
                    "sl": r["Close"] * (1 + sl_pct / 100),
                    "tp": r["Close"] * (1 - tp_pct / 100),
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
                "total_return_pct": 0, "avg_win_pct": 0, "avg_loss_pct": 0,
                "profit_factor": 0, "max_drawdown_pct": 0, "avg_rr": 0,
                "expectancy_pct": 0}

    wins = [t for t in trades if t["result"] == "WIN"]
    losses = [t for t in trades if t["result"] == "LOSS"]
    total_return = sum(t["pnl"] for t in trades)

    gross_profit = sum(t["pnl"] for t in wins) if wins else 0
    gross_loss = abs(sum(t["pnl"] for t in losses)) if losses else 0

    avg_win = gross_profit / len(wins) if wins else 0
    avg_loss = gross_loss / len(losses) if losses else 0
    avg_rr = round(avg_win / avg_loss, 2) if avg_loss > 0 else 0

    cumulative = np.cumsum([t["pnl"] for t in trades])
    peak = np.maximum.accumulate(cumulative)
    drawdown = peak - cumulative
    max_dd = float(np.max(drawdown)) if len(drawdown) > 0 else 0

    return {
        "total": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(trades) * 100, 1),
        "total_return_pct": round(total_return, 2),
        "avg_win_pct": round(avg_win, 2),
        "avg_loss_pct": round(-avg_loss, 2),
        "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else float("inf"),
        "max_drawdown_pct": round(max_dd, 2),
        "avg_rr": avg_rr,
        "expectancy_pct": round(total_return / len(trades), 2),
    }


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

def optimize_instrument(symbol, df, min_trades=10):
    """Run parameter sweep for a single instrument."""
    results = []

    # RSI Mean Reversion sweep
    for rsi_os in [25, 30, 35]:
        for rsi_ob in [65, 70, 75, 80]:
            for sl in [5, 8, 10, 12, 15]:
                for tp in [3, 5, 8, 10, 15]:
                    trades = strategy_rsi_mean_reversion(df, rsi_os, rsi_ob, sl, tp)
                    if len(trades) >= min_trades:
                        s = calc_stats(trades)
                        results.append({
                            "strategy": f"RSI({rsi_os}/{rsi_ob})",
                            "params": f"SL:{sl}% TP:{tp}%", **s})

    # Spike Fade sweep
    for spike in [10, 15, 20, 25]:
        for sl in [8, 10, 12, 15, 20]:
            for tp in [5, 8, 10, 12, 15]:
                trades = strategy_vix_spike_fade(df, spike, sl, tp)
                if len(trades) >= min_trades:
                    s = calc_stats(trades)
                    results.append({
                        "strategy": f"SpikeFade({spike}%)",
                        "params": f"SL:{sl}% TP:{tp}%", **s})

    # Bollinger sweep
    for sl in [5, 8, 10, 12, 15]:
        for tp in [3, 5, 8, 10, 12]:
            trades = strategy_bollinger_reversion(df, sl, tp)
            if len(trades) >= min_trades:
                s = calc_stats(trades)
                results.append({
                    "strategy": "Bollinger",
                    "params": f"SL:{sl}% TP:{tp}%", **s})

    # EMA Crossover sweep
    for sl in [5, 8, 10, 12, 15]:
        for tp in [3, 5, 8, 10, 12]:
            trades = strategy_ema_crossover(df, sl, tp)
            if len(trades) >= min_trades:
                s = calc_stats(trades)
                results.append({
                    "strategy": "EMA(9/21)",
                    "params": f"SL:{sl}% TP:{tp}%", **s})

    # MACD sweep
    for sl in [5, 8, 10, 12, 15]:
        for tp in [3, 5, 8, 10, 12]:
            trades = strategy_macd_momentum(df, sl, tp)
            if len(trades) >= min_trades:
                s = calc_stats(trades)
                results.append({
                    "strategy": "MACD",
                    "params": f"SL:{sl}% TP:{tp}%", **s})

    # Percentile Reversion sweep
    for low_p in [0.10, 0.15, 0.20]:
        for high_p in [0.80, 0.85, 0.90]:
            for sl in [8, 10, 12, 15]:
                for tp in [5, 8, 10, 12]:
                    trades = strategy_percentile_reversion(df, low_p, high_p, sl, tp)
                    if len(trades) >= min_trades:
                        s = calc_stats(trades)
                        results.append({
                            "strategy": f"Percentile({low_p}/{high_p})",
                            "params": f"SL:{sl}% TP:{tp}%", **s})

    # Contango Decay sweep (short-only, for UVXY/VIXY)
    info = VIX_INSTRUMENTS.get(symbol, {})
    if info.get("type") == "long_vix":
        for sl in [10, 12, 15, 20]:
            for tp in [3, 4, 5, 6, 8]:
                trades = strategy_contango_decay(df, sl, tp)
                if len(trades) >= min_trades:
                    s = calc_stats(trades)
                    results.append({
                        "strategy": "ContangoDecay",
                        "params": f"SL:{sl}% TP:{tp}%", **s})

    return results


# ---------------------------------------------------------------------------
# Main Runner
# ---------------------------------------------------------------------------

def run_all():
    """Run backtests across all VIX instruments."""
    all_results = {}

    for symbol, info in VIX_INSTRUMENTS.items():
        print(f"\n{'='*80}")
        print(f"  Fetching {symbol} - {info['name']}...")
        print(f"{'='*80}")

        df = fetch_data(symbol, period="5y")
        if df is None or len(df) < 100:
            print(f"  Skipped: insufficient data")
            continue

        df = add_indicators(df)
        print(f"  Data: {len(df)} bars ({df.index[0].strftime('%Y-%m-%d')} to "
              f"{df.index[-1].strftime('%Y-%m-%d')})")
        print(f"  Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
        print(f"  Current: ${df['Close'].iloc[-1]:.2f} | RSI: {df['RSI'].iloc[-1]:.1f}")

        # Run default strategies first
        print(f"\n  --- Default Strategy Results ---")
        strategies = {
            "RSI Mean Reversion": strategy_rsi_mean_reversion(df),
            "Spike Fade": strategy_vix_spike_fade(df),
            "Bollinger Reversion": strategy_bollinger_reversion(df),
            "EMA Crossover": strategy_ema_crossover(df),
            "MACD Momentum": strategy_macd_momentum(df),
            "Percentile Reversion": strategy_percentile_reversion(df),
        }
        if info.get("type") == "long_vix":
            strategies["Contango Decay (Short)"] = strategy_contango_decay(df)

        print(f"\n  {'Strategy':<25} {'#':>4} {'WR%':>6} {'Return':>8} {'PF':>6} {'R:R':>5} {'Exp':>7}")
        print(f"  {'-'*65}")
        for name, trades in sorted(strategies.items(),
                                    key=lambda x: calc_stats(x[1])["total_return_pct"],
                                    reverse=True):
            s = calc_stats(trades)
            print(f"  {name:<25} {s['total']:>4} {s['win_rate']:>5.1f}% "
                  f"{s['total_return_pct']:>7.1f}% {s['profit_factor']:>5.2f} "
                  f"{s['avg_rr']:>5.2f} {s['expectancy_pct']:>6.2f}%")

        # Run optimizer
        print(f"\n  --- Optimizing (parameter sweep)... ---")
        opt_results = optimize_instrument(symbol, df)

        if opt_results:
            df_opt = pd.DataFrame(opt_results)
            profitable = df_opt[df_opt["total_return_pct"] > 0]
            print(f"  Configs tested: {len(df_opt)} | Profitable: {len(profitable)}")

            # Balanced score
            df_opt["score"] = (
                (df_opt["win_rate"] / 100) *
                df_opt["profit_factor"].clip(upper=10) *
                df_opt["expectancy_pct"].clip(lower=0) *
                np.log(df_opt["total"].clip(lower=1))
            )

            top = df_opt.sort_values("score", ascending=False).head(10)
            print(f"\n  TOP 10 OPTIMIZED (by balanced score):")
            print(f"  {'Strategy':<30} {'Params':<15} {'#':>4} {'WR%':>6} {'Ret%':>8} {'PF':>6} {'Exp':>7}")
            print(f"  {'-'*80}")
            for _, r in top.iterrows():
                print(f"  {r['strategy']:<30} {r['params']:<15} "
                      f"{r['total']:>4} {r['win_rate']:>5.1f}% "
                      f"{r['total_return_pct']:>7.1f}% {r['profit_factor']:>5.2f} "
                      f"{r['expectancy_pct']:>6.2f}%")

            best = df_opt.sort_values("score", ascending=False).iloc[0]
            all_results[symbol] = {
                "name": info["name"],
                "bars": len(df),
                "best_strategy": best["strategy"],
                "best_params": best["params"],
                "trades": best["total"],
                "win_rate": best["win_rate"],
                "total_return": best["total_return_pct"],
                "profit_factor": best["profit_factor"],
                "expectancy": best["expectancy_pct"],
                "max_drawdown": best["max_drawdown_pct"],
            }
        else:
            print(f"  No profitable configs found with enough trades")

    # Final Summary
    print(f"\n\n{'='*90}")
    print(f"  BEST STRATEGY FOR EACH VIX INSTRUMENT")
    print(f"{'='*90}")
    print(f"{'Symbol':<8} {'Name':<25} {'Strategy':<30} {'Params':<15} {'#':>4} {'WR%':>6} {'Ret%':>8} {'PF':>6}")
    print(f"{'-'*90}")
    for symbol, r in all_results.items():
        print(f"{symbol:<8} {r['name']:<25} {r['best_strategy']:<30} {r['best_params']:<15} "
              f"{r['trades']:>4} {r['win_rate']:>5.1f}% "
              f"{r['total_return']:>7.1f}% {r['profit_factor']:>5.2f}")
    print(f"{'='*90}")

    return all_results


if __name__ == "__main__":
    run_all()
