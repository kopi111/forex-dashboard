"""
Strategy Optimizer: Sweep parameters to find best 50+ trade configurations.
"""

import itertools
from backtest import (
    fetch_gold_data, add_indicators, calc_stats,
    _check_exit
)
import numpy as np
import pandas as pd


def _run_ema_cross(df, fast, slow, trend, sl_m, tp_m):
    """Parameterized EMA crossover."""
    from ta.trend import EMAIndicator
    close = df["Close"]
    ema_f = EMAIndicator(close, window=fast).ema_indicator()
    ema_s = EMAIndicator(close, window=slow).ema_indicator()
    ema_t = EMAIndicator(close, window=trend).ema_indicator()

    trades, pos = [], None
    for i in range(1, len(df)):
        r, p = df.iloc[i], df.iloc[i - 1]
        ef, es, et = ema_f.iloc[i], ema_s.iloc[i], ema_t.iloc[i]
        pef, pes = ema_f.iloc[i - 1], ema_s.iloc[i - 1]

        if pos is None:
            if pef <= pes and ef > es and r["Close"] > et:
                pos = {"type": "BUY", "entry": r["Close"],
                       "sl": r["Close"] - r["ATR"] * sl_m,
                       "tp": r["Close"] + r["ATR"] * tp_m,
                       "entry_date": r.name}
            elif pef >= pes and ef < es and r["Close"] < et:
                pos = {"type": "SELL", "entry": r["Close"],
                       "sl": r["Close"] + r["ATR"] * sl_m,
                       "tp": r["Close"] - r["ATR"] * tp_m,
                       "entry_date": r.name}
        else:
            closed, pnl, _ = _check_exit(r, pos)
            if closed:
                pos["exit_date"] = r.name
                pos["pnl"] = round(pnl, 2)
                pos["result"] = "WIN" if pnl > 0 else "LOSS"
                trades.append(pos)
                pos = None
    return trades


def _run_rsi_strat(df, oversold, overbought, sl_m, tp_m):
    """Parameterized RSI strategy."""
    trades, pos = [], None
    for i in range(1, len(df)):
        r, p = df.iloc[i], df.iloc[i - 1]
        if pos is None:
            if p["RSI"] < oversold and r["RSI"] >= oversold:
                pos = {"type": "BUY", "entry": r["Close"],
                       "sl": r["Close"] - r["ATR"] * sl_m,
                       "tp": r["Close"] + r["ATR"] * tp_m,
                       "entry_date": r.name}
            elif p["RSI"] > overbought and r["RSI"] <= overbought:
                pos = {"type": "SELL", "entry": r["Close"],
                       "sl": r["Close"] + r["ATR"] * sl_m,
                       "tp": r["Close"] - r["ATR"] * tp_m,
                       "entry_date": r.name}
        else:
            closed, pnl, _ = _check_exit(r, pos)
            if closed:
                pos["exit_date"] = r.name
                pos["pnl"] = round(pnl, 2)
                pos["result"] = "WIN" if pnl > 0 else "LOSS"
                trades.append(pos)
                pos = None
    return trades


def _run_macd_strat(df, sl_m, tp_m):
    """Parameterized MACD strategy."""
    trades, pos = [], None
    for i in range(1, len(df)):
        r, p = df.iloc[i], df.iloc[i - 1]
        if pos is None:
            if p["MACD_Hist"] <= 0 and r["MACD_Hist"] > 0:
                pos = {"type": "BUY", "entry": r["Close"],
                       "sl": r["Close"] - r["ATR"] * sl_m,
                       "tp": r["Close"] + r["ATR"] * tp_m,
                       "entry_date": r.name}
            elif p["MACD_Hist"] >= 0 and r["MACD_Hist"] < 0:
                pos = {"type": "SELL", "entry": r["Close"],
                       "sl": r["Close"] + r["ATR"] * sl_m,
                       "tp": r["Close"] - r["ATR"] * tp_m,
                       "entry_date": r.name}
        else:
            closed, pnl, _ = _check_exit(r, pos)
            if closed:
                pos["exit_date"] = r.name
                pos["pnl"] = round(pnl, 2)
                pos["result"] = "WIN" if pnl > 0 else "LOSS"
                trades.append(pos)
                pos = None
    return trades


def _run_trend_follow(df, ema_len, adx_thresh, rsi_low, rsi_high, sl_m, tp_m):
    """Trend following: trade in direction of EMA when ADX strong + RSI confirms."""
    from ta.trend import EMAIndicator
    ema = EMAIndicator(df["Close"], window=ema_len).ema_indicator()

    trades, pos = [], None
    for i in range(2, len(df)):
        r, p = df.iloc[i], df.iloc[i - 1]

        if pos is None:
            uptrend = r["Close"] > ema.iloc[i] and ema.iloc[i] > ema.iloc[i - 1]
            downtrend = r["Close"] < ema.iloc[i] and ema.iloc[i] < ema.iloc[i - 1]

            if uptrend and r["ADX"] > adx_thresh and rsi_low < r["RSI"] < rsi_high:
                pos = {"type": "BUY", "entry": r["Close"],
                       "sl": r["Close"] - r["ATR"] * sl_m,
                       "tp": r["Close"] + r["ATR"] * tp_m,
                       "entry_date": r.name}
            elif downtrend and r["ADX"] > adx_thresh and (100 - rsi_high) < r["RSI"] < (100 - rsi_low):
                pos = {"type": "SELL", "entry": r["Close"],
                       "sl": r["Close"] + r["ATR"] * sl_m,
                       "tp": r["Close"] - r["ATR"] * tp_m,
                       "entry_date": r.name}
        else:
            closed, pnl, _ = _check_exit(r, pos)
            if closed:
                pos["exit_date"] = r.name
                pos["pnl"] = round(pnl, 2)
                pos["result"] = "WIN" if pnl > 0 else "LOSS"
                trades.append(pos)
                pos = None
    return trades


def optimize(min_trades=50):
    """Run parameter sweep and return best configs with 50+ trades."""
    print("Fetching max data (5y)...")
    df = fetch_gold_data(period="5y")
    df = add_indicators(df)
    print(f"Data: {len(df)} bars\n")

    results = []

    # --- EMA Crossover sweep ---
    print("Sweeping EMA Crossover...")
    for fast, slow, trend in [(5, 13, 50), (7, 21, 50), (9, 21, 50),
                               (5, 13, 30), (7, 15, 40), (9, 21, 100),
                               (5, 10, 30), (8, 21, 50), (10, 30, 50)]:
        for sl in [1.0, 1.5, 2.0, 2.5, 3.0]:
            for tp in [0.5, 0.75, 1.0, 1.5, 2.0]:
                trades = _run_ema_cross(df, fast, slow, trend, sl, tp)
                if len(trades) >= min_trades:
                    s = calc_stats(trades)
                    results.append({
                        "strategy": f"EMA({fast}/{slow}/{trend})",
                        "params": f"SL:{sl} TP:{tp}",
                        **s
                    })

    # --- RSI sweep ---
    print("Sweeping RSI...")
    for os_val in [25, 30, 35, 40]:
        for ob_val in [60, 65, 70, 75]:
            for sl in [1.0, 1.5, 2.0, 2.5, 3.0]:
                for tp in [0.5, 0.75, 1.0, 1.5, 2.0]:
                    trades = _run_rsi_strat(df, os_val, ob_val, sl, tp)
                    if len(trades) >= min_trades:
                        s = calc_stats(trades)
                        results.append({
                            "strategy": f"RSI({os_val}/{ob_val})",
                            "params": f"SL:{sl} TP:{tp}",
                            **s
                        })

    # --- MACD sweep ---
    print("Sweeping MACD...")
    for sl in [1.0, 1.5, 2.0, 2.5, 3.0]:
        for tp in [0.5, 0.75, 1.0, 1.5, 2.0]:
            trades = _run_macd_strat(df, sl, tp)
            if len(trades) >= min_trades:
                s = calc_stats(trades)
                results.append({
                    "strategy": "MACD",
                    "params": f"SL:{sl} TP:{tp}",
                    **s
                })

    # --- Trend Follow sweep ---
    print("Sweeping Trend Follow...")
    for ema_len in [21, 30, 50]:
        for adx_t in [15, 20, 25]:
            for rsi_lo, rsi_hi in [(40, 70), (35, 75), (45, 65), (30, 70)]:
                for sl in [1.5, 2.0, 2.5, 3.0]:
                    for tp in [0.5, 0.75, 1.0, 1.5]:
                        trades = _run_trend_follow(df, ema_len, adx_t, rsi_lo, rsi_hi, sl, tp)
                        if len(trades) >= min_trades:
                            s = calc_stats(trades)
                            results.append({
                                "strategy": f"TrendFollow(EMA{ema_len},ADX>{adx_t},RSI{rsi_lo}-{rsi_hi})",
                                "params": f"SL:{sl} TP:{tp}",
                                **s
                            })

    if not results:
        print("No configurations found with 50+ trades.")
        return

    # Sort by different criteria
    df_results = pd.DataFrame(results)

    print(f"\nTotal configs with {min_trades}+ trades: {len(df_results)}")

    print(f"\n{'='*90}")
    print("TOP 10 BY WIN RATE (50+ trades)")
    print(f"{'='*90}")
    top_wr = df_results.sort_values("win_rate", ascending=False).head(10)
    for _, r in top_wr.iterrows():
        print(f"{r['strategy']:<45} {r['params']:<15} "
              f"Trades:{r['total']:>3}  WR:{r['win_rate']:>5.1f}%  "
              f"P&L:${r['total_pnl']:>8.2f}  PF:{r['profit_factor']:>5.2f}")

    print(f"\n{'='*90}")
    print("TOP 10 BY PROFIT FACTOR (50+ trades)")
    print(f"{'='*90}")
    top_pf = df_results[df_results["profit_factor"] < float("inf")].sort_values(
        "profit_factor", ascending=False).head(10)
    for _, r in top_pf.iterrows():
        print(f"{r['strategy']:<45} {r['params']:<15} "
              f"Trades:{r['total']:>3}  WR:{r['win_rate']:>5.1f}%  "
              f"P&L:${r['total_pnl']:>8.2f}  PF:{r['profit_factor']:>5.2f}")

    print(f"\n{'='*90}")
    print("TOP 10 BY TOTAL P&L (50+ trades)")
    print(f"{'='*90}")
    top_pnl = df_results.sort_values("total_pnl", ascending=False).head(10)
    for _, r in top_pnl.iterrows():
        print(f"{r['strategy']:<45} {r['params']:<15} "
              f"Trades:{r['total']:>3}  WR:{r['win_rate']:>5.1f}%  "
              f"P&L:${r['total_pnl']:>8.2f}  PF:{r['profit_factor']:>5.2f}")

    # Best balanced: score = win_rate * profit_factor * log(trades)
    df_results["score"] = (df_results["win_rate"] / 100 *
                           df_results["profit_factor"].clip(upper=10) *
                           np.log(df_results["total"]))
    print(f"\n{'='*90}")
    print("TOP 10 BALANCED SCORE (WR * PF * log(trades))")
    print(f"{'='*90}")
    top_bal = df_results.sort_values("score", ascending=False).head(10)
    for _, r in top_bal.iterrows():
        print(f"{r['strategy']:<45} {r['params']:<15} "
              f"Trades:{r['total']:>3}  WR:{r['win_rate']:>5.1f}%  "
              f"P&L:${r['total_pnl']:>8.2f}  PF:{r['profit_factor']:>5.2f}  "
              f"Score:{r['score']:>.2f}")

    return df_results


if __name__ == "__main__":
    optimize(min_trades=50)
