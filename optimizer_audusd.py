"""
AUD/USD Strategy Optimizer: Sweep parameters to find best configurations.
P&L in pips (1 pip = 0.0001 for AUD/USD).
"""

import numpy as np
import pandas as pd
from backtest_audusd import (
    fetch_audusd_data, add_indicators, calc_stats, _check_exit, PIP
)
from ta.trend import EMAIndicator


def _run_ema_cross(df, fast, slow, trend, sl_m, tp_m):
    ema_f = EMAIndicator(df["Close"], window=fast).ema_indicator()
    ema_s = EMAIndicator(df["Close"], window=slow).ema_indicator()
    ema_t = EMAIndicator(df["Close"], window=trend).ema_indicator()

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
                pos["pnl"] = round(pnl, 1)
                pos["result"] = "WIN" if pnl > 0 else "LOSS"
                trades.append(pos)
                pos = None
    return trades


def _run_rsi_strat(df, oversold, overbought, sl_m, tp_m):
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
                pos["pnl"] = round(pnl, 1)
                pos["result"] = "WIN" if pnl > 0 else "LOSS"
                trades.append(pos)
                pos = None
    return trades


def _run_macd_strat(df, sl_m, tp_m):
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
                pos["pnl"] = round(pnl, 1)
                pos["result"] = "WIN" if pnl > 0 else "LOSS"
                trades.append(pos)
                pos = None
    return trades


def _run_trend_follow(df, ema_len, adx_thresh, rsi_low, rsi_high, sl_m, tp_m):
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
                pos["pnl"] = round(pnl, 1)
                pos["result"] = "WIN" if pnl > 0 else "LOSS"
                trades.append(pos)
                pos = None
    return trades


def _run_triple_confluence(df, adx_thresh, rsi_low_buy, rsi_high_buy,
                           rsi_low_sell, rsi_high_sell, sl_m, tp_m):
    """Parameterized triple confluence."""
    trades, pos = [], None
    for i in range(2, len(df)):
        r, p = df.iloc[i], df.iloc[i - 1]
        if pos is None:
            uptrend = r["EMA_50"] > r["EMA_200"] and r["Close"] > r["EMA_50"]
            downtrend = r["EMA_50"] < r["EMA_200"] and r["Close"] < r["EMA_50"]

            if (uptrend and rsi_low_buy < r["RSI"] < rsi_high_buy and
                    r["MACD_Hist"] > p["MACD_Hist"] and r["MACD_Hist"] > 0 and
                    r["ADX"] > adx_thresh):
                pos = {"type": "BUY", "entry": r["Close"],
                       "sl": r["Close"] - r["ATR"] * sl_m,
                       "tp": r["Close"] + r["ATR"] * tp_m,
                       "entry_date": r.name}
            elif (downtrend and rsi_low_sell < r["RSI"] < rsi_high_sell and
                  r["MACD_Hist"] < p["MACD_Hist"] and r["MACD_Hist"] < 0 and
                  r["ADX"] > adx_thresh):
                pos = {"type": "SELL", "entry": r["Close"],
                       "sl": r["Close"] + r["ATR"] * sl_m,
                       "tp": r["Close"] - r["ATR"] * tp_m,
                       "entry_date": r.name}
        else:
            closed, pnl, _ = _check_exit(r, pos)
            if closed:
                pos["exit_date"] = r.name
                pos["pnl"] = round(pnl, 1)
                pos["result"] = "WIN" if pnl > 0 else "LOSS"
                trades.append(pos)
                pos = None
    return trades


def optimize(min_trades=30):
    """Run parameter sweep for AUD/USD and return best configs."""
    print("Fetching AUD/USD data (5y)...")
    df = fetch_audusd_data(period="5y")
    df = add_indicators(df)
    print(f"Data: {len(df)} bars ({df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')})")
    print(f"ATR range: {df['ATR'].min()*10000:.0f} - {df['ATR'].max()*10000:.0f} pips\n")

    results = []

    # --- EMA Crossover sweep ---
    print("Sweeping EMA Crossover...")
    for fast, slow, trend in [(5, 13, 50), (7, 21, 50), (9, 21, 50),
                               (5, 13, 30), (7, 15, 40), (9, 21, 100),
                               (5, 10, 30), (8, 21, 50), (10, 30, 50),
                               (12, 26, 50), (5, 20, 100), (8, 34, 50)]:
        for sl in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]:
            for tp in [0.5, 0.75, 1.0, 1.5, 2.0, 2.5]:
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
    for os_val in [20, 25, 30, 35, 40]:
        for ob_val in [60, 65, 70, 75, 80]:
            for sl in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]:
                for tp in [0.5, 0.75, 1.0, 1.5, 2.0, 2.5]:
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
    for sl in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]:
        for tp in [0.5, 0.75, 1.0, 1.5, 2.0, 2.5]:
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
    for ema_len in [21, 30, 50, 100]:
        for adx_t in [15, 20, 25, 30]:
            for rsi_lo, rsi_hi in [(40, 70), (35, 75), (45, 65), (30, 70),
                                    (35, 80), (30, 80)]:
                for sl in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
                    for tp in [0.5, 0.75, 1.0, 1.5, 2.0]:
                        trades = _run_trend_follow(df, ema_len, adx_t,
                                                    rsi_lo, rsi_hi, sl, tp)
                        if len(trades) >= min_trades:
                            s = calc_stats(trades)
                            results.append({
                                "strategy": f"TrendFollow(EMA{ema_len},ADX>{adx_t},RSI{rsi_lo}-{rsi_hi})",
                                "params": f"SL:{sl} TP:{tp}",
                                **s
                            })

    # --- Triple Confluence sweep ---
    print("Sweeping Triple Confluence...")
    for adx_t in [15, 20, 25]:
        for rsi_lb, rsi_hb in [(30, 55), (35, 60), (30, 65), (40, 60)]:
            for rsi_ls, rsi_hs in [(40, 65), (35, 70), (45, 65), (40, 70)]:
                for sl in [1.5, 2.0, 2.5, 3.0]:
                    for tp in [0.75, 1.0, 1.2, 1.5, 2.0]:
                        trades = _run_triple_confluence(df, adx_t,
                                                         rsi_lb, rsi_hb,
                                                         rsi_ls, rsi_hs,
                                                         sl, tp)
                        if len(trades) >= min_trades:
                            s = calc_stats(trades)
                            results.append({
                                "strategy": f"TripleConf(ADX>{adx_t},B:{rsi_lb}-{rsi_hb},S:{rsi_ls}-{rsi_hs})",
                                "params": f"SL:{sl} TP:{tp}",
                                **s
                            })

    if not results:
        print("No configurations found with enough trades.")
        return

    df_results = pd.DataFrame(results)
    # Filter out negative P&L configs for ranking
    profitable = df_results[df_results["total_pips"] > 0]

    print(f"\nTotal configs tested: {len(df_results)}")
    print(f"Profitable configs: {len(profitable)}")

    def _print_table(subset, title):
        print(f"\n{'='*100}")
        print(title)
        print(f"{'='*100}")
        print(f"{'Strategy':<50} {'Params':<15} {'#':>4} {'WR%':>6} {'Pips':>8} {'PF':>6} {'R:R':>5} {'Exp':>6}")
        print(f"{'-'*100}")
        for _, r in subset.iterrows():
            print(f"{r['strategy']:<50} {r['params']:<15} "
                  f"{r['total']:>4} {r['win_rate']:>5.1f}% "
                  f"{r['total_pips']:>7.0f} {r['profit_factor']:>5.2f} "
                  f"{r['avg_rr']:>5.2f} {r['expectancy_pips']:>5.1f}p")

    _print_table(
        df_results.sort_values("win_rate", ascending=False).head(15),
        "TOP 15 BY WIN RATE"
    )

    _print_table(
        df_results[df_results["profit_factor"] < float("inf")]
        .sort_values("profit_factor", ascending=False).head(15),
        "TOP 15 BY PROFIT FACTOR"
    )

    _print_table(
        df_results.sort_values("total_pips", ascending=False).head(15),
        "TOP 15 BY TOTAL PIPS"
    )

    _print_table(
        df_results.sort_values("expectancy_pips", ascending=False).head(15),
        "TOP 15 BY EXPECTANCY (pips/trade)"
    )

    # Balanced score: combines win rate, PF, expectancy, and trade count
    df_results["score"] = (
        (df_results["win_rate"] / 100) *
        df_results["profit_factor"].clip(upper=10) *
        df_results["expectancy_pips"].clip(lower=0) *
        np.log(df_results["total"].clip(lower=1))
    )
    _print_table(
        df_results.sort_values("score", ascending=False).head(15),
        "TOP 15 BALANCED SCORE (WR * PF * Expectancy * log(trades))"
    )

    # Print the absolute best
    best = df_results.sort_values("score", ascending=False).iloc[0]
    print(f"\n{'*'*60}")
    print(f"  BEST OVERALL: {best['strategy']}")
    print(f"  Params: {best['params']}")
    print(f"  Trades: {best['total']} | Win Rate: {best['win_rate']}%")
    print(f"  Total Pips: {best['total_pips']} | PF: {best['profit_factor']}")
    print(f"  Avg Win: {best['avg_win_pips']}p | Avg Loss: {best['avg_loss_pips']}p")
    print(f"  R:R = {best['avg_rr']} | Expectancy: {best['expectancy_pips']}p/trade")
    print(f"  Max Drawdown: {best['max_drawdown_pips']} pips")
    print(f"{'*'*60}")

    return df_results


if __name__ == "__main__":
    optimize(min_trades=30)
