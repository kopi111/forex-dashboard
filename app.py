"""
Trading Dashboard - Flask App
Serves: XAU/USD, AUD/USD, Deriv Synthetic Indices dashboards
"""

from flask import Flask, render_template, jsonify, request
from backtest import run_all_strategies, fetch_gold_data, add_indicators
import json

app = Flask(__name__)


# ---------------------------------------------------------------------------
# Main Pages
# ---------------------------------------------------------------------------

@app.route("/")
def home():
    """Landing page with links to all dashboards."""
    return render_template("home.html")


@app.route("/gold")
def gold_dashboard():
    """XAU/USD Gold Dashboard."""
    return render_template("index.html")


@app.route("/audusd")
def audusd_dashboard():
    """AUD/USD Dashboard (uses same template with different data)."""
    return render_template("index.html")


# ---------------------------------------------------------------------------
# Deriv Synthetic Indices Pages (from GitHub dashboard)
# ---------------------------------------------------------------------------

@app.route("/deriv")
def deriv_dashboard():
    """Deriv Live Trading Dashboard (V75 + Scanner)."""
    return render_template("deriv_dashboard.html")


@app.route("/deriv/signals")
def deriv_signals():
    """Deriv Trading Signals Dashboard."""
    return render_template("deriv_signals.html")


@app.route("/deriv/rsi")
def deriv_rsi():
    """Deriv RSI Strategy Signals."""
    return render_template("deriv_rsi.html")


@app.route("/deriv/price-action")
def deriv_price_action():
    """Deriv Price Action Analysis."""
    return render_template("deriv_price_action.html")


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.route("/api/backtest")
def api_backtest():
    """Run XAU/USD backtest."""
    period = request.args.get("period", "2y")
    try:
        results, price_data = run_all_strategies(period=period)

        response = {}
        for name, data in results.items():
            trades_serialized = []
            for t in data["trades"]:
                trades_serialized.append({
                    "type": t["type"],
                    "entry": t["entry"],
                    "pnl": t["pnl"],
                    "result": t["result"],
                    "entry_date": str(t["entry_date"]),
                    "exit_date": str(t["exit_date"]),
                    "sl": round(t["sl"], 2),
                    "tp": round(t["tp"], 2),
                })
            response[name] = {
                "stats": data["stats"],
                "trades": trades_serialized,
            }

        return jsonify({
            "strategies": response,
            "prices": json.loads(price_data.to_json(orient="index")),
            "dates": list(price_data.index),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/backtest/audusd")
def api_backtest_audusd():
    """Run AUD/USD backtest."""
    try:
        from backtest_audusd import run_all_strategies as run_audusd
        results, df = run_audusd(period=request.args.get("period", "5y"))

        response = {}
        for name, data in results.items():
            trades_serialized = []
            for t in data["trades"]:
                trades_serialized.append({
                    "type": t["type"],
                    "entry": round(t["entry"], 5),
                    "pnl": t["pnl"],
                    "result": t["result"],
                    "entry_date": str(t["entry_date"]),
                    "exit_date": str(t.get("exit_date", "")),
                    "sl": round(t["sl"], 5),
                    "tp": round(t["tp"], 5),
                })
            response[name] = {
                "stats": data["stats"],
                "trades": trades_serialized,
            }

        return jsonify({"strategies": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/backtest/deriv")
def api_backtest_deriv():
    """Run Deriv synthetic indices backtest (cached)."""
    try:
        from backtest_deriv import fetch_candles, add_indicators, calc_stats
        from backtest_deriv import (strategy_rsi_reversal, strategy_ema_crossover,
                                     strategy_macd_momentum, strategy_bollinger_bounce)

        symbol = request.args.get("symbol", "R_75")
        granularity = int(request.args.get("granularity", "3600"))

        df = fetch_candles(symbol, granularity=granularity, count=5000)
        if df is None:
            return jsonify({"error": "Failed to fetch data"}), 500

        df = add_indicators(df)

        strategies = {
            "EMA(9/21)": strategy_ema_crossover(df),
            "RSI(30/70)": strategy_rsi_reversal(df),
            "MACD": strategy_macd_momentum(df),
            "Bollinger": strategy_bollinger_bounce(df),
        }

        response = {}
        for name, trades in strategies.items():
            stats = calc_stats(trades)
            trades_ser = [{
                "type": t["type"], "entry": round(t["entry"], 2),
                "pnl": t["pnl"], "result": t["result"],
                "entry_date": str(t["entry_date"]),
                "exit_date": str(t.get("exit_date", "")),
            } for t in trades]
            response[name] = {"stats": stats, "trades": trades_ser}

        return jsonify({
            "symbol": symbol,
            "bars": len(df),
            "strategies": response,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/strategies")
def api_strategies():
    """Return best strategies summary for all instruments."""
    strategies = {
        "XAU/USD": {
            "strategy": "Optimized Trend (EMA50 + ADX + RSI)",
            "params": "SL:3.0 ATR, TP:0.75 ATR",
            "win_rate": "87.4%",
            "profit_factor": 2.33,
        },
        "AUD/USD": {
            "strategy": "RSI(30/65)",
            "params": "SL:2.5 ATR, TP:1.0 ATR",
            "win_rate": "90.0%",
            "profit_factor": 5.41,
        },
        "R_75 (Volatility 75)": {
            "strategy": "EMA(5/21) Crossover",
            "params": "SL:3.0 ATR, TP:1.5 ATR",
            "win_rate": "73.4%",
            "total_return": "+89.4%",
            "profit_factor": 1.35,
        },
        "R_100 (Volatility 100)": {
            "strategy": "RSI(25/70)",
            "params": "SL:3.0 ATR, TP:1.0 ATR",
            "win_rate": "84.2%",
            "total_return": "+60.6%",
            "profit_factor": 1.72,
        },
        "1HZ100V (Vol 100 1s)": {
            "strategy": "RSI(25/75)",
            "params": "SL:3.0 ATR, TP:2.0 ATR",
            "win_rate": "67.7%",
            "total_return": "+44.4%",
            "profit_factor": 1.41,
        },
        "1HZ75V (Vol 75 1s)": {
            "strategy": "RSI(25/70)",
            "params": "SL:3.0 ATR, TP:2.5 ATR",
            "win_rate": "61.0%",
            "total_return": "+45.2%",
            "profit_factor": 1.30,
        },
        "stpRNG (Step Index)": {
            "strategy": "RSI(30/75)",
            "params": "SL:2.0 ATR, TP:0.75 ATR",
            "win_rate": "83.7%",
            "total_return": "+4.7%",
            "profit_factor": 1.97,
        },
    }
    return jsonify(strategies)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
