"""
AUD/USD Trading Bot - RSI(30/65) Strategy
Platform: MetaTrader 5 (Windows) or Signal-Only mode (Linux/Mac)
Timeframe: Daily (D1)

Strategy:
  BUY  -> RSI(14) crosses back above 30 (oversold recovery)
  SELL -> RSI(14) crosses back below 65 (overbought reversal)
  SL   -> 2.5 x ATR(14)
  TP   -> 1.0 x ATR(14)

Usage:
  python3 bot_audusd.py                          # Signal mode (works anywhere)
  python3 bot_audusd.py --mt5                     # MT5 auto-execute (Windows only)
  python3 bot_audusd.py --risk 1.0                # Risk 1% per trade
  python3 bot_audusd.py --telegram --token XXX --chat-id YYY  # Telegram alerts
"""

import pandas as pd
import numpy as np
import yfinance as yf
import time
import logging
import json
import argparse
import platform
from datetime import datetime
from pathlib import Path

# Try to import MT5 (Windows only)
MT5_AVAILABLE = False
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    pass

# Try to import requests for Telegram
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG = {
    # Mode: "signal" (alert only) or "mt5" (auto-execute on MT5)
    "mode": "signal",

    # MT5 Connection (only needed for mt5 mode)
    "mt5_path": None,
    "mt5_login": None,
    "mt5_password": None,
    "mt5_server": None,

    # Trading
    "symbol": "AUDUSD",
    "symbol_yf": "AUDUSD=X",
    "magic_number": 202602,
    "max_spread_pips": 3.0,
    "max_slippage_points": 30,

    # Strategy Parameters (from optimizer - best on 5y backtest)
    "rsi_period": 14,
    "rsi_oversold": 30,
    "rsi_overbought": 65,
    "atr_period": 14,
    "sl_atr_mult": 2.5,
    "tp_atr_mult": 1.0,

    # Risk Management
    "risk_percent": 2.0,
    "account_balance": 10000,    # Used in signal mode for lot calculation
    "max_open_trades": 1,
    "max_daily_trades": 2,
    "max_daily_loss_pct": 5.0,

    # Telegram Alerts (optional)
    "telegram_enabled": False,
    "telegram_token": "",
    "telegram_chat_id": "",

    # Bot Settings
    "check_interval_seconds": 300,  # 5 min check for daily TF
}

PIP = 0.0001
CONFIG_FILE = Path(__file__).parent / "bot_config.json"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_DIR = Path(__file__).parent / "bot_logs"
LOG_DIR.mkdir(exist_ok=True)

log_file = LOG_DIR / f"audusd_bot_{datetime.now().strftime('%Y%m%d')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger("AUDUSD_BOT")


# ---------------------------------------------------------------------------
# Trade History (for signal mode)
# ---------------------------------------------------------------------------

TRADES_FILE = LOG_DIR / "trades_history.json"
OPEN_TRADE_FILE = LOG_DIR / "open_trade.json"


def save_trade(trade):
    """Append trade to history file."""
    trades = []
    if TRADES_FILE.exists():
        with open(TRADES_FILE) as f:
            trades = json.load(f)
    trades.append(trade)
    with open(TRADES_FILE, "w") as f:
        json.dump(trades, f, indent=2)


def save_open_trade(trade):
    """Save current open trade."""
    if trade is None:
        OPEN_TRADE_FILE.unlink(missing_ok=True)
    else:
        with open(OPEN_TRADE_FILE, "w") as f:
            json.dump(trade, f, indent=2)


def load_open_trade():
    """Load current open trade if any."""
    if OPEN_TRADE_FILE.exists():
        with open(OPEN_TRADE_FILE) as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Telegram Alerts
# ---------------------------------------------------------------------------

def send_telegram(message):
    """Send alert via Telegram bot."""
    if not CONFIG["telegram_enabled"] or not REQUESTS_AVAILABLE:
        return
    try:
        url = f"https://api.telegram.org/bot{CONFIG['telegram_token']}/sendMessage"
        payload = {
            "chat_id": CONFIG["telegram_chat_id"],
            "text": message,
            "parse_mode": "Markdown",
        }
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        logger.warning(f"Telegram send failed: {e}")


# ---------------------------------------------------------------------------
# MT5 Functions (Windows only)
# ---------------------------------------------------------------------------

def connect_mt5():
    if not MT5_AVAILABLE:
        logger.error("MetaTrader5 package not available (Windows only)")
        return False

    kwargs = {}
    if CONFIG["mt5_path"]:
        kwargs["path"] = CONFIG["mt5_path"]
    if CONFIG["mt5_login"]:
        kwargs["login"] = int(CONFIG["mt5_login"])
    if CONFIG["mt5_password"]:
        kwargs["password"] = CONFIG["mt5_password"]
    if CONFIG["mt5_server"]:
        kwargs["server"] = CONFIG["mt5_server"]

    if not mt5.initialize(**kwargs):
        logger.error(f"MT5 init failed: {mt5.last_error()}")
        return False

    account = mt5.account_info()
    if account is None:
        logger.error("Failed to get account info")
        return False

    logger.info(f"MT5 Connected: {account.server}")
    logger.info(f"Account: {account.login} | Balance: ${account.balance:.2f} | "
                f"Leverage: 1:{account.leverage}")
    CONFIG["account_balance"] = account.balance

    # Find symbol
    symbol = CONFIG["symbol"]
    info = mt5.symbol_info(symbol)
    if info is None:
        for variant in [symbol, f"{symbol}m", f"{symbol}.raw", f"{symbol}.",
                        f"{symbol[:3]}/{symbol[3:]}"]:
            info = mt5.symbol_info(variant)
            if info is not None:
                CONFIG["symbol"] = variant
                logger.info(f"Symbol found as: {variant}")
                break
        if info is None:
            logger.error(f"Symbol {symbol} not found")
            return False

    if not info.visible:
        mt5.symbol_select(CONFIG["symbol"], True)

    logger.info(f"Symbol: {CONFIG['symbol']} | Spread: {info.spread}pts | Digits: {info.digits}")
    return True


def disconnect_mt5():
    if MT5_AVAILABLE:
        mt5.shutdown()
        logger.info("MT5 disconnected")


# ---------------------------------------------------------------------------
# Data Fetching
# ---------------------------------------------------------------------------

def get_candles_yfinance(count=200):
    """Fetch candles from Yahoo Finance (works on any platform)."""
    try:
        ticker = yf.Ticker(CONFIG["symbol_yf"])
        df = ticker.history(period="1y", interval="1d")
        if df is None or len(df) == 0:
            logger.error("yfinance returned no data")
            return None
        df = df.dropna()
        return df.tail(count)
    except Exception as e:
        logger.error(f"yfinance error: {e}")
        return None


def get_candles_mt5(count=200):
    """Fetch candles from MT5."""
    rates = mt5.copy_rates_from_pos(CONFIG["symbol"], mt5.TIMEFRAME_D1, 0, count)
    if rates is None or len(rates) == 0:
        logger.error(f"MT5 candle fetch failed: {mt5.last_error()}")
        return None

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)
    df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                        "close": "Close", "tick_volume": "Volume"}, inplace=True)
    return df


def get_candles(count=200):
    """Fetch candles from best available source."""
    if CONFIG["mode"] == "mt5" and MT5_AVAILABLE:
        df = get_candles_mt5(count)
        if df is not None:
            return df
    return get_candles_yfinance(count)


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


def compute_atr(df, period=14):
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def analyze(df):
    """Compute indicators and return analysis dict."""
    df = df.copy()
    df["RSI"] = compute_rsi(df["Close"], CONFIG["rsi_period"])
    df["ATR"] = compute_atr(df, CONFIG["atr_period"])
    df.dropna(inplace=True)

    if len(df) < 2:
        return None

    current = df.iloc[-1]
    previous = df.iloc[-2]

    spread = 0
    if CONFIG["mode"] == "mt5" and MT5_AVAILABLE:
        info = mt5.symbol_info(CONFIG["symbol"])
        if info:
            spread = info.spread

    return {
        "time": df.index[-1],
        "close": current["Close"],
        "high": current["High"],
        "low": current["Low"],
        "rsi": current["RSI"],
        "rsi_prev": previous["RSI"],
        "atr": current["ATR"],
        "atr_pips": current["ATR"] / PIP,
        "spread": spread,
    }


# ---------------------------------------------------------------------------
# Signal Detection
# ---------------------------------------------------------------------------

def check_signal(data):
    if data is None:
        return None

    rsi = data["rsi"]
    rsi_prev = data["rsi_prev"]

    # BUY: RSI crosses back above 30
    if rsi_prev < CONFIG["rsi_oversold"] and rsi >= CONFIG["rsi_oversold"]:
        logger.info(f">>> BUY SIGNAL: RSI crossed above {CONFIG['rsi_oversold']} "
                    f"(prev={rsi_prev:.1f}, now={rsi:.1f})")
        return "BUY"

    # SELL: RSI crosses back below 65
    if rsi_prev > CONFIG["rsi_overbought"] and rsi <= CONFIG["rsi_overbought"]:
        logger.info(f">>> SELL SIGNAL: RSI crossed below {CONFIG['rsi_overbought']} "
                    f"(prev={rsi_prev:.1f}, now={rsi:.1f})")
        return "SELL"

    return None


# ---------------------------------------------------------------------------
# Position Sizing
# ---------------------------------------------------------------------------

def calculate_lot_size(sl_distance):
    """Calculate lot size based on risk % and SL distance."""
    if CONFIG["mode"] == "mt5" and MT5_AVAILABLE:
        account = mt5.account_info()
        balance = account.balance if account else CONFIG["account_balance"]
        symbol_info = mt5.symbol_info(CONFIG["symbol"])
        if symbol_info:
            sl_pips = sl_distance / (symbol_info.point * 10)
            pip_value = symbol_info.trade_tick_value * 10
            if sl_pips > 0 and pip_value > 0:
                risk_amount = balance * CONFIG["risk_percent"] / 100
                lot_size = risk_amount / (sl_pips * pip_value)
                lot_step = symbol_info.volume_step
                lot_size = max(symbol_info.volume_min,
                               min(symbol_info.volume_max,
                                   round(lot_size / lot_step) * lot_step))
                return round(lot_size, 2)

    # Fallback: standard calculation
    balance = CONFIG["account_balance"]
    risk_amount = balance * CONFIG["risk_percent"] / 100
    sl_pips = sl_distance / PIP
    pip_value = 10  # $10 per pip per standard lot for AUD/USD
    if sl_pips > 0:
        lot_size = risk_amount / (sl_pips * pip_value)
        return round(max(0.01, lot_size), 2)
    return 0.01


# ---------------------------------------------------------------------------
# Check Open Trade (signal mode)
# ---------------------------------------------------------------------------

def check_open_trade(data):
    """Check if open trade hit SL or TP."""
    trade = load_open_trade()
    if trade is None:
        return

    hit_tp = False
    hit_sl = False

    if trade["signal"] == "BUY":
        if data["high"] >= trade["tp"]:
            hit_tp = True
        elif data["low"] <= trade["sl"]:
            hit_sl = True
    else:
        if data["low"] <= trade["tp"]:
            hit_tp = True
        elif data["high"] >= trade["sl"]:
            hit_sl = True

    if hit_tp or hit_sl:
        result = "WIN (TP hit)" if hit_tp else "LOSS (SL hit)"
        exit_price = trade["tp"] if hit_tp else trade["sl"]

        if trade["signal"] == "BUY":
            pnl_pips = (exit_price - trade["entry"]) / PIP
        else:
            pnl_pips = (trade["entry"] - exit_price) / PIP

        pnl_dollar = pnl_pips * 10 * trade["lots"]  # $10 per pip per lot

        trade["exit_price"] = round(exit_price, 5)
        trade["exit_time"] = str(data["time"])
        trade["result"] = result
        trade["pnl_pips"] = round(pnl_pips, 1)
        trade["pnl_dollar"] = round(pnl_dollar, 2)

        save_trade(trade)
        save_open_trade(None)

        logger.info(f"TRADE CLOSED: {result} | P&L: {pnl_pips:.0f} pips (${pnl_dollar:.2f})")

        msg = (
            f"*AUD/USD TRADE CLOSED*\n"
            f"Result: {result}\n"
            f"Entry: {trade['entry']} -> Exit: {exit_price}\n"
            f"P&L: {pnl_pips:.0f} pips (${pnl_dollar:.2f})\n"
            f"Lots: {trade['lots']}"
        )
        send_telegram(msg)


# ---------------------------------------------------------------------------
# Order Execution
# ---------------------------------------------------------------------------

def execute_signal(signal, data):
    """Execute trade signal based on mode."""
    atr = data["atr"]
    sl_distance = atr * CONFIG["sl_atr_mult"]
    tp_distance = atr * CONFIG["tp_atr_mult"]
    price = data["close"]

    if signal == "BUY":
        sl = round(price - sl_distance, 5)
        tp = round(price + tp_distance, 5)
    else:
        sl = round(price + sl_distance, 5)
        tp = round(price - tp_distance, 5)

    lot_size = calculate_lot_size(sl_distance)
    risk_dollar = CONFIG["account_balance"] * CONFIG["risk_percent"] / 100

    # --- MT5 Auto-Execute ---
    if CONFIG["mode"] == "mt5" and MT5_AVAILABLE:
        return execute_mt5(signal, data, sl, tp, sl_distance, tp_distance, lot_size)

    # --- Signal Mode (log + alert) ---
    trade = {
        "time": str(data["time"]),
        "signal": signal,
        "entry": round(price, 5),
        "sl": sl,
        "tp": tp,
        "sl_pips": round(sl_distance / PIP, 1),
        "tp_pips": round(tp_distance / PIP, 1),
        "lots": lot_size,
        "risk_dollar": round(risk_dollar, 2),
        "rsi": round(data["rsi"], 1),
        "atr_pips": round(data["atr_pips"], 1),
    }

    save_open_trade(trade)

    logger.info("=" * 50)
    logger.info(f"  >>> {signal} AUD/USD <<<")
    logger.info(f"  Entry:  {price:.5f}")
    logger.info(f"  SL:     {sl:.5f}  ({sl_distance/PIP:.0f} pips)")
    logger.info(f"  TP:     {tp:.5f}  ({tp_distance/PIP:.0f} pips)")
    logger.info(f"  Lots:   {lot_size}")
    logger.info(f"  Risk:   ${risk_dollar:.2f} ({CONFIG['risk_percent']}%)")
    logger.info(f"  RSI:    {data['rsi']:.1f}")
    logger.info(f"  ATR:    {data['atr_pips']:.0f} pips")
    logger.info("=" * 50)

    msg = (
        f"*{'='*30}*\n"
        f"*>>> {signal} AUD/USD <<<*\n"
        f"*{'='*30}*\n\n"
        f"*Entry:* `{price:.5f}`\n"
        f"*SL:* `{sl:.5f}` ({sl_distance/PIP:.0f} pips)\n"
        f"*TP:* `{tp:.5f}` ({tp_distance/PIP:.0f} pips)\n"
        f"*Lots:* {lot_size}\n"
        f"*Risk:* ${risk_dollar:.2f} ({CONFIG['risk_percent']}%)\n"
        f"*RSI:* {data['rsi']:.1f} | *ATR:* {data['atr_pips']:.0f}p\n\n"
        f"_Place this trade on MT5 manually_"
    )
    send_telegram(msg)

    return trade


def execute_mt5(signal, data, sl, tp, sl_distance, tp_distance, lot_size):
    """Execute trade on MT5."""
    symbol_info = mt5.symbol_info(CONFIG["symbol"])
    tick = mt5.symbol_info_tick(CONFIG["symbol"])
    if not symbol_info or not tick:
        logger.error("Cannot get MT5 symbol/tick info")
        return None

    digits = symbol_info.digits

    if signal == "BUY":
        order_type = mt5.ORDER_TYPE_BUY
        price = tick.ask
        sl = round(price - sl_distance, digits)
        tp = round(price + tp_distance, digits)
    else:
        order_type = mt5.ORDER_TYPE_SELL
        price = tick.bid
        sl = round(price + sl_distance, digits)
        tp = round(price - tp_distance, digits)

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": CONFIG["symbol"],
        "volume": lot_size,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": CONFIG["max_slippage_points"],
        "magic": CONFIG["magic_number"],
        "comment": "AUDUSD_RSI_BOT",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    logger.info(f"MT5 Sending {signal}: {lot_size} lots @ {price} SL={sl} TP={tp}")
    result = mt5.order_send(request)

    if result is None:
        logger.error(f"MT5 order failed: {mt5.last_error()}")
        return None
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.error(f"MT5 order rejected: {result.retcode} - {result.comment}")
        return None

    logger.info(f"MT5 ORDER FILLED: #{result.order} | {signal} {lot_size}lots @ {result.price}")

    account = mt5.account_info()
    risk_dollar = account.balance * CONFIG["risk_percent"] / 100 if account else 0

    msg = (
        f"*AUD/USD {signal} - MT5 FILLED*\n"
        f"Ticket: #{result.order}\n"
        f"Entry: {result.price}\n"
        f"SL: {sl} ({sl_distance/PIP:.0f}p) | TP: {tp} ({tp_distance/PIP:.0f}p)\n"
        f"Lots: {lot_size} | Risk: ${risk_dollar:.2f}"
    )
    send_telegram(msg)
    return result


# ---------------------------------------------------------------------------
# Risk Checks (MT5 mode)
# ---------------------------------------------------------------------------

def check_risk_limits_mt5(data):
    """Check risk limits when in MT5 mode."""
    positions = mt5.positions_get(symbol=CONFIG["symbol"])
    bot_positions = [p for p in (positions or []) if p.magic == CONFIG["magic_number"]]
    if len(bot_positions) >= CONFIG["max_open_trades"]:
        return False, "Max open trades reached"

    spread_pips = data["spread"] / 10
    if spread_pips > CONFIG["max_spread_pips"]:
        return False, f"Spread too wide ({spread_pips:.1f}p)"

    return True, "OK"


# ---------------------------------------------------------------------------
# Config Management
# ---------------------------------------------------------------------------

def save_config():
    save_data = {k: v for k, v in CONFIG.items()}
    with open(CONFIG_FILE, "w") as f:
        json.dump(save_data, f, indent=2)
    logger.info(f"Config saved to {CONFIG_FILE}")


def load_config():
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            saved = json.load(f)
        for key, val in saved.items():
            if key in CONFIG:
                CONFIG[key] = val
        logger.info(f"Config loaded from {CONFIG_FILE}")


# ---------------------------------------------------------------------------
# Status Display
# ---------------------------------------------------------------------------

def print_status(data):
    if data is None:
        return

    # RSI zone indicator
    rsi = data["rsi"]
    if rsi < CONFIG["rsi_oversold"]:
        zone = "OVERSOLD <<<"
    elif rsi > CONFIG["rsi_overbought"]:
        zone = "OVERBOUGHT >>>"
    elif rsi < 40:
        zone = "Near oversold"
    elif rsi > 60:
        zone = "Near overbought"
    else:
        zone = "Neutral"

    logger.info(
        f"AUD/USD {data['close']:.5f} | RSI: {rsi:.1f} ({zone}) | "
        f"ATR: {data['atr_pips']:.0f}p | {data['time']}"
    )

    # Check open trade status
    trade = load_open_trade()
    if trade:
        if trade["signal"] == "BUY":
            current_pips = (data["close"] - trade["entry"]) / PIP
        else:
            current_pips = (trade["entry"] - data["close"]) / PIP
        logger.info(f"  Open: {trade['signal']} @ {trade['entry']} | "
                    f"Current P&L: {current_pips:.0f} pips")


# ---------------------------------------------------------------------------
# Main Bot Loop
# ---------------------------------------------------------------------------

def run_bot():
    logger.info("=" * 60)
    logger.info("    AUD/USD RSI(30/65) Trading Bot")
    logger.info("=" * 60)
    logger.info(f"  Mode:     {'MT5 Auto-Execute' if CONFIG['mode'] == 'mt5' else 'Signal Alerts'}")
    logger.info(f"  Strategy: RSI({CONFIG['rsi_oversold']}/{CONFIG['rsi_overbought']})")
    logger.info(f"  SL: {CONFIG['sl_atr_mult']}x ATR | TP: {CONFIG['tp_atr_mult']}x ATR")
    logger.info(f"  Risk: {CONFIG['risk_percent']}% per trade")
    logger.info(f"  Balance: ${CONFIG['account_balance']:,.0f}")
    logger.info(f"  Telegram: {'ON' if CONFIG['telegram_enabled'] else 'OFF'}")
    logger.info(f"  Platform: {platform.system()}")
    logger.info("=" * 60)

    send_telegram(
        f"*AUD/USD Bot Started*\n"
        f"Mode: {'MT5' if CONFIG['mode'] == 'mt5' else 'Signals'}\n"
        f"RSI({CONFIG['rsi_oversold']}/{CONFIG['rsi_overbought']}) | "
        f"Risk: {CONFIG['risk_percent']}%"
    )

    last_signal_date = None
    check_count = 0

    while True:
        try:
            # MT5 connection check
            if CONFIG["mode"] == "mt5" and MT5_AVAILABLE:
                if not mt5.terminal_info():
                    logger.warning("MT5 disconnected, reconnecting...")
                    if not connect_mt5():
                        time.sleep(30)
                        continue

            # Fetch data
            df = get_candles(200)
            if df is None:
                logger.warning("No data received, retrying...")
                time.sleep(CONFIG["check_interval_seconds"])
                continue

            data = analyze(df)
            if data is None:
                time.sleep(CONFIG["check_interval_seconds"])
                continue

            # Status update (every check)
            check_count += 1
            print_status(data)

            # Check if open trade hit SL/TP (signal mode)
            if CONFIG["mode"] == "signal":
                check_open_trade(data)

            # Only check for new signals on new candles
            candle_date = str(data["time"].date()) if hasattr(data["time"], "date") else str(data["time"])[:10]
            if candle_date == last_signal_date:
                time.sleep(CONFIG["check_interval_seconds"])
                continue

            # Check if we already have an open trade (signal mode)
            if CONFIG["mode"] == "signal" and load_open_trade() is not None:
                logger.debug("Open trade exists, skipping new signals")
                time.sleep(CONFIG["check_interval_seconds"])
                continue

            # Check for signal
            signal = check_signal(data)

            if signal:
                if CONFIG["mode"] == "mt5" and MT5_AVAILABLE:
                    allowed, reason = check_risk_limits_mt5(data)
                    if not allowed:
                        logger.info(f"Trade skipped: {reason}")
                        send_telegram(f"*Signal skipped*: {signal} - {reason}")
                        last_signal_date = candle_date
                        time.sleep(CONFIG["check_interval_seconds"])
                        continue

                result = execute_signal(signal, data)
                if result:
                    last_signal_date = candle_date
            else:
                # Log RSI proximity for awareness
                rsi = data["rsi"]
                if rsi < CONFIG["rsi_oversold"] + 5:
                    logger.info(f"  --> RSI approaching oversold zone ({rsi:.1f})")
                elif rsi > CONFIG["rsi_overbought"] - 5:
                    logger.info(f"  --> RSI approaching overbought zone ({rsi:.1f})")

            time.sleep(CONFIG["check_interval_seconds"])

        except KeyboardInterrupt:
            logger.info("Bot stopped by user (Ctrl+C)")
            send_telegram("*AUD/USD Bot Stopped*")
            break
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            send_telegram(f"*Bot Error*: {e}")
            time.sleep(60)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="AUD/USD RSI(30/65) Trading Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 bot_audusd.py                        # Signal alert mode
  python3 bot_audusd.py --mt5                   # MT5 auto-execute
  python3 bot_audusd.py --risk 1.0              # 1% risk per trade
  python3 bot_audusd.py --balance 5000          # Set account balance
  python3 bot_audusd.py --telegram --token BOT_TOKEN --chat-id CHAT_ID
        """)

    parser.add_argument("--mt5", action="store_true",
                        help="Enable MT5 auto-execute mode (Windows only)")
    parser.add_argument("--risk", type=float, help="Risk %% per trade (default: 2.0)")
    parser.add_argument("--balance", type=float, help="Account balance for lot sizing")
    parser.add_argument("--symbol", type=str, help="MT5 symbol name (default: AUDUSD)")
    parser.add_argument("--telegram", action="store_true", help="Enable Telegram alerts")
    parser.add_argument("--token", type=str, help="Telegram bot token")
    parser.add_argument("--chat-id", type=str, help="Telegram chat ID")
    parser.add_argument("--login", type=int, help="MT5 account login")
    parser.add_argument("--password", type=str, help="MT5 password")
    parser.add_argument("--server", type=str, help="MT5 server")
    parser.add_argument("--interval", type=int, help="Check interval in seconds")
    parser.add_argument("--save-config", action="store_true", help="Save settings to file")
    parser.add_argument("--status", action="store_true", help="Show current status and exit")
    args = parser.parse_args()

    load_config()

    # Apply CLI args
    if args.mt5:
        CONFIG["mode"] = "mt5"
    if args.risk:
        CONFIG["risk_percent"] = args.risk
    if args.balance:
        CONFIG["account_balance"] = args.balance
    if args.symbol:
        CONFIG["symbol"] = args.symbol
    if args.telegram:
        CONFIG["telegram_enabled"] = True
    if args.token:
        CONFIG["telegram_token"] = args.token
    if args.chat_id:
        CONFIG["telegram_chat_id"] = args.chat_id
    if args.login:
        CONFIG["mt5_login"] = args.login
    if args.password:
        CONFIG["mt5_password"] = args.password
    if args.server:
        CONFIG["mt5_server"] = args.server
    if args.interval:
        CONFIG["check_interval_seconds"] = args.interval

    if args.save_config:
        save_config()
        print("Config saved.")
        return

    # --status: quick check and exit
    if args.status:
        df = get_candles(200)
        if df is not None:
            data = analyze(df)
            if data:
                print(f"\nAUD/USD Status ({data['time']})")
                print(f"  Price: {data['close']:.5f}")
                print(f"  RSI:   {data['rsi']:.1f}")
                print(f"  ATR:   {data['atr_pips']:.0f} pips")
                signal = check_signal(data)
                if signal:
                    print(f"\n  >>> ACTIVE SIGNAL: {signal} <<<")
                else:
                    print(f"\n  No signal (RSI between {CONFIG['rsi_oversold']}-{CONFIG['rsi_overbought']})")
                trade = load_open_trade()
                if trade:
                    if trade["signal"] == "BUY":
                        pips = (data["close"] - trade["entry"]) / PIP
                    else:
                        pips = (trade["entry"] - data["close"]) / PIP
                    print(f"\n  Open trade: {trade['signal']} @ {trade['entry']} ({pips:.0f} pips)")
        return

    # Connect MT5 if needed
    if CONFIG["mode"] == "mt5":
        if not MT5_AVAILABLE:
            logger.error("MetaTrader5 package not installed. Install on Windows: pip install MetaTrader5")
            logger.info("Falling back to signal mode...")
            CONFIG["mode"] = "signal"
        elif not connect_mt5():
            logger.error("MT5 connection failed. Falling back to signal mode...")
            CONFIG["mode"] = "signal"

    try:
        run_bot()
    finally:
        if CONFIG["mode"] == "mt5" and MT5_AVAILABLE:
            disconnect_mt5()


if __name__ == "__main__":
    main()
