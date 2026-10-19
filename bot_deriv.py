"""
Deriv Synthetic Indices Trading Bot
Uses optimized strategies from backtesting:
  R_75:    EMA(5/21) Crossover  | SL:3.0 ATR, TP:1.5 ATR | 73.4% WR, +89.4%
  R_100:   RSI(25/70)           | SL:3.0 ATR, TP:1.0 ATR | 84.2% WR, +60.6%
  1HZ100V: RSI(25/75)           | SL:3.0 ATR, TP:2.0 ATR | 67.7% WR, +44.4%
  1HZ75V:  RSI(25/70)           | SL:3.0 ATR, TP:2.5 ATR | 61.0% WR, +45.2%
  stpRNG:  RSI(30/75)           | SL:2.0 ATR, TP:0.75 ATR| 83.7% WR, +4.7%

Modes:
  --signal  : Alert-only mode (default) - prints signals + Telegram
  --auto    : Auto-execute via Deriv API (requires --deriv-token)

Usage:
  python3 bot_deriv.py                              # Signal mode, all symbols
  python3 bot_deriv.py --symbols R_75 R_100         # Specific symbols
  python3 bot_deriv.py --auto --deriv-token ABC123  # Auto-trade
  python3 bot_deriv.py --status                     # Show current state
"""

import asyncio
import json
import argparse
import os
import sys
import signal
import numpy as np
import pandas as pd
import websockets
from datetime import datetime, timedelta
from pathlib import Path


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DERIV_WS = "wss://ws.derivws.com/websockets/v3?app_id=1089"

# Optimized strategy parameters per symbol (from backtesting)
STRATEGIES = {
    "R_75": {
        "name": "Volatility 75 Index",
        "type": "EMA_CROSS",
        "fast_ema": 5,
        "slow_ema": 21,
        "sl_atr": 3.0,
        "tp_atr": 1.5,
        "backtest_wr": 73.4,
        "backtest_return": 89.4,
    },
    "R_100": {
        "name": "Volatility 100 Index",
        "type": "RSI",
        "rsi_oversold": 25,
        "rsi_overbought": 70,
        "sl_atr": 3.0,
        "tp_atr": 1.0,
        "backtest_wr": 84.2,
        "backtest_return": 60.6,
    },
    "1HZ100V": {
        "name": "Volatility 100 (1s)",
        "type": "RSI",
        "rsi_oversold": 25,
        "rsi_overbought": 75,
        "sl_atr": 3.0,
        "tp_atr": 2.0,
        "backtest_wr": 67.7,
        "backtest_return": 44.4,
    },
    "1HZ75V": {
        "name": "Volatility 75 (1s)",
        "type": "RSI",
        "rsi_oversold": 25,
        "rsi_overbought": 70,
        "sl_atr": 3.0,
        "tp_atr": 2.5,
        "backtest_wr": 61.0,
        "backtest_return": 45.2,
    },
    "stpRNG": {
        "name": "Step Index",
        "type": "RSI",
        "rsi_oversold": 30,
        "rsi_overbought": 75,
        "sl_atr": 2.0,
        "tp_atr": 0.75,
        "backtest_wr": 83.7,
        "backtest_return": 4.7,
    },
}

BOT_DIR = Path(__file__).parent
TRADE_LOG = BOT_DIR / "deriv_trades.json"
STATE_FILE = BOT_DIR / "deriv_bot_state.json"


# ---------------------------------------------------------------------------
# Indicators (same as backtest_deriv.py)
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


def add_indicators(df, symbol):
    """Add indicators relevant to the symbol's strategy."""
    cfg = STRATEGIES[symbol]
    close = df["Close"]

    df["ATR"] = compute_atr(df, 14)
    df["RSI"] = compute_rsi(close, 14)

    if cfg["type"] == "EMA_CROSS":
        df["EMA_Fast"] = compute_ema(close, cfg["fast_ema"])
        df["EMA_Slow"] = compute_ema(close, cfg["slow_ema"])

    df.dropna(inplace=True)
    return df


# ---------------------------------------------------------------------------
# Deriv WebSocket Client
# ---------------------------------------------------------------------------

class DerivClient:
    """Handles WebSocket connection to Deriv API."""

    def __init__(self, token=None):
        self.token = token
        self.ws = None
        self.authorized = False

    async def connect(self):
        self.ws = await websockets.connect(DERIV_WS)
        if self.token:
            await self._authorize()

    async def _authorize(self):
        await self.ws.send(json.dumps({"authorize": self.token}))
        resp = json.loads(await self.ws.recv())
        if "error" in resp:
            print(f"[AUTH ERROR] {resp['error']['message']}")
            self.authorized = False
        else:
            self.authorized = True
            bal = resp["authorize"]["balance"]
            currency = resp["authorize"]["currency"]
            print(f"[AUTH] Authorized. Balance: {bal} {currency}")

    async def disconnect(self):
        if self.ws:
            await self.ws.close()

    async def fetch_candles(self, symbol, granularity=3600, count=500):
        """Fetch historical candles."""
        try:
            async with websockets.connect(DERIV_WS) as ws:
                req = {
                    "ticks_history": symbol,
                    "adjust_start_time": 1,
                    "count": count,
                    "end": "latest",
                    "granularity": granularity,
                    "style": "candles",
                }
                await ws.send(json.dumps(req))
                resp = json.loads(await ws.recv())

                if "error" in resp:
                    print(f"[DATA ERROR] {symbol}: {resp['error']['message']}")
                    return None

                candles = resp.get("candles", [])
                if not candles:
                    return None

                df = pd.DataFrame(candles)
                df["time"] = pd.to_datetime(df["epoch"], unit="s")
                df.set_index("time", inplace=True)
                df = df.rename(columns={
                    "open": "Open", "high": "High",
                    "low": "Low", "close": "Close"
                })
                for col in ["Open", "High", "Low", "Close"]:
                    df[col] = df[col].astype(float)
                df = df[~df.index.duplicated(keep="last")].sort_index()
                return df
        except Exception as e:
            print(f"[FETCH ERROR] {symbol}: {e}")
            return None

    async def buy_contract(self, symbol, direction, amount, duration=5, duration_unit="m"):
        """Buy a contract on Deriv (for auto mode)."""
        if not self.ws or not self.authorized:
            print("[TRADE ERROR] Not authorized")
            return None

        contract_type = "CALL" if direction == "BUY" else "PUT"
        req = {
            "buy": 1,
            "price": amount,
            "parameters": {
                "contract_type": contract_type,
                "symbol": symbol,
                "duration": duration,
                "duration_unit": duration_unit,
                "currency": "USD",
                "amount": amount,
                "basis": "stake",
            }
        }
        await self.ws.send(json.dumps(req))
        resp = json.loads(await self.ws.recv())

        if "error" in resp:
            print(f"[BUY ERROR] {resp['error']['message']}")
            return None

        contract_id = resp["buy"]["contract_id"]
        buy_price = resp["buy"]["buy_price"]
        print(f"[TRADE] {contract_type} {symbol} | Contract #{contract_id} | Cost: ${buy_price}")
        return {
            "contract_id": contract_id,
            "buy_price": buy_price,
            "contract_type": contract_type,
            "symbol": symbol,
        }


# ---------------------------------------------------------------------------
# Signal Engine
# ---------------------------------------------------------------------------

class SignalEngine:
    """Generates signals based on optimized strategies."""

    @staticmethod
    def check_signal(df, symbol):
        """
        Returns signal dict or None.
        Signal: {"direction": "BUY"/"SELL", "entry", "sl", "tp", "reason"}
        """
        cfg = STRATEGIES[symbol]
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        price = curr["Close"]
        atr = curr["ATR"]

        if cfg["type"] == "RSI":
            return SignalEngine._check_rsi(curr, prev, price, atr, cfg)
        elif cfg["type"] == "EMA_CROSS":
            return SignalEngine._check_ema(curr, prev, price, atr, cfg)
        return None

    @staticmethod
    def _check_rsi(curr, prev, price, atr, cfg):
        oversold = cfg["rsi_oversold"]
        overbought = cfg["rsi_overbought"]
        sl_mult = cfg["sl_atr"]
        tp_mult = cfg["tp_atr"]

        # RSI crosses UP through oversold -> BUY
        if prev["RSI"] < oversold and curr["RSI"] >= oversold:
            return {
                "direction": "BUY",
                "entry": price,
                "sl": round(price - atr * sl_mult, 2),
                "tp": round(price + atr * tp_mult, 2),
                "reason": f"RSI crossed above {oversold} (prev={prev['RSI']:.1f}, now={curr['RSI']:.1f})",
            }
        # RSI crosses DOWN through overbought -> SELL
        if prev["RSI"] > overbought and curr["RSI"] <= overbought:
            return {
                "direction": "SELL",
                "entry": price,
                "sl": round(price + atr * sl_mult, 2),
                "tp": round(price - atr * tp_mult, 2),
                "reason": f"RSI crossed below {overbought} (prev={prev['RSI']:.1f}, now={curr['RSI']:.1f})",
            }
        return None

    @staticmethod
    def _check_ema(curr, prev, price, atr, cfg):
        sl_mult = cfg["sl_atr"]
        tp_mult = cfg["tp_atr"]

        # EMA fast crosses above slow -> BUY
        if prev["EMA_Fast"] <= prev["EMA_Slow"] and curr["EMA_Fast"] > curr["EMA_Slow"]:
            return {
                "direction": "BUY",
                "entry": price,
                "sl": round(price - atr * sl_mult, 2),
                "tp": round(price + atr * tp_mult, 2),
                "reason": f"EMA({cfg['fast_ema']}) crossed above EMA({cfg['slow_ema']})",
            }
        # EMA fast crosses below slow -> SELL
        if prev["EMA_Fast"] >= prev["EMA_Slow"] and curr["EMA_Fast"] < curr["EMA_Slow"]:
            return {
                "direction": "SELL",
                "entry": price,
                "sl": round(price + atr * sl_mult, 2),
                "tp": round(price - atr * tp_mult, 2),
                "reason": f"EMA({cfg['fast_ema']}) crossed below EMA({cfg['slow_ema']})",
            }
        return None


# ---------------------------------------------------------------------------
# Trade Manager
# ---------------------------------------------------------------------------

class TradeManager:
    """Tracks open positions, logs trades."""

    def __init__(self):
        self.open_trades = {}  # symbol -> trade dict
        self.history = []
        self._load_state()

    def _load_state(self):
        if STATE_FILE.exists():
            try:
                data = json.loads(STATE_FILE.read_text())
                self.open_trades = data.get("open_trades", {})
                self.history = data.get("history", [])
            except Exception:
                pass

    def _save_state(self):
        data = {
            "open_trades": self.open_trades,
            "history": self.history,
            "last_update": datetime.now().isoformat(),
        }
        STATE_FILE.write_text(json.dumps(data, indent=2, default=str))

    def has_open(self, symbol):
        return symbol in self.open_trades

    def open_trade(self, symbol, signal):
        trade = {
            "symbol": symbol,
            "direction": signal["direction"],
            "entry": signal["entry"],
            "sl": signal["sl"],
            "tp": signal["tp"],
            "reason": signal["reason"],
            "opened_at": datetime.now().isoformat(),
        }
        self.open_trades[symbol] = trade
        self._save_state()
        self._log_trade("OPEN", trade)
        return trade

    def check_exits(self, symbol, current_price, high, low):
        """Check if open trade hit SL or TP."""
        if symbol not in self.open_trades:
            return None

        trade = self.open_trades[symbol]
        result = None

        if trade["direction"] == "BUY":
            if low <= trade["sl"]:
                pnl = (trade["sl"] - trade["entry"]) / trade["entry"] * 100
                result = {"result": "LOSS", "exit_price": trade["sl"], "pnl": round(pnl, 2)}
            elif high >= trade["tp"]:
                pnl = (trade["tp"] - trade["entry"]) / trade["entry"] * 100
                result = {"result": "WIN", "exit_price": trade["tp"], "pnl": round(pnl, 2)}
        else:  # SELL
            if high >= trade["sl"]:
                pnl = (trade["entry"] - trade["sl"]) / trade["entry"] * 100
                result = {"result": "LOSS", "exit_price": trade["sl"], "pnl": round(pnl, 2)}
            elif low <= trade["tp"]:
                pnl = (trade["entry"] - trade["tp"]) / trade["entry"] * 100
                result = {"result": "WIN", "exit_price": trade["tp"], "pnl": round(pnl, 2)}

        if result:
            trade.update(result)
            trade["closed_at"] = datetime.now().isoformat()
            self.history.append(trade)
            del self.open_trades[symbol]
            self._save_state()
            self._log_trade("CLOSE", trade)

        return result

    def _log_trade(self, action, trade):
        """Append to trade log file."""
        entry = {"action": action, "time": datetime.now().isoformat(), **trade}
        logs = []
        if TRADE_LOG.exists():
            try:
                logs = json.loads(TRADE_LOG.read_text())
            except Exception:
                logs = []
        logs.append(entry)
        TRADE_LOG.write_text(json.dumps(logs, indent=2, default=str))

    def get_stats(self):
        if not self.history:
            return {"total": 0, "wins": 0, "losses": 0, "win_rate": 0, "total_pnl": 0}
        wins = [t for t in self.history if t.get("result") == "WIN"]
        total_pnl = sum(t.get("pnl", 0) for t in self.history)
        return {
            "total": len(self.history),
            "wins": len(wins),
            "losses": len(self.history) - len(wins),
            "win_rate": round(len(wins) / len(self.history) * 100, 1),
            "total_pnl": round(total_pnl, 2),
        }


# ---------------------------------------------------------------------------
# Telegram Alerts
# ---------------------------------------------------------------------------

async def send_telegram(token, chat_id, message):
    """Send alert via Telegram bot."""
    if not token or not chat_id:
        return
    import urllib.request
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = json.dumps({"chat_id": chat_id, "text": message, "parse_mode": "HTML"}).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        print(f"[TELEGRAM ERROR] {e}")


# ---------------------------------------------------------------------------
# Main Bot
# ---------------------------------------------------------------------------

class DerivBot:
    """Main bot loop - scans symbols, generates signals, manages trades."""

    def __init__(self, symbols, auto_trade=False, deriv_token=None,
                 risk_pct=2.0, balance=1000, interval=300,
                 tg_token=None, tg_chat=None):
        self.symbols = symbols
        self.auto_trade = auto_trade
        self.deriv_token = deriv_token
        self.risk_pct = risk_pct
        self.balance = balance
        self.interval = interval  # seconds between scans
        self.tg_token = tg_token
        self.tg_chat = tg_chat
        self.client = DerivClient(token=deriv_token if auto_trade else None)
        self.trade_mgr = TradeManager()
        self.running = True
        self.scan_count = 0

    async def start(self):
        print("=" * 70)
        print("  DERIV SYNTHETIC INDICES TRADING BOT")
        print("=" * 70)
        print(f"  Mode:     {'AUTO-TRADE' if self.auto_trade else 'SIGNAL ONLY'}")
        print(f"  Symbols:  {', '.join(self.symbols)}")
        print(f"  Interval: {self.interval}s ({self.interval // 60}m)")
        print(f"  Risk:     {self.risk_pct}% per trade")
        print(f"  Telegram: {'Enabled' if self.tg_token else 'Disabled'}")
        print()

        for sym in self.symbols:
            cfg = STRATEGIES[sym]
            print(f"  {sym:<10} {cfg['name']:<25} Strategy: {cfg['type']}")
            if cfg["type"] == "RSI":
                print(f"             RSI({cfg['rsi_oversold']}/{cfg['rsi_overbought']}) "
                      f"| SL:{cfg['sl_atr']} ATR, TP:{cfg['tp_atr']} ATR "
                      f"| BT: {cfg['backtest_wr']}% WR, +{cfg['backtest_return']}%")
            else:
                print(f"             EMA({cfg['fast_ema']}/{cfg['slow_ema']}) "
                      f"| SL:{cfg['sl_atr']} ATR, TP:{cfg['tp_atr']} ATR "
                      f"| BT: {cfg['backtest_wr']}% WR, +{cfg['backtest_return']}%")

        print(f"\n{'='*70}")
        print(f"  Bot started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Press Ctrl+C to stop")
        print(f"{'='*70}\n")

        if self.auto_trade and self.deriv_token:
            await self.client.connect()

        # Main loop
        while self.running:
            try:
                await self._scan_cycle()
                self.scan_count += 1

                # Print periodic status
                if self.scan_count % 12 == 0:  # Every ~hour at 5min intervals
                    self._print_status()

                await asyncio.sleep(self.interval)

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"[ERROR] Scan cycle failed: {e}")
                await asyncio.sleep(30)

        print("\n[BOT] Shutting down...")
        if self.auto_trade:
            await self.client.disconnect()

    async def _scan_cycle(self):
        """One scan cycle: fetch data, check signals, manage trades."""
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"\n--- Scan #{self.scan_count + 1} at {ts} ---")

        for symbol in self.symbols:
            try:
                # Fetch recent candles (1H timeframe, 200 bars for indicators)
                df = await self.client.fetch_candles(symbol, granularity=3600, count=200)
                if df is None or len(df) < 50:
                    print(f"  {symbol}: Insufficient data")
                    continue

                df = add_indicators(df, symbol)
                curr = df.iloc[-1]
                price = curr["Close"]
                rsi = curr["RSI"]
                atr = curr["ATR"]

                # Check open trade exits
                exit_result = self.trade_mgr.check_exits(
                    symbol, price, curr["High"], curr["Low"]
                )
                if exit_result:
                    emoji = "WIN" if exit_result["result"] == "WIN" else "LOSS"
                    msg = (f"[{emoji}] {symbol} CLOSED | "
                           f"Exit: ${exit_result['exit_price']:.2f} | "
                           f"PnL: {exit_result['pnl']:+.2f}%")
                    print(f"  {msg}")
                    if self.tg_token:
                        await send_telegram(self.tg_token, self.tg_chat,
                                            f"<b>{emoji}</b> {symbol}\n"
                                            f"PnL: {exit_result['pnl']:+.2f}%")

                # Check for new signals (only if no open trade)
                if not self.trade_mgr.has_open(symbol):
                    sig = SignalEngine.check_signal(df, symbol)
                    if sig:
                        trade = self.trade_mgr.open_trade(symbol, sig)
                        cfg = STRATEGIES[symbol]
                        msg = (f"[SIGNAL] {sig['direction']} {symbol} @ ${sig['entry']:.2f}\n"
                               f"  SL: ${sig['sl']:.2f} | TP: ${sig['tp']:.2f}\n"
                               f"  Reason: {sig['reason']}")
                        print(f"  {msg}")

                        # Telegram alert
                        if self.tg_token:
                            tg_msg = (f"<b>SIGNAL: {sig['direction']} {symbol}</b>\n"
                                      f"{cfg['name']}\n"
                                      f"Entry: ${sig['entry']:.2f}\n"
                                      f"SL: ${sig['sl']:.2f}\n"
                                      f"TP: ${sig['tp']:.2f}\n"
                                      f"Reason: {sig['reason']}")
                            await send_telegram(self.tg_token, self.tg_chat, tg_msg)

                        # Auto-trade execution
                        if self.auto_trade and self.client.authorized:
                            stake = self.balance * self.risk_pct / 100
                            result = await self.client.buy_contract(
                                symbol, sig["direction"], stake
                            )
                            if result:
                                print(f"  [EXECUTED] Contract #{result['contract_id']}")
                    else:
                        # Status line
                        status_parts = [f"${price:.2f}", f"RSI:{rsi:.1f}", f"ATR:{atr:.2f}"]
                        cfg = STRATEGIES[symbol]
                        if cfg["type"] == "EMA_CROSS":
                            ema_f = curr["EMA_Fast"]
                            ema_s = curr["EMA_Slow"]
                            status_parts.append(f"EMA:{ema_f:.2f}/{ema_s:.2f}")
                        print(f"  {symbol}: {' | '.join(status_parts)} -- No signal")
                else:
                    ot = self.trade_mgr.open_trades[symbol]
                    dist_sl = abs(price - ot["sl"]) / price * 100
                    dist_tp = abs(price - ot["tp"]) / price * 100
                    print(f"  {symbol}: ${price:.2f} | OPEN {ot['direction']} "
                          f"@ ${ot['entry']:.2f} | SL:{dist_sl:.1f}% away, TP:{dist_tp:.1f}% away")

            except Exception as e:
                print(f"  {symbol}: Error - {e}")

    def _print_status(self):
        stats = self.trade_mgr.get_stats()
        print(f"\n{'='*50}")
        print(f"  BOT STATUS (Scan #{self.scan_count})")
        print(f"  Trades: {stats['total']} | Wins: {stats['wins']} | "
              f"Losses: {stats['losses']} | WR: {stats['win_rate']}%")
        print(f"  Total PnL: {stats['total_pnl']:+.2f}%")
        print(f"  Open: {list(self.trade_mgr.open_trades.keys())}")
        print(f"{'='*50}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def show_status():
    """Show current bot state from saved files."""
    print("=" * 60)
    print("  DERIV BOT STATUS")
    print("=" * 60)

    if STATE_FILE.exists():
        data = json.loads(STATE_FILE.read_text())
        print(f"\n  Last Update: {data.get('last_update', 'Unknown')}")

        open_trades = data.get("open_trades", {})
        if open_trades:
            print(f"\n  OPEN TRADES ({len(open_trades)}):")
            for sym, t in open_trades.items():
                print(f"    {sym}: {t['direction']} @ ${t['entry']:.2f} "
                      f"| SL: ${t['sl']:.2f} | TP: ${t['tp']:.2f}")
                print(f"      Opened: {t['opened_at']} | {t['reason']}")
        else:
            print("\n  No open trades")

        history = data.get("history", [])
        if history:
            wins = sum(1 for t in history if t.get("result") == "WIN")
            total_pnl = sum(t.get("pnl", 0) for t in history)
            print(f"\n  TRADE HISTORY ({len(history)} trades):")
            print(f"    Wins: {wins} | Losses: {len(history) - wins} | "
                  f"WR: {wins/len(history)*100:.1f}%")
            print(f"    Total PnL: {total_pnl:+.2f}%")
            print(f"\n  Last 5 trades:")
            for t in history[-5:]:
                tag = "WIN " if t.get("result") == "WIN" else "LOSS"
                print(f"    [{tag}] {t['symbol']} {t['direction']} "
                      f"| Entry: ${t['entry']:.2f} | PnL: {t.get('pnl', 0):+.2f}%")
        else:
            print("  No trade history")
    else:
        print("\n  No state file found. Bot hasn't run yet.")

    if TRADE_LOG.exists():
        logs = json.loads(TRADE_LOG.read_text())
        print(f"\n  Trade log: {len(logs)} entries ({TRADE_LOG})")

    print(f"\n{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Deriv Synthetic Indices Trading Bot")
    parser.add_argument("--symbols", nargs="+",
                        default=list(STRATEGIES.keys()),
                        help="Symbols to trade (default: all)")
    parser.add_argument("--auto", action="store_true",
                        help="Auto-execute trades via Deriv API")
    parser.add_argument("--deriv-token", type=str, default=None,
                        help="Deriv API token (required for --auto)")
    parser.add_argument("--risk", type=float, default=2.0,
                        help="Risk per trade in %% (default: 2.0)")
    parser.add_argument("--balance", type=float, default=1000,
                        help="Account balance for position sizing (default: 1000)")
    parser.add_argument("--interval", type=int, default=300,
                        help="Scan interval in seconds (default: 300 = 5min)")
    parser.add_argument("--telegram-token", type=str, default=None,
                        help="Telegram bot token for alerts")
    parser.add_argument("--telegram-chat", type=str, default=None,
                        help="Telegram chat ID for alerts")
    parser.add_argument("--status", action="store_true",
                        help="Show current bot state and exit")

    args = parser.parse_args()

    if args.status:
        show_status()
        return

    # Validate symbols
    valid = []
    for s in args.symbols:
        if s in STRATEGIES:
            valid.append(s)
        else:
            print(f"[WARN] Unknown symbol: {s} (available: {list(STRATEGIES.keys())})")
    if not valid:
        print("[ERROR] No valid symbols")
        sys.exit(1)

    if args.auto and not args.deriv_token:
        print("[ERROR] --deriv-token required for auto-trade mode")
        sys.exit(1)

    bot = DerivBot(
        symbols=valid,
        auto_trade=args.auto,
        deriv_token=args.deriv_token,
        risk_pct=args.risk,
        balance=args.balance,
        interval=args.interval,
        tg_token=args.telegram_token,
        tg_chat=args.telegram_chat,
    )

    # Handle Ctrl+C gracefully
    def handle_signal(sig, frame):
        bot.running = False
    signal.signal(signal.SIGINT, handle_signal)

    asyncio.run(bot.start())


if __name__ == "__main__":
    main()
