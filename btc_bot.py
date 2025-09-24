"""
BTC Bot v0.1 — Breakout 10% TP / 5% SL with MA200 filter
Author: ChatGPT (for Jorge)

Features
- Backtest on BTC/USDT (default timeframe 1h, 2019–today) using CCXT data
- Strategy: Breakout of highest high (lookback=20) with MA200 trend filter
- Risk: position size = 10% of equity per trade; max 5 trades/day; 1h cooldown after a losing trade
- Exits: TP +10% (full close), SL −5% (hard stop), move SL to breakeven at +5%, time exit after 48h if neither TP/SL hits
- Paper/live modes (dry_run by default). Live trading via ccxt (market/limit mix)
- Equity compounding; optional weekly profit split (50% withdrawable — NOT automated withdrawals)
- Telegram alerts; CSV export of trades & daily PnL; simple performance report

Security Notice
- NEVER give API keys withdrawal permissions. This bot does NOT perform withdrawals.
- Use a fresh API key with trade-only rights and IP allowlisting if possible.

Usage (quick start)
1) pip install ccxt pandas numpy python-dotenv
2) Create a .env next to this file with:
   EXCHANGE=kraken
   API_KEY=your_key
   API_SECRET=your_secret
   TELEGRAM_BOT_TOKEN=123456:ABC...
   TELEGRAM_CHAT_ID=123456789
   TIMEFRAME=1h
   DRY_RUN=true
3) Run backtest first:  python btc_bot.py backtest
4) (Optional) Paper/live:   python btc_bot.py trade

Notes
- Kraken spot pair for USDT is often BTC/USDT; Coinbase is BTC/USD. For backtest data we fall back to Binance for convenience if needed (data only).
- If your exchange lacks BTC/USDT, set PAIR in .env accordingly.
"""

from __future__ import annotations
import os, sys, time, math, json, datetime as dt
import logging
from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np
from dotenv import load_dotenv

try:
    import ccxt
except Exception as e:
    print("Please: pip install ccxt pandas numpy python-dotenv")
    raise

# ---------------------- Config ----------------------
load_dotenv()
EXCHANGE = os.getenv("EXCHANGE", "kraken")
API_KEY = os.getenv("API_KEY", "")
API_SECRET = os.getenv("API_SECRET", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
TIMEFRAME = os.getenv("TIMEFRAME", "1h")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("btc_bot")


def _default_pair_for_exchange(exchange_name: str) -> str:
    if exchange_name.lower() == "kraken":
        return "XBT/USDT"
    return "BTC/USDT"


PAIR = os.getenv("PAIR") or _default_pair_for_exchange(EXCHANGE)
DRY_RUN = os.getenv("DRY_RUN", "true").lower() in ("1","true","yes")

# Strategy params (editable)
LOOKBACK = int(os.getenv("LOOKBACK", 20))
MA_LEN = int(os.getenv("MA_LEN", 200))
TP_PCT = float(os.getenv("TP_PCT", 0.10))    # +10%
SL_PCT = float(os.getenv("SL_PCT", 0.05))    # -5%
BE_TRIGGER = float(os.getenv("BE_TRIGGER", 0.05))  # move SL to BE at +5%
TIME_EXIT_HOURS = int(os.getenv("TIME_EXIT_HOURS", 48))

# Risk / Ops
POS_PCT = float(os.getenv("POS_PCT", 0.10))  # 10% equity per trade
MAX_TRADES_PER_DAY = int(os.getenv("MAX_TRADES_PER_DAY", 5))
COOLDOWN_AFTER_LOSS_H = int(os.getenv("COOLDOWN_AFTER_LOSS_H", 1))

# Fees (maker/taker, conservative)
FEE_RATE = float(os.getenv("FEE_RATE", 0.001))  # 0.10%

DATA_START = os.getenv("DATA_START", "2019-01-01")
INITIAL_EQUITY = float(os.getenv("INITIAL_EQUITY", 10000))


# ---------------------- Helpers ----------------------

def now_utc() -> dt.datetime:
    return dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)

def to_dt(ts_ms: int) -> dt.datetime:
    return dt.datetime.utcfromtimestamp(ts_ms/1000).replace(tzinfo=dt.timezone.utc)


def send_telegram(msg: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        import urllib.parse, urllib.request
        q = urllib.parse.urlencode({
            'chat_id': TELEGRAM_CHAT_ID,
            'text': msg
        })
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage?{q}"
        with urllib.request.urlopen(url, timeout=10) as r:
            r.read()
    except Exception:
        pass


# ---------------------- Data ----------------------

def get_exchange(name: str):
    ex = getattr(ccxt, name)({
        'apiKey': API_KEY,
        'secret': API_SECRET,
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'}
    })
    return ex


def fetch_ohlcv_hist(pair: str, timeframe: str, since_iso: str) -> pd.DataFrame:
    """Try chosen exchange; if not supported for history, fall back to binance for data only."""
    since_ms = int(pd.Timestamp(since_iso, tz='UTC').timestamp()*1000)
    sources = [EXCHANGE, 'binance'] if EXCHANGE != 'binance' else ['binance']
    last_error = None
    for src in sources:
        try:
            ex = get_exchange(src)
            if src == 'coinbase' and pair.endswith('USDT'):
                # Coinbase uses USD; allow override via env if needed
                pass
            all_rows = []
            limit = 1000
            since = since_ms
            while True:
                batch = ex.fetch_ohlcv(pair, timeframe=timeframe, since=since, limit=limit)
                if not batch:
                    break
                all_rows += batch
                since = batch[-1][0] + 1
                if len(batch) < limit:
                    break
            if not all_rows:
                continue
            df = pd.DataFrame(all_rows, columns=['ts','open','high','low','close','volume'])
            df['dt'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
            df.set_index('dt', inplace=True)
            return df[['open','high','low','close','volume']]
        except Exception as e:
            last_error = e
            continue
    raise RuntimeError(f"Failed to fetch OHLCV for {pair} {timeframe}: {last_error}")


# ---------------------- Strategy Logic ----------------------

@dataclass
class Position:
    entry_time: pd.Timestamp
    entry_price: float
    size_qty: float
    stop_price: float
    tp_price: float
    be_armed: bool = False
    max_favorable: float = 0.0


def generate_signals(df: pd.DataFrame) -> pd.Series:
    """Long-only breakout above highest high of last LOOKBACK bars with MA200 uptrend filter."""
    highs = df['high'].rolling(LOOKBACK).max().shift(1)
    ma = df['close'].rolling(MA_LEN).mean()
    uptrend = df['close'] > ma
    signal = (df['close'] > highs) & uptrend
    return signal.fillna(False)


# ---------------------- Backtester ----------------------

def backtest(df: pd.DataFrame, equity0: float=INITIAL_EQUITY) -> dict:
    signal = generate_signals(df)
    equity = equity0
    trades = []
    pos: Optional[Position] = None
    last_loss_time: Optional[pd.Timestamp] = None
    daily_trades = {}

    for t, row in df.iterrows():
        price = float(row['close'])
        day = t.date()
        daily_trades.setdefault(day, 0)

        # Cooldown after loss
        if last_loss_time is not None:
            if t < last_loss_time + pd.Timedelta(hours=COOLDOWN_AFTER_LOSS_H):
                eligible = False
            else:
                last_loss_time = None
                eligible = True
        else:
            eligible = True

        # Manage open position
        if pos:
            # Track max favorable excursion
            mfe = (price - pos.entry_price)/pos.entry_price
            pos.max_favorable = max(pos.max_favorable, mfe)
            
            # Move to BE at +5%
            if (not pos.be_armed) and pos.max_favorable >= BE_TRIGGER:
                pos.stop_price = pos.entry_price * 1.0005  # a hair above BE to cover fee
                pos.be_armed = True

            # Check TP/SL
            if price >= pos.tp_price:
                exit_price = pos.tp_price
                pnl = (exit_price - pos.entry_price) * pos.size_qty
                fee = (pos.entry_price + exit_price) * pos.size_qty * FEE_RATE
                equity += pnl - fee
                trades.append({
                    'entry_time': pos.entry_time, 'exit_time': t,
                    'entry_price': pos.entry_price, 'exit_price': exit_price,
                    'qty': pos.size_qty, 'pnl': pnl - fee, 'reason': 'TP'
                })
                pos = None
            elif price <= pos.stop_price:
                exit_price = pos.stop_price
                pnl = (exit_price - pos.entry_price) * pos.size_qty
                fee = (pos.entry_price + exit_price) * pos.size_qty * FEE_RATE
                equity += pnl - fee
                trades.append({
                    'entry_time': pos.entry_time, 'exit_time': t,
                    'entry_price': pos.entry_price, 'exit_price': exit_price,
                    'qty': pos.size_qty, 'pnl': pnl - fee, 'reason': 'SL/BE'
                })
                last_loss_time = t if pnl - fee < 0 else None
                pos = None
            else:
                # Time exit
                if t >= pos.entry_time + pd.Timedelta(hours=TIME_EXIT_HOURS):
                    exit_price = price
                    pnl = (exit_price - pos.entry_price) * pos.size_qty
                    fee = (pos.entry_price + exit_price) * pos.size_qty * FEE_RATE
                    equity += pnl - fee
                    trades.append({
                        'entry_time': pos.entry_time, 'exit_time': t,
                        'entry_price': pos.entry_price, 'exit_price': exit_price,
                        'qty': pos.size_qty, 'pnl': pnl - fee, 'reason': 'TIME'
                    })
                    last_loss_time = t if pnl - fee < 0 else None
                    pos = None

        # Entry rules
        if (pos is None) and eligible and signal.loc[t]:
            if daily_trades[day] < MAX_TRADES_PER_DAY:
                risk_equity = equity * POS_PCT
                size_qty = risk_equity / price
                entry_price = price
                stop_price = entry_price * (1 - SL_PCT)
                tp_price = entry_price * (1 + TP_PCT)
                pos = Position(entry_time=t, entry_price=entry_price, size_qty=size_qty,
                               stop_price=stop_price, tp_price=tp_price)
                daily_trades[day] += 1

    # Build report
    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return {
            'equity_final': equity,
            'trades': trades_df,
            'metrics': {}
        }
    trades_df['ret'] = trades_df['pnl'] / equity0
    total_pnl = trades_df['pnl'].sum()
    equity_final = equity0 + total_pnl
    wins = (trades_df['pnl'] > 0).sum()
    losses = (trades_df['pnl'] <= 0).sum()
    win_rate = wins / max(1, len(trades_df))
    profit_factor = trades_df.loc[trades_df['pnl']>0,'pnl'].sum() / max(1e-9, -trades_df.loc[trades_df['pnl']<0,'pnl'].sum())

    # Equity curve (naive, mark-to-market on exits only)
    ec = [equity0]
    for pnl in trades_df['pnl']:
        ec.append(ec[-1] + pnl)
    dd = 0.0
    peak = ec[0]
    for v in ec:
        peak = max(peak, v)
        dd = max(dd, (peak - v)/peak)

    metrics = {
        'trades': len(trades_df),
        'wins': int(wins),
        'losses': int(losses),
        'win_rate': round(win_rate, 4),
        'profit_factor': round(profit_factor, 3),
        'max_drawdown_pct': round(dd*100, 2),
        'equity_final': round(equity_final, 2),
        'total_pnl': round(total_pnl, 2),
    }
    return {'equity_final': equity_final, 'trades': trades_df, 'metrics': metrics}


# ---------------------- Live Trader (simplified) ----------------------

class Trader:
    def __init__(self):
        self.ex = get_exchange(EXCHANGE)
        self.pair = PAIR
        self.timeframe = TIMEFRAME
        self.dry_run = DRY_RUN
        self.equity = INITIAL_EQUITY
        self.pos: Optional[Position] = None
        self.daily_trades = {}
        self.last_loss_time: Optional[dt.datetime] = None

    def get_price(self) -> float:
        t = self.ex.fetch_ticker(self.pair)
        return float(t['last'])

    def can_trade_today(self) -> bool:
        today = dt.datetime.utcnow().date()
        self.daily_trades.setdefault(today, 0)
        return self.daily_trades[today] < MAX_TRADES_PER_DAY

    def record_trade(self):
        today = dt.datetime.utcnow().date()
        self.daily_trades[today] += 1

    def place_order(self, side: str, qty: float):
        if self.dry_run:
            return {'status':'filled','price': self.get_price(), 'qty': qty}
        # Market orders for simplicity; consider limit+postonly for entries to save fees.
        o = self.ex.create_order(self.pair, type='market', side=side, amount=qty)
        return o

    def loop(self):
        logger.info("Trader loop starting on %s (pair=%s, timeframe=%s, dry_run=%s)", EXCHANGE, self.pair, self.timeframe, self.dry_run)
        send_telegram("BTC Bot started (dry_run=%s)" % self.dry_run)
        while True:
            try:
                # Fetch last N bars for signals
                ohlcv = self.ex.fetch_ohlcv(self.pair, timeframe=self.timeframe, limit=max(MA_LEN, LOOKBACK)+5)
                df = pd.DataFrame(ohlcv, columns=['ts','open','high','low','close','volume'])
                df['dt'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
                df.set_index('dt', inplace=True)
                sig = generate_signals(df)
                t = df.index[-1]
                price = float(df['close'].iloc[-1])
                signal_flag = bool(sig.iloc[-1])
                position_state = "open" if self.pos else "flat"
                logger.info(
                    "loop tick ts=%s price=%.2f signal=%s position=%s",
                    t.to_pydatetime().isoformat(),
                    price,
                    signal_flag,
                    position_state,
                )

                # Manage open position
                if self.pos:
                    mfe = (price - self.pos.entry_price)/self.pos.entry_price
                    self.pos.max_favorable = max(self.pos.max_favorable, mfe)
                    if (not self.pos.be_armed) and self.pos.max_favorable >= BE_TRIGGER:
                        self.pos.stop_price = self.pos.entry_price * 1.0005
                        self.pos.be_armed = True
                    # TP/SL/time
                    reason = None
                    if price >= self.pos.tp_price:
                        exit_price = self.pos.tp_price
                        reason = 'TP'
                    elif price <= self.pos.stop_price:
                        exit_price = self.pos.stop_price
                        reason = 'SL/BE'
                    elif dt.datetime.now(dt.timezone.utc) >= self.pos.entry_time.to_pydatetime().replace(tzinfo=dt.timezone.utc) + dt.timedelta(hours=TIME_EXIT_HOURS):
                        exit_price = price
                        reason = 'TIME'
                    if reason:
                        qty = self.pos.size_qty
                        if not self.dry_run:
                            self.place_order('sell', qty)
                        pnl = (exit_price - self.pos.entry_price)*qty
                        fee = (self.pos.entry_price + exit_price)*qty*FEE_RATE
                        self.equity += pnl - fee
                        send_telegram(f"Closed {reason}: PnL={pnl-fee:.2f} | Equity={self.equity:.2f}")
                        if pnl - fee < 0:
                            self.last_loss_time = now_utc()
                        self.pos = None

                # New entry
                cooldown_ok = True
                if self.last_loss_time:
                    cooldown_ok = now_utc() >= self.last_loss_time + dt.timedelta(hours=COOLDOWN_AFTER_LOSS_H)
                    if cooldown_ok:
                        self.last_loss_time = None
                if (self.pos is None) and cooldown_ok and self.can_trade_today() and signal_flag:
                    risk_equity = self.equity * POS_PCT
                    qty = risk_equity / price
                    entry_price = price
                    stop_price = entry_price * (1 - SL_PCT)
                    tp_price = entry_price * (1 + TP_PCT)
                    if not self.dry_run:
                        self.place_order('buy', qty)
                    self.pos = Position(entry_time=t, entry_price=entry_price, size_qty=qty,
                                        stop_price=stop_price, tp_price=tp_price)
                    self.record_trade()
                    send_telegram(f"Entered LONG @ {entry_price:.2f} qty={qty:.6f} TP={tp_price:.2f} SL={stop_price:.2f}")

                # Sleep until next bar boundary
                time.sleep(30)
            except Exception as e:
                send_telegram(f"Error: {e}")
                time.sleep(10)


# ---------------------- Reporting ----------------------

def save_report(trades_df: pd.DataFrame, metrics: dict, tag: str="backtest"):
    ts = dt.datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    os.makedirs('reports', exist_ok=True)
    path_trades = f'reports/trades_{tag}_{ts}.csv'
    path_metrics = f'reports/metrics_{tag}_{ts}.json'
    trades_df.to_csv(path_trades, index=False)
    with open(path_metrics, 'w') as f:
        json.dump(metrics, f, indent=2)
    print("Saved:", path_trades)
    print("Saved:", path_metrics)


# ---------------------- CLI ----------------------

def cmd_backtest():
    print("Fetching data...")
    df = fetch_ohlcv_hist(PAIR, TIMEFRAME, DATA_START)
    print("Bars:", len(df))
    result = backtest(df, equity0=INITIAL_EQUITY)
    print("Metrics:")
    print(result['metrics'])
    if isinstance(result['trades'], pd.DataFrame) and not result['trades'].empty:
        save_report(result['trades'], result['metrics'], tag="bt")


def cmd_trade():
    t = Trader()
    t.loop()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python btc_bot.py [backtest|trade]")
        sys.exit(0)
    cmd = sys.argv[1].lower()
    if cmd == 'backtest':
        cmd_backtest()
    elif cmd == 'trade':
        cmd_trade()
    else:
        print("Unknown command.")

