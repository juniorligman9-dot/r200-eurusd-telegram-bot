Absolutely. Below is the complete Python file in one piece.

This version is designed for EURUSD, not gold, and is deliberately conservative for a R200 account. It does not guarantee profit—with R200, the broker's minimum lot size and margin requirements can make some trades too risky, so the bot will skip trades when the calculated risk is too high.

Important: do not put your Telegram token directly in this file or post it in chat. We'll add it as an environment variable when deploying.

Save this as:

r200_eurusd_telegram_bot.py

"""
R200 EURUSD Telegram + MetaTrader 5 Trading Bot

Strategy:
- EURUSD
- M15 timeframe
- EMA 9 / EMA 21 trend crossover
- RSI confirmation
- ATR-based Stop Loss / Take Profit
- Maximum 1 open position
- Risk-based position sizing
- Daily loss protection
- Consecutive-loss protection
- Telegram commands and alerts

IMPORTANT:
This bot does NOT guarantee profits.
Test on an MT5 DEMO account before using real money.

Required environment variables:
    TELEGRAM_BOT_TOKEN
    MT5_LOGIN
    MT5_PASSWORD
    MT5_SERVER

Optional:
    MT5_PATH
    ALLOWED_CHAT_ID

Python package:
    MetaTrader5
"""

import os
import time
import math
import logging
import datetime as dt
from typing import Optional

import requests
import MetaTrader5 as mt5


# ============================================================
# CONFIGURATION
# ============================================================

SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_M15

# Account protection
RISK_PER_TRADE = 0.01          # 1% of account
MAX_DAILY_LOSS = 0.03          # 3% maximum daily loss
MAX_CONSECUTIVE_LOSSES = 3

# Strategy
FAST_EMA = 9
SLOW_EMA = 21
RSI_PERIOD = 14
ATR_PERIOD = 14

# ATR stop / target
SL_ATR_MULTIPLIER = 1.5
TP_ATR_MULTIPLIER = 2.0

# Trading limits
MAX_OPEN_POSITIONS = 1
MAGIC_NUMBER = 2002608

# Bot checks every 30 seconds
CHECK_INTERVAL = 30

# Don't trade immediately after a new candle appears until enough data exists
MIN_BARS = 100

# Spread protection
MAX_SPREAD_POINTS = 30

# Do not force the broker's minimum lot if it exceeds our calculated risk.
# This is especially important for very small accounts.
ALLOW_MINIMUM_LOT_IF_RISK_TOO_HIGH = False


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("R200_EURUSD_BOT")


# ============================================================
# ENVIRONMENT VARIABLES
# ============================================================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()

MT5_LOGIN_RAW = os.getenv("MT5_LOGIN", "").strip()
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "").strip()
MT5_SERVER = os.getenv("MT5_SERVER", "").strip()
MT5_PATH = os.getenv("MT5_PATH", "").strip()

ALLOWED_CHAT_ID_RAW = os.getenv("ALLOWED_CHAT_ID", "").strip()


def get_int(value: str) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


MT5_LOGIN = get_int(MT5_LOGIN_RAW)
ALLOWED_CHAT_ID = get_int(ALLOWED_CHAT_ID_RAW)


# ============================================================
# TELEGRAM
# ============================================================

last_update_id = 0


def telegram_request(method: str, params=None):
    if not TELEGRAM_BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN is missing.")
        return None

    url = (
        f"https://api.telegram.org/bot"
        f"{TELEGRAM_BOT_TOKEN}/{method}"
    )

    try:
        response = requests.get(
            url,
            params=params or {},
            timeout=20
        )

        if response.status_code != 200:
            logger.error(
                "Telegram HTTP error: %s %s",
                response.status_code,
                response.text[:500]
            )
            return None

        return response.json()

    except Exception as e:
        logger.error("Telegram error: %s", e)
        return None


def send_message(chat_id: int, message: str):
    telegram_request(
        "sendMessage",
        {
            "chat_id": chat_id,
            "text": message
        }
    )


def broadcast(message: str):
    """
    Send to the configured Telegram chat.
    """
    if ALLOWED_CHAT_ID is not None:
        send_message(ALLOWED_CHAT_ID, message)


# ============================================================
# MT5 CONNECTION
# ============================================================

def connect_mt5() -> bool:
    logger.info("Connecting to MetaTrader 5...")

    try:
        if MT5_PATH:
            initialized = mt5.initialize(path=MT5_PATH)
        else:
            initialized = mt5.initialize()

        if not initialized:
            logger.error(
                "MT5 initialize failed: %s",
                mt5.last_error()
            )
            return False

        if MT5_LOGIN and MT5_PASSWORD and MT5_SERVER:
            logged_in = mt5.login(
                MT5_LOGIN,
                password=MT5_PASSWORD,
                server=MT5_SERVER
            )

            if not logged_in:
                logger.error(
                    "MT5 login failed: %s",
                    mt5.last_error()
                )
                return False

        account = mt5.account_info()

        if account is None:
            logger.error("Could not read MT5 account information.")
            return False

        logger.info(
            "Connected | Login=%s | Balance=%.2f | Equity=%.2f",
            account.login,
            account.balance,
            account.equity
        )

        symbol_info = mt5.symbol_info(SYMBOL)

        if symbol_info is None:
            logger.error("%s is not available.", SYMBOL)
            return False

        if not symbol_info.visible:
            if not mt5.symbol_select(SYMBOL, True):
                logger.error("Could not select %s.", SYMBOL)
                return False

        return True

    except Exception as e:
        logger.error("MT5 connection exception: %s", e)
        return False


# ============================================================
# MARKET DATA
# ============================================================

def get_rates(count=MIN_BARS):
    rates = mt5.copy_rates_from_pos(
        SYMBOL,
        TIMEFRAME,
        0,
        count
    )

    if rates is None:
        logger.error(
            "Could not retrieve candles: %s",
            mt5.last_error()
        )
        return None

    if len(rates) < count:
        logger.warning(
            "Only %s candles available.",
            len(rates)
        )
        return None

    return rates


def ema(values, period):
    """
    Simple EMA implementation.
    """
    if len(values) < period:
        return []

    multiplier = 2 / (period + 1)

    result = [None] * len(values)

    initial = sum(values[:period]) / period
    result[period - 1] = initial

    previous = initial

    for i in range(period, len(values)):
        current = (
            (values[i] - previous) * multiplier
            + previous
        )

        result[i] = current
        previous = current

    return result


def rsi(values, period=14):
    if len(values) <= period:
        return []

    gains = []
    losses = []

    for i in range(1, len(values)):
        change = values[i] - values[i - 1]

        gains.append(max(change, 0))
        losses.append(max(-change, 0))

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    output = [None] * (period)

    if avg_loss == 0:
        output.append(100)
    else:
        rs = avg_gain / avg_loss
        output.append(100 - (100 / (1 + rs)))

    for i in range(period, len(gains)):
        avg_gain = (
            (avg_gain * (period - 1)) + gains[i]
        ) / period

        avg_loss = (
            (avg_loss * (period - 1)) + losses[i]
        ) / period

        if avg_loss == 0:
            current_rsi = 100
        else:
            rs = avg_gain / avg_loss
            current_rsi = 100 - (100 / (1 + rs))

        output.append(current_rsi)

    return output


def atr(highs, lows, closes, period=14):
    if len(closes) <= period:
        return []

    true_ranges = []

    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1])
        )

        true_ranges.append(tr)

    if len(true_ranges) < period:
        return []

    first_atr = sum(true_ranges[:period]) / period

    output = [None] * period
    output.append(first_atr)

    previous = first_atr

    for i in range(period, len(true_ranges)):
        previous = (
            (previous * (period - 1))
            + true_ranges[i]
        ) / period

        output.append(previous)

    return output


# ============================================================
# SIGNAL
# ============================================================

def get_signal():
    rates = get_rates()

    if rates is None:
        return None

    closes = [float(x["close"]) for x in rates]
    highs = [float(x["high"]) for x in rates]
    lows = [float(x["low"]) for x in rates]

    fast = ema(closes, FAST_EMA)
    slow = ema(closes, SLOW_EMA)
    rsi_values = rsi(closes, RSI_PERIOD)
    atr_values = atr(
        highs,
        lows,
        closes,
        ATR_PERIOD
    )

    # Use completed candles.
    i = len(closes) - 2
    previous = i - 1

    if previous < 0:
        return None

    if (
        fast[i] is None
        or slow[i] is None
        or fast[previous] is None
        or slow[previous] is None
        or rsi_values[i] is None
        or atr_values[i] is None
    ):
        return None

    fast_now = fast[i]
    slow_now = slow[i]

    fast_previous = fast[previous]
    slow_previous = slow[previous]

    current_rsi = rsi_values[i]
    current_atr = atr_values[i]

    signal = None

    # Bullish crossover + RSI confirmation
    if (
        fast_previous <= slow_previous
        and fast_now > slow_now
        and current_rsi >= 50
    ):
        signal = "BUY"

    # Bearish crossover + RSI confirmation
    elif (
        fast_previous >= slow_previous
        and fast_now < slow_now
        and current_rsi <= 50
    ):
        signal = "SELL"

    if signal is None:
        return None

    return {
        "signal": signal,
        "atr": float(current_atr),
        "rsi": float(current_rsi),
        "price": float(closes[i])
    }


# ============================================================
# ACCOUNT PROTECTION
# ============================================================

def get_account():
    account = mt5.account_info()

    if account is None:
        logger.error("Account info unavailable.")
        return None

    return account


def get_today_start():
    now = dt.datetime.now()
    return dt.datetime(
        now.year,
        now.month,
        now.day
    )


def get_today_profit():
    account = get_account()

    if account is None:
        return 0.0

    start = get_today_start()
    end = dt.datetime.now()

    deals = mt5.history_deals_get(
        start,
        end
    )

    if deals is None:
        return 0.0

    profit = 0.0

    for deal in deals:
        if deal.symbol != SYMBOL:
            continue

        # Entry OUT / OUT_BY represent closed positions.
        if deal.entry in (
            mt5.DEAL_ENTRY_OUT,
            mt5.DEAL_ENTRY_OUT_BY
        ):
            profit += float(deal.profit)
            profit += float(deal.swap)
            profit += float(deal.commission)

    return profit


def daily_loss_limit_hit():
    account = get_account()

    if account is None:
        return True

    today_profit = get_today_profit()

    max_loss_money = account.balance * MAX_DAILY_LOSS

    if today_profit <= -max_loss_money:
        logger.warning(
            "Daily loss limit reached: %.2f",
            today_profit
        )
        return True

    return False


def get_consecutive_losses():
    start = dt.datetime.now() - dt.timedelta(days=30)
    end = dt.datetime.now()

    deals = mt5.history_deals_get(
        start,
        end
    )

    if deals is None:
        return 0

    closed = []

    for deal in deals:
        if deal.symbol != SYMBOL:
            continue

        if deal.entry in (
            mt5.DEAL_ENTRY_OUT,
            mt5.DEAL_ENTRY_OUT_BY
        ):
            net = (
                float(deal.profit)
                + float(deal.swap)
                + float(deal.commission)
            )

            closed.append(
                (
                    deal.time,
                    net
                )
            )

    closed.sort(
        key=lambda x: x[0],
        reverse=True
    )

    losses = 0

    for _, profit in closed:
        if profit < 0:
            losses += 1

            if losses >= MAX_CONSECUTIVE_LOSSES:
                break
        else:
            break

    return losses


# ============================================================
# POSITION INFORMATION
# ============================================================

def get_open_positions():
    positions = mt5.positions_get(
        symbol=SYMBOL
    )

    if positions is None:
        return []

    return list(positions)


def has_open_position():
    return len(get_open_positions()) >= MAX_OPEN_POSITIONS


# ============================================================
# LOT SIZE CALCULATION
# ============================================================

def normalize_volume(volume, info):
    step = float(info.volume_step)
    minimum = float(info.volume_min)
    maximum = float(info.volume_max)

    if step <= 0:
        return minimum

    volume = math.floor(volume / step) * step

    volume = max(0.0, volume)
    volume = min(volume, maximum)

    # Round to avoid floating-point errors.
    decimals = 2

    if step < 0.01:
        decimals = 3

    volume = round(volume, decimals)

    if volume < minimum:
        return 0.0

    return volume


def calculate_lot_size(stop_distance):
    account = get_account()

    if account is None:
        return 0.0

    info = mt5.symbol_info(SYMBOL)

    if info is None:
        return 0.0

    if stop_distance <= 0:
        return 0.0

    risk_money = float(account.equity) * RISK_PER_TRADE

    tick_size = float(info.trade_tick_size)
    tick_value = float(info.trade_tick_value)

    if tick_size <= 0 or tick_value <= 0:
        logger.error("Invalid tick size/value.")
        return 0.0

    # Money risk for 1 lot at this stop distance.
    money_per_lot = (
        stop_distance / tick_size
    ) * tick_value

    if money_per_lot <= 0:
        return 0.0

    raw_volume = risk_money / money_per_lot

    volume = normalize_volume(
        raw_volume,
        info
    )

    minimum = float(info.volume_min)

    # For a tiny account, using the broker minimum lot
    # can exceed the intended risk. Skip instead.
    if volume <= 0:
        if not ALLOW_MINIMUM_LOT_IF_RISK_TOO_HIGH:
            logger.warning(
                "Minimum lot would exceed calculated risk. "
                "Trade skipped."
            )
            return 0.0

        volume = minimum

    return volume


# ============================================================
# SPREAD CHECK
# ============================================================

def spread_is_safe():
    info = mt5.symbol_info(SYMBOL)

    if info is None:
        return False

    tick = mt5.symbol_info_tick(SYMBOL)

    if tick is None:
        return False

    spread = (
        float(tick.ask) - float(tick.bid)
    )

    point = float(info.point)

    if point <= 0:
        return False

    spread_points = spread / point

    logger.info(
        "Spread: %.1f points",
        spread_points
    )

    return spread_points <= MAX_SPREAD_POINTS


# ============================================================
# ORDER EXECUTION
# ============================================================

def place_trade(signal_data):
    signal = signal_data["signal"]
    atr_value = signal_data["atr"]

    info = mt5.symbol_info(SYMBOL)

    if info is None:
        logger.error("Symbol info unavailable.")
        return False

    tick = mt5.symbol_info_tick(SYMBOL)

    if tick is None:
        logger.error("Market tick unavailable.")
        return False

    if signal == "BUY":
        entry = float(tick.ask)

        stop_distance = atr_value * SL_ATR_MULTIPLIER
        target_distance = atr_value * TP_ATR_MULTIPLIER

        sl = entry - stop_distance
        tp = entry + target_distance

        order_type = mt5.ORDER_TYPE_BUY

    elif signal == "SELL":
        entry = float(tick.bid)

        stop_distance = atr_value * SL_ATR_MULTIPLIER
        target_distance = atr_value * TP_ATR_MULTIPLIER

        sl = entry + stop_distance
        tp = entry - target_distance

        order_type = mt5.ORDER_TYPE_SELL

    else:
        return False

    # Respect broker's minimum stop distance.
    minimum_stop_points = float(
        info.trade_stops_level
    )

    minimum_stop_distance = (
        minimum_stop_points
        * float(info.point)
    )

    if stop_distance < minimum_stop_distance:
        stop_distance = minimum_stop_distance * 1.2

        if signal == "BUY":
            sl = entry - stop_distance
            tp = entry + (
                stop_distance
                * TP_ATR_MULTIPLIER
                / SL_ATR_MULTIPLIER
            )
        else:
            sl = entry + stop_distance
            tp = entry - (
                stop_distance
                * TP_ATR_MULTIPLIER
                / SL_ATR_MULTIPLIER
            )

    volume = calculate_lot_size(
        stop_distance
    )

    if volume <= 0:
        logger.warning(
            "No safe volume available. Trade skipped."
        )
        return False

    digits = int(info.digits)

    sl = round(sl, digits)
    tp = round(tp, digits)

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": volume,
        "type": order_type,
        "price": entry,
        "sl": sl,
        "tp": tp,
        "deviation": 20,
        "magic": MAGIC_NUMBER,
        "comment": "R200 EURUSD Bot",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    logger.info(
        "Sending %s | %.2f lots | Entry %.5f | SL %.5f | TP %.5f",
        signal,
        volume,
        entry,
        sl,
        tp
    )

    result = mt5.order_send(request)

    if result is None:
        logger.error(
            "order_send returned None: %s",
            mt5.last_error()
        )
        return False

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.error(
            "Trade rejected | retcode=%s | comment=%s",
            result.retcode,
            result.comment
        )

        broadcast(
            "❌ Trade rejected\n"
            f"Symbol: {SYMBOL}\n"
            f"Signal: {signal}\n"
            f"Reason: {result.comment}"
        )

        return False

    message = (
        f"✅ TRADE OPENED\n\n"
        f"Pair: {SYMBOL}\n"
        f"Direction: {signal}\n"
        f"Lot: {volume}\n"
        f"Entry: {entry:.5f}\n"
        f"SL: {sl:.5f}\n"
        f"TP: {tp:.5f}\n"
        f"RSI: {signal_data['rsi']:.1f}"
    )

    logger.info(message.replace("\n", " | "))

    broadcast(message)

    return True


# ============================================================
# BOT STATUS
# ============================================================

def status_message():
    account = get_account()

    if account is None:
        return "❌ MT5 account information unavailable."

    positions = get_open_positions()

    today_profit = get_today_profit()
    consecutive_losses = get_consecutive_losses()

    position_text = "None"

    if p
