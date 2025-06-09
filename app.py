import logging
import re
import requests
import pandas as pd
import ta
import matplotlib.pyplot as plt
import os
import asyncio
import pytz
from datetime import datetime, timedelta
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes, MessageHandler, filters
from binance import AsyncClient

# Logging setup
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
BOT_TOKEN = "7551132622:AAGteuMu94y1op7qnVz4hs6e0K3XQUCpTCI" # Replace with your actual bot token
BINANCE_API_KEY = "FvuS8Q9qbrWe1PF8gCOkAJEy8Eysd3MiDdggnrSxjWY9SoAToUAiWGghHU4cRwiP"
BINANCE_API_SECRET = "lVu5sY39nIN6QzAYTGy0IQNWlJKSSpbvUwagvkbP26jsIm3jOsa42okZR70Ixuke"
# Default Prediction Confidence Threshold (will be dynamically adjusted)
DEFAULT_CONFIDENCE_THRESHOLD = 0.6

# --- Helper Functions ---

def escape_markdown_v2(text: str) -> str:
    """
    Escape special MarkdownV2 characters.
    This function is crucial for ensuring messages are parsed correctly by Telegram.
    """
    if not isinstance(text, str):
        text = str(text)
    special_chars = r'([_*[\]()~`>#+-=<>{}|.!\\])'
    return re.sub(special_chars, r'\\\1', text)

def format_volume(volume: float) -> str:
    """Format volume in millions or thousands USDT."""
    if volume >= 1_000_000:
        return f"{volume / 1_000_000:.2f}M USDT"
    elif volume >= 1_000:
        return f"{volume / 1_000:.2f}K USDT"
    return f"{volume:.2f} USDT"

def detect_candlestick_pattern(df):
    """Detect simple candlestick pattern for essential output."""
    if len(df) < 2:
        return None
    last = df.iloc[-1]
    prev = df.iloc[-2]
    # Bullish Engulfing
    if (last['close'] > last['open'] and prev['close'] < prev['open'] and
            last['close'] > prev['open'] and last['open'] < prev['close']):
        return 'Bullish Engulfing'
    # Bearish Engulfing
    if (last['close'] < last['open'] and prev['close'] > prev['open'] and
            last['open'] > prev['close'] and last['close'] < prev['open']):
        return 'Bearish Engulfing'
    # Doji (open ≈ close)
    if abs(last['close'] - last['open']) / last['open'] < 0.001:
        return 'Doji' # Simplified for minimal output
    return None

def calculate_pivot_points(high: float, low: float, close: float) -> dict:
    """Calculate Classic Pivot Points, Supports, and Resistances."""
    pivot = (high + low + close) / 3
    r1 = (2 * pivot) - low
    s1 = (2 * pivot) - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = r1 + (high - low)
    s3 = s1 - (high - low)
    return {'P': pivot, 'R1': r1, 'R2': r2, 'R3': r3, 'S1': s1, 'S2': s2, 'S3': s3}

async def get_btc_price():
    """Fetch current BTC price from Binance."""
    try:
        url = "https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT"
        response = await asyncio.to_thread(requests.get, url, timeout=5)
        response.raise_for_status() # Raise an exception for bad status codes
        data = response.json()
        return {
            'price': float(data['lastPrice']),
            'change_24h': float(data['priceChangePercent']) / 100,
            'volume': float(data['quoteVolume'])
        }
    except requests.exceptions.RequestException as e:
        logger.error(f"Price fetch error: {e}")
        return {'price': 0.0, 'change_24h': 0.0, 'volume': 0.0} # Return default on error
    except (ValueError, KeyError) as e:
        logger.error(f"Price data parsing error: {e}")
        return {'price': 0.0, 'change_24h': 0.0, 'volume': 0.0}

async def get_historical_data(timeframe: str, limit: int = 200) -> pd.DataFrame:
    """Fetch historical BTC/USDT data from Binance."""
    try:
        url = f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval={timeframe}&limit={limit}"
        response = await asyncio.to_thread(requests.get, url, timeout=5)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except requests.exceptions.RequestException as e:
        logger.error(f"Historical data fetch error for {timeframe}: {e}")
        return pd.DataFrame() # Return empty DataFrame on error
    except (ValueError, KeyError) as e:
        logger.error(f"Historical data parsing error for {timeframe}: {e}")
        return pd.DataFrame()

async def get_order_book():
    """Fetch Binance order book for BTC/USDT."""
    try:
        client = AsyncClient(BINANCE_API_KEY, BINANCE_API_SECRET)
        order_book = await client.get_order_book(symbol='BTCUSDT', limit=20) # Increased limit for better depth
        await client.close_connection()
        
        bid_volume = sum(float(bid[1]) * float(bid[0]) for bid in order_book['bids']) # Volume weighted by price
        ask_volume = sum(float(ask[1]) * float(ask[0]) for ask in order_book['asks'])
        total_volume = bid_volume + ask_volume
        
        if total_volume > 0:
            bid_pct = (bid_volume / total_volume) * 100
            ask_pct = (ask_volume / total_volume) * 100
        else:
            bid_pct, ask_pct = 50, 50 # Default to neutral if no volume
        return {
            'bid_volume': bid_volume,
            'ask_volume': ask_volume,
            'bid_pct': bid_pct,
            'ask_pct': ask_pct
        }
    except Exception as e:
        logger.error(f"Order book error: {e}")
        return {'bid_volume': 0, 'ask_volume': 0, 'bid_pct': 50, 'ask_pct': 50}

async def get_fear_greed_index():
    """Fetch Crypto Fear & Greed Index from alternative.me."""
    try:
        url = "https://api.alternative.me/fng/?limit=1"
        response = await asyncio.to_thread(requests.get, url, timeout=5)
        response.raise_for_status()
        data = response.json()['data'][0]
        return {'value': int(data['value']), 'value_classification': data['value_classification']}
    except requests.exceptions.RequestException as e:
        logger.warning(f"Fear & Greed Index fetch error: {e}")
        return {'value': 50, 'value_classification': 'Neutral'} # Default to neutral on error
    except (ValueError, KeyError) as e:
        logger.warning(f"Fear & Greed Index parsing error: {e}")
        return {'value': 50, 'value_classification': 'Neutral'}

async def predict_price_movement(df_5m: pd.DataFrame, df_15m: pd.DataFrame, df_1h: pd.DataFrame, df_1d: pd.DataFrame, order_book, current_price, fear_greed_index):
    """
    Predicts price movement based on multiple timeframes, dynamic confidence,
    volume indicators, order book pressure, divergence, ADX, Stochastic,
    Pivot Points, and Fear & Greed Index.
    Returns: (prediction: str, target_price: float, confidence: float, reasons: list)
    """
    if df_5m.empty or len(df_5m) < 50: # Need sufficient data for 5m indicators
        return "SIDEWAYS", current_price, 0.0, ["Insufficient 5-minute historical data for prediction\\."]

    # --- Dynamic Confidence Threshold (based on 5m ATR) ---
    global CONFIDENCE_THRESHOLD 
    
    df_5m['atr'] = ta.volatility.AverageTrueRange(high=df_5m['high'], low=df_5m['low'], close=df_5m['close'], window=14).average_true_range()
    latest_atr_5m = df_5m['atr'].iloc[-1]
    avg_atr_5m = df_5m['atr'].mean()

    if latest_atr_5m > avg_atr_5m * 1.5:  # High volatility
        CONFIDENCE_THRESHOLD = 0.75 # Require stronger signals
        logger.info(f"High volatility detected (5m ATR). Adjusted CONFIDENCE_THRESHOLD to {CONFIDENCE_THRESHOLD}")
    elif latest_atr_5m < avg_atr_5m * 0.7: # Low volatility
        CONFIDENCE_THRESHOLD = 0.5 # Relax in stable markets
        logger.info(f"Low volatility detected (5m ATR). Adjusted CONFIDENCE_THRESHOLD to {CONFIDENCE_THRESHOLD}")
    else:
        CONFIDENCE_THRESHOLD = DEFAULT_CONFIDENCE_THRESHOLD # Revert to default
        logger.info(f"Normal volatility detected (5m ATR). CONFIDENCE_THRESHOLD is {CONFIDENCE_THRESHOLD}")

    # --- Technical Indicator Analysis for 5m ---
    df_5m['rsi'] = ta.momentum.RSIIndicator(close=df_5m['close'], window=14).rsi()
    df_5m['ema20'] = ta.trend.EMAIndicator(close=df_5m['close'], window=20).ema_indicator()
    df_5m['ema50'] = ta.trend.EMAIndicator(close=df_5m['close'], window=50).ema_indicator()
    macd_5m = ta.trend.MACD(close=df_5m['close'], window_slow=26, window_fast=12, window_sign=9)
    df_5m['macd'] = macd_5m.macd()
    df_5m['macd_signal'] = macd_5m.macd_signal()
    bollinger_5m = ta.volatility.BollingerBands(close=df_5m['close'], window=20, window_dev=2)
    df_5m['bb_high'] = bollinger_5m.bollinger_hband()
    df_5m['bb_low'] = bollinger_5m.bollinger_lband()
    # Stochastic Oscillator
    stoch_5m = ta.momentum.StochasticOscillator(high=df_5m['high'], low=df_5m['low'], close=df_5m['close'], window=14, smooth_window=3)
    df_5m['stoch_k'] = stoch_5m.stoch()
    df_5m['stoch_d'] = stoch_5m.stoch_signal()
    # ADX
    adx_5m = ta.trend.ADXIndicator(high=df_5m['high'], low=df_5m['low'], close=df_5m['close'], window=14)
    df_5m['adx'] = adx_5m.adx()
    df_5m['plus_di'] = adx_5m.adx_pos()
    df_5m['minus_di'] = adx_5m.adx_neg()
    
    # Get latest 5m indicator values
    latest_close_5m = df_5m['close'].iloc[-1]
    latest_rsi_5m = df_5m['rsi'].iloc[-1]
    latest_ema20_5m = df_5m['ema20'].iloc[-1]
    latest_ema50_5m = df_5m['ema50'].iloc[-1]
    latest_macd_5m = df_5m['macd'].iloc[-1]
    latest_macd_signal_5m = df_5m['macd_signal'].iloc[-1]
    latest_bb_high_5m = df_5m['bb_high'].iloc[-1]
    latest_bb_low_5m = df_5m['bb_low'].iloc[-1]
    latest_stoch_k_5m = df_5m['stoch_k'].iloc[-1]
    latest_stoch_d_5m = df_5m['stoch_d'].iloc[-1]
    latest_adx_5m = df_5m['adx'].iloc[-1]
    latest_plus_di_5m = df_5m['plus_di'].iloc[-1]
    latest_minus_di_5m = df_5m['minus_di'].iloc[-1]
    
    candle_pattern_5m = detect_candlestick_pattern(df_5m)

    bullish_score = 0
    bearish_score = 0
    reasons = []

    # --- RSI ---
    if latest_rsi_5m < 35: bullish_score += 1; reasons.append("🔻 5m RSI oversold\\.")
    elif latest_rsi_5m > 65: bearish_score += 1; reasons.append("🔺 5m RSI overbought\\.")
    
    # --- EMA Crossover ---
    if latest_ema20_5m > latest_ema50_5m and df_5m['ema20'].iloc[-2] <= df_5m['ema50'].iloc[-2]: bullish_score += 1.5; reasons.append("⬆️ 5m EMA Bullish Cross\\.")
    elif latest_ema20_5m < latest_ema50_5m and df_5m['ema20'].iloc[-2] >= df_5m['ema50'].iloc[-2]: bearish_score += 1.5; reasons.append("⬇️ 5m EMA Bearish Cross\\.")
    elif latest_ema20_5m > latest_ema50_5m: bullish_score += 0.5; reasons.append("🟢 5m EMAs bullish aligned\\.")
    elif latest_ema20_5m < latest_ema50_5m: bearish_score += 0.5; reasons.append("🔴 5m EMAs bearish aligned\\.")
        
    # --- MACD ---
    if latest_macd_5m > latest_macd_signal_5m and df_5m['macd'].iloc[-2] <= df_5m['macd_signal'].iloc[-2]: bullish_score += 1.5; reasons.append("⬆️ 5m MACD Bullish Cross\\.")
    elif latest_macd_5m < latest_macd_signal_5m and df_5m['macd'].iloc[-2] >= df_5m['macd_signal'].iloc[-2]: bearish_score += 1.5; reasons.append("⬇️ 5m MACD Bearish Cross\\.")
    elif latest_macd_5m > 0 and latest_macd_signal_5m > 0: bullish_score += 0.5; reasons.append("📈 5m MACD positive\\.")
    elif latest_macd_5m < 0 and latest_macd_signal_5m < 0: bearish_score += 0.5; reasons.append("📉 5m MACD negative\\.")

    # --- Bollinger Bands ---
    if current_price < latest_bb_low_5m * 1.002: bullish_score += 1; reasons.append("🔵 5m Price near lower Bollinger Band\\.")
    elif current_price > latest_bb_high_5m * 0.998: bearish_score += 1; reasons.append("🔴 5m Price near upper Bollinger Band\\.")
    
    # --- Candlestick Pattern ---
    if candle_pattern_5m:
        if 'Bullish' in candle_pattern_5m: bullish_score += 1; reasons.append(f"🕯️ 5m {escape_markdown_v2(candle_pattern_5m)}\\.")
        elif 'Bearish' in candle_pattern_5m: bearish_score += 1; reasons.append(f"🕯️ 5m {escape_markdown_v2(candle_pattern_5m)}\\.")

    # --- Stochastic Oscillator ---
    if latest_stoch_k_5m < 20 and latest_stoch_k_5m > latest_stoch_d_5m and df_5m['stoch_k'].iloc[-2] <= df_5m['stoch_d'].iloc[-2]:
        bullish_score += 1.0; reasons.append("🟢 5m Stochastics oversold, bullish cross\\.")
    elif latest_stoch_k_5m > 80 and latest_stoch_k_5m < latest_stoch_d_5m and df_5m['stoch_k'].iloc[-2] >= df_5m['stoch_d'].iloc[-2]:
        bearish_score += 1.0; reasons.append("🔴 5m Stochastics overbought, bearish cross\\.")

    # --- ADX (Trend Strength) ---
    if latest_adx_5m > 25: # Strong trend
        if latest_plus_di_5m > latest_minus_di_5m: bullish_score += 0.5; reasons.append("💪 5m Strong upward trend (ADX)\\.")
        else: bearish_score += 0.5; reasons.append("🔻 5m Strong downward trend (ADX)\\.")
    elif latest_adx_5m < 20: # Weak/Ranging trend
        reasons.append("↔️ 5m Weak/Ranging trend (low ADX)\\.")
        # Reduce confidence slightly if expecting strong movement
        if abs(bullish_score - bearish_score) > 1:
            if bullish_score > bearish_score: bullish_score -= 0.5
            else: bearish_score -= 0.5

    # --- Volume-Price Relationship Nuances ---
    # Check if last candle volume is significantly above average recent volume
    avg_vol_5m = df_5m['volume'].iloc[-10:].mean()
    last_candle_vol = df_5m['volume'].iloc[-1]
    
    if last_candle_vol > avg_vol_5m * 1.5: # 50% above average
        if df_5m['close'].iloc[-1] > df_5m['open'].iloc[-1]: # Bullish candle
            bullish_score += 0.7; reasons.append("⬆️ 5m Strong bullish candle on high volume\\.")
        elif df_5m['close'].iloc[-1] < df_5m['open'].iloc[-1]: # Bearish candle
            bearish_score += 0.7; reasons.append("⬇️ 5m Strong bearish candle on high volume\\.")
    elif last_candle_vol < avg_vol_5m * 0.5: # 50% below average (lack of conviction)
        if abs(df_5m['close'].iloc[-1] - df_5m['close'].iloc[-2]) / df_5m['close'].iloc[-2] > 0.001: # Check for price movement
             reasons.append("⚠️ 5m Price movement on low volume (lack of conviction)\\.")
             # Slightly penalize if movement happens without conviction
             if df_5m['close'].iloc[-1] > df_5m['close'].iloc[-2]: bullish_score -= 0.2
             else: bearish_score -= 0.2

    # --- Divergence Detection (RSI & MACD) ---
    # Need at least 5-10 candles to look for swings for divergence
    if len(df_5m) > 10:
        # Simple divergence check: compare last swing high/low to previous
        # This is a basic approach; real divergence detection is more complex.
        
        # Bearish Divergence: Price HH, Indicator LH
        if (df_5m['close'].iloc[-1] > df_5m['close'].iloc[-5:-1].max() and # Last close is new high
            latest_rsi_5m < df_5m['rsi'].iloc[-5:-1].max()): # But RSI is lower than its recent high
            bearish_score += 1.5; reasons.append("📉 5m Bearish RSI Divergence\\.")
        if (df_5m['close'].iloc[-1] > df_5m['close'].iloc[-5:-1].max() and
            latest_macd_5m < df_5m['macd'].iloc[-5:-1].max()):
            bearish_score += 1.5; reasons.append("📉 5m Bearish MACD Divergence\\.")

        # Bullish Divergence: Price LL, Indicator HL
        if (df_5m['close'].iloc[-1] < df_5m['close'].iloc[-5:-1].min() and # Last close is new low
            latest_rsi_5m > df_5m['rsi'].iloc[-5:-1].min()): # But RSI is higher than its recent low
            bullish_score += 1.5; reasons.append("📈 5m Bullish RSI Divergence\\.")
        if (df_5m['close'].iloc[-1] < df_5m['close'].iloc[-5:-1].min() and
            latest_macd_5m > df_5m['macd'].iloc[-5:-1].min()):
            bullish_score += 1.5; reasons.append("📈 5m Bullish MACD Divergence\\.")

    # --- Multi-Timeframe Confirmation (15m, 1h) & OBV (1d) ---
    has_bullish_higher_tf = False
    has_bearish_higher_tf = False

    # 15m Analysis (EMA crossover)
    if not df_15m.empty and len(df_15m) >= 50:
        df_15m['ema20'] = ta.trend.EMAIndicator(close=df_15m['close'], window=20).ema_indicator()
        df_15m['ema50'] = ta.trend.EMAIndicator(close=df_15m['close'], window=50).ema_indicator()
        
        latest_ema20_15m = df_15m['ema20'].iloc[-1]
        latest_ema50_15m = df_15m['ema50'].iloc[-1]

        if latest_ema20_15m > latest_ema50_15m:
            has_bullish_higher_tf = True
            reasons.append("↗️ 15m EMAs indicate upward trend\\.")
        elif latest_ema20_15m < latest_ema50_15m:
            has_bearish_higher_tf = True
            reasons.append("↘️ 15m EMAs indicate downward trend\\.")
    else:
        reasons.append("⚠️ Insufficient 15m data for trend confirmation\\.")

    # 1h Analysis (EMA crossover)
    if not df_1h.empty and len(df_1h) >= 50:
        df_1h['ema20'] = ta.trend.EMAIndicator(close=df_1h['close'], window=20).ema_indicator()
        df_1h['ema50'] = ta.trend.EMAIndicator(close=df_1h['close'], window=50).ema_indicator()
        
        latest_ema20_1h = df_1h['ema20'].iloc[-1]
        latest_ema50_1h = df_1h['ema50'].iloc[-1]

        if latest_ema20_1h > latest_ema50_1h:
            has_bullish_higher_tf = True
            reasons.append("⬆️ 1h EMAs indicate upward trend\\.")
        elif latest_ema20_1h < latest_ema50_1h:
            has_bearish_higher_tf = True
            reasons.append("⬇️ 1h EMAs indicate downward trend\\.")
    else:
        reasons.append("⚠️ Insufficient 1h data for trend confirmation\\.")

    # --- 1-Day OBV Analysis ---
    if not df_1d.empty and len(df_1d) >= 2: # Need at least 2 points to compare
        df_1d['obv'] = ta.volume.OnBalanceVolumeIndicator(close=df_1d['close'], volume=df_1d['volume']).on_balance_volume()
        
        latest_obv = df_1d['obv'].iloc[-1]
        previous_obv = df_1d['obv'].iloc[-2]

        if latest_obv > previous_obv:
            if has_bullish_higher_tf: # Boost if higher timeframes are also bullish
                bullish_score += 1.0; reasons.append("📊 1d OBV trending up \\(Bullish volume, multi-TF confirmed\\)\\.")
            else:
                bullish_score += 0.5; reasons.append("📊 1d OBV trending up \\(Bullish volume\\)\\.")
        elif latest_obv < previous_obv:
            if has_bearish_higher_tf: # Boost if higher timeframes are also bearish
                bearish_score += 1.0; reasons.append("📉 1d OBV trending down \\(Bearish volume, multi-TF confirmed\\)\\.")
            else:
                bearish_score += 0.5; reasons.append("📉 1d OBV trending down \\(Bearish volume\\)\\.")
        else:
            reasons.append("↔️ 1d OBV is neutral\\.")
    else:
        reasons.append("⚠️ Insufficient 1d data for OBV analysis\\.")

    # --- Refine Multi-Timeframe Agreement (Point 4) ---
    # Give a stronger boost if the 5m trend aligns with at least one higher TF
    if bullish_score > bearish_score and has_bullish_higher_tf:
        bullish_score += 0.7 # Stronger confirmation
        reasons.append("✅ 5m Bullish signal confirmed by higher timeframes\\.")
    elif bearish_score > bullish_score and has_bearish_higher_tf:
        bearish_score += 0.7 # Stronger confirmation
        reasons.append("❌ 5m Bearish signal confirmed by higher timeframes\\.")

    # --- Daily Pivot Points (Point 6) ---
    if not df_1d.empty and len(df_1d) >= 2:
        # Use previous day's HLC for today's pivots
        prev_day = df_1d.iloc[-2]
        pivots = calculate_pivot_points(prev_day['high'], prev_day['low'], prev_day['close'])
        
        # Check proximity to pivot points
        proximity_threshold = current_price * 0.002 # e.g., 0.2% of current price
        
        for level_name, level_price in pivots.items():
            if abs(current_price - level_price) < proximity_threshold:
                reasons.append(f"📍 Price near {escape_markdown_v2(level_name)} \\(${level_price:.2f}\\)\\.")
                if 'R' in level_name: # Resistance
                    bearish_score += 0.5 # Potential reversal down or resistance
                elif 'S' in level_name: # Support
                    bullish_score += 0.5 # Potential bounce up or support
                break # Only report the closest one

    # --- Order Book Pressure Analysis ---
    bid_pct = order_book['bid_pct']
    ask_pct = order_book['ask_pct']

    if bid_pct > ask_pct * 1.15: # Significant buying pressure (15% more bids)
        bullish_score += 1.0 # Add to 5m score for direct impact
        reasons.append(f"📚 Strong Buy Wall \\({bid_pct:.1f}\\% bids\\)\\.")
    elif ask_pct > bid_pct * 1.15: # Significant selling pressure
        bearish_score += 1.0 # Add to 5m score for direct impact
        reasons.append(f"📚 Strong Sell Wall \\({ask_pct:.1f}\\% asks\\)\\.")
    elif bid_pct > ask_pct:
        bullish_score += 0.2
        reasons.append(f"📚 Slight Buy Pressure \\({bid_pct:.1f}\\% bids\\)\\.")
    elif ask_pct > bid_pct:
        bearish_score += 0.2
        reasons.append(f"📚 Slight Sell Pressure \\({ask_pct:.1f}\\% asks\\)\\.")
    else:
        reasons.append("📚 Order book is balanced\\.")
        
    # --- Fear & Greed Index Sentiment (Point 7) ---
    fng_value = fear_greed_index['value']
    fng_classification = fear_greed_index['value_classification']
    reasons.append(f"👻 Fear & Greed Index: {escape_markdown_v2(fng_classification)} \\({fng_value}\\)\\.")

    # Adjust scores based on sentiment
    if fng_value <= 20: # Extreme Fear
        bullish_score += 0.5 # Potential for bounce
        reasons.append("🟢 Extreme Fear sentiment \\(potential bounce\\)\\.")
    elif fng_value >= 80: # Extreme Greed
        bearish_score += 0.5 # Potential for correction
        reasons.append("🔴 Extreme Greed sentiment \\(potential correction\\)\\.")

    # --- Final Prediction and Confidence Calculation ---
    total_score = bullish_score + bearish_score
    
    if total_score == 0:
        return "SIDEWAYS", current_price, 0.0, ["No strong signals detected\\. Price expected to \\*sustain\\* current levels\\."]
        
    confidence = abs(bullish_score - bearish_score) / total_score if total_score > 0 else 0
    
    prediction = "SIDEWAYS"
    target_price = current_price
    
    price_change_atr = latest_atr_5m * 0.5 # A fraction of 5m ATR for 5-min target

    if bullish_score > bearish_score and confidence >= CONFIDENCE_THRESHOLD:
        prediction = "UP"
        target_price = current_price + price_change_atr
        reasons.append(f"🎯 Target: ~\\${target_price:.2f} \\(based on current price \\+ 0\\.5 ATR\\)\\.")
    elif bearish_score > bullish_score and confidence >= CONFIDENCE_THRESHOLD:
        prediction = "DOWN"
        target_price = current_price - price_change_atr
        reasons.append(f"🎯 Target: ~\\${target_price:.2f} \\(based on current price \\- 0\\.5 ATR\\)\\.")
    else:
        prediction = "SIDEWAYS"
        target_price = current_price # No significant movement expected
        reasons.append("No clear strong trend identified based on confidence threshold\\. Price expected to \\*sustain\\* near current levels\\.")
        confidence = 0.0 # Reset confidence for sideways to indicate uncertainty

    return prediction, target_price, confidence, reasons

async def analyze_and_predict(timeframe='5m', price_input=None):
    """Perform technical analysis and prediction for the next 5 minutes."""
    
    # Fetch data for multiple timeframes, including 1-day for OBV and Pivot Points
    df_5m = await get_historical_data('5m', limit=200)
    df_15m = await get_historical_data('15m', limit=200)
    df_1h = await get_historical_data('1h', limit=200)
    df_1d = await get_historical_data('1d', limit=200) # Fetch 1-day data for OBV and Pivots

    if df_5m.empty:
        raise Exception("Not enough 5-minute historical data for analysis\\.")

    current_price = df_5m.iloc[-1]['close'] if price_input is None else price_input

    order_book = await get_order_book()
    fear_greed_index = await get_fear_greed_index() # Fetch Fear & Greed Index
    
    prediction, target_price, confidence, reasons = await predict_price_movement(df_5m, df_15m, df_1h, df_1d, order_book, current_price, fear_greed_index)
    
    ist_timezone = pytz.timezone('Asia/Kolkata')
    timestamp = datetime.now(ist_timezone).strftime('%Y-%m-%d %H:%M:%S IST')
    
    return {
        'current_price': current_price,
        'prediction': prediction,
        'target_price': target_price,
        'confidence': confidence,
        'reasons': reasons,
        'timestamp': timestamp,
        'order_book': order_book,
        'fear_greed_index': fear_greed_index,
        'data': df_5m
    }

# --- Telegram Handlers ---

refresh_message_id_map = {} # Maps chat_id to message_id for the last prediction message

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Welcome message with inline keyboard."""
    user_name = update.effective_user.first_name if update.effective_user else "there"
    keyboard = [
        [InlineKeyboardButton("📈 Current Price", callback_data="price")],
        [InlineKeyboardButton("🔮 Predict Next 5 Min", callback_data="predict_5min")],
        [InlineKeyboardButton("🔄 Refresh Analysis", callback_data="refresh_5min_prediction")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        f"🚀 \\*Hey {escape_markdown_v2(user_name)}\\! Welcome to your BTC Price Predictor Bot\\!\\* ✨\n\n"
        f"I can provide an opinion on BTC/USDT price movement for the next 5 minutes\\.\n\n"
        f"Choose an option below or type a price to get a prediction\\! 👇",
        reply_markup=reply_markup,
        parse_mode="MarkdownV2"
    )

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle inline keyboard buttons."""
    query = update.callback_query
    await query.answer() # Acknowledge the button press
    
    chat_id = query.message.chat_id
    message_id = query.message.message_id

    try:
        if query.data == "price":
            price_data = await get_btc_price()
            price_change_emoji = "🟢" if price_data['change_24h'] >= 0 else "🔴"
            
            message = (
                f"📈 \\*Current BTC/USDT Price\\*\n\n"
                f"Price: \\${escape_markdown_v2(f"{price_data['price']:.2f}")}\n"
                f"24h Change: {price_change_emoji} {escape_markdown_v2(f"{price_data['change_24h']:.2%}")}\n"
                f"24h Volume: {escape_markdown_v2(format_volume(price_data['volume']))}"
            )
            await query.message.reply_text(message, parse_mode="MarkdownV2")
            # No need to store for refresh if it's just a price lookup
            if chat_id in refresh_message_id_map:
                del refresh_message_id_map[chat_id]

        elif query.data == "predict_5min":
            # For new predictions, send "Analyzing..." message and then edit it.
            msg = await query.message.reply_text(
                f"🔮 \\*Analyzing BTC for 5\\-minute prediction\\! Please wait\\!\\* 📊",
                parse_mode="MarkdownV2"
            )
            # Store the message ID for future refresh
            refresh_message_id_map[chat_id] = msg.message_id
            await run_prediction_and_send_result(query.message, msg)

        elif query.data == "refresh_5min_prediction":
            # Check if there's a previous prediction message to refresh
            if chat_id in refresh_message_id_map:
                previous_message_id = refresh_message_id_map[chat_id]
                try:
                    # Attempt to edit the previous message for refresh animation
                    msg_to_edit = await context.bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=previous_message_id,
                        text="🔄 \\*Refreshing analysis\\! Please wait\\!\\* 🔄",
                        parse_mode="MarkdownV2"
                    )
                    await run_prediction_and_send_result(query.message, msg_to_edit, is_refresh=True)
                except Exception as e:
                    logger.warning(f"Failed to edit message {previous_message_id} for refresh: {e}. Sending new message.")
                    # If edit fails (e.g., message too old), send a new "Analyzing..." message
                    msg = await query.message.reply_text(
                        f"🔮 \\*Analyzing BTC for 5\\-minute prediction\\! Please wait\\!\\* 📊",
                        parse_mode="MarkdownV2"
                    )
                    refresh_message_id_map[chat_id] = msg.message_id # Update stored message ID
                    await run_prediction_and_send_result(query.message, msg)
            else:
                # No previous prediction to refresh, trigger a new one
                await query.message.reply_text(
                    f"🤔 No previous 5\\-minute prediction to refresh\\. Starting a new one\\.\n"
                    f"🔮 \\*Analyzing BTC for 5\\-minute prediction\\! Please wait\\!\\* 📊",
                    parse_mode="MarkdownV2"
                )
                msg = await query.message.reply_text(
                    f"🔮 \\*Analyzing BTC for 5\\-minute prediction\\! Please wait\\!\\* 📊",
                    parse_mode="MarkdownV2"
                )
                refresh_message_id_map[chat_id] = msg.message_id
                await run_prediction_and_send_result(query.message, msg)

    except Exception as e:
        logger.error(f"Button callback error: {e}")
        await query.message.reply_text(
            "❌ An error occurred\\. Please try again\\! 😔",
            parse_mode="MarkdownV2"
        )


async def run_prediction_and_send_result(telegram_message_obj, msg_to_edit, price_input=None, is_refresh=False):
    """Run prediction and send results with refresh animation."""
    chat_id = telegram_message_obj.chat_id # Get chat_id from the original message object

    # Animation frames for refreshing
    animation_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    frame_index = 0
    
    # Task to update "Analyzing..." message
    async def update_animation():
        nonlocal frame_index
        try:
            while True:
                animation_text = f"{animation_frames[frame_index]} \\*Analyzing BTC for 5\\-minute prediction\\! Please wait\\!\\* {animation_frames[frame_index]}"
                if is_refresh:
                    animation_text = f"{animation_frames[frame_index]} \\*Refreshing analysis\\! Please wait\\!\\* {animation_frames[frame_index]}"

                await msg_to_edit.edit_text(
                    text=animation_text,
                    parse_mode="MarkdownV2"
                )
                frame_index = (frame_index + 1) % len(animation_frames)
                await asyncio.sleep(0.3) # Adjust speed of animation
        except Exception as e:
            # This exception is expected if the message is deleted/edited by main logic
            logger.debug(f"Animation update stopped: {e}")

    animation_task = asyncio.create_task(update_animation())

    try:
        prediction_result = await analyze_and_predict('5m', price_input)
        
        # Stop animation task
        animation_task.cancel()
        
        current_price_escaped = escape_markdown_v2(f"{prediction_result['current_price']:.2f}")
        target_price_escaped = escape_markdown_v2(f"{prediction_result['target_price']:.2f}")
        confidence_escaped = escape_markdown_v2(f"{prediction_result['confidence']:.2%}")
        timestamp_escaped = escape_markdown_v2(prediction_result['timestamp'])
        
        reasons_formatted = "\n".join([f"• {escape_markdown_v2(reason)}" for reason in prediction_result['reasons']])
        
        order_book_info = (
            f"Buy Volume: {escape_markdown_v2(f"{prediction_result['order_book']['bid_pct']:.2f}")}\\% \n"
            f"Sell Volume: {escape_markdown_v2(f"{prediction_result['order_book']['ask_pct']:.2f}")}\\%"
        )
        
        prediction_text_for_message = ""
        recommendation_text_for_message = ""

        if prediction_result['prediction'] == "UP":
            emoji = "🟢"
            prediction_text_for_message = "UP"
            recommendation_text_for_message = "Price is likely to go \\*up\\*\\."
        elif prediction_result['prediction'] == "DOWN":
            emoji = "🔴"
            prediction_text_for_message = "DOWN"
            recommendation_text_for_message = "Price is likely to go \\*down\\*\\."
        else: # SIDEWAYS
            emoji = "⚪️"
            prediction_text_for_message = "SIDEWAYS"
            recommendation_text_for_message = "No clear strong trend identified\\. Price expected to \\*sustain\\* near current levels\\."

        # Construct the final message with strict MarkdownV2 escaping
        message = (
            f"*{emoji} 5\\-Minute Price Opinion: {escape_markdown_v2(prediction_text_for_message)} {emoji}*\n"
            f"Confidence: {confidence_escaped} \\(Threshold: {escape_markdown_v2(f'{CONFIDENCE_THRESHOLD:.2%}')}\\)\n"
            f"Recommendation: {recommendation_text_for_message}\n"
            f"Current Price: \\${current_price_escaped}\n"
            f"Expected Target \\(5min\\): \\${target_price_escaped}\n"
            f"\n\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\n"
            f"🔍 \\*Key Factors\\*:\n{reasons_formatted}\n"
            f"\n\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\n"
            f"📈 \\*Order Book Insights\\*:\n"
            f"{order_book_info}\n"
            f"\n\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\n"
            f"⏰ \\*Last Updated\\*:\\ {timestamp_escaped}"
        )
        
        # Try to edit the message, if it fails, send a new one
        try:
            await msg_to_edit.edit_text(
                text=message,
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔄 Refresh Analysis", callback_data="refresh_5min_prediction")]]),
                parse_mode="MarkdownV2"
            )
            # Update the stored message ID in case it was a new message after edit failure
            refresh_message_id_map[chat_id] = msg_to_edit.message_id

        except Exception as e:
            logger.error(f"Failed to edit message {msg_to_edit.message_id}: {e}. Sending new message.")
            # If editing fails, send a new message and update the stored message ID
            new_msg = await telegram_message_obj.reply_text(
                text=message,
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔄 Refresh Analysis", callback_data="refresh_5min_prediction")]]),
                parse_mode="MarkdownV2"
            )
            refresh_message_id_map[chat_id] = new_msg.message_id
            
    except Exception as e:
        # Ensure animation task is cancelled even on prediction failure
        animation_task.cancel()
        logger.error(f"Prediction failed: {e}")
        try:
            # Try to edit the analysis message to an error message
            await msg_to_edit.edit_text(
                f"❌ Prediction failed\\. Please try again\\! 📉\nError: {escape_markdown_v2(str(e))}",
                parse_mode="MarkdownV2"
            )
        except Exception as edit_e:
            logger.warning(f"Failed to edit error message: {edit_e}. Sending new error message.")
            await telegram_message_obj.reply_text(
                f"❌ Prediction failed\\. Please try again\\! 📉\nError: {escape_markdown_v2(str(e))}",
                parse_mode="MarkdownV2"
            )
        # Clear the stored message ID as the prediction failed
        if chat_id in refresh_message_id_map:
            del refresh_message_id_map[chat_id]


async def price_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Get current BTC price."""
    price_data = await get_btc_price()
    price_change_emoji = "🟢" if price_data['change_24h'] >= 0 else "🔴"
    
    message = (
        f"📈 \\*Current BTC/USDT Price\\*\n\n"
        f"Price: \\${escape_markdown_v2(f"{price_data['price']:.2f}")}\n"
        f"24h Change: {price_change_emoji} {escape_markdown_v2(f"{price_data['change_24h']:.2%}")}\n"
        f"24h Volume: {escape_markdown_v2(format_volume(price_data['volume']))}"
    )
    await update.message.reply_text(message, parse_mode="MarkdownV2")

async def predict_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Perform 5-minute prediction triggered by /predict command."""
    chat_id = update.message.chat_id
    msg = await update.message.reply_text(
        f"🔮 \\*Analyzing BTC for 5\\-minute prediction\\! Please wait\\!\\* 📊",
        parse_mode="MarkdownV2"
    )
    refresh_message_id_map[chat_id] = msg.message_id
    await run_prediction_and_send_result(update.message, msg)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle general messages, including numeric price input for prediction."""
    text = update.message.text
    user_name = update.effective_user.first_name if update.effective_user else "there"
    user_name_escaped = escape_markdown_v2(user_name)
    chat_id = update.message.chat_id

    try:
        price_input = float(text)
        if price_input > 0: # Basic validation for price
            msg = await update.message.reply_text(
                f"🔮 \\*Analyzing BTC trend based on your input price \\${escape_markdown_v2(f'{price_input:.2f}')} for 5\\-minute prediction\\! Please wait\\!\\* 📊",
                parse_mode="MarkdownV2"
            )
            refresh_message_id_map[chat_id] = msg.message_id
            await run_prediction_and_send_result(update.message, msg, price_input=price_input)
            return
    except ValueError:
        pass # Not a number, proceed to check other messages

    if "how are you" in text.lower():
        await update.message.reply_text("✨ I'm a bot, but I'm running smoothly and ready to assist you\\! 😎", parse_mode="MarkdownV2")
    elif "thank you" in text.lower() or "thanks" in text.lower():
        await update.message.reply_text("💖 You're most welcome\\! Happy to help\\! 🚀", parse_mode="MarkdownV2")
    elif "your name" in text.lower():
        await update.message.reply_text("🤖 Call me your BTC Price Predictor Bot\\! Ready to serve\\! 💡", parse_mode="MarkdownV2")
    elif "test" in text.lower():
        await update.message.reply_text("✅ Test passed\\! I'm here for you and working perfectly\\! 👍", parse_mode="MarkdownV2")
    else:
        await update.message.reply_text(
            f"🤔 Sorry, I didn't quite understand that\\. Try tapping a button from /start, use a command like /price or /predict, or type a BTC price for a custom prediction\\! 💬",
            parse_mode="MarkdownV2"
        )

def main():
    """Run the bot."""
    application = Application.builder().token(BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("price", price_command))
    application.add_handler(CommandHandler("predict", predict_command))

    application.add_handler(CallbackQueryHandler(button_callback))

    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Bot started polling...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    # Ensure charts directory exists (even if not sending, generate_chart writes to it)
    os.makedirs("charts", exist_ok=True)
    main()
