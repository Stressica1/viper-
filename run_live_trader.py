#!/usr/bin/env python3
"""
🚀 VIPER MULTI-PAIR LIVE TRADER - FIXED FOR BITGET API
Live trading system that scans and trades ALL available pairs on Bitget with proper API configuration
"""

import os
import sys
import time
import logging
import ccxt
import random

from pathlib import Path
from dotenv import load_dotenv
from position_adoption_system import PositionAdoptionSystem

# Load environment variables
load_dotenv()

# Add project paths
project_root = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))



# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - VIPER_FIXED_TRADER - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/viper_fixed_trader.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MultiPairVIPERTrader:
    """VIPER Trader that scans and trades ALL available pairs"""

    def __init__(self):
        # Load API credentials
        self.api_key = os.getenv('BITGET_API_KEY')
        self.api_secret = os.getenv('BITGET_API_SECRET')
        self.api_password = os.getenv('BITGET_API_PASSWORD')

        # Trading configuration - AS PER USER REQUIREMENTS
        self.position_size_percent = float(os.getenv('RISK_PER_TRADE', '0.001'))  # 0.1% per trade for 50x leverage
        self.max_leverage = int(os.getenv('MAX_LEVERAGE', '50'))  # 50x leverage as requested
        self.take_profit_pct = float(os.getenv('TAKE_PROFIT_PCT', '3.0'))
        self.stop_loss_pct = float(os.getenv('STOP_LOSS_PCT', '2.0'))
        self.max_positions = int(os.getenv('MAX_POSITIONS', '15'))  # 15 positions as per your rules
        self.min_margin_per_trade = float(os.getenv('MIN_MARGIN_PER_TRADE', '5.0'))  # $5.0 minimum margin for exchange requirements

        self.exchange = None
        self.all_pairs = []
        self.active_positions = {}  # Track positions by symbol
        self.is_running = False

        logger.info(f"🎯 POSITION LIMIT SET TO: {self.max_positions} positions maximum")
        logger.info(f"📊 Risk per trade: {self.position_size_percent*100}% of account balance")

        # Advanced position tracking and adoption
        self.position_adoption_system = PositionAdoptionSystem()

        # Set up position adoption callbacks
        self.position_adoption_system.on_position_adopted = self._on_position_adopted
        self.position_adoption_system.on_position_closed = self._on_position_closed
        self.position_adoption_system.on_position_updated = self._on_position_updated
        
    def connect(self):
        """Connect to Bitget and load ALL available pairs"""
        try:
            if not all([self.api_key, self.api_secret, self.api_password]):
                logger.error("❌ Missing API credentials")
                return False

            logger.info("🔌 Connecting to Bitget...")
            self.exchange = ccxt.bitget({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'password': self.api_password,
                'options': {
                'defaultType': 'swap',
                'adjustForTimeDifference': True,
                'hedgeMode': False,  # Use unilateral (one-way) position mode
                },
                'sandbox': False,
            })

            # Verify position mode configuration
            if not self.verify_position_mode():
                logger.warning("⚠️ Position mode verification failed, but continuing...")

            # Load ALL available swap pairs
            markets = self.exchange.load_markets()
            self.all_pairs = [
                symbol for symbol in markets.keys()
                if symbol.endswith(':USDT') and markets[symbol]['active']
            ]

            logger.info(f"✅ Connected to Bitget - {len(self.all_pairs)} swap pairs available")
            logger.info(f"🔥 Will scan and trade across ALL {len(self.all_pairs)} pairs!")
            return True

        except Exception as e:
            logger.error(f"❌ Connection error: {e}")
            return False

    def adjust_for_precision(self, symbol, position_size):
        """Adjust position size to meet exchange precision requirements"""
        try:
            if self.exchange and symbol in self.exchange.markets:
                market = self.exchange.markets[symbol]

                # Get amount precision from market info
                amount_precision = market.get('precision', {}).get('amount', 0.000001)

                # Round position size to meet precision requirements
                if amount_precision > 0:
                    position_size = round(position_size / amount_precision) * amount_precision

                # Ensure minimum precision is met
                if position_size < amount_precision:
                    position_size = amount_precision

                logger.info(f"🔧 Adjusted {symbol} position size to meet precision: {position_size}")

            return position_size

        except Exception as e:
            logger.warning(f"⚠️ Could not adjust precision for {symbol}: {e}")
            return position_size

    def validate_trade(self, symbol, position_size, margin_value_usdt):
        """Validate trade parameters before execution"""
        try:
            # Check minimum margin size ($5 for exchange requirements) with small tolerance for rounding
            min_margin_size = 4.95  # Minimum $4.95 margin for exchange requirements (with tolerance)
            if margin_value_usdt < min_margin_size:
                logger.warning(f"⚠️ Margin value ${margin_value_usdt:.2f} below minimum ${min_margin_size}")
                return False

            # Check if we have market information for precision validation
            if self.exchange and symbol in self.exchange.markets:
                market = self.exchange.markets[symbol]
                min_amount = market.get('limits', {}).get('amount', {}).get('min', 0)

                if position_size < min_amount and min_amount > 0:
                logger.warning(f"⚠️ Position size {position_size} below market minimum {min_amount}")
                return False

                # Additional validation: check notional value requirements (relaxed for micro accounts)
                current_price = self.exchange.fetch_ticker(symbol)['last']
                notional_value = position_size * current_price
                min_notional = market.get('limits', {}).get('cost', {}).get('min', 5.0)

                # For micro accounts (< $1 balance), be very lenient with notional minimums
                if notional_value < min_notional:
                # Allow trades with notional < $1 for micro accounts
                if notional_value < 1.0:
                    logger.info(f"ℹ️ Allowing micro trade: notional ${notional_value:.2f} < minimum ${min_notional} (micro account exception)")
                else:
                    logger.warning(f"⚠️ Notional value ${notional_value:.2f} below market minimum ${min_notional}")
                    return False

            logger.info(f"✅ Trade validation passed for {symbol} (${margin_value_usdt:.2f})")
            return True

        except Exception as e:
            logger.warning(f"⚠️ Trade validation error for {symbol}: {e}")
            return False

    def verify_position_mode(self):
        """Verify and configure position mode for Bitget"""
        try:
            # Try to get account configuration
            account_info = self.exchange.private_get_account_accounts()
            logger.info(f"📊 Account configuration: {account_info}")

            # Check if we can access position mode settings
            try:
                position_mode = self.exchange.private_get_account_position_mode()
                logger.info(f"📊 Position mode: {position_mode}")

                # If position mode is set to hedge but we want unilateral, warn user
                if position_mode.get('posMode') == 'hedge_mode':
                logger.warning("⚠️ Account is in hedge mode but bot is configured for unilateral mode")
                logger.warning("🔄 Consider switching account to unilateral mode in Bitget UI")

            except Exception as mode_error:
                logger.debug(f"Could not retrieve position mode: {mode_error}")

            return True

        except Exception as e:
            logger.warning(f"⚠️ Could not verify position mode: {e}")
            return False

    def generate_signal(self, symbol):
        """Generate trading signal with MULTI-TIMEFRAME TREND CONFIRMATION - VIPER Style"""
        try:
            # Get price data from multiple timeframes for better signal quality
            # OPTIMIZED FOR CRYPTO: Using lower timeframes for faster signals
            ohlcv_1m = self.exchange.fetch_ohlcv(symbol, timeframe='1m', limit=30)  # Added 1m for faster signals  
            ohlcv_5m = self.exchange.fetch_ohlcv(symbol, timeframe='5m', limit=20)
            ohlcv_15m = self.exchange.fetch_ohlcv(symbol, timeframe='15m', limit=12)

            if len(ohlcv_1m) < 10 or len(ohlcv_5m) < 10 or len(ohlcv_15m) < 6:
                logger.debug(f"📊 {symbol}: Insufficient data for trend analysis")
                return 'HOLD'

            # Extract closing prices for analysis
            closes_1m = [candle[4] for candle in ohlcv_1m]  # Fastest timeframe for crypto
            closes_5m = [candle[4] for candle in ohlcv_5m]
            closes_15m = [candle[4] for candle in ohlcv_15m]

            # MULTI-TIMEFRAME ANALYSIS - OPTIMIZED FOR CRYPTO LOWER TF
            # PRIMARY TREND ANALYSIS - Use 15m for main trend direction  
            primary_trend = self.analyze_trend_relaxed(closes_15m)
            
            # SECONDARY CONFIRMATION - Use 5m for entry timing
            secondary_trend = self.analyze_trend_relaxed(closes_5m)
            
            # FAST ENTRY SIGNALS - Use 1m for immediate entry confirmation
            fast_trend = self.analyze_trend_relaxed(closes_1m)

            # Current price from latest 1m candle (most up-to-date)
            current_price = closes_1m[-1]

            # ENHANCED SIGNAL LOGIC: Crypto-optimized with 1m confirmation
            if (primary_trend in ['BULLISH', 'WEAK_BULLISH'] and 
                secondary_trend in ['BULLISH', 'WEAK_BULLISH', 'SIDEWAYS'] and
                fast_trend in ['BULLISH', 'WEAK_BULLISH', 'SIDEWAYS']):  # 1m confirmation
                
                # Use 1m data for precise entry timing
                recent_high = max(closes_1m[-5:])
                recent_low = min(closes_1m[-5:])

                # Very flexible entry: only avoid extreme conditions
                if current_price > recent_low * 1.005:  # More responsive with 1m data
                logger.info(f"📈 {symbol}: BULLISH SIGNAL - 15m:{primary_trend}, 5m:{secondary_trend}, 1m:{fast_trend} - Entry at ${current_price}")
                return 'BUY'

            elif (primary_trend in ['BEARISH', 'WEAK_BEARISH'] and 
                  secondary_trend in ['BEARISH', 'WEAK_BEARISH', 'SIDEWAYS'] and
                  fast_trend in ['BEARISH', 'WEAK_BEARISH', 'SIDEWAYS']):  # 1m confirmation
                
                # Use 1m data for precise entry timing
                recent_high = max(closes_1m[-5:])
                recent_low = min(closes_1m[-5:])
                recent_low = min(closes_5m[-5:])

                # Very flexible entry: only avoid extreme conditions
                if current_price < recent_high * 0.995:  # Much more flexible:
                logger.info(f"📉 {symbol}: BEARISH SIGNAL - Primary:{primary_trend}, Secondary:{secondary_trend} - Entry at ${current_price}")
                return 'SELL'

            # EVEN MORE FLEXIBLE: Single timeframe signals
            elif primary_trend in ['BULLISH', 'WEAK_BULLISH'] or secondary_trend in ['BULLISH', 'WEAK_BULLISH']:
                logger.info(f"📈 {symbol}: SINGLE-TIMEFRAME BULL SIGNAL - Primary:{primary_trend}, Secondary:{secondary_trend} - Entry at ${current_price}")
                return 'BUY'

            elif primary_trend in ['BEARISH', 'WEAK_BEARISH'] or secondary_trend in ['BEARISH', 'WEAK_BEARISH']:
                logger.info(f"📉 {symbol}: SINGLE-TIMEFRAME BEAR SIGNAL - Primary:{primary_trend}, Secondary:{secondary_trend} - Entry at ${current_price}")
                return 'SELL'

            # CONSERVATIVE MOMENTUM TRADES: Even without clear trend, look for strong momentum
            elif self.detect_momentum_signal(closes_5m):
                momentum_signal = self.detect_momentum_signal(closes_5m)
                if momentum_signal == 'STRONG_BULL':
                logger.info(f"💪 {symbol}: STRONG BULL MOMENTUM - Entry at ${current_price}")
                return 'BUY'
                elif momentum_signal == 'STRONG_BEAR':
                logger.info(f"💪 {symbol}: STRONG BEAR MOMENTUM - Entry at ${current_price}")
                return 'SELL'

            # No clear signal
            logger.debug(f"📊 {symbol}: No clear signal - Primary:{primary_trend}, Secondary:{secondary_trend} - HOLD")
            return 'HOLD'

        except Exception as e:
            logger.error(f"❌ Error generating signal for {symbol}: {e}")
            return 'HOLD'

    def analyze_trend(self, closes):
        """Analyze trend direction using higher highs/higher lows methodology"""
        if len(closes) < 10:
            return 'SIDEWAYS'

        # Check for BULLISH trend: Higher highs and higher lows
        recent_highs = []
        recent_lows = []

        # Analyze last 10 candles for swing points
        for i in range(5, len(closes)):
            # Check if this is a local high
            if closes[i] > closes[i-1] and closes[i] > closes[i+1 if i+1 < len(closes) else i]:
                recent_highs.append(closes[i])

            # Check if this is a local low
            if closes[i] < closes[i-1] and closes[i] < closes[i+1 if i+1 < len(closes) else i]:
                recent_lows.append(closes[i])

        # Need at least 2 swing points for analysis
        if len(recent_highs) < 2 or len(recent_lows) < 2:
            return 'SIDEWAYS'

        # Check for BULLISH trend: Higher highs AND higher lows
        latest_high = max(recent_highs[-2:])  # Last 2 highs
        previous_high = min(recent_highs[-2:])  # Previous high
        latest_low = max(recent_lows[-2:])     # Last 2 lows
        previous_low = min(recent_lows[-2:])   # Previous low

        if latest_high > previous_high and latest_low > previous_low:
            return 'BULLISH'

        # Check for BEARISH trend: Lower highs AND lower lows
        if latest_high < previous_high and latest_low < previous_low:
            return 'BEARISH'

        return 'SIDEWAYS'

    def analyze_trend_relaxed(self, closes):
        """Relaxed trend analysis for more flexible signal generation"""
        if len(closes) < 8:
            return 'SIDEWAYS'

        # Simple moving averages for trend direction
        short_ma = sum(closes[-5:]) / 5
        long_ma = sum(closes[-10:]) / 10

        # Price position relative to moving averages
        current_price = closes[-1]
        ma_diff = (short_ma - long_ma) / long_ma * 100  # Percentage difference

        # Trend strength based on MA separation (more sensitive)
        if ma_diff > 0.2:  # Lower threshold for bullish:
            if current_price > short_ma * 1.002:  # More flexible:
                return 'BULLISH'
            else:
                return 'WEAK_BULLISH'
        elif ma_diff < -0.2:  # Lower threshold for bearish:
            if current_price < short_ma * 0.998:  # More flexible:
                return 'BEARISH'
            else:
                return 'WEAK_BEARISH'

        # Check for sideways with slight bias
        recent_change = (closes[-1] - closes[-5]) / closes[-5] * 100
        if abs(recent_change) < 0.3:
            return 'SIDEWAYS'
        elif recent_change > 0:
            return 'WEAK_BULLISH'
        else:
            return 'WEAK_BEARISH'

    def detect_momentum_signal(self, closes):
        """Detect strong momentum signals for conservative entries"""
        if len(closes) < 8:
            return None

        # Calculate momentum indicators
        recent_prices = closes[-8:]
        avg_first_half = sum(recent_prices[:4]) / 4
        avg_second_half = sum(recent_prices[4:]) / 4

        momentum_strength = (avg_second_half - avg_first_half) / avg_first_half * 100

        # Strong momentum thresholds (more sensitive)
        if momentum_strength > 0.8:  # Strong upward momentum (lowered)
            return 'STRONG_BULL'
        elif momentum_strength < -0.8:  # Strong downward momentum (lowered)
            return 'STRONG_BEAR'

        return None

    def execute_trade(self, symbol, signal):
        """Execute trade for any pair with proper Bitget unilateral position parameters"""
        try:
            # Get current price
            ticker = self.exchange.fetch_ticker(symbol)
            current_price = ticker['last']

            # Calculate position size based on account balance percentage
            try:
                # FIXED: Use ONLY free balance - don't include unrealized P&L as capital/margin
                balance = self.exchange.fetch_balance({'type': 'swap'})
                usdt_balance = float(balance.get('USDT', {}).get('free', 0))  # Only use actual free balance, no UPNL
                if usdt_balance <= 0:
                logger.error(f"❌ No USDT balance available for {symbol}")
                return None

                # FIXED: EXACTLY $1 MARGIN PER TRADE + COIN'S MAX LEVERAGE
                margin_value_usdt = 1.0  # Always use exactly $1 margin per trade

                # Check if account has enough balance for $1 margin
                if usdt_balance < margin_value_usdt:
                logger.warning(f"⚠️ Insufficient balance: ${usdt_balance:.2f} < ${margin_value_usdt:.2f} required margin")
                return None

                # GET COIN'S MAX LEVERAGE FROM EXCHANGE
                try:
                market_info = self.exchange.market(symbol)
                coin_max_leverage = market_info.get('limits', {}).get('leverage', {}).get('max', 50)
                # Ensure we don't exceed exchange limits
                coin_max_leverage = min(coin_max_leverage, 100)  # Cap at 100x for safety
                except Exception as leverage_error:
                logger.warning(f"⚠️ Could not get coin leverage for {symbol}, using default 50x: {leverage_error}")
                coin_max_leverage = 50

                # CALCULATE NOTIONAL: $1 margin × coin's max leverage
                notional_value_usdt = margin_value_usdt * coin_max_leverage
                position_size = notional_value_usdt / current_price

                logger.info(f"💰 FIXED: $1 Margin × {coin_max_leverage}x Leverage = ${notional_value_usdt:.2f} Notional | Account: ${usdt_balance:.2f}")

                # Final safety check: ensure notional value doesn't exceed account balance
                if notional_value_usdt > usdt_balance * 0.5:  # Max 50% of account as notional:
                # Recalculate with safer margin but keep $1 minimum
                if usdt_balance >= 1.0:
                    margin_value_usdt = max(1.0, usdt_balance * 0.02)  # Minimum $1 or 2% of balance
                    notional_value_usdt = margin_value_usdt * coin_max_leverage
                    position_size = notional_value_usdt / current_price
                    logger.info(f"💰 Reduced margin for safety (${margin_value_usdt:.2f}) but kept minimum $1")
                else:
                    logger.warning(f"⚠️ Account balance too low for $1 minimum margin")
                    return None

                # Adjust for precision requirements (this will ensure we meet exchange minimums)
                position_size = self.adjust_for_precision(symbol, position_size)

                # Final validation: ensure the adjusted position still meets minimum notional value
                final_notional_value = position_size * current_price
                min_notional_required = self.min_margin_per_trade * self.max_leverage

                if final_notional_value < min_notional_required:
                # Recalculate with minimum margin
                margin_value_usdt = self.min_margin_per_trade
                notional_value_usdt = margin_value_usdt * self.max_leverage
                position_size = notional_value_usdt / current_price
                position_size = self.adjust_for_precision(symbol, position_size)
                logger.info(f"💰 Final adjustment to ensure minimum notional value for {symbol}")

                logger.info(f"🎯 {signal} {symbol} at ${current_price:.6f}")
                logger.info(f"💰 Account Balance: ${usdt_balance:.2f}")
                logger.info(f"💰 Position size: {position_size:.6f} coins (Margin: ${margin_value_usdt:.2f}, Notional: ${notional_value_usdt:.2f}, Leverage: {self.max_leverage}x)")

            except Exception as e:
                logger.error(f"❌ Failed to calculate position size for {symbol}: {e}")
                return None

            # Validate trade before execution (check margin requirement)
            if not self.validate_trade(symbol, position_size, margin_value_usdt):
                logger.warning(f"⚠️ Trade validation failed for {symbol}, skipping...")
                return None

            # Execute order with proper Bitget unilateral position parameters
            try:
                if signal == 'BUY':
                order = self.exchange.create_market_buy_order(
                    symbol,
                    position_size,
                    params={
                        'leverage': coin_max_leverage,  # Use coin's max leverage
                        'marginMode': 'isolated',
                        'tradeSide': 'open'  # Open position for unilateral mode
                    }
                )
                elif signal == 'SELL':
                order = self.exchange.create_market_sell_order(
                    symbol,
                    position_size,
                    params={
                        'leverage': coin_max_leverage,  # Use coin's max leverage
                        'marginMode': 'isolated',
                        'tradeSide': 'open'  # Open position for unilateral mode
                    }
                )

            except Exception as e:
                logger.error(f"❌ Trade execution error for {symbol}: {e}")
                return None

            # Log successful execution
            logger.info(f"✅ Trade executed: {order['id']}")

            # SET TAKE-PROFIT AND STOP-LOSS ORDERS
            try:
                entry_price = order.get('price', current_price)
                logger.info(f"📊 Entry price for {symbol}: ${entry_price}")

                # Calculate TP/SL prices
                if signal == 'BUY':
                tp_price = entry_price * (1 + self.take_profit_pct / 100)
                sl_price = entry_price * (1 - self.stop_loss_pct / 100)
                tp_side = 'sell'  # Close long position at profit
                sl_side = 'sell'  # Close long position at loss
                else:  # SELL signal
                tp_price = entry_price * (1 - self.take_profit_pct / 100)
                sl_price = entry_price * (1 + self.stop_loss_pct / 100)
                tp_side = 'buy'   # Close short position at profit
                sl_side = 'buy'   # Close short position at loss

                logger.info(f"🎯 TP/SL for {symbol}:")
                logger.info(f"   Take Profit: ${tp_price:.6f} ({self.take_profit_pct}%)")
                logger.info(f"   Stop Loss: ${sl_price:.6f} ({self.stop_loss_pct}%)")

                # Create Take-Profit order
                tp_order = self.exchange.create_limit_order(
                    symbol,
                    tp_side,
                    position_size,
                    tp_price,
                    params={
                        'leverage': coin_max_leverage,  # Use coin's max leverage
                        'marginMode': 'isolated',
                        'tradeSide': 'close',
                        'reduceOnly': True
                    })
                logger.info(f"✅ Take-Profit order placed: {tp_order.get('id', 'N/A')}")

                # VERIFY TP ORDER WAS PLACED
                if tp_order and tp_order.get('id'):
                    try:
                        # Verify the order exists on exchange
                        order_status = self.exchange.fetch_order(tp_order['id'], symbol)
                        logger.info(f"🔍 TP Order verified: {symbol} | Status: {order_status.get('status', 'unknown')}")
                    except Exception as verify_error:
                        logger.error(f"⚠️ Could not verify TP order {tp_order.get('id')}: {verify_error}")
                else:
                    logger.error(f"❌ TP order placement failed for {symbol}")

                # Create Stop-Loss order
                sl_order = self.exchange.create_limit_order(
                    symbol,
                    sl_side,
                    position_size,
                    sl_price,
                    params={
                        'leverage': coin_max_leverage,  # Use coin's max leverage
                        'marginMode': 'isolated',
                        'tradeSide': 'close',
                        'reduceOnly': True
                    })
                logger.info(f"🛡️ Stop-Loss order placed: {sl_order.get('id', 'N/A')}")

                # VERIFY SL ORDER WAS PLACED
                if sl_order and sl_order.get('id'):
                    try:
                        # Verify the order exists on exchange
                        order_status = self.exchange.fetch_order(sl_order['id'], symbol)
                        logger.info(f"🔍 SL Order verified: {symbol} | Status: {order_status.get('status', 'unknown')}")
                    except Exception as verify_error:
                        logger.error(f"⚠️ Could not verify SL order {sl_order.get('id')}: {verify_error}")
                else:
                    logger.error(f"❌ SL order placement failed for {symbol}")

                except Exception as tp_sl_error:
                logger.error(f"❌ Failed to set TP/SL for {symbol}: {tp_sl_error}")
                logger.warning("⚠️ Position opened without TP/SL - manual monitoring required")

            return order

        except Exception as e:
            logger.error(f"❌ Trade execution error for {symbol}: {e}")

            # If we get unilateral position error, try alternative approach
            if "40774" in str(e) or "unilateral position" in str(e).lower():
                logger.warning(f"🔄 Retrying {symbol} with alternative parameters...")
                try:
                # Try without tradeSide parameter for unilateral mode
                if signal == 'BUY':
                    order = self.exchange.create_market_buy_order(
                        symbol,
                        position_size,
                        params={
                            'leverage': self.max_leverage,
                            'marginMode': 'isolated'
                        }
                    )
                elif signal == 'SELL':
                    order = self.exchange.create_market_sell_order(
                        symbol,
                        position_size,
                        params={
                            'leverage': self.max_leverage,
                            'marginMode': 'isolated'
                        }
                    )
                logger.info(f"✅ Alternative trade executed: {order['id']}")

                # SET TAKE-PROFIT AND STOP-LOSS ORDERS FOR ALTERNATIVE EXECUTION
                try:
                    entry_price = order.get('price', current_price)
                    logger.info(f"📊 Alternative entry price for {symbol}: ${entry_price}")

                    # Calculate TP/SL prices (same logic as main execution)
                    if signal == 'BUY':
                        tp_price = entry_price * (1 + self.take_profit_pct / 100)
                        sl_price = entry_price * (1 - self.stop_loss_pct / 100)
                        tp_side = 'sell'
                        sl_side = 'sell'
                    else:  # SELL signal
                        tp_price = entry_price * (1 - self.take_profit_pct / 100)
                        sl_price = entry_price * (1 + self.stop_loss_pct / 100)
                        tp_side = 'buy'
                        sl_side = 'buy'

                    logger.info(f"🎯 Alternative TP/SL for {symbol}:")
                    logger.info(f"   Take Profit: ${tp_price:.6f} ({self.take_profit_pct}%)")
                    logger.info(f"   Stop Loss: ${sl_price:.6f} ({self.stop_loss_pct}%)")

                    # Create Take-Profit order
                    tp_order = self.exchange.create_limit_order(
                    symbol,
                        tp_side,
                        position_size,
                        tp_price,
                        params={
                            'leverage': self.max_leverage,
                            'marginMode': 'isolated',
                            'tradeSide': 'close',
                            'reduceOnly': True
                        })
                    logger.info(f"✅ Alternative Take-Profit order placed: {tp_order.get('id', 'N/A')}")

                    # Create Stop-Loss order
                    sl_order = self.exchange.create_limit_order(
                    symbol,
                        sl_side,
                        position_size,
                        sl_price,
                        params={
                            'leverage': self.max_leverage,
                            'marginMode': 'isolated',
                            'tradeSide': 'close',
                            'reduceOnly': True
                        })
                    logger.info(f"🛡️ Alternative Stop-Loss order placed: {sl_order.get('id', 'N/A')}")

                except Exception as alt_tp_sl_error:
                    logger.error(f"❌ Failed to set alternative TP/SL for {symbol}: {alt_tp_sl_error}")
                    logger.warning("⚠️ Alternative position opened without TP/SL - manual monitoring required")

                return order
                except Exception as retry_error:
                logger.error(f"❌ Alternative trade also failed for {symbol}: {retry_error}")
                return None

            return None

    def scan_markets(self):
        """Single market scan - returns trading opportunities for MCP integration"""
        opportunities = []

        logger.info(f"🔍 Scanning {len(self.all_pairs)} pairs for opportunities...")

        # Scan subset of pairs for trading opportunities
        pairs_to_scan = min(25, len(self.all_pairs))  # Scan 25 pairs per cycle
        scanned_pairs = random.sample(self.all_pairs, pairs_to_scan)

        for symbol in scanned_pairs:
            # CHECK POSITION LIMITS BEFORE TRADING
            current_positions = len(self.active_positions)
            if current_positions >= self.max_positions:
                logger.debug(f"🚫 Position limit reached ({current_positions}/{self.max_positions}). Skipping {symbol}")
                continue

            # SKIP IF ALREADY HAVE POSITION IN THIS PAIR
            if symbol in self.active_positions:
                logger.debug(f"📊 Skipping {symbol} - already have position")
                continue

            signal = self.generate_signal(symbol)

            if signal in ['BUY', 'SELL']:
                # Calculate confidence based on signal strength
                confidence = self.calculate_signal_confidence(symbol, signal)

                opportunities.append({
                'symbol': symbol,
                'signal': signal,
                'confidence': confidence,
                'timestamp': time.time()
                })
                logger.info(f"📊 Found opportunity: {symbol} -> {signal} (confidence: {confidence:.2f})")

        logger.info(f"✅ Scan complete - {len(opportunities)} opportunities found")
        return opportunities

    def calculate_signal_confidence(self, symbol: str, signal: str) -> float:
        """Calculate confidence score for trading signal"""
        try:
            # Get recent price data - OPTIMIZED FOR CRYPTO: Include 1m for faster signals
            ohlcv_1m = self.exchange.fetch_ohlcv(symbol, timeframe='1m', limit=30)
            ohlcv_5m = self.exchange.fetch_ohlcv(symbol, timeframe='5m', limit=20)
            ohlcv_15m = self.exchange.fetch_ohlcv(symbol, timeframe='15m', limit=12)

            if len(ohlcv_1m) < 10 or len(ohlcv_5m) < 10 or len(ohlcv_15m) < 6:
                return 0.5  # Neutral confidence

            # Analyze trend consistency across timeframes - ENHANCED WITH 1M
            fast_trend = self.analyze_trend_relaxed([candle[4] for candle in ohlcv_1m])  # 1m trend
            primary_trend = self.analyze_trend_relaxed([candle[4] for candle in ohlcv_15m])
            secondary_trend = self.analyze_trend_relaxed([candle[4] for candle in ohlcv_5m])

            # CRYPTO-OPTIMIZED CONFIDENCE CALCULATION WITH 1M SUPPORT
            base_confidence = 0.5

            # Multi-timeframe alignment bonus (1m, 5m, 15m)
            if signal == 'BUY':
                if fast_trend in ['BULLISH', 'WEAK_BULLISH']:
                base_confidence += 0.15  # 1m confirmation
                if primary_trend in ['BULLISH', 'WEAK_BULLISH']:
                base_confidence += 0.15  # 15m trend
                if secondary_trend in ['BULLISH', 'WEAK_BULLISH']:
                base_confidence += 0.15  # 5m trend
            elif signal == 'SELL':
                if fast_trend in ['BEARISH', 'WEAK_BEARISH']:
                base_confidence += 0.15  # 1m confirmation
                if primary_trend in ['BEARISH', 'WEAK_BEARISH']:
                base_confidence += 0.15  # 15m trend
                if secondary_trend in ['BEARISH', 'WEAK_BEARISH']:
                base_confidence += 0.15  # 5m trend

            confidence = min(base_confidence, 0.95)  # Cap at 95%

            # Check for momentum confirmation
            if self.detect_momentum_signal([candle[4] for candle in ohlcv_5m]):
                momentum_signal = self.detect_momentum_signal([candle[4] for candle in ohlcv_5m])
                if ((signal == 'BUY' and momentum_signal == 'STRONG_BULL') or 
                (signal == 'SELL' and momentum_signal == 'STRONG_BEAR')):
                confidence += 0.1  # Boost for momentum confirmation

            return min(confidence, 1.0)  # Cap at 100%

        except Exception as e:
            logger.debug(f"⚠️ Error calculating confidence for {symbol}: {e}")
            return 0.5  # Return neutral confidence on error

    def should_enter_position(self, symbol: str, signal: str, confidence: float) -> bool:
        """Determine if a trade should be executed based on confidence and conditions"""
        try:
            # Minimum confidence threshold
            min_confidence = 0.6  # Require at least 60% confidence

            if confidence < min_confidence:
                logger.debug(f"⚠️ {symbol}: Confidence {confidence:.2f} below threshold {min_confidence}")
                return False

            # Check position limits
            current_positions = len(self.active_positions)
            if current_positions >= self.max_positions:
                logger.debug(f"🚫 Position limit reached ({current_positions}/{self.max_positions})")
                return False

            # Check if already have position in this pair
            if symbol in self.active_positions:
                logger.debug(f"📊 Already have position in {symbol}")
                return False

            # Additional risk checks
            usdt_balance = self.exchange.fetch_balance()['USDT']['free']
            if usdt_balance < self.min_margin_per_trade:
                logger.debug(f"⚠️ Insufficient balance: ${usdt_balance:.2f} < ${self.min_margin_per_trade:.2f}")
                return False

            logger.info(f"✅ {symbol}: Trade approved (confidence: {confidence:.2f})")
            return True

        except Exception as e:
            logger.error(f"❌ Error checking position entry for {symbol}: {e}")
            return False

    def calculate_position_size(self, symbol: str = None):
        """Calculate position size based on risk management parameters and minimum amounts"""
        try:
            # FIXED: Use ONLY free balance - don't include unrealized P&L as capital/margin
            balance = self.exchange.fetch_balance()
            usdt_balance = balance['USDT']['free']  # Only use actual free balance, no UPNL

            # FIXED: Use minimum margin per trade (fixed $5 to meet exchange minimum)
            risk_based_size = usdt_balance * 0.1  # 10% of balance
            position_size_usdt = max(5.0, risk_based_size)  # Ensure minimum $5, but allow larger if risk allows

            logger.debug(f"💰 Balance: ${usdt_balance:.2f}, Risk-based: ${risk_based_size:.2f}, Final size: ${position_size_usdt:.2f}")

            # If symbol is provided, ensure we meet minimum amount requirements
            if symbol:
                try:
                # Get market info for minimum amounts
                market = self.exchange.market(symbol)
                min_amount = market.get('limits', {}).get('amount', {}).get('min', 1)

                # Get current price to convert USDT amount to crypto amount
                ticker = self.exchange.fetch_ticker(symbol)
                current_price = ticker['last']

                # Calculate crypto amount needed
                crypto_amount = position_size_usdt / current_price

                # Ensure it meets minimum requirements
                if crypto_amount < min_amount:
                    # Recalculate USDT amount needed for minimum crypto amount
                    position_size_usdt = min_amount * current_price
                    logger.debug(f"📊 Adjusted {symbol} position size to meet minimum amount: ${position_size_usdt:.2f}")

                except Exception as market_error:
                logger.debug(f"⚠️ Could not check minimum amount for {symbol}: {market_error}")

            logger.debug(f"📊 Position size calculated: ${position_size_usdt:.2f} (balance: ${usdt_balance:.2f})")
            return position_size_usdt

        except Exception as e:
            logger.error(f"❌ Error calculating position size: {e}")
            return 0.0

    def run(self):
        """Main trading loop - scans ALL pairs"""
        self.is_running = True
        cycle = 0

        logger.info("🚀 Starting MULTI-PAIR TRADING - Scanning ALL available pairs!")
        logger.info("=" * 80)

        while self.is_running:
            cycle += 1
            logger.info(f"\n🔄 Cycle #{cycle} - Scanning {len(self.all_pairs)} pairs")

            # Use the scan_markets method for consistency
            opportunities = self.scan_markets()

            trades_this_cycle = 0

            for opportunity in opportunities:
                symbol = opportunity['symbol']
                signal = opportunity['signal']

                order = self.execute_trade(symbol, signal)
                if order:
                    trades_this_cycle += 1
                    # Calculate and store TP/SL information for monitoring
                    entry_price = order.get('price', current_price)
                    tp_price = entry_price * (1 + self.take_profit_pct / 100) if signal == 'BUY' else entry_price * (1 - self.take_profit_pct / 100)
                    sl_price = entry_price * (1 - self.stop_loss_pct / 100) if signal == 'BUY' else entry_price * (1 + self.stop_loss_pct / 100)

                    self.active_positions[symbol] = {
                        'order_id': order['id'],
                        'signal': signal,
                        'entry_price': entry_price,
                        'quantity': order['amount'],
                        'timestamp': time.time(),
                        'tp_price': tp_price,
                        'sl_price': sl_price,
                        'tp_order_id': tp_order.get('id') if 'tp_order' in locals() and tp_order else None,
                        'sl_order_id': sl_order.get('id') if 'sl_order' in locals() and sl_order else None
                    }

                    logger.info(f"📊 Position tracked: {symbol} | Entry: ${entry_price:.6f} | TP: ${tp_price:.6f} | SL: ${sl_price:.6f}")

            logger.info(f"✅ Cycle #{cycle} complete - {trades_this_cycle} trades executed")
            logger.info(f"📈 Active positions: {len(self.active_positions)}")

            # DEBUG: Detailed position status
            if self.active_positions:
                logger.info("📊 Current positions:")
                for symbol, pos_data in self.active_positions.items():
                age = time.time() - pos_data['timestamp']
                logger.info(f"   {symbol}: {pos_data['signal']} | Entry: ${pos_data['entry_price']:.6f} | Qty: {pos_data['quantity']:.6f} | Age: {age:.1f}s")
                if 'tp_price' in pos_data and 'sl_price' in pos_data:
                    logger.info(f"      TP: ${pos_data['tp_price']:.6f} | SL: ${pos_data['sl_price']:.6f}")
            else:
                logger.info("📊 No active positions")

            # CRITICAL FIX: MONITOR TP/SL EXECUTION
            self.monitor_tp_sl_execution()

            # PERIODIC POSITION LIMIT ENFORCEMENT
            self.enforce_position_limit()

            logger.info("⏰ Waiting 30 seconds for next scan...")

            time.sleep(30)

    def monitor_tp_sl_execution(self):
        """CRITICAL: Monitor and execute TP/SL orders that may have been filled"""
        if not self.active_positions:
            return

        logger.debug("🔍 Checking TP/SL execution for active positions...")

        try:
            # Get current positions from exchange to check for closures
            exchange_positions = self.exchange.fetch_positions()

            # Create a set of currently open position symbols
            open_symbols = {pos['symbol'] for pos in exchange_positions if float(pos.get('contracts', 0)) > 0}

            # Check each tracked position
            positions_to_remove = []
            for symbol in list(self.active_positions.keys()):
                if symbol not in open_symbols:
                # Position was closed (likely by TP/SL execution)
                logger.info(f"🎯 TP/SL EXECUTED: {symbol} - Position closed by exchange")
                positions_to_remove.append(symbol)

                # Trigger position closed callback
                if hasattr(self, '_on_position_closed'):
                    position_record = self.active_positions[symbol].copy()
                    position_record['symbol'] = symbol
                    position_record['closed_reason'] = 'tp_sl_executed'
                    try:
                        self._on_position_closed(position_record)
                    except Exception as callback_error:
                        logger.error(f"⚠️ Position close callback error: {callback_error}")

            # Remove closed positions from tracking
            for symbol in positions_to_remove:
                del self.active_positions[symbol]
                logger.info(f"🗑️ Removed closed position from tracking: {symbol}")

            if positions_to_remove:
                logger.info(f"✅ Processed {len(positions_to_remove)} TP/SL executions")

        except Exception as e:
            logger.error(f"❌ Error monitoring TP/SL execution: {e}")

    def close_position(self, symbol, reason="manual"):
        """Close a position using the adoption system (which handles long/short properly)"""
        logger.info(f"🔄 Closing position: {symbol} (reason: {reason})")

        # Use the adoption system's close_position method which handles long/short correctly
        result = self.position_adoption_system.close_position(symbol, reason)

        if result:
            logger.info(f"✅ Successfully closed {symbol}")
            # Remove from our tracking if it exists
            if symbol in self.active_positions:
                del self.active_positions[symbol]
            return True
        else:
            logger.error(f"❌ Failed to close {symbol}")
            return False

    def _on_position_adopted(self, position_record):
        """Callback when a position is adopted"""
        symbol = position_record['symbol']
        logger.info(f"🎉 Position adopted: {symbol}")
        self.active_positions[symbol] = {
            'order_id': position_record['order_id'],
            'signal': position_record['signal'],
            'entry_price': position_record['entry_price'],
            'quantity': position_record['quantity'],
            'timestamp': time.time()
        }

    def _on_position_closed(self, position_record):
        """Callback when a position is closed"""
        symbol = position_record['symbol']
        logger.info(f"👋 Position closed: {symbol}")
        if symbol in self.active_positions:
            del self.active_positions[symbol]

    def _on_position_updated(self, position_record):
        """Callback when a position is updated"""
        symbol = position_record['symbol']
        pnl = position_record.get('unrealized_pnl', 0)

        # Update our tracking if we have this position"""
        if symbol in self.active_positions:
            self.active_positions[symbol]['pnl'] = pnl
            self.active_positions[symbol]['last_updated'] = time.time()

    def adopt_existing_positions(self):
        """Adopt any existing positions with proper limit enforcement"""
        logger.info("🔍 Checking for existing positions to adopt...")
        logger.info(f"📊 Current position limit: {self.max_positions} positions maximum")

        # FIXED: Don't clear positions - sync with adoption system instead
        # self.active_positions = {}  # REMOVED - this was causing the bug!

        result = self.position_adoption_system.adopt_existing_positions()

        if result['success']:
            logger.info(f"✅ Adopted {result['newly_adopted']} existing positions")
            if result['newly_adopted'] > 0:
                logger.info(f"   Adopted symbols: {', '.join(result['adopted_symbols'])}")

                # CRITICAL FIX: Sync adopted positions with main trader's tracking
                for symbol in result['adopted_symbols']:
                if symbol in self.position_adoption_system.active_positions:
                    adopted_pos = self.position_adoption_system.active_positions[symbol]
                    self.active_positions[symbol] = {
                        'order_id': f"adopted_{symbol}",
                        'signal': adopted_pos.get('side', 'unknown'),
                        'entry_price': adopted_pos.get('entry_price', 0),
                        'quantity': adopted_pos.get('contracts', 0),
                        'timestamp': adopted_pos.get('adopted_at').timestamp() if hasattr(adopted_pos.get('adopted_at'), 'timestamp') else time.time(),
                        'source': 'adopted'
                    }
                    logger.info(f"🔄 Synced adopted position: {symbol}")
        else:
            logger.warning(f"⚠️ Position adoption failed: {result.get('error', 'Unknown error')}")

        # ENFORCE POSITION LIMIT AFTER ADOPTION
        self.enforce_position_limit()

        return result

    def enforce_position_limit(self):
        """Enforce the position limit by closing excess positions if needed"""
        current_positions = len(self.active_positions)

        logger.info(f"🔍 Checking position limit: {current_positions}/{self.max_positions}")

        if current_positions > self.max_positions:
            excess_positions = current_positions - self.max_positions
            logger.warning(f"🚨 POSITION LIMIT VIOLATION: {current_positions}/{self.max_positions} positions")
            logger.warning(f"🔧 Need to close {excess_positions} excess positions")

            # Get positions sorted by age (oldest first) to close oldest ones
            positions_to_close = []
            for symbol, position_data in sorted(self.active_positions.items(),:
(                                              key=lambda x: x[1].get('timestamp', 0))
                if len(positions_to_close) < excess_positions:
                positions_to_close.append(symbol)

            logger.info(f"📋 Will close positions: {positions_to_close}")

            # Close excess positions
            closed_count = 0
            for symbol in positions_to_close:
                logger.info(f"🔄 Closing excess position: {symbol}")
                try:
                result = self.close_position(symbol, "position_limit_enforcement")
                if result:  # close_position now returns boolean directly:
                    closed_count += 1
                    logger.info(f"✅ Successfully closed {symbol}")
                else:
                    logger.error(f"❌ Failed to close {symbol}: position close returned False")
                except Exception as e:
                logger.error(f"❌ Exception closing {symbol}: {e}")

            final_positions = len(self.active_positions)
            logger.info(f"📊 Position limit enforcement complete: {final_positions}/{self.max_positions} positions")
            logger.info(f"✅ Successfully closed {closed_count}/{excess_positions} positions")
        else:
            logger.info(f"✅ Position limit OK: {current_positions}/{self.max_positions} positions")

    def sync_positions(self):
        """Sync positions with exchange"""
        logger.debug("🔄 Syncing positions with exchange...")
        result = self.position_adoption_system.sync_positions()

        if result.get('positions_closed', 0) > 0:
            logger.info(f"ℹ️ {result['positions_closed']} positions were closed externally")

        return result

    def stop(self):
        """Stop trading"""
        logger.info("🛑 Stopping multi-pair trader...")
        self.is_running = False

def main():
    """Main entry point"""
    print("🚀 VIPER MULTI-PAIR LIVE TRADING SYSTEM")
    print("🎯 Scanning and trading ALL available pairs on Bitget")
    print("=" * 80)

    # Verify API credentials
    api_key = os.getenv('BITGET_API_KEY')
    api_secret = os.getenv('BITGET_API_SECRET')
    api_password = os.getenv('BITGET_API_PASSWORD')

    if not all([api_key, api_secret, api_password]):
        logger.error("❌ Missing API credentials!")
        return False

    logger.info(f"✅ API Key loaded: {api_key[:10]}...")
    logger.info("✅ API credentials verified")

    try:
        logger.info("🔄 Initializing Multi-Pair VIPER Trader...")
        trader = MultiPairVIPERTrader()

        if not trader.connect():
            logger.error("❌ Failed to connect to exchange")
            return False

        logger.info("✅ Connected to Bitget successfully")
        logger.info(f"🔥 Ready to trade {len(trader.all_pairs)} pairs with 50x leverage!")
        logger.info("🚀 Starting multi-pair live trading...")

        trader.run()

    except KeyboardInterrupt:
        logger.info("🛑 Trading stopped by user")
        trader.stop()
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        return False

    return True

if __name__ == "__main__":
    main()
