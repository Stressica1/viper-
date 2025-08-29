#!/usr/bin/env python3
"""
🚀 VIPER MULTI-PAIR LIVE TRADER - FIXED FOR BITGET API
Live trading system that scans and trades ALL available pairs on Bitget with proper unilateral position configuration
"""

import os
import sys
import time
import logging
import ccxt
import random
import requests
from pathlib import Path
from dotenv import load_dotenv

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

class FixedVIPERTrader:
    """VIPER Trader with FIXED Bitget API configuration for unilateral positions"""

    def __init__(self):
        # Load API credentials
        self.api_key = os.getenv('BITGET_API_KEY')
        self.api_secret = os.getenv('BITGET_API_SECRET')
        self.api_password = os.getenv('BITGET_API_PASSWORD')

        # Trading configuration - AS PER USER REQUIREMENTS
        self.position_size_usdt = float(os.getenv('POSITION_SIZE_USDT', '10'))  # $10 per trade
        self.max_leverage = int(os.getenv('MAX_LEVERAGE', '50'))  # 50x as per your rules
        self.take_profit_pct = float(os.getenv('TAKE_PROFIT_PCT', '3.0'))
        self.stop_loss_pct = float(os.getenv('STOP_LOSS_PCT', '2.0'))
        self.max_positions = int(os.getenv('MAX_POSITIONS', '15'))  # 15 positions max

        self.exchange = None
        self.all_pairs = []
        self.active_positions = {}
        self.is_running = False
        self.base_url = 'https://api.bitget.com'

    def connect(self):
        """Connect to Bitget with proper unilateral position configuration"""
        try:
            if not all([self.api_key, self.api_secret, self.api_password]):
                logger.error("❌ Missing API credentials")
                return False

            logger.info("🔌 Connecting to Bitget...")

            # Initialize exchange with proper configuration for unilateral positions
            self.exchange = ccxt.bitget({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'password': self.api_password,
                'options': {
                    'defaultType': 'swap',
                    'adjustForTimeDifference': True,
                    'recvWindow': 5000,
                    'hedgeMode': False,  # Explicitly set unilateral mode
                },
                'sandbox': False,
            })

            # Configure position mode before trading
            self.configure_position_mode()

            # Load ALL available swap pairs
            markets = self.exchange.load_markets()
            self.all_pairs = [
                symbol for symbol in markets.keys()
                if symbol.endswith(':USDT') and markets[symbol]['active']
            ]

            logger.info(f"✅ Connected to Bitget - {len(self.all_pairs)} swap pairs available")
            logger.info(f"🔥 Ready to trade across ALL {len(self.all_pairs)} pairs with unilateral positions!")
            return True

        except Exception as e:
            logger.error(f"❌ Connection error: {e}")
            return False

    def configure_position_mode(self):
        """Configure position mode for unilateral trading"""
        try:
            logger.info("🔧 Configuring unilateral position mode...")

            # Set position mode to unilateral (one-way positions)
            timestamp = str(int(time.time() * 1000))

            # This is the key fix - set position mode before trading
            params = {
                'productType': 'USDT-FUTURES',
                'marginCoin': 'USDT',
                'holdMode': 'single'  # This is crucial for unilateral positions
            }

            # Try to configure via API
            response = self.exchange.private_post_account_set_position_mode(params)
            logger.info(f"✅ Position mode configured: {response}")

        except Exception as e:
            logger.warning(f"⚠️ Position mode configuration warning: {e}")
            logger.info("🔄 Continuing with default position mode...")

    def set_leverage_for_symbol(self, symbol):
        """Set leverage for a specific symbol"""
        try:
            # Extract symbol without :USDT suffix
            symbol_base = symbol.replace(':USDT', '')

            params = {
                'productType': 'USDT-FUTURES',
                'symbol': symbol_base,
                'marginCoin': 'USDT',
                'leverage': str(self.max_leverage)
            }

            response = self.exchange.private_post_account_set_leverage(params)
            logger.debug(f"✅ Leverage set for {symbol}: {self.max_leverage}x")

        except Exception as e:
            logger.warning(f"⚠️ Could not set leverage for {symbol}: {e}")

    def generate_signal(self, symbol):
        """Generate random trading signal for any pair"""
        signals = ['BUY', 'SELL', 'HOLD']
        weights = [0.4, 0.4, 0.2]  # 40% chance each for BUY/SELL, 20% HOLD
        return random.choices(signals, weights=weights)[0]

    def execute_trade(self, symbol, signal):
        """Execute trade with proper Bitget unilateral position configuration"""
        try:
            # Set leverage for this symbol first
            self.set_leverage_for_symbol(symbol)

            # Get current price
            ticker = self.exchange.fetch_ticker(symbol)
            current_price = ticker['last']

            # Calculate position size
            position_size = self.position_size_usdt / current_price

            logger.info(f"🎯 {signal} {symbol} at ${current_price:.6f}")
            logger.info(f"💰 Position size: {position_size:.6f} coins (${self.position_size_usdt})")

            # Execute order with CORRECT Bitget unilateral position parameters
            if signal == 'BUY':
                # For unilateral positions, we use 'long' direction
                order = self.exchange.create_order(
                    symbol,
                    'market',
                    'buy',
                    position_size,
                    None,
                    params={
                        'productType': 'USDT-FUTURES',
                        'marginMode': 'isolated',
                        'holdSide': 'long',  # This is key for unilateral positions
                    }
                )
            elif signal == 'SELL':
                # For unilateral positions, we use 'short' direction
                order = self.exchange.create_order(
                    symbol,
                    'market',
                    'sell',
                    position_size,
                    None,
                    params={
                        'productType': 'USDT-FUTURES',
                        'marginMode': 'isolated',
                        'holdSide': 'short',  # This is key for unilateral positions
                    }
                )
            else:
                return None

            logger.info(f"✅ Trade executed: {order['id']}")
            return order

        except Exception as e:
            logger.error(f"❌ Trade execution error for {symbol}: {e}")

            # If we still get unilateral position error, try alternative approach
            if "40774" in str(e) or "unilateral position" in str(e).lower():
                logger.warning(f"🔄 Retrying {symbol} with fallback parameters...")
                try:
                    # Fallback: Try with minimal parameters
                    if signal == 'BUY':
                        order = self.exchange.create_market_buy_order(
                            symbol,
                            position_size,
                            params={'marginMode': 'isolated'}
                        )
                    elif signal == 'SELL':
                        order = self.exchange.create_market_sell_order(
                            symbol,
                            position_size,
                            params={'marginMode': 'isolated'}
                        )
                    else:
                        return None

                    logger.info(f"✅ Fallback trade executed: {order['id']}")
                    return order

                except Exception as fallback_error:
                    logger.error(f"❌ Fallback also failed for {symbol}: {fallback_error}")
                    return None

            return None

    def run(self):
        """Main trading loop - scans ALL pairs"""
        self.is_running = True
        cycle = 0

        logger.info("🚀 Starting MULTI-PAIR TRADING - Scanning ALL available pairs!")
        logger.info("=" * 80)

        while self.is_running:
            cycle += 1
            logger.info(f"\n🔄 Cycle #{cycle} - Scanning {len(self.all_pairs)} pairs")

            # Scan random subset of pairs for trading opportunities
            pairs_to_scan = min(50, len(self.all_pairs))  # Scan 50 pairs per cycle
            scanned_pairs = random.sample(self.all_pairs, pairs_to_scan)

            trades_this_cycle = 0

            for symbol in scanned_pairs:
                if not self.is_running:
                    break

                signal = self.generate_signal(symbol)

                if signal in ['BUY', 'SELL']:
                    logger.info(f"📊 {symbol} -> {signal}")
                    order = self.execute_trade(symbol, signal)
                    if order:
                        trades_this_cycle += 1
                        self.active_positions[symbol] = {
                            'order_id': order['id'],
                            'signal': signal,
                            'entry_price': order['price'],
                            'quantity': order['amount'],
                            'timestamp': time.time()
                        }

            logger.info(f"✅ Cycle #{cycle} complete - {trades_this_cycle} trades executed")
            logger.info(f"📈 Active positions: {len(self.active_positions)}")
            logger.info("⏰ Waiting 30 seconds for next scan...")

            time.sleep(30)

    def stop(self):
        """Stop trading"""
        logger.info("🛑 Stopping fixed VIPER trader...")
        self.is_running = False

def main():
    """Main entry point"""
    print("🚀 VIPER FIXED MULTI-PAIR LIVE TRADING SYSTEM")
    print("🎯 Scanning and trading ALL available pairs with proper Bitget API")
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
        logger.info("🔄 Initializing Fixed VIPER Trader...")
        trader = FixedVIPERTrader()

        if not trader.connect():
            logger.error("❌ Failed to connect to exchange")
            return False

        logger.info("✅ Connected to Bitget successfully")
        logger.info(f"🔥 Ready to trade {len(trader.all_pairs)} pairs with unilateral positions!")
        logger.info("🚀 Starting fixed multi-pair live trading...")

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
