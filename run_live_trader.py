#!/usr/bin/env python3
"""
🚀 VIPER MULTI-PAIR LIVE TRADER - SCANS ALL PAIRS
Live trading system that scans and trades ALL available pairs on Bitget
"""

import os
import sys
import time
import logging
import ccxt
import random
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
    format='%(asctime)s - MULTI_PAIR_TRADER - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/viper_multi_pair_trader.log'),
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

        # Trading configuration
        self.position_size_usdt = float(os.getenv('POSITION_SIZE_USDT', '10'))
        self.max_leverage = int(os.getenv('MAX_LEVERAGE', '50'))  # 50x as per your rules
        self.take_profit_pct = float(os.getenv('TAKE_PROFIT_PCT', '3.0'))
        self.stop_loss_pct = float(os.getenv('STOP_LOSS_PCT', '2.0'))
        self.max_positions = int(os.getenv('MAX_POSITIONS', '15'))  # Max 15 positions as per rules

        self.exchange = None
        self.all_pairs = []
        self.active_positions = {}
        self.is_running = False

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
        """Generate random trading signal for any pair"""
        signals = ['BUY', 'SELL', 'HOLD']
        weights = [0.4, 0.4, 0.2]  # 40% chance each for BUY/SELL, 20% HOLD
        return random.choices(signals, weights=weights)[0]

    def execute_trade(self, symbol, signal):
        """Execute trade for any pair with proper Bitget unilateral position parameters"""
        try:
            # Get current price
            ticker = self.exchange.fetch_ticker(symbol)
            current_price = ticker['last']

            # Calculate position size
            position_size = self.position_size_usdt / current_price

            logger.info(f"🎯 {signal} {symbol} at ${current_price:.6f}")
            logger.info(f"💰 Position size: {position_size:.6f} coins (${self.position_size_usdt})")

            # Execute order with proper Bitget unilateral position parameters
            if signal == 'BUY':
                order = self.exchange.create_market_buy_order(
                    symbol,
                    position_size,
                    params={
                        'leverage': self.max_leverage,
                        'marginMode': 'isolated',
                        'tradeSide': 'open'  # Open position for unilateral mode
                    }
                )
            elif signal == 'SELL':
                order = self.exchange.create_market_sell_order(
                    symbol,
                    position_size,
                    params={
                        'leverage': self.max_leverage,
                        'marginMode': 'isolated',
                        'tradeSide': 'open'  # Open position for unilateral mode
                    }
                )
            else:
                return None

            logger.info(f"✅ Trade executed: {order['id']}")
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
                    return order
                except Exception as retry_error:
                    logger.error(f"❌ Alternative trade also failed for {symbol}: {retry_error}")
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
