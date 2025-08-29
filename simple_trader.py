#!/usr/bin/env python3
"""
🚀 SIMPLE VIPER TRADER - WORKING VERSION
"""

import os
import sys
import time
import logging
import ccxt
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleVIPERTrader:
    def __init__(self):
        # Load API credentials
        self.api_key = os.getenv('BITGET_API_KEY', '')
        self.api_secret = os.getenv('BITGET_API_SECRET', '')
        self.api_password = os.getenv('BITGET_API_PASSWORD', '')
        
        # Trading config
        self.symbol = "BTC/USDT:USDT"
        self.position_size_usdt = float(os.getenv('POSITION_SIZE_USDT', '10'))
        self.max_leverage = int(os.getenv('MAX_LEVERAGE', '20'))
        self.take_profit_pct = float(os.getenv('TAKE_PROFIT_PCT', '3.0'))
        self.stop_loss_pct = float(os.getenv('STOP_LOSS_PCT', '2.0'))
        
        self.exchange = None
        self.active_positions = {}
        self.is_running = False

    def connect(self):
        """Connect to Bitget"""
        try:
            if not all([self.api_key, self.api_secret, self.api_password]):
                logger.error("❌ Missing API credentials")
                return False
                
            logger.info("🔌 Connecting to Bitget...")
            self.exchange = ccxt.bitget({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'password': self.api_password,
                'options': {'defaultType': 'swap', 'adjustForTimeDifference': True},
                'sandbox': False,
            })
            
            markets = self.exchange.load_markets()
            logger.info(f"✅ Connected - {len(markets)} markets loaded")
            return True
            
        except Exception as e:
            logger.error(f"❌ Connection failed: {e}")
            return False

    def get_signal(self):
        """Simple signal generation"""
        try:
            ticker = self.exchange.fetch_ticker(self.symbol)
            price = ticker['last']
            change_24h = ticker.get('percentage', 0)
            
            # Simple momentum signal
            if change_24h > 1.0:
                return 'buy'
            elif change_24h < -1.0:
                return 'sell'
            return None
            
        except Exception as e:
            logger.error(f"❌ Signal error: {e}")
            return None

    def execute_trade(self, side):
        """Execute trade"""
        try:
            ticker = self.exchange.fetch_ticker(self.symbol)
            current_price = ticker['last']
            
            # Calculate position size
            position_value = self.position_size_usdt * self.max_leverage
            position_size = position_value / current_price
            
            logger.info(f"🚀 Executing {side.upper()} order")
            logger.info(f"   Price: ${current_price:.6f}")
            logger.info(f"   Size: {position_size:.6f}")
            
            order = self.exchange.create_order(
                symbol=self.symbol,
                type='market',
                side=side,
                amount=position_size,
                params={
                    'marginCoin': 'USDT',
                    'leverage': self.max_leverage,
                    'marginMode': 'isolated',
                    'holdSide': 'long' if side == 'buy' else 'short',
                    'tradeSide': 'open'
                }
            )
            
            # Store position
            self.active_positions[self.symbol] = {
                'side': side,
                'entry_price': current_price,
                'size': position_size,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"✅ Trade executed: {order['id']}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Trade failed: {e}")
            return False

    def monitor_positions(self):
        """Monitor positions"""
        if not self.active_positions:
            return
            
        try:
            ticker = self.exchange.fetch_ticker(self.symbol)
            current_price = ticker['last']
            
            position = self.active_positions[self.symbol]
            entry_price = position['entry_price']
            side = position['side']
            
            # Calculate P&L
            if side == 'buy':
                pnl_pct = (current_price - entry_price) / entry_price * 100
            else:
                pnl_pct = (entry_price - current_price) / entry_price * 100
            
            logger.info(f"📊 {self.symbol}: {pnl_pct:.2f}% P&L")
            
            # Risk management
            if pnl_pct >= self.take_profit_pct:
                logger.info(f"💰 Taking profit ({pnl_pct:.1f}%)")
                self.close_position("PROFIT")
            elif pnl_pct <= -self.stop_loss_pct:
                logger.info(f"🛑 Stopping loss ({pnl_pct:.1f}%)")
                self.close_position("STOP_LOSS")
                
        except Exception as e:
            logger.error(f"❌ Monitor error: {e}")

    def close_position(self, reason):
        """Close position"""
        try:
            if self.symbol not in self.active_positions:
                return
                
            position = self.active_positions[self.symbol]
            opposite_side = 'sell' if position['side'] == 'buy' else 'buy'
            
            close_order = self.exchange.create_order(
                symbol=self.symbol,
                type='market',
                side=opposite_side,
                amount=position['size'],
                params={
                    'marginCoin': 'USDT',
                    'holdSide': 'long' if position['side'] == 'buy' else 'short',
                    'tradeSide': 'close'
                }
            )
            
            logger.info(f"✅ Position closed: {self.symbol} ({reason})")
            del self.active_positions[self.symbol]
            
        except Exception as e:
            logger.error(f"❌ Close error: {e}")

    def run(self):
        """Main trading loop"""
        logger.info("🚀 Starting VIPER Trading Bot")
        logger.info("=" * 60)
        
        self.is_running = True
        cycle_count = 0
        
        try:
            while self.is_running:
                cycle_count += 1
                logger.info(f"\n🔄 Cycle #{cycle_count}")
                
                # Monitor positions
                self.monitor_positions()
                
                # Look for new opportunities (if no position)
                if self.symbol not in self.active_positions:
                    signal = self.get_signal()
                    if signal:
                        logger.info(f"🎯 Signal: {signal.upper()}")
                        self.execute_trade(signal)
                        time.sleep(2)
                
                # Status
                logger.info(f"📊 Active positions: {len(self.active_positions)}")
                
                # Wait
                logger.info("⏰ Waiting 30 seconds...")
                time.sleep(30)
                
        except KeyboardInterrupt:
            logger.info("\n🛑 Trading stopped by user")
        finally:
            # Close all positions
            if self.active_positions:
                logger.info("🔄 Closing all positions...")
                self.close_position("SHUTDOWN")
            
            logger.info("✅ Trading bot shutdown complete")

def main():
    """Main entry point"""
    logger.info("🚀 VIPER TRADING BOT STARTING...")
    
    trader = SimpleVIPERTrader()
    
    if not trader.connect():
        logger.error("❌ Failed to connect")
        return
    
    logger.info("⏳ Starting in 3 seconds...")
    time.sleep(3)
    
    try:
        trader.run()
    except KeyboardInterrupt:
        logger.info("🛑 Cancelled by user")
    except Exception as e:
        logger.error(f"❌ Error: {e}")
    
    logger.info("✅ Bot shutdown complete")

if __name__ == "__main__":
    main()
