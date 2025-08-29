#!/usr/bin/env python3
"""
🚀 VIPER BITGET UNLIMITED TRADER
NO LIMITS - NO MINIMUM BALANCE - NO DAILY LOSS LIMITS
PURE AGGRESSIVE TRADING WITH 50X LEVERAGE
"""

import os
import time
import logging
import ccxt
import json
import random
from datetime import datetime
from typing import List, Dict, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BitgetUnlimitedTrader:
    """Unlimited trader - NO BALANCE LIMITS - NO LOSS LIMITS"""

    def __init__(self):
        self.exchange = None
        self.real_balance = 0.0
        self.swap_pairs_50x = []
        self.active_positions = {}
        self.is_running = False
        self.total_trades = 0
        self.total_pnl = 0.0
        
        # Bitget API configuration
        self.api_key = os.getenv('BITGET_API_KEY', 'bg_d20a392139710bc38b8ab39e970114eb')
        self.api_secret = os.getenv('BITGET_API_SECRET', '23ed4a7fe10b9c947d41a15223647f1b263f0d932b7d5e9e7bdfac01d3b84b36')
        self.api_password = os.getenv('BITGET_API_PASSWORD', '22672267')
        
        # AGGRESSIVE trading configuration - NO LIMITS
        self.max_leverage = 50
        self.risk_per_trade = 0.05  # 5% risk per trade - AGGRESSIVE
        self.max_positions = 50     # 50 positions - NO LIMITS
        
        # AGGRESSIVE coin groups - MAXIMUM LEVERAGE EVERYWHERE
        self.aggressive_groups = {
            "high_vol_majors": {
                "coins": ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT"],
                "leverage": 50,  # MAX LEVERAGE
                "risk_pct": 0.08,  # 8% risk - AGGRESSIVE
                "take_profit": 0.15,  # 15% TP - BIG MOVES
                "stop_loss": 0.10,   # 10% SL - WIDE STOPS
            },
            "meme_explosion": {
                "coins": ["DOGEUSDT", "SHIBUSDT", "PEPEUSDT", "WIFUSDT", "BONKUSDT"],
                "leverage": 50,  # MAX LEVERAGE
                "risk_pct": 0.10,  # 10% risk - MAXIMUM
                "take_profit": 0.30,  # 30% TP - HUGE MOVES
                "stop_loss": 0.15,   # 15% SL - WIDE
            },
            "defi_explosive": {
                "coins": ["LINKUSDT", "UNIUSDT", "AAVEUSDT", "COMPUSDT", "CRVUSDT"],
                "leverage": 50,  # MAX LEVERAGE
                "risk_pct": 0.07,  # 7% risk
                "take_profit": 0.20,  # 20% TP
                "stop_loss": 0.12,   # 12% SL
            },
            "new_narrative": {
                "coins": ["ORDIUSDT", "INJUSDT", "SUIUSDT", "APTOS", "SEUSDT"],
                "leverage": 50,  # MAX LEVERAGE
                "risk_pct": 0.12,  # 12% risk - MAXIMUM AGGRESSION
                "take_profit": 0.50,  # 50% TP - MASSIVE MOVES
                "stop_loss": 0.20,   # 20% SL - VERY WIDE
            },
            "volatile_alts": {
                "coins": ["LTCUSDT", "XRPUSDT", "TRXUSDT", "DOTUSDT", "AVAXUSDT"],
                "leverage": 50,  # MAX LEVERAGE
                "risk_pct": 0.06,  # 6% risk
                "take_profit": 0.18,  # 18% TP
                "stop_loss": 0.10,   # 10% SL
            }
        }
        
        logger.info("🚀 VIPER BITGET UNLIMITED TRADER INITIALIZED")
        logger.info("⚡ NO LIMITS - NO MINIMUM BALANCE - NO LOSS LIMITS")
        logger.info("💥 PURE AGGRESSIVE 50X LEVERAGE TRADING")
        logger.info(f"🎯 Max Leverage: {self.max_leverage}x EVERYWHERE")
        logger.info(f"🔢 Max Positions: {self.max_positions} (NO LIMIT)")

    def connect_bitget_unlimited(self):
        """Connect to Bitget - NO RESTRICTIONS"""
        try:
            logger.info("🔌 Connecting to Bitget for UNLIMITED trading...")
            self.exchange = ccxt.bitget({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'password': self.api_password,
                'options': {
                    'defaultType': 'swap',  # Futures only
                    'adjustForTimeDifference': True,
                    'hedgeMode': True,  # Enable hedge mode for proper position management
                },
                'sandbox': False,  # LIVE TRADING ONLY
            })
            
            # Load all markets
            markets = self.exchange.load_markets()
            logger.info(f"✅ Bitget Connected - {len(markets)} markets loaded")
            
            # Get ALL swap pairs with any leverage
            for symbol, market in markets.items():
                if (market.get('type') == 'swap' and 
                    market.get('active', False) and
                    market.get('quote') == 'USDT'):
                    self.swap_pairs_50x.append(symbol)
            
            logger.info(f"💥 Found {len(self.swap_pairs_50x)} USDT swap pairs for trading")
            
            return True

        except Exception as e:
            logger.error(f"❌ Failed to connect to Bitget: {e}")
            return False

    def get_real_balance_unlimited(self):
        """Get real balance - TRADE WITH ANY AMOUNT"""
        try:
            logger.info("💰 Fetching REAL balance for UNLIMITED trading...")
            
            balance = self.exchange.fetch_balance({'type': 'swap'})
            
            if 'USDT' in balance:
                self.real_balance = float(balance['USDT']['free'])
                logger.info(f"✅ REAL Balance: ${self.real_balance:,.2f}")
                logger.info("💥 TRADING WITH ANY AMOUNT - NO MINIMUM LIMITS!")
                
                return self.real_balance
            else:
                logger.warning("⚠️  No USDT found but CONTINUING ANYWAY!")
                self.real_balance = 0.0
                return 0.0
                
        except Exception as e:
            logger.error(f"❌ Error fetching balance: {e}")
            logger.info("💥 CONTINUING WITHOUT BALANCE CHECK!")
            return 0.0

    def calculate_aggressive_position(self, symbol: str, price: float) -> Dict:
        """Calculate AGGRESSIVE position size - NO LIMITS"""
        try:
            # Find group for symbol
            group_config = None
            for group_name, config in self.aggressive_groups.items():
                if symbol in config['coins']:
                    group_config = config
                    break
            
            if not group_config:
                # Default AGGRESSIVE settings for unknown symbols
                group_config = {
                    "leverage": 50,
                    "risk_pct": 0.05,  # 5% default risk
                    "take_profit": 0.10,
                    "stop_loss": 0.08
                }
            
            # AGGRESSIVE position calculation
            risk_amount = max(self.real_balance * group_config['risk_pct'], 10)  # Minimum $10 position
            position_value = risk_amount * group_config['leverage']
            position_size = position_value / price
            margin_required = position_value / group_config['leverage']
            
            # OVERRIDE: If balance is low, use minimum viable position
            if margin_required > self.real_balance and self.real_balance < 50:
                margin_required = max(self.real_balance * 0.9, 1)  # Use 90% of balance, minimum $1
                position_value = margin_required * group_config['leverage']
                position_size = position_value / price
            
            return {
                "symbol": symbol,
                "group": group_config,
                "position_size": position_size,
                "margin_required": margin_required,
                "leverage": group_config['leverage'],
                "risk_amount": risk_amount,
                "take_profit": group_config['take_profit'],
                "stop_loss": group_config['stop_loss']
            }
            
        except Exception as e:
            logger.error(f"❌ Error calculating position for {symbol}: {e}")
            # Return minimum viable position anyway
            return {
                "symbol": symbol,
                "position_size": 0.001,
                "margin_required": 1.0,
                "leverage": 50,
                "risk_amount": 1.0,
                "take_profit": 0.10,
                "stop_loss": 0.08
            }

    def execute_unlimited_trade(self, symbol: str, side: str) -> Optional[Dict]:
        """Execute trade with NO LIMITS"""
        try:
            # Get current price
            ticker = self.exchange.fetch_ticker(symbol)
            current_price = ticker['last']
            
            # Calculate AGGRESSIVE position
            position_calc = self.calculate_aggressive_position(symbol, current_price)
            
            logger.info(f"💥 EXECUTING UNLIMITED {side.upper()} ORDER")
            logger.info(f"   📊 Symbol: {symbol}")
            logger.info(f"   💰 Price: ${current_price:.6f}")
            logger.info(f"   📏 Size: {position_calc['position_size']:.6f}")
            logger.info(f"   🔄 Leverage: {position_calc['leverage']}x")
            logger.info(f"   💵 Margin: ${position_calc['margin_required']:.2f}")
            logger.info(f"   🎯 NO BALANCE CHECKS - EXECUTING ANYWAY!")
            
            # Execute REAL order with correct Bitget parameters
            order = self.exchange.create_order(
                symbol=symbol,
                type='market',
                side=side,
                amount=position_calc['position_size'],
                params={
                    'leverage': position_calc['leverage'],
                    'marginMode': 'isolated',
                    'holdSide': 'long' if side == 'buy' else 'short',  # Required for hedge mode
                    'tradeSide': 'open'  # Open position
                }
            )
            
            logger.info(f"✅ UNLIMITED ORDER EXECUTED: {order['id']}")
            
            # Store position
            self.active_positions[symbol] = {
                'order_id': order['id'],
                'side': side,
                'size': position_calc['position_size'],
                'entry_price': current_price,
                'leverage': position_calc['leverage'],
                'margin': position_calc['margin_required'],
                'take_profit': position_calc['take_profit'],
                'stop_loss': position_calc['stop_loss'],
                'timestamp': datetime.now().isoformat()
            }
            
            self.total_trades += 1
            
            return order
            
        except Exception as e:
            logger.error(f"❌ Failed to execute UNLIMITED trade for {symbol}: {e}")
            logger.info("💥 CONTINUING DESPITE ERROR!")
            return None

    def monitor_unlimited_positions(self):
        """Monitor positions - AGGRESSIVE MANAGEMENT"""
        try:
            if not self.active_positions:
                return
                
            logger.info(f"👁️  Monitoring {len(self.active_positions)} UNLIMITED positions...")
            
            # Get current positions
            try:
                positions = self.exchange.fetch_positions()
            except Exception as e:
                logger.warning(f"⚠️  Could not fetch positions: {e}")
                return
            
            for position in positions:
                symbol = position['symbol']
                if symbol in self.active_positions and position.get('contracts', 0) > 0:
                    unrealized_pnl = position.get('unrealizedPnl', 0) or 0
                    percentage = position.get('percentage', 0) or 0
                    
                    position_config = self.active_positions[symbol]
                    take_profit_pct = position_config['take_profit'] * 100
                    stop_loss_pct = position_config['stop_loss'] * 100
                    
                    logger.info(f"📊 {symbol}: P&L ${unrealized_pnl:.2f} ({percentage:.2f}%)")
                    
                    # AGGRESSIVE profit taking
                    if percentage >= take_profit_pct:
                        logger.info(f"💰 BIG PROFIT! Closing {symbol} at {percentage:.2f}%")
                        self.close_unlimited_position(symbol, "HUGE_PROFIT")
                        
                    # AGGRESSIVE stop loss (wider than normal)
                    elif percentage <= -stop_loss_pct:
                        logger.info(f"🛑 STOP LOSS! Closing {symbol} at {percentage:.2f}%")
                        self.close_unlimited_position(symbol, "STOP_LOSS")
                        
        except Exception as e:
            logger.error(f"❌ Error monitoring positions: {e}")

    def close_unlimited_position(self, symbol: str, reason: str):
        """Close position - NO RESTRICTIONS"""
        try:
            if symbol not in self.active_positions:
                return
                
            position_info = self.active_positions[symbol]
            
            # Close with opposite order
            opposite_side = 'sell' if position_info['side'] == 'buy' else 'buy'
            
            close_order = self.exchange.create_order(
                symbol=symbol,
                type='market',
                side=opposite_side,
                amount=position_info['size'],
                params={
                    'holdSide': 'long' if position_info['side'] == 'buy' else 'short',
                    'tradeSide': 'close'  # Close position
                }
            )
            
            logger.info(f"✅ POSITION CLOSED: {symbol} ({reason})")
            logger.info(f"   📋 Close Order: {close_order['id']}")
            
            # Remove from active positions
            del self.active_positions[symbol]
            
        except Exception as e:
            logger.error(f"❌ Error closing position {symbol}: {e}")
            # Force remove from tracking anyway
            if symbol in self.active_positions:
                del self.active_positions[symbol]

    def scan_unlimited_opportunities(self) -> List[str]:
        """Scan for UNLIMITED trading opportunities"""
        opportunities = []
        
        try:
            # Get random sample of pairs for speed
            pairs_to_check = random.sample(
                self.swap_pairs_50x,
                min(30, len(self.swap_pairs_50x))
            )
            
            for symbol in pairs_to_check:
                # Skip if already have position
                if symbol in self.active_positions:
                    continue
                
                try:
                    ticker = self.exchange.fetch_ticker(symbol)
                    change_24h = abs(ticker.get('percentage', 0) or 0)
                    volume = ticker.get('quoteVolume', 0) or 0
                    
                    # AGGRESSIVE opportunity criteria - ANY movement
                    if volume > 50000 or change_24h > 0.5:  # $50k+ volume OR >0.5% movement
                        opportunities.append(symbol)
                        
                except Exception as e:
                    continue
                    
        except Exception as e:
            logger.error(f"❌ Error scanning opportunities: {e}")
            
        return opportunities

    def run_unlimited_trading(self):
        """Run UNLIMITED trading system"""
        logger.info("💥 STARTING UNLIMITED TRADING SYSTEM")
        logger.info("🚨 NO LIMITS - NO BALANCE MINIMUMS - NO LOSS LIMITS")
        logger.info("⚡ PURE AGGRESSIVE 50X LEVERAGE TRADING")
        self.is_running = True
        
        # Get balance but continue regardless
        self.get_real_balance_unlimited()
        
        logger.info("=" * 100)
        logger.info("💥 VIPER UNLIMITED TRADING SYSTEM ACTIVATED!")
        logger.info("📊 System Status: UNLIMITED LIVE TRADING")
        logger.info(f"💰 Balance: ${self.real_balance:.2f} (TRADING REGARDLESS)")
        logger.info(f"🎯 Available Pairs: {len(self.swap_pairs_50x)}")
        logger.info(f"📊 Aggressive Groups: {len(self.aggressive_groups)}")
        logger.info("💥 NO DAILY LOSS LIMITS - NO POSITION LIMITS")
        logger.info("=" * 100)
        
        cycle_count = 0
        
        while self.is_running:  # UNLIMITED CYCLES
            try:
                cycle_count += 1
                logger.info(f"\n💥 UNLIMITED CYCLE #{cycle_count}")
                
                # Update balance
                self.get_real_balance_unlimited()
                
                # Monitor existing positions
                self.monitor_unlimited_positions()
                
                # Always look for new opportunities - NO POSITION LIMITS
                opportunities = self.scan_unlimited_opportunities()
                
                if opportunities:
                    logger.info(f"🎯 Found {len(opportunities)} opportunities: {opportunities[:10]}")
                    
                    # Execute multiple trades per cycle - AGGRESSIVE
                    for symbol in opportunities[:5]:  # Up to 5 new trades per cycle
                        try:
                            ticker = self.exchange.fetch_ticker(symbol)
                            change_24h = ticker.get('percentage', 0) or 0
                            
                            # Trade direction based on momentum + randomness for volatility
                            if abs(change_24h) > 2:  # Strong momentum
                                side = 'buy' if change_24h > 0 else 'sell'
                            else:  # Random direction for low momentum
                                side = random.choice(['buy', 'sell'])
                            
                            order = self.execute_unlimited_trade(symbol, side)
                            if order:
                                time.sleep(1)  # Brief pause
                                
                        except Exception as e:
                            logger.error(f"❌ Error trading {symbol}: {e}")
                            continue
                
                # Status update
                logger.info(f"💥 Balance: ${self.real_balance:.2f} | "
                          f"Active: {len(self.active_positions)} | "
                          f"Total Trades: {self.total_trades}")
                
                # Aggressive cycle timing
                logger.info("⏰ Next cycle in 30 seconds...")
                time.sleep(30)
                
            except KeyboardInterrupt:
                logger.info("🛑 UNLIMITED trading interrupted")
                break
            except Exception as e:
                logger.error(f"❌ Error in UNLIMITED cycle: {e}")
                logger.info("💥 CONTINUING DESPITE ERROR!")
                time.sleep(10)
        
        logger.info("🏁 UNLIMITED trading system stopped")

def main():
    """Main function"""
    logger.info("💥 VIPER BITGET UNLIMITED TRADER STARTING...")
    logger.info("🚨 WARNING: UNLIMITED TRADING - NO SAFETY LIMITS")
    
    trader = BitgetUnlimitedTrader()
    
    if not trader.connect_bitget_unlimited():
        logger.error("❌ Failed to connect to Bitget.")
        logger.info("💥 ATTEMPTING TO CONTINUE ANYWAY...")
    
    logger.info("\n" + "="*100)
    logger.info("🚨 FINAL WARNING: UNLIMITED TRADING WITH NO LIMITS!")
    logger.info("   - NO MINIMUM BALANCE REQUIREMENTS")
    logger.info("   - NO DAILY LOSS LIMITS") 
    logger.info("   - NO POSITION LIMITS")
    logger.info("   - AGGRESSIVE 50X LEVERAGE ON ALL TRADES")
    logger.info("   Press Ctrl+C within 5 seconds to cancel...")
    logger.info("="*100)
    
    try:
        time.sleep(5)
        trader.run_unlimited_trading()
    except KeyboardInterrupt:
        logger.info("🛑 UNLIMITED trading cancelled")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        logger.info("💥 SYSTEM WILL CONTINUE DESPITE ERRORS!")
    
    logger.info("✅ UNLIMITED trader shutdown complete")

if __name__ == "__main__":
    main()
