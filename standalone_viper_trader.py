#!/usr/bin/env python3
"""
🚀 VIPER Trading System - Standalone Trading Component
Complete Scan → Score → Trade → TP/SL Flow in One Script

Features:
- Market scanning for multiple trading pairs
- VIPER score calculation (Volume, Price, External, Range)  
- Automated trade execution with risk management
- Take Profit (TP) and Stop Loss (SL) monitoring
- Real-time position management
- Comprehensive logging and safety controls

Usage:
    python standalone_viper_trader.py

Requirements:
    - Bitget API credentials in .env file
    - Python packages: ccxt, requests, python-dotenv
"""

import os
import sys
import time
import json
import logging
import ccxt
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from dotenv import load_dotenv
import threading
import signal

# Load environment variables
load_dotenv()

@dataclass
class TradingPosition:
    """Data class for tracking active positions"""
    symbol: str
    side: str  # 'buy' or 'sell'
    size: float
    entry_price: float
    stop_loss: float
    take_profit: float
    timestamp: str
    order_id: str = None
    unrealized_pnl: float = 0.0

@dataclass 
class VIPERSignal:
    """Data class for VIPER trading signals"""
    symbol: str
    signal: str  # 'LONG', 'SHORT', or 'HOLD'
    viper_score: float
    price: float
    price_change: float
    volume: float
    confidence: float
    stop_loss: float
    take_profit: float
    timestamp: str

class StandaloneVIPERTrader:
    """Complete standalone VIPER trading system"""
    
    def __init__(self):
        """Initialize the trading system"""
        self.setup_logging()
        self.logger.info("🚀 Initializing VIPER Standalone Trader...")
        
        # Configuration
        self.config = self.load_configuration()
        self.exchange = None
        self.active_positions = {}  # {symbol: TradingPosition}
        self.running = True
        
        # Trading parameters
        self.max_positions = self.config.get('max_positions', 5)
        self.risk_per_trade = self.config.get('risk_per_trade', 0.02)  # 2%
        self.viper_threshold = self.config.get('viper_threshold', 85.0)
        self.scan_interval = self.config.get('scan_interval', 30)  # seconds
        
        # Trading pairs to monitor
        self.trading_pairs = [
            'BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT', 
            'BNB/USDT:USDT', 'ADA/USDT:USDT', 'DOT/USDT:USDT',
            'MATIC/USDT:USDT', 'AVAX/USDT:USDT', 'LINK/USDT:USDT'
        ]
        
        # Initialize exchange connection
        self.initialize_exchange()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
    def setup_logging(self):
        """Setup comprehensive logging"""
        log_format = '%(asctime)s | %(levelname)-8s | %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler('viper_trader.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_configuration(self) -> Dict:
        """Load trading configuration from environment"""
        return {
            'max_positions': int(os.getenv('MAX_POSITIONS', '5')),
            'risk_per_trade': float(os.getenv('RISK_PER_TRADE', '0.02')),
            'viper_threshold': float(os.getenv('VIPER_THRESHOLD', '85.0')),
            'scan_interval': int(os.getenv('SCAN_INTERVAL', '30')),
            'stop_loss_percent': float(os.getenv('STOP_LOSS_PERCENT', '0.02')),  # 2%
            'take_profit_percent': float(os.getenv('TAKE_PROFIT_PERCENT', '0.04')),  # 4%
            'daily_loss_limit': float(os.getenv('DAILY_LOSS_LIMIT', '0.05')),  # 5%
        }
    
    def initialize_exchange(self):
        """Initialize Bitget exchange connection"""
        try:
            self.logger.info("🔗 Connecting to Bitget exchange...")
            
            api_key = os.getenv('BITGET_API_KEY')
            api_secret = os.getenv('BITGET_API_SECRET') 
            api_password = os.getenv('BITGET_API_PASSWORD')
            
            if not all([api_key, api_secret, api_password]):
                raise ValueError("Missing Bitget API credentials in environment")
                
            self.exchange = ccxt.bitget({
                'apiKey': api_key,
                'secret': api_secret,
                'password': api_password,
                'sandbox': False,  # Set to True for testing
                'options': {
                    'defaultType': 'swap',  # Use perpetual futures
                    'adjustForTimeDifference': True,
                }
            })
            
            # Test connection
            self.exchange.load_markets()
            balance = self.exchange.fetch_balance()
            self.logger.info(f"✅ Exchange connected. Account balance: {balance.get('USDT', {}).get('free', 0)} USDT")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize exchange: {e}")
            raise
    
    def fetch_market_data(self, symbol: str) -> Optional[Dict]:
        """Fetch comprehensive market data for a symbol"""
        try:
            # Get ticker data
            ticker = self.exchange.fetch_ticker(symbol)
            
            # Get recent OHLCV data for additional analysis
            ohlcv = self.exchange.fetch_ohlcv(symbol, '1h', limit=24)
            recent_candle = ohlcv[-1] if ohlcv else None
            
            if not ticker or not recent_candle:
                return None
                
            return {
                'symbol': symbol,
                'price': ticker['last'],
                'high': ticker['high'],
                'low': ticker['low'],
                'volume': ticker['baseVolume'],
                'price_change': ticker['percentage'],
                'bid': ticker['bid'],
                'ask': ticker['ask'],
                'spread': (ticker['ask'] - ticker['bid']) / ticker['last'] * 100 if ticker['ask'] and ticker['bid'] else 0,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error fetching market data for {symbol}: {e}")
            return None
    
    def calculate_viper_score(self, market_data: Dict) -> float:
        """
        Calculate VIPER score using Volume, Price, External, Range factors
        Returns score from 0-100 (higher = better opportunity)
        """
        try:
            volume = market_data.get('volume', 0)
            price_change = abs(market_data.get('price_change', 0))
            high = market_data.get('high', 1)
            low = market_data.get('low', 0) 
            current_price = market_data.get('price', 1)
            spread = market_data.get('spread', 0)
            
            # Volume Score (0-100): Higher volume = better liquidity
            volume_score = min((volume / 1_000_000) * 25, 100)  # Scale by 1M volume
            
            # Price Score (0-100): Momentum strength  
            price_score = min(price_change * 20, 100)  # Scale by price change %
            
            # External Score (0-100): Lower spread = better execution
            external_score = max(100 - (spread * 1000), 0)  # Penalize high spreads
            
            # Range Score (0-100): Volatility within reasonable bounds
            if current_price > 0:
                volatility = ((high - low) / current_price) * 100
                range_score = min(volatility * 10, 100) if volatility > 0.5 else volatility * 5
            else:
                range_score = 0
            
            # Weighted VIPER score
            viper_score = (
                volume_score * 0.30 +     # 30% volume weight
                price_score * 0.35 +      # 35% momentum weight  
                external_score * 0.20 +   # 20% execution cost weight
                range_score * 0.15        # 15% volatility weight
            )
            
            return min(max(viper_score, 0), 100)  # Clamp to 0-100 range
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating VIPER score: {e}")
            return 0.0
    
    def generate_signal(self, symbol: str, market_data: Dict) -> Optional[VIPERSignal]:
        """Generate trading signal based on VIPER score and market conditions"""
        try:
            viper_score = self.calculate_viper_score(market_data)
            
            if viper_score < self.viper_threshold:
                return None  # Score too low for trading
                
            price_change = market_data.get('price_change', 0)
            current_price = market_data.get('price', 0)
            volume = market_data.get('volume', 0)
            
            # Determine signal direction based on momentum
            signal = None
            if price_change > 1.0:  # Strong upward momentum (>1%)
                signal = "LONG"
            elif price_change < -1.0:  # Strong downward momentum (<-1%)
                signal = "SHORT"
            else:
                return None  # Insufficient momentum
            
            # Calculate stop loss and take profit levels
            stop_loss_pct = self.config['stop_loss_percent']
            take_profit_pct = self.config['take_profit_percent']
            
            if signal == "LONG":
                stop_loss = current_price * (1 - stop_loss_pct)
                take_profit = current_price * (1 + take_profit_pct)
            else:  # SHORT
                stop_loss = current_price * (1 + stop_loss_pct)
                take_profit = current_price * (1 - take_profit_pct)
            
            return VIPERSignal(
                symbol=symbol,
                signal=signal,
                viper_score=viper_score,
                price=current_price,
                price_change=price_change,
                volume=volume,
                confidence=min(viper_score / 100, 1.0),
                stop_loss=stop_loss,
                take_profit=take_profit,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error generating signal for {symbol}: {e}")
            return None
    
    def scan_markets(self) -> List[VIPERSignal]:
        """Scan all trading pairs for opportunities"""
        self.logger.info(f"🔍 Scanning {len(self.trading_pairs)} trading pairs...")
        
        signals = []
        for symbol in self.trading_pairs:
            try:
                market_data = self.fetch_market_data(symbol)
                if market_data:
                    signal = self.generate_signal(symbol, market_data)
                    if signal:
                        signals.append(signal)
                        self.logger.info(f"📊 {symbol}: VIPER Score {signal.viper_score:.1f} → {signal.signal} Signal")
                        
            except Exception as e:
                self.logger.error(f"❌ Error scanning {symbol}: {e}")
                
        self.logger.info(f"🎯 Found {len(signals)} trading opportunities")
        return signals
    
    def calculate_position_size(self, signal: VIPERSignal) -> float:
        """Calculate position size based on risk management"""
        try:
            # Get account balance
            balance = self.exchange.fetch_balance()
            available_usdt = balance.get('USDT', {}).get('free', 0)
            
            if available_usdt <= 0:
                self.logger.warning("⚠️ Insufficient USDT balance")
                return 0
            
            # Risk amount per trade
            risk_amount = available_usdt * self.risk_per_trade
            
            # Calculate position size based on stop loss distance
            price_diff = abs(signal.price - signal.stop_loss)
            if price_diff <= 0:
                return 0
                
            position_value = risk_amount / (price_diff / signal.price)
            position_size = position_value / signal.price
            
            # Apply minimum size constraints
            market = self.exchange.market(signal.symbol)
            min_size = market['limits']['amount']['min'] or 0.001
            position_size = max(position_size, min_size)
            
            return round(position_size, 6)
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating position size: {e}")
            return 0
    
    def execute_trade(self, signal: VIPERSignal) -> bool:
        """Execute trade based on VIPER signal"""
        try:
            # Check position limits
            if len(self.active_positions) >= self.max_positions:
                self.logger.warning(f"⚠️ Maximum positions ({self.max_positions}) reached")
                return False
            
            # Check if already have position in this symbol
            if signal.symbol in self.active_positions:
                self.logger.info(f"⚠️ Already have position in {signal.symbol}")
                return False
            
            # Calculate position size
            position_size = self.calculate_position_size(signal)
            if position_size <= 0:
                self.logger.warning(f"⚠️ Invalid position size for {signal.symbol}")
                return False
            
            # Determine order side
            side = 'buy' if signal.signal == 'LONG' else 'sell'
            
            self.logger.info(f"🎯 Executing {signal.signal} trade for {signal.symbol}")
            self.logger.info(f"   Size: {position_size}, Price: ${signal.price:.2f}")
            self.logger.info(f"   Stop Loss: ${signal.stop_loss:.2f}, Take Profit: ${signal.take_profit:.2f}")
            
            # Execute market order
            order = self.exchange.create_order(
                symbol=signal.symbol,
                type='market',
                side=side,
                amount=position_size,
                price=None  # Market order
            )
            
            if order and order.get('id'):
                # Create position tracking
                position = TradingPosition(
                    symbol=signal.symbol,
                    side=side,
                    size=position_size,
                    entry_price=signal.price,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    timestamp=datetime.now().isoformat(),
                    order_id=order['id']
                )
                
                self.active_positions[signal.symbol] = position
                
                self.logger.info(f"✅ Trade executed successfully for {signal.symbol}")
                self.logger.info(f"   Order ID: {order['id']}")
                
                return True
            else:
                self.logger.error(f"❌ Trade execution failed for {signal.symbol}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error executing trade for {signal.symbol}: {e}")
            return False
    
    def monitor_positions(self):
        """Monitor active positions for TP/SL conditions"""
        if not self.active_positions:
            return
        
        self.logger.info(f"📊 Monitoring {len(self.active_positions)} active positions...")
        
        positions_to_close = []
        
        for symbol, position in self.active_positions.items():
            try:
                # Get current price
                ticker = self.exchange.fetch_ticker(symbol)
                current_price = ticker['last']
                
                # Calculate P&L
                if position.side == 'buy':
                    pnl = (current_price - position.entry_price) / position.entry_price
                else:  # sell
                    pnl = (position.entry_price - current_price) / position.entry_price
                
                position.unrealized_pnl = pnl
                
                self.logger.info(f"📈 {symbol} ({position.side.upper()}): "
                               f"${current_price:.2f} | P&L: {pnl*100:.2f}%")
                
                # Check stop loss condition
                stop_loss_triggered = False
                if position.side == 'buy' and current_price <= position.stop_loss:
                    stop_loss_triggered = True
                elif position.side == 'sell' and current_price >= position.stop_loss:
                    stop_loss_triggered = True
                
                # Check take profit condition  
                take_profit_triggered = False
                if position.side == 'buy' and current_price >= position.take_profit:
                    take_profit_triggered = True
                elif position.side == 'sell' and current_price <= position.take_profit:
                    take_profit_triggered = True
                
                # Close position if TP or SL triggered
                if stop_loss_triggered:
                    self.logger.info(f"🛑 Stop Loss triggered for {symbol}")
                    positions_to_close.append((symbol, 'stop_loss'))
                elif take_profit_triggered:
                    self.logger.info(f"🎯 Take Profit triggered for {symbol}")
                    positions_to_close.append((symbol, 'take_profit'))
                    
            except Exception as e:
                self.logger.error(f"❌ Error monitoring position {symbol}: {e}")
        
        # Close positions that hit TP/SL
        for symbol, reason in positions_to_close:
            self.close_position(symbol, reason)
    
    def close_position(self, symbol: str, reason: str = 'manual'):
        """Close an active position"""
        try:
            if symbol not in self.active_positions:
                self.logger.warning(f"⚠️ No active position found for {symbol}")
                return False
            
            position = self.active_positions[symbol]
            
            # Determine closing side (opposite of opening)
            close_side = 'sell' if position.side == 'buy' else 'buy'
            
            self.logger.info(f"🔄 Closing {symbol} position ({reason})")
            
            # Execute closing order
            order = self.exchange.create_order(
                symbol=symbol,
                type='market',
                side=close_side,
                amount=position.size,
                price=None
            )
            
            if order and order.get('id'):
                # Calculate final P&L
                current_price = self.exchange.fetch_ticker(symbol)['last']
                if position.side == 'buy':
                    final_pnl = (current_price - position.entry_price) / position.entry_price
                else:
                    final_pnl = (position.entry_price - current_price) / position.entry_price
                
                self.logger.info(f"✅ Position closed for {symbol}")
                self.logger.info(f"   Final P&L: {final_pnl*100:.2f}%")
                self.logger.info(f"   Reason: {reason}")
                
                # Remove from active positions
                del self.active_positions[symbol]
                
                return True
            else:
                self.logger.error(f"❌ Failed to close position for {symbol}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error closing position {symbol}: {e}")
            return False
    
    def print_status(self):
        """Print current trading status"""
        print("\n" + "="*80)
        print("🚀 VIPER STANDALONE TRADER STATUS")
        print("="*80)
        print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 Active Positions: {len(self.active_positions)}/{self.max_positions}")
        print(f"🎯 VIPER Threshold: {self.viper_threshold}")
        print(f"💰 Risk per Trade: {self.risk_per_trade*100:.1f}%")
        
        if self.active_positions:
            print("\n📈 ACTIVE POSITIONS:")
            print("-"*60)
            for symbol, position in self.active_positions.items():
                pnl_pct = position.unrealized_pnl * 100 if position.unrealized_pnl else 0
                pnl_icon = "🟢" if pnl_pct > 0 else "🔴" if pnl_pct < 0 else "⚪"
                print(f"  {pnl_icon} {symbol} | {position.side.upper()} | "
                      f"P&L: {pnl_pct:.2f}% | Entry: ${position.entry_price:.2f}")
        else:
            print("\n💤 No active positions")
        
        print("="*80)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info(f"🛑 Received signal {signum}, shutting down gracefully...")
        self.running = False
    
    def run(self):
        """Main trading loop"""
        self.logger.info("🚀 Starting VIPER Standalone Trading System...")
        self.logger.info(f"📊 Monitoring {len(self.trading_pairs)} trading pairs")
        self.logger.info(f"⏰ Scan interval: {self.scan_interval}s")
        
        try:
            while self.running:
                start_time = time.time()
                
                # Print current status
                self.print_status()
                
                # 1. SCAN: Look for trading opportunities
                signals = self.scan_markets()
                
                # 2. SCORE & TRADE: Execute trades for high-scoring signals
                for signal in signals:
                    if len(self.active_positions) >= self.max_positions:
                        self.logger.info(f"⚠️ Position limit reached, skipping new trades")
                        break
                    
                    self.logger.info(f"🎯 Processing {signal.signal} signal for {signal.symbol}")
                    self.logger.info(f"   VIPER Score: {signal.viper_score:.1f}")
                    self.logger.info(f"   Confidence: {signal.confidence:.1%}")
                    
                    # Execute trade
                    success = self.execute_trade(signal)
                    if success:
                        self.logger.info(f"✅ Trade executed successfully")
                    else:
                        self.logger.warning(f"⚠️ Trade execution failed")
                
                # 3. TP/SL: Monitor existing positions
                self.monitor_positions()
                
                # Sleep until next scan cycle
                elapsed = time.time() - start_time
                sleep_time = max(0, self.scan_interval - elapsed)
                
                if sleep_time > 0:
                    self.logger.info(f"💤 Sleeping for {sleep_time:.1f}s until next scan...")
                    time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            self.logger.info("🛑 Received keyboard interrupt")
        except Exception as e:
            self.logger.error(f"❌ Unexpected error in main loop: {e}")
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Graceful shutdown procedure"""
        self.logger.info("🛑 Shutting down VIPER Trader...")
        
        # Close all positions if desired (optional safety measure)
        # Uncomment the following lines to close all positions on shutdown
        # for symbol in list(self.active_positions.keys()):
        #     self.close_position(symbol, 'shutdown')
        
        self.logger.info("✅ Shutdown complete")


def main():
    """Main entry point"""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                🚀 VIPER STANDALONE TRADING COMPONENT                         ║
║                Complete Scan → Score → Trade → TP/SL Flow                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    try:
        # Check for required environment variables
        required_vars = ['BITGET_API_KEY', 'BITGET_API_SECRET', 'BITGET_API_PASSWORD']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
            print("Please configure your .env file with Bitget API credentials")
            return 1
        
        # Create and run the trader
        trader = StandaloneVIPERTrader()
        trader.run()
        
        return 0
        
    except Exception as e:
        print(f"❌ Failed to start VIPER Trader: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())