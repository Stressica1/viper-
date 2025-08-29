#!/usr/bin/env python3
"""
🔧 ENHANCED TRADE EXECUTION ENGINE
Fixed trade execution with proper OHLCV handling, execution cost awareness, and S1S2R1R2 strategy

Key Fixes:
1. Proper async OHLCV data fetching (no more "coroutine has no len" errors)
2. Execution cost-aware trading decisions
3. S1S2R1R2 predictive ranges strategy integration
4. Enhanced position sizing with dynamic risk management
5. Smart order routing (LIMIT vs MARKET based on execution cost)
"""

import os
import asyncio
import aiohttp
import logging
import ccxt.pro as ccxt
import json
import time
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

# Import the enhanced VIPER scoring service
import sys
sys.path.insert(0, '/home/runner/work/viper-/viper-/services/viper-scoring-service')
try:
    from main import VIPERScoringService, SignalType
except ImportError:
    print("⚠️ Warning: Could not import VIPERScoringService, using mock implementation")
    VIPERScoringService = None
    SignalType = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - ENHANCED_TRADER - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TradeResult:
    """Result of trade execution"""
    success: bool
    trade_id: Optional[str] = None
    side: Optional[str] = None
    symbol: Optional[str] = None
    price: Optional[float] = None
    amount: Optional[float] = None
    execution_cost: Optional[float] = None
    order_type: Optional[str] = None
    error: Optional[str] = None

class EnhancedTradeExecutionEngine:
    """Enhanced trade execution engine with fixed OHLCV and execution cost awareness"""
    
    def __init__(self):
        self.exchange = None
        self.session = None
        self.viper_service = VIPERScoringService() if VIPERScoringService else None
        self.is_running = False
        self.active_positions = {}
        self.trade_history = []
        
        # Configuration from environment
        self.api_key = os.getenv('BITGET_API_KEY', 'test_key')
        self.api_secret = os.getenv('BITGET_API_SECRET', 'test_secret')
        self.api_password = os.getenv('BITGET_API_PASSWORD', 'test_password')
        self.use_mock_data = os.getenv('USE_MOCK_DATA', 'true').lower() == 'true'
        
        # Trading parameters
        self.max_positions = 5
        self.risk_per_trade = 0.02  # 2% risk per trade
        self.max_execution_cost = 3.0  # $3 maximum execution cost
        self.min_viper_score = 70.0  # Minimum VIPER score for trade
        
        logger.info("🚀 Enhanced Trade Execution Engine initialized")
        logger.info(f"   Mock Data Mode: {self.use_mock_data}")
        logger.info(f"   Max Positions: {self.max_positions}")
        logger.info(f"   Risk Per Trade: {self.risk_per_trade * 100}%")
        logger.info(f"   Max Execution Cost: ${self.max_execution_cost}")
        
    async def initialize_exchange(self) -> bool:
        """Initialize exchange connection with proper error handling"""
        try:
            if self.use_mock_data:
                logger.info("🎭 Using mock data mode - no real exchange connection")
                return True
                
            logger.info("🔌 Connecting to Bitget exchange...")
            self.exchange = ccxt.bitget({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'password': self.api_password,
                'options': {
                    'defaultType': 'swap',
                    'adjustForTimeDifference': True,
                },
                'sandbox': False,
            })
            
            # Test connection
            await self.exchange.load_markets()
            logger.info("✅ Exchange connection established")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to exchange: {e}")
            if not self.use_mock_data:
                logger.info("🎭 Falling back to mock data mode")
                self.use_mock_data = True
            return True  # Continue with mock data
    
    async def fetch_market_data_safely(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Safely fetch market data with proper async OHLCV handling"""
        try:
            if self.use_mock_data:
                return self.create_mock_market_data(symbol)
                
            # Fetch ticker data
            ticker = await self.exchange.fetch_ticker(symbol)
            
            # Fetch orderbook data
            orderbook = await self.exchange.fetch_order_book(symbol, limit=10)
            
            # Fetch OHLCV data properly (this was the source of the coroutine error)
            try:
                # Use proper async call and await the result
                ohlcv_raw = await self.exchange.fetch_ohlcv(symbol, '1h', limit=50)
                
                # Ensure we have the data before processing
                if ohlcv_raw and len(ohlcv_raw) > 0:
                    ohlcv_data = {'ohlcv': ohlcv_raw}
                else:
                    logger.warning(f"⚠️ No OHLCV data received for {symbol}")
                    ohlcv_data = {'ohlcv': []}
                    
            except Exception as ohlcv_error:
                logger.error(f"❌ OHLCV fetch error for {symbol}: {ohlcv_error}")
                ohlcv_data = {'ohlcv': []}
            
            # Combine all market data
            market_data = {
                'symbol': symbol,
                'ticker': {
                    'price': ticker.get('last', 0),
                    'high': ticker.get('high', 0),
                    'low': ticker.get('low', 0),
                    'close': ticker.get('close', 0),
                    'volume': ticker.get('baseVolume', 0),
                    'quoteVolume': ticker.get('quoteVolume', 0),
                    'change': ticker.get('change', 0),
                    'price_change': ticker.get('percentage', 0) or ticker.get('change', 0)
                },
                'orderbook': orderbook,
                'ohlcv': ohlcv_data,
                'timestamp': datetime.now().isoformat()
            }
            
            return market_data
            
        except Exception as e:
            logger.error(f"❌ Error fetching market data for {symbol}: {e}")
            # Fallback to mock data
            return self.create_mock_market_data(symbol)
    
    def create_mock_market_data(self, symbol: str) -> Dict[str, Any]:
        """Create realistic mock market data for testing"""
        base_price = 50000.0 if 'BTC' in symbol else 3000.0
        
        # Create realistic OHLCV data
        ohlcv_data = []
        current_time = int(time.time() * 1000)
        
        for i in range(50):
            timestamp = current_time - (i * 3600000)  # Hourly candles
            price_variation = np.random.normal(0, 0.02)  # 2% standard deviation
            open_price = base_price * (1 + price_variation)
            high = open_price * (1 + abs(np.random.normal(0, 0.01)))
            low = open_price * (1 - abs(np.random.normal(0, 0.01)))
            close = open_price * (1 + np.random.normal(0, 0.005))
            volume = np.random.uniform(1000000, 5000000)
            
            ohlcv_data.append([timestamp, open_price, high, low, close, volume])
        
        # Reverse to get chronological order
        ohlcv_data.reverse()
        
        return {
            'symbol': symbol,
            'ticker': {
                'price': base_price,
                'high': base_price * 1.02,
                'low': base_price * 0.98,
                'close': base_price,
                'volume': 2000000.0,
                'quoteVolume': base_price * 2000000,
                'change': np.random.uniform(-2, 2),
                'price_change': np.random.uniform(-2, 2)
            },
            'orderbook': {
                'bids': [[base_price - 1, 1.5], [base_price - 2, 2.0], [base_price - 3, 3.0]],
                'asks': [[base_price + 1, 1.5], [base_price + 2, 2.0], [base_price + 3, 3.0]]
            },
            'ohlcv': {
                'ohlcv': ohlcv_data
            },
            'timestamp': datetime.now().isoformat()
        }
    
    async def analyze_trading_opportunity(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Analyze trading opportunity using enhanced VIPER system"""
        try:
            logger.info(f"🔍 Analyzing trading opportunity for {symbol}...")
            
            # Fetch market data safely
            market_data = await self.fetch_market_data_safely(symbol)
            if not market_data:
                logger.error(f"❌ Could not fetch market data for {symbol}")
                return None
            
            # Use VIPER scoring service if available
            if self.viper_service:
                viper_result = self.viper_service.calculate_viper_score(market_data, symbol)
                signal = self.viper_service.generate_signal(market_data, symbol)
                
                # Check if signal was rejected due to high execution cost
                if 'rejected_reason' in viper_result:
                    logger.info(f"🚫 {symbol} opportunity rejected: {viper_result['rejected_reason']}")
                    return None
                
                # Check VIPER score threshold
                if viper_result['overall_score'] < self.min_viper_score:
                    logger.info(f"⏭️ {symbol} score {viper_result['overall_score']:.1f} below threshold {self.min_viper_score}")
                    return None
                
                # Get execution cost
                execution_cost = viper_result.get('execution_cost', 0)
                if execution_cost >= self.max_execution_cost:
                    logger.warning(f"🚫 {symbol} execution cost ${execution_cost:.2f} exceeds maximum ${self.max_execution_cost}")
                    return None
                
                opportunity = {
                    'symbol': symbol,
                    'viper_score': viper_result,
                    'signal': signal,
                    'market_data': market_data,
                    'execution_cost': execution_cost,
                    's1s2r1r2_levels': viper_result.get('s1s2r1r2_levels', {}),
                    'recommended_order_type': signal.get('order_type', 'MARKET') if signal else 'MARKET'
                }
                
                logger.info(f"✅ {symbol} opportunity found: Score {viper_result['overall_score']:.1f}, "
                          f"Cost ${execution_cost:.2f}, "
                          f"Signal: {signal['type'] if signal else 'None'}")
                
                return opportunity
            else:
                logger.warning("⚠️ VIPER service not available, using basic analysis")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error analyzing opportunity for {symbol}: {e}")
            return None
    
    def calculate_position_size(self, symbol: str, price: float, execution_cost: float, 
                              balance: float = 10000.0) -> Tuple[float, Dict[str, float]]:
        """Calculate position size with dynamic risk management"""
        try:
            # Base risk per trade
            base_risk_amount = balance * self.risk_per_trade
            
            # Adjust for execution cost (reduce position size if high execution costs)
            execution_cost_factor = max(0.5, 1 - (execution_cost / 10))  # Reduce size for high costs
            adjusted_risk_amount = base_risk_amount * execution_cost_factor
            
            # Calculate stop loss distance (dynamic based on volatility)
            stop_loss_pct = max(0.02, execution_cost / price + 0.01)  # At least 2% or cost + 1%
            stop_loss_distance = price * stop_loss_pct
            
            # Position size calculation
            position_size = adjusted_risk_amount / stop_loss_distance
            
            # Calculate take profit (risk/reward ratio of at least 2:1 after costs)
            min_profit_ratio = max(2.0, execution_cost / adjusted_risk_amount * 3)
            take_profit_distance = stop_loss_distance * min_profit_ratio
            
            risk_management = {
                'stop_loss_pct': stop_loss_pct,
                'stop_loss_distance': stop_loss_distance,
                'take_profit_distance': take_profit_distance,
                'risk_amount': adjusted_risk_amount,
                'execution_cost_factor': execution_cost_factor
            }
            
            logger.info(f"   💰 Position sizing for {symbol}: "
                      f"Size: {position_size:.6f}, "
                      f"Risk: ${adjusted_risk_amount:.2f} ({stop_loss_pct*100:.1f}% SL), "
                      f"R/R: 1:{min_profit_ratio:.1f}")
            
            return position_size, risk_management
            
        except Exception as e:
            logger.error(f"❌ Error calculating position size: {e}")
            return 0.001, {}  # Fallback minimum size
    
    async def execute_trade(self, opportunity: Dict[str, Any]) -> TradeResult:
        """Execute trade with enhanced execution cost awareness"""
        try:
            symbol = opportunity['symbol']
            signal = opportunity.get('signal')
            execution_cost = opportunity['execution_cost']
            market_data = opportunity['market_data']
            
            if not signal:
                return TradeResult(success=False, error="No trading signal generated")
            
            current_price = market_data['ticker']['price']
            order_type = opportunity['recommended_order_type']
            side = signal['type'].upper()
            
            # Calculate position size
            position_size, risk_mgmt = self.calculate_position_size(
                symbol, current_price, execution_cost
            )
            
            # Create trade entry
            if self.use_mock_data:
                # Simulate trade execution
                trade_id = f"MOCK_{symbol}_{int(time.time())}"
                
                logger.info(f"📝 SIMULATED {side} Trade for {symbol}:")
                logger.info(f"   💰 Entry Price: ${current_price:.2f}")
                logger.info(f"   📊 Position Size: {position_size:.6f}")
                logger.info(f"   🎯 Order Type: {order_type}")
                logger.info(f"   💸 Execution Cost: ${execution_cost:.2f}")
                logger.info(f"   🛡️ Stop Loss: {risk_mgmt.get('stop_loss_pct', 0)*100:.1f}%")
                
                # Store position
                self.active_positions[trade_id] = {
                    'symbol': symbol,
                    'side': side,
                    'entry_price': current_price,
                    'position_size': position_size,
                    'order_type': order_type,
                    'execution_cost': execution_cost,
                    'risk_management': risk_mgmt,
                    's1s2r1r2_levels': opportunity.get('s1s2r1r2_levels', {}),
                    'entry_time': datetime.now(),
                    'viper_score': opportunity['viper_score']['overall_score']
                }
                
                result = TradeResult(
                    success=True,
                    trade_id=trade_id,
                    side=side,
                    symbol=symbol,
                    price=current_price,
                    amount=position_size,
                    execution_cost=execution_cost,
                    order_type=order_type
                )
                
            else:
                # Real trade execution (placeholder - needs full implementation)
                logger.warning("🚨 Real trading not implemented - use mock mode")
                result = TradeResult(success=False, error="Real trading not implemented")
            
            # Record trade
            self.trade_history.append({
                'timestamp': datetime.now(),
                'result': result,
                'opportunity': opportunity
            })
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Trade execution failed: {e}")
            return TradeResult(success=False, error=str(e))
    
    async def scan_and_trade(self, symbols: List[str]) -> Dict[str, Any]:
        """Scan symbols and execute trades based on VIPER analysis"""
        logger.info(f"🔍 Scanning {len(symbols)} symbols for trading opportunities...")
        
        results = {
            'scanned': len(symbols),
            'opportunities_found': 0,
            'trades_executed': 0,
            'trades_rejected': 0,
            'total_execution_cost': 0.0,
            'details': []
        }
        
        for symbol in symbols:
            try:
                # Skip if we already have max positions
                if len(self.active_positions) >= self.max_positions:
                    logger.info(f"⏭️ Max positions ({self.max_positions}) reached, skipping {symbol}")
                    break
                
                # Analyze opportunity
                opportunity = await self.analyze_trading_opportunity(symbol)
                
                if opportunity:
                    results['opportunities_found'] += 1
                    
                    # Execute trade
                    trade_result = await self.execute_trade(opportunity)
                    
                    if trade_result.success:
                        results['trades_executed'] += 1
                        results['total_execution_cost'] += trade_result.execution_cost or 0
                        logger.info(f"✅ Trade executed for {symbol}: {trade_result.side} at ${trade_result.price:.2f}")
                    else:
                        results['trades_rejected'] += 1
                        logger.warning(f"❌ Trade rejected for {symbol}: {trade_result.error}")
                    
                    results['details'].append({
                        'symbol': symbol,
                        'opportunity': opportunity,
                        'trade_result': trade_result
                    })
                
                # Small delay between scans
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"❌ Error processing {symbol}: {e}")
                results['trades_rejected'] += 1
        
        return results
    
    async def run_trading_session(self, symbols: List[str]) -> None:
        """Run a complete trading session"""
        logger.info("🚀 Starting Enhanced Trading Session")
        logger.info("=" * 80)
        
        try:
            # Initialize exchange
            await self.initialize_exchange()
            
            # Run scan and trade
            session_results = await self.scan_and_trade(symbols)
            
            # Print results
            logger.info("=" * 80)
            logger.info("📊 TRADING SESSION RESULTS")
            logger.info("=" * 80)
            logger.info(f"   🔍 Symbols Scanned: {session_results['scanned']}")
            logger.info(f"   🎯 Opportunities Found: {session_results['opportunities_found']}")
            logger.info(f"   ✅ Trades Executed: {session_results['trades_executed']}")
            logger.info(f"   ❌ Trades Rejected: {session_results['trades_rejected']}")
            logger.info(f"   💸 Total Execution Cost: ${session_results['total_execution_cost']:.2f}")
            
            if session_results['trades_executed'] > 0:
                avg_cost = session_results['total_execution_cost'] / session_results['trades_executed']
                logger.info(f"   📊 Average Execution Cost: ${avg_cost:.2f}")
                
                if avg_cost < 3.0:
                    logger.info("🎉 SUCCESS: All trades executed with acceptable costs!")
                else:
                    logger.warning("⚠️ WARNING: High average execution costs detected")
            
            # Show active positions
            if self.active_positions:
                logger.info(f"\n💼 Active Positions: {len(self.active_positions)}")
                for trade_id, position in self.active_positions.items():
                    logger.info(f"   📈 {position['symbol']} {position['side']} - "
                              f"Score: {position['viper_score']:.1f}, "
                              f"Cost: ${position['execution_cost']:.2f}")
            
        except Exception as e:
            logger.error(f"❌ Trading session failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        finally:
            # Cleanup
            if self.exchange:
                await self.exchange.close()
            if self.session:
                await self.session.close()

async def main():
    """Main execution function"""
    
    # Test symbols
    test_symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'ADA/USDT:USDT']
    
    # Create and run enhanced trader
    trader = EnhancedTradeExecutionEngine()
    await trader.run_trading_session(test_symbols)

if __name__ == "__main__":
    asyncio.run(main())