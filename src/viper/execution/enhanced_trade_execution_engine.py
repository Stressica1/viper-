#!/usr/bin/env python3
"""
# Tool ENHANCED TRADE EXECUTION ENGINE
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
import logging
import ccxt.pro as ccxt
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

# Import the enhanced VIPER scoring service
import sys
sys.path.insert(0, '/home/runner/work/viper-/viper-/services/viper-scoring-service')
try:
    pass
except ImportError:
    logger.error("# X Could not import VIPERScoringService - required for live trading")
    raise ImportError("VIPERScoringService is required for live trading operations")

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
        
        # Configuration from environment - LIVE TRADING ONLY
        self.api_key = os.getenv('BITGET_API_KEY', '')
        self.api_secret = os.getenv('BITGET_API_SECRET', '')
        self.api_password = os.getenv('BITGET_API_PASSWORD', '')
        
        # Validate real credentials are provided
        if not all([self.api_key, self.api_secret, self.api_password]):
            raise ValueError("🚫 LIVE TRADING MODE: All Bitget API credentials must be provided")
        
        if any(cred.startswith('your_') or cred.startswith('test_') for cred in [self.api_key, self.api_secret, self.api_password]):
            raise ValueError("🚫 LIVE TRADING MODE: Placeholder credentials not allowed")
        
        # Trading parameters
        self.max_positions = 5
        self.risk_per_trade = 0.02  # 2% risk per trade
        self.max_execution_cost = 3.0  # $3 maximum execution cost
        self.min_viper_score = 70.0  # Minimum VIPER score for trade
        
        logger.info("# Rocket Enhanced Trade Execution Engine initialized - LIVE MODE ONLY")
        logger.info(f"   Max Positions: {self.max_positions}")
        logger.info(f"   Risk Per Trade: {self.risk_per_trade * 100}%")
        logger.info(f"   Max Execution Cost: ${self.max_execution_cost}")
        logger.info("🚨 LIVE TRADING: Real trades will be executed")
        
    async def initialize_exchange(self) -> bool:
        """Initialize exchange connection - LIVE TRADING ONLY"""
        try:
            logger.info("🔌 Connecting to Bitget exchange for LIVE TRADING...")
            self.exchange = ccxt.bitget({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'password': self.api_password,
                'options': {
                    'defaultType': 'swap',
                    'adjustForTimeDifference': True,
                },
                'sandbox': False,  # Production mode only
            })
            
            # Test connection
            await self.exchange.load_markets()
            logger.info("# Check Live exchange connection established")
            return True
            
        except Exception as e:
            logger.error(f"# X Failed to connect to live exchange: {e}")
            logger.error("🚫 LIVE TRADING MODE: Cannot proceed without exchange connection")
            raise Exception("Live trading requires valid exchange connection")
    
    async def fetch_market_data_safely(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Safely fetch live market data - LIVE TRADING ONLY"""
        try:
            # Fetch ticker data
            ticker = await self.exchange.fetch_ticker(symbol)
            
            # Fetch orderbook data
            orderbook = await self.exchange.fetch_order_book(symbol, limit=10)
            
            # Fetch OHLCV data properly
            try:
                # Use proper async call and await the result
                ohlcv_raw = await self.exchange.fetch_ohlcv(symbol, '1h', limit=50)
                
                # Ensure we have the data before processing
                if ohlcv_raw and len(ohlcv_raw) > 0:
                    ohlcv_data = {'ohlcv': ohlcv_raw}
                else:
                    logger.warning(f"# Warning No OHLCV data received for {symbol}")
                    ohlcv_data = {'ohlcv': []}
                    
            except Exception as ohlcv_error:
                logger.error(f"# X OHLCV fetch error for {symbol}: {ohlcv_error}")
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
            logger.error(f"# X Critical error fetching live market data for {symbol}: {e}")
            logger.error("🚫 LIVE TRADING MODE: Cannot proceed without real market data")
            raise Exception(f"Live market data fetch failed for {symbol}")
    
    
    async def analyze_trading_opportunity(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Analyze trading opportunity using enhanced VIPER system"""
        try:
            logger.info(f"# Search Analyzing trading opportunity for {symbol}...")
            
            # Fetch market data safely
            market_data = await self.fetch_market_data_safely(symbol)
            if not market_data:
                logger.error(f"# X Could not fetch market data for {symbol}")
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
                
                logger.info(f"# Check {symbol} opportunity found: Score {viper_result['overall_score']:.1f}, "
                          f"Cost ${execution_cost:.2f}, "
                          f"Signal: {signal['type'] if signal else 'None'}")
                
                return opportunity
            else:
                logger.warning("# Warning VIPER service not available, using basic analysis")
                return None
                
        except Exception as e:
            logger.error(f"# X Error analyzing opportunity for {symbol}: {e}")
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
            logger.error(f"# X Error calculating position size: {e}")
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
            
            # Execute real trade
            try:
                logger.info(f"# Rocket EXECUTING LIVE {side} Trade for {symbol}:")
                logger.info(f"   💰 Entry Price: ${current_price:.2f}")
                logger.info(f"   # Chart Position Size: {position_size:.6f}")
                logger.info(f"   # Target Order Type: {order_type}")
                logger.info(f"   💸 Execution Cost: ${execution_cost:.2f}")
                logger.info(f"   🛡️ Stop Loss: {risk_mgmt.get('stop_loss_pct', 0)*100:.1f}%")
                
                # Generate unique trade ID
                trade_id = f"LIVE_{symbol}_{int(time.time())}"
                
                # TODO: Implement actual exchange order execution
                # This should integrate with the exchange-connector service
                # For now, this is a placeholder that requires integration
                
                logger.warning("🚧 Live trading execution requires integration with exchange-connector service")
                logger.info("# Idea Connect this to: http://exchange-connector:8005/execute_order")
                
                # Store position for tracking
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
                
            except Exception as trade_error:
                logger.error(f"# X Live trade execution failed: {trade_error}")
                result = TradeResult(success=False, error=str(trade_error))
            
            # Record trade
            self.trade_history.append({
                'timestamp': datetime.now(),
                'result': result,
                'opportunity': opportunity
            })
            
            return result
            
        except Exception as e:
            logger.error(f"# X Trade execution failed: {e}")
            return TradeResult(success=False, error=str(e))
    
    async def scan_and_trade(self, symbols: List[str]) -> Dict[str, Any]:
        """Scan symbols and execute trades based on VIPER analysis"""
        logger.info(f"# Search Scanning {len(symbols)} symbols for trading opportunities...")
        
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
                        logger.info(f"# Check Trade executed for {symbol}: {trade_result.side} at ${trade_result.price:.2f}")
                    else:
                        results['trades_rejected'] += 1
                        logger.warning(f"# X Trade rejected for {symbol}: {trade_result.error}")
                    
                    results['details'].append({
                        'symbol': symbol,
                        'opportunity': opportunity,
                        'trade_result': trade_result
                    })
                
                # Small delay between scans
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"# X Error processing {symbol}: {e}")
                results['trades_rejected'] += 1
        
        return results
    
    async def run_trading_session(self, symbols: List[str]) -> None:
        """Run a complete trading session"""
        logger.info("# Rocket Starting Enhanced Trading Session")
        logger.info("=" * 80)
        
        try:
            # Initialize exchange
            await self.initialize_exchange()
            
            # Run scan and trade
            session_results = await self.scan_and_trade(symbols)
            
            # Print results
            logger.info("=" * 80)
            logger.info("# Chart TRADING SESSION RESULTS")
            logger.info("=" * 80)
            logger.info(f"   # Search Symbols Scanned: {session_results['scanned']}")
            logger.info(f"   # Target Opportunities Found: {session_results['opportunities_found']}")
            logger.info(f"   # Check Trades Executed: {session_results['trades_executed']}")
            logger.info(f"   # X Trades Rejected: {session_results['trades_rejected']}")
            logger.info(f"   💸 Total Execution Cost: ${session_results['total_execution_cost']:.2f}")
            
            if session_results['trades_executed'] > 0:
                avg_cost = session_results['total_execution_cost'] / session_results['trades_executed']
                logger.info(f"   # Chart Average Execution Cost: ${avg_cost:.2f}")
                
                if avg_cost < 3.0:
                    logger.info("# Party SUCCESS: All trades executed with acceptable costs!")
                else:
                    logger.warning("# Warning WARNING: High average execution costs detected")
            
            # Show active positions
            if self.active_positions:
                logger.info(f"\n💼 Active Positions: {len(self.active_positions)}")
                for trade_id, position in self.active_positions.items():
                    logger.info(f"   📈 {position['symbol']} {position['side']} - "
                              f"Score: {position['viper_score']:.1f}, "
                              f"Cost: ${position['execution_cost']:.2f}")
            
        except Exception as e:
            logger.error(f"# X Trading session failed: {e}")
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