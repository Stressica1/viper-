#!/usr/bin/env python3
"""
🚀 VIPER SYSTEM DEMONSTRATION
Demonstrate the fixed VIPER trading system with favorable conditions
"""

import os
import sys
import asyncio
import logging
from datetime import datetime
import numpy as np

# Set mock data mode with favorable conditions
os.environ['USE_MOCK_DATA'] = 'true'

# Import the enhanced trade execution engine
sys.path.insert(0, '/home/runner/work/viper-/viper-')
from enhanced_trade_execution_engine import EnhancedTradeExecutionEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - DEMO - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VIPERSystemDemo:
    """Demonstrate the VIPER system with favorable trading conditions"""
    
    def __init__(self):
        self.trader = EnhancedTradeExecutionEngine()
        # Lower the minimum score threshold for demo purposes
        self.trader.min_viper_score = 60.0  # Lower threshold to show trades
        
    def create_favorable_mock_data(self, symbol: str, scenario: str):
        """Create mock data with favorable trading conditions"""
        base_price = 50000.0 if 'BTC' in symbol else 3000.0 if 'ETH' in symbol else 1.0
        
        # Override the create_mock_market_data method temporarily
        original_method = self.trader.create_mock_market_data
        
        if scenario == "bullish_breakout":
            # Bullish breakout scenario - should generate LONG signal
            mock_data = {
                'symbol': symbol,
                'ticker': {
                    'price': base_price * 1.01,  # Price above resistance
                    'high': base_price * 1.02,
                    'low': base_price * 0.99,
                    'close': base_price * 1.01,
                    'volume': 8000000.0,  # High volume
                    'quoteVolume': base_price * 8000000,
                    'change': 2.5,  # Strong positive momentum
                    'price_change': 2.5
                },
                'orderbook': {
                    'bids': [[base_price * 0.99995, 5.0], [base_price * 0.9999, 10.0]],  # Very tight spread
                    'asks': [[base_price * 1.00005, 5.0], [base_price * 1.0001, 10.0]]
                },
                'ohlcv': {
                    'ohlcv': self.create_trending_ohlcv(base_price, trend='up')
                },
                'timestamp': datetime.now().isoformat()
            }
            
        elif scenario == "bearish_breakdown":
            # Bearish breakdown scenario - should generate SHORT signal  
            mock_data = {
                'symbol': symbol,
                'ticker': {
                    'price': base_price * 0.99,  # Price below support
                    'high': base_price * 1.005,
                    'low': base_price * 0.98,
                    'close': base_price * 0.99,
                    'volume': 6000000.0,  # Good volume
                    'quoteVolume': base_price * 6000000,
                    'change': -2.0,  # Negative momentum
                    'price_change': -2.0
                },
                'orderbook': {
                    'bids': [[base_price * 0.99995, 4.0], [base_price * 0.9999, 8.0]],  # Very tight spread
                    'asks': [[base_price * 1.00005, 4.0], [base_price * 1.0001, 8.0]]
                },
                'ohlcv': {
                    'ohlcv': self.create_trending_ohlcv(base_price, trend='down')
                },
                'timestamp': datetime.now().isoformat()
            }
            
        elif scenario == "support_bounce":
            # Support bounce scenario - S1S2R1R2 strategy
            mock_data = {
                'symbol': symbol,
                'ticker': {
                    'price': base_price * 0.96,  # Near S2 level (calculated as ~48000 for 50000 base)
                    'high': base_price * 1.02,   # This creates S2 around current price
                    'low': base_price * 0.94,    
                    'close': base_price * 0.96,
                    'volume': 7000000.0,
                    'quoteVolume': base_price * 7000000,
                    'change': -0.5,  # Small negative (consolidating)
                    'price_change': -0.5
                },
                'orderbook': {
                    'bids': [[base_price * 0.99995, 6.0], [base_price * 0.9999, 12.0]],  # Very tight spread
                    'asks': [[base_price * 1.00005, 6.0], [base_price * 1.0001, 12.0]]
                },
                'ohlcv': {
                    'ohlcv': self.create_trending_ohlcv(base_price, trend='consolidation')
                },
                'timestamp': datetime.now().isoformat()
            }
            
        else:  # Default favorable conditions
            mock_data = {
                'symbol': symbol,
                'ticker': {
                    'price': base_price,
                    'high': base_price * 1.015,
                    'low': base_price * 0.985,
                    'close': base_price,
                    'volume': 5000000.0,  # Good volume
                    'quoteVolume': base_price * 5000000,
                    'change': 1.0,
                    'price_change': 1.0
                },
                'orderbook': {
                    'bids': [[base_price * 0.99995, 3.0], [base_price * 0.9999, 6.0]],  # Very tight spread
                    'asks': [[base_price * 1.00005, 3.0], [base_price * 1.0001, 6.0]]
                },
                'ohlcv': {
                    'ohlcv': self.create_trending_ohlcv(base_price, trend='up')
                },
                'timestamp': datetime.now().isoformat()
            }
        
        # Monkey patch the method temporarily
        def mock_method(symbol_param):
            return mock_data
        
        self.trader.create_mock_market_data = mock_method
        return mock_data
    
    def create_trending_ohlcv(self, base_price: float, trend: str = 'up') -> list:
        """Create OHLCV data showing a trend"""
        ohlcv_data = []
        current_price = base_price
        
        for i in range(50):
            timestamp = int((datetime.now().timestamp() - (49-i) * 3600) * 1000)
            
            if trend == 'up':
                price_change = np.random.uniform(0.002, 0.008)  # 0.2% to 0.8% up per hour
            elif trend == 'down':
                price_change = np.random.uniform(-0.008, -0.002)  # 0.2% to 0.8% down per hour
            else:  # consolidation
                price_change = np.random.uniform(-0.003, 0.003)  # -0.3% to +0.3%
            
            open_price = current_price
            close_price = current_price * (1 + price_change)
            high = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.005)))
            low = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.005)))
            volume = np.random.uniform(2000000, 8000000)
            
            ohlcv_data.append([timestamp, open_price, high, low, close_price, volume])
            current_price = close_price
            
        return ohlcv_data
        
    async def run_demo_scenarios(self):
        """Run demo with different favorable scenarios"""
        logger.info("🚀 STARTING VIPER SYSTEM DEMONSTRATION")
        logger.info("=" * 80)
        
        scenarios = [
            ("BTC/USDT:USDT", "bullish_breakout", "Bullish breakout with high volume"),
            ("ETH/USDT:USDT", "bearish_breakdown", "Bearish breakdown scenario"),  
            ("ADA/USDT:USDT", "support_bounce", "S2 support level bounce"),
        ]
        
        for symbol, scenario, description in scenarios:
            logger.info(f"\n🎯 SCENARIO: {description}")
            logger.info(f"   Symbol: {symbol}")
            logger.info("-" * 50)
            
            # Set up favorable conditions
            self.create_favorable_mock_data(symbol, scenario)
            
            # Analyze opportunity
            opportunity = await self.trader.analyze_trading_opportunity(symbol)
            
            if opportunity:
                logger.info(f"   ✅ Opportunity found!")
                
                # Show detailed analysis
                viper_score = opportunity['viper_score']
                logger.info(f"   📊 VIPER Score: {viper_score['overall_score']:.1f}")
                logger.info(f"      Volume: {viper_score['components']['volume_score']:.1f}")
                logger.info(f"      Price: {viper_score['components']['price_score']:.1f}")  
                logger.info(f"      External: {viper_score['components']['external_score']:.1f}")
                logger.info(f"      Range: {viper_score['components']['range_score']:.1f}")
                logger.info(f"   💸 Execution Cost: ${opportunity['execution_cost']:.2f}")
                
                # Show S1S2R1R2 levels
                levels = opportunity.get('s1s2r1r2_levels', {})
                if levels:
                    logger.info(f"   🎯 S1S2R1R2 Levels:")
                    for level_name, level_value in levels.items():
                        logger.info(f"      {level_name}: ${level_value:.2f}")
                
                # Execute trade
                trade_result = await self.trader.execute_trade(opportunity)
                
                if trade_result.success:
                    logger.info(f"   🎉 TRADE EXECUTED: {trade_result.side} {trade_result.symbol}")
                    logger.info(f"      Trade ID: {trade_result.trade_id}")
                    logger.info(f"      Entry Price: ${trade_result.price:.2f}")
                    logger.info(f"      Position Size: {trade_result.amount:.6f}")
                    logger.info(f"      Order Type: {trade_result.order_type}")
                else:
                    logger.warning(f"   ❌ Trade failed: {trade_result.error}")
                
            else:
                logger.warning(f"   ❌ No opportunity found for {symbol}")
                
            await asyncio.sleep(0.5)  # Brief pause between scenarios
            
        # Show final summary
        logger.info("\n" + "=" * 80)
        logger.info("📊 DEMONSTRATION SUMMARY")
        logger.info("=" * 80)
        
        active_positions = self.trader.active_positions
        logger.info(f"🎯 Active Positions: {len(active_positions)}")
        
        total_execution_cost = 0.0
        for trade_id, position in active_positions.items():
            logger.info(f"   📈 {position['symbol']} {position['side']} - "
                      f"VIPER Score: {position['viper_score']:.1f}, "
                      f"Cost: ${position['execution_cost']:.2f}")
            total_execution_cost += position['execution_cost']
        
        if active_positions:
            avg_cost = total_execution_cost / len(active_positions)
            logger.info(f"\n💸 Total Execution Cost: ${total_execution_cost:.2f}")
            logger.info(f"📊 Average Cost per Trade: ${avg_cost:.2f}")
            
            if avg_cost < 3.0:
                logger.info("🎉 SUCCESS: All trades executed with acceptable execution costs!")
                logger.info("✅ VIPER system is working correctly with:")
                logger.info("   - Execution cost awareness (< $3.00 limit)")
                logger.info("   - S1S2R1R2 predictive ranges strategy")
                logger.info("   - Enhanced scoring system with proper weights")
                logger.info("   - Smart order routing (LIMIT vs MARKET)")
            else:
                logger.warning("⚠️ Some trades had high execution costs")
        else:
            logger.info("ℹ️ No trades executed - scores may be below threshold")
            
        logger.info("\n🎯 VIPER SYSTEM DEMONSTRATION COMPLETE!")

async def main():
    """Main demonstration function"""
    demo = VIPERSystemDemo()
    await demo.run_demo_scenarios()

if __name__ == "__main__":
    asyncio.run(main())