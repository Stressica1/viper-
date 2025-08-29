#!/usr/bin/env python3
"""
🚀 EXECUTE LIVE TRADE - COMPLETE TRADING CYCLE
Advanced live trading execution with all optimized components
"""

import asyncio
import logging
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import trading components
from viper_async_trader import ViperAsyncTrader
from predictive_ranges_strategy import get_predictive_strategy
from optimized_trade_entry_system import get_optimized_entry_system
from emergency_stop_system import get_emergency_system
from github_mcp_integration import GitHubMCPIntegration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - LIVE_TRADE - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LiveTradeExecutor:
    """Complete live trade execution system"""

    def __init__(self):
        self.trader = ViperAsyncTrader()
        self.predictive_strategy = get_predictive_strategy()
        self.entry_system = get_optimized_entry_system()
        self.emergency_system = get_emergency_system()
        self.github_mcp = GitHubMCPIntegration()

        logger.info("🚀 Live Trade Executor initialized")

    async def execute_complete_trade_cycle(self):
        """Execute complete trade cycle from signal to order"""

        print("🚀 VIPER LIVE TRADE EXECUTION")
        print("=" * 60)

        try:
            # Step 1: System Health Check
            print("📊 STEP 1: SYSTEM HEALTH CHECK")
            await self.system_health_check()

            # Step 2: Generate Trading Signals
            print("\\n🎯 STEP 2: GENERATE TRADING SIGNALS")
            signals = await self.generate_trading_signals()

            if not signals:
                print("❌ No trading signals generated")
                return False

            # Step 3: Validate and Optimize Signals
            print("\\n🔍 STEP 3: VALIDATE SIGNALS")
            validated_signals = await self.validate_signals(signals)

            if not validated_signals:
                print("❌ No signals passed validation")
                return False

            # Step 4: Execute Trade Order
            print("\\n💰 STEP 4: EXECUTE TRADE ORDER")
            trade_result = await self.execute_trade_order(validated_signals[0])

            if not trade_result:
                print("❌ Trade execution failed")
                return False

            # Step 5: Monitor and Manage Position
            print("\\n📈 STEP 5: MONITOR POSITION")
            await self.monitor_position(trade_result)

            # Step 6: Report Results
            print("\\n📊 STEP 6: GENERATE REPORT")
            await self.generate_trade_report(trade_result)

            print("\\n🎉 TRADE EXECUTION COMPLETE!")
            return True

        except Exception as e:
            logger.error(f"❌ Trade execution failed: {e}")
            await self.emergency_system.manual_emergency_stop(f"Trade execution error: {e}")
            return False

    async def system_health_check(self):
        """Perform comprehensive system health check"""

        # Check account balance
        balance = await self.trader.check_account_balance()
        print(f"💰 Account Balance: ${balance:.2f}")

        if balance < 1.0:
            raise Exception("❌ Insufficient account balance for trading")

        # Check emergency system
        health = await self.emergency_system.check_system_health()
        print(f"🛡️ Emergency System: {health['system_status']}")

        if health['system_status'] == 'EMERGENCY_STOP':
            raise Exception("❌ Emergency stop is active")

        # Check GitHub MCP
        print("🔗 GitHub MCP Status: Checking...")

        # Verify API connectivity
        try:
            await self.trader.connect_exchange()
            print("✅ Exchange Connection: SUCCESS")
        except Exception as e:
            print(f"❌ Exchange Connection: FAILED - {e}")
            raise

        print("✅ System Health Check: PASSED")

    async def generate_trading_signals(self):
        """Generate optimized trading signals"""

        # Get market data
        symbol = "BTCUSDT"
        market_data = await self.get_market_data(symbol)

        if not market_data:
            print("❌ Failed to get market data")
            return []

        # Calculate predictive ranges
        current_price = market_data['1h']['close'].iloc[-1]
        predictive_ranges = self.predictive_strategy.calculate_predictive_ranges(
            market_data['1h'], symbol, '1h'
        )

        # Generate optimized signals
        signals = await self.entry_system.analyze_optimal_entries(
            symbol, market_data, current_price, account_balance=2.84
        )

        print(f"🎯 Generated {len(signals)} optimized signals")
        for i, signal in enumerate(signals[:3]):
            print(f"   {i+1}. {signal.direction.upper()} {signal.symbol} @ ${signal.entry_price:.2f} "
                  f"(Conf: {signal.confidence_score:.1%}, Quality: {signal.entry_quality})")

        return signals

    async def get_market_data(self, symbol):
        """Get comprehensive market data"""

        try:
            # Use trader's market data fetching
            market_data = {}

            # Get 1-hour data
            ohlcv_1h = await self.trader.fetch_market_data(symbol, '1h', 100)
            if ohlcv_1h:
                df_1h = self.ohlcv_to_dataframe(ohlcv_1h)
                market_data['1h'] = df_1h

            # Get 4-hour data
            ohlcv_4h = await self.trader.fetch_market_data(symbol, '4h', 100)
            if ohlcv_4h:
                df_4h = self.ohlcv_to_dataframe(ohlcv_4h)
                market_data['4h'] = df_4h

            return market_data

        except Exception as e:
            logger.error(f"❌ Failed to get market data: {e}")
            return None

    def ohlcv_to_dataframe(self, ohlcv):
        """Convert OHLCV data to DataFrame"""
        import pandas as pd

        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        # Convert to numeric
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])

        return df

    async def validate_signals(self, signals):
        """Validate signals against risk parameters"""

        validated_signals = []

        for signal in signals:
            # Check confidence threshold
            if signal.confidence_score < 0.7:
                print(f"⚠️ Signal rejected: Low confidence ({signal.confidence_score:.1%})")
                continue

            # Check risk-reward ratio
            if signal.risk_reward_ratio < 2.0:
                print(f"⚠️ Signal rejected: Poor RR ratio ({signal.risk_reward_ratio:.2f})")
                continue

            # Check entry quality
            if signal.entry_quality not in ['PREMIUM', 'EXCELLENT', 'GOOD']:
                print(f"⚠️ Signal rejected: Poor quality ({signal.entry_quality})")
                continue

            # Check position size
            if signal.position_size <= 0:
                print("⚠️ Signal rejected: Invalid position size")
                continue

            validated_signals.append(signal)
            print(f"✅ Signal validated: {signal.direction.upper()} {signal.symbol}")

        return validated_signals

    async def execute_trade_order(self, signal):
        """Execute actual trade order"""

        print(f"💰 EXECUTING TRADE: {signal.direction.upper()} {signal.symbol}")
        print(f"   Entry Price: ${signal.entry_price:.2f}")
        print(f"   Position Size: {signal.position_size:.6f}")
        print(f"   Stop Loss: ${signal.stop_loss:.2f}")
        print(f"   Take Profit: ${signal.take_profit:.2f}")

        try:
            # Execute the trade
            trade_result = await self.trader.execute_trade(
                symbol=signal.symbol,
                direction=signal.direction,
                position_size=signal.position_size,
                entry_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit
            )

            if trade_result:
                print("✅ TRADE EXECUTED SUCCESSFULLY!")
                print(f"   Order ID: {trade_result.get('order_id', 'N/A')}")
                print(f"   Executed Price: ${trade_result.get('price', 0):.2f}")
                print(f"   Executed Size: {trade_result.get('size', 0):.6f}")

                return trade_result
            else:
                print("❌ Trade execution failed")
                return None

        except Exception as e:
            logger.error(f"❌ Trade execution error: {e}")
            return None

    async def monitor_position(self, trade_result):
        """Monitor and manage the executed position"""

        if not trade_result:
            return

        symbol = trade_result.get('symbol')
        print(f"📈 MONITORING POSITION: {symbol}")

        try:
            # Monitor for 60 seconds
            for i in range(6):
                await asyncio.sleep(10)

                # Check position status
                position_status = await self.trader.monitor_positions()
                if position_status:
                    print(f"   Position Status: {position_status}")

                # Check for TP/SL hits
                if await self.check_tp_sl_hit(trade_result):
                    print("🎯 Take Profit or Stop Loss hit!")
                    break

            print("✅ Position monitoring complete")

        except Exception as e:
            logger.error(f"❌ Position monitoring error: {e}")

    async def check_tp_sl_hit(self, trade_result):
        """Check if TP or SL has been hit"""

        # This would integrate with exchange API to check current position status
        # For now, return False to continue monitoring
        return False

    async def generate_trade_report(self, trade_result):
        """Generate comprehensive trade report"""

        print("📊 GENERATING TRADE REPORT")

        report = {
            'timestamp': datetime.now().isoformat(),
            'trade_result': trade_result,
            'system_status': 'TRADE_EXECUTED',
            'performance_metrics': {
                'signal_confidence': getattr(trade_result, 'confidence_score', 0),
                'risk_reward_ratio': getattr(trade_result, 'risk_reward_ratio', 0),
                'position_size': trade_result.get('size', 0),
                'execution_price': trade_result.get('price', 0)
            }
        }

        # Save to GitHub
        try:
            await self.github_mcp.create_performance_issue(report)
            print("✅ Trade report saved to GitHub")
        except Exception as e:
            logger.error(f"❌ Failed to save report to GitHub: {e}")

        return report

async def main():
    """Main execution function"""

    print("🚀 STARTING COMPLETE LIVE TRADE EXECUTION")
    print("⚠️  WARNING: This will execute REAL trades with REAL money!")
    print("=" * 60)

    # Confirm execution
    confirm = input("Are you sure you want to execute a LIVE trade? (yes/no): ").strip().lower()
    if confirm != 'yes':
        print("❌ Trade execution cancelled")
        return

    # Initialize and run trade executor
    executor = LiveTradeExecutor()
    success = await executor.execute_complete_trade_cycle()

    if success:
        print("\\n🎉 LIVE TRADE EXECUTION SUCCESSFUL!")
        print("💰 A trade has been placed and is being monitored")
    else:
        print("\\n❌ LIVE TRADE EXECUTION FAILED!")
        print("🔍 Check logs for detailed error information")

if __name__ == "__main__":
    asyncio.run(main())
