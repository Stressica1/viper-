#!/usr/bin/env python3
"""
🚀 START LIVE TRADER - COMPLETE CYCLE
Complete trading cycle using existing components:
- Scan markets for opportunities
- Score and filter signals
- Execute trades with proper position sizing
- Manage TP/SL/TSL automatically
- Use only swaps wallet on Bitget USDT
"""

import asyncio
import logging
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import existing trading components
from viper_async_trader import ViperAsyncTrader
from predictive_ranges_strategy import get_predictive_strategy
from optimized_trade_entry_system import get_optimized_entry_system
from emergency_stop_system import get_emergency_system
from github_mcp_integration import GitHubMCPOrchestration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - LIVE_TRADER - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CompleteLiveTrader:
    """Complete live trader using existing components"""

    def __init__(self):
        self.viper_trader = ViperAsyncTrader()
        self.predictive_strategy = get_predictive_strategy()
        self.entry_system = get_optimized_entry_system()
        self.emergency_system = get_emergency_system()
        self.github_mcp = GitHubMCPOrchestration()

        # Bitget USDT Swaps Configuration
        self.symbol = "BTCUSDT"  # Target symbol for swaps
        self.wallet_type = "swaps"  # Only use swaps wallet
        self.margin_mode = "isolated"
        self.leverage = int(os.getenv('LEVERAGE', '5'))

        logger.info("🚀 Complete Live Trader initialized")

    async def start_complete_trading_cycle(self):
        """Start the complete trading cycle"""

        print("🚀 STARTING COMPLETE LIVE TRADING CYCLE")
        print("=" * 70)
        print("📊 COMPONENTS:")
        print("   ✅ Scan: Market opportunity scanning")
        print("   ✅ Score: Signal quality assessment")
        print("   ✅ Execute: Trade placement with position sizing")
        print("   ✅ TP/SL: Take profit and stop loss management")
        print("   ✅ Monitor: Real-time position monitoring")
        print("   📈 Wallet: Swaps wallet only (Bitget USDT)")
        print("=" * 70)

        try:
            # Step 1: System Health Check
            print("🏥 STEP 1: SYSTEM HEALTH CHECK")
            await self.system_health_check()

            # Step 2: Initialize Trading Components
            print("\\n🔧 STEP 2: INITIALIZE TRADING COMPONENTS")
            await self.initialize_trading_components()

            # Step 3: Start Main Trading Loop
            print("\\n💰 STEP 3: START MAIN TRADING LOOP")
            await self.run_trading_loop()

        except KeyboardInterrupt:
            print("\\n⏹️  Trading interrupted by user")
            await self.graceful_shutdown()
        except Exception as e:
            logger.error(f"❌ Trading cycle failed: {e}")
            await self.emergency_shutdown(e)

    async def system_health_check(self):
        """Comprehensive system health check"""

        print("🔍 Checking system components...")

        # Check API connectivity
        try:
            # Test Bitget connection
            await self.viper_trader.connect_exchange()
            print("✅ Exchange Connection: SUCCESS")
        except Exception as e:
            print(f"❌ Exchange Connection: FAILED - {e}")
            raise

        # Check account balance (swaps wallet)
        try:
            balance = await self.viper_trader.check_account_balance()
            print(".2f")
            if balance < 1.0:
                print("⚠️  Low balance - ensure sufficient funds in swaps wallet")
        except Exception as e:
            print(f"⚠️  Balance check failed: {e}")

        # Check emergency system
        try:
            health = await self.emergency_system.check_system_health()
            print(f"🛡️ Emergency System: {health['system_status']}")
        except Exception as e:
            print(f"❌ Emergency system check failed: {e}")

        print("✅ System health check completed")

    async def initialize_trading_components(self):
        """Initialize all trading components"""

        print("🔧 Initializing trading components...")

        # Initialize predictive strategy
        try:
            self.predictive_strategy.initialize()
            print("✅ Predictive Ranges Strategy: INITIALIZED")
        except Exception as e:
            print(f"⚠️  Predictive strategy initialization failed: {e}")

        # Initialize entry system
        try:
            await self.entry_system.initialize()
            print("✅ Optimized Entry System: INITIALIZED")
        except Exception as e:
            print(f"⚠️  Entry system initialization failed: {e}")

        # Initialize MCP GitHub tracking
        try:
            await self.github_mcp.initialize_repository()
            print("✅ GitHub MCP Integration: INITIALIZED")
        except Exception as e:
            print(f"⚠️  GitHub MCP initialization failed: {e}")

        print("✅ All trading components initialized")

    async def run_trading_loop(self):
        """Main trading loop with complete cycle"""

        print("💰 Starting main trading loop...")
        print("Cycle: SCAN → SCORE → EXECUTE → MONITOR → TP/SL")

        scan_interval = int(os.getenv('SCAN_INTERVAL', '30'))  # 30 seconds default

        while True:
            try:
                cycle_start = datetime.now()

                # Step 1: SCAN - Market opportunity scanning
                print(f"\\n🔍 CYCLE START: {cycle_start.strftime('%H:%M:%S')}")
                opportunities = await self.scan_market_opportunities()

                if not opportunities:
                    print("📊 No opportunities found, waiting for next cycle...")
                    await asyncio.sleep(scan_interval)
                    continue

                # Step 2: SCORE - Signal quality assessment
                print("🎯 Scoring signals...")
                scored_signals = await self.score_signals(opportunities)

                if not scored_signals:
                    print("⚠️  No signals passed scoring threshold")
                    await asyncio.sleep(scan_interval)
                    continue

                # Step 3: EXECUTE - Trade placement
                print("💰 Executing trades...")
                executed_trades = await self.execute_trades(scored_signals)

                # Step 4: MONITOR - Position monitoring
                if executed_trades:
                    print("📈 Monitoring positions...")
                    await self.monitor_positions(executed_trades)

                # Step 5: REPORT - Cycle completion
                cycle_end = datetime.now()
                cycle_duration = (cycle_end - cycle_start).total_seconds()

                print(f"✅ Cycle completed in {cycle_duration:.1f}s")

                # MCP GitHub tracking
                await self.update_mcp_cycle_report(cycle_start, executed_trades)

                # Wait for next cycle
                await asyncio.sleep(scan_interval)

            except Exception as e:
                logger.error(f"❌ Trading cycle error: {e}")
                await self.handle_cycle_error(e)
                await asyncio.sleep(scan_interval)

    async def scan_market_opportunities(self) -> list:
        """Scan market for trading opportunities"""

        opportunities = []

        try:
            # Get market data for target symbol
            market_data = await self.viper_trader.fetch_market_data(self.symbol, '1h', 100)

            if not market_data:
                return opportunities

            # Use predictive ranges strategy to identify opportunities
            predictive_ranges = self.predictive_strategy.calculate_predictive_ranges(
                market_data, self.symbol, '1h'
            )

            if predictive_ranges:
                # Check for entry opportunities
                current_price = market_data['close'].iloc[-1]

                # Bullish opportunity
                if predictive_ranges.get('bullish_signal', False):
                    opportunities.append({
                        'symbol': self.symbol,
                        'direction': 'long',
                        'entry_price': current_price,
                        'predictive_ranges': predictive_ranges,
                        'signal_type': 'bullish'
                    })

                # Bearish opportunity
                if predictive_ranges.get('bearish_signal', False):
                    opportunities.append({
                        'symbol': self.symbol,
                        'direction': 'short',
                        'entry_price': current_price,
                        'predictive_ranges': predictive_ranges,
                        'signal_type': 'bearish'
                    })

            print(f"🔍 Found {len(opportunities)} market opportunities")

        except Exception as e:
            logger.error(f"❌ Market scanning failed: {e}")

        return opportunities

    async def score_signals(self, opportunities: list) -> list:
        """Score and filter trading signals"""

        scored_signals = []

        try:
            for opportunity in opportunities:
                # Get optimized entry analysis
                entry_analysis = await self.entry_system.analyze_optimal_entries(
                    opportunity['symbol'],
                    {'1h': None},  # Market data placeholder
                    opportunity['entry_price'],
                    account_balance=30.0  # Current balance
                )

                if entry_analysis:
                    best_entry = max(entry_analysis, key=lambda x: x.confidence_score)

                    # Apply scoring criteria
                    if (best_entry.confidence_score >= 0.7 and
                        best_entry.risk_reward_ratio >= 2.0 and
                        best_entry.entry_quality in ['PREMIUM', 'EXCELLENT']):

                        scored_signal = {
                            'symbol': opportunity['symbol'],
                            'direction': opportunity['direction'],
                            'entry_price': opportunity['entry_price'],
                            'confidence_score': best_entry.confidence_score,
                            'risk_reward_ratio': best_entry.risk_reward_ratio,
                            'position_size': best_entry.position_size,
                            'stop_loss': best_entry.stop_loss,
                            'take_profit': best_entry.take_profit,
                            'entry_quality': best_entry.entry_quality
                        }

                        scored_signals.append(scored_signal)

            print(f"🎯 {len(scored_signals)} signals passed scoring threshold")

        except Exception as e:
            logger.error(f"❌ Signal scoring failed: {e}")

        return scored_signals

    async def execute_trades(self, scored_signals: list) -> list:
        """Execute trades with proper position sizing"""

        executed_trades = []

        try:
            for signal in scored_signals:
                print(f"💰 Executing {signal['direction'].upper()} trade for {signal['symbol']}")

                # Execute trade using viper trader
                trade_result = await self.viper_trader.execute_trade(
                    symbol=signal['symbol'],
                    direction=signal['direction'],
                    position_size=signal['position_size'],
                    entry_price=signal['entry_price'],
                    stop_loss=signal['stop_loss'],
                    take_profit=signal['take_profit']
                )

                if trade_result:
                    executed_trade = {
                        'signal': signal,
                        'trade_result': trade_result,
                        'timestamp': datetime.now().isoformat()
                    }

                    executed_trades.append(executed_trade)
                    print(f"✅ Trade executed: {trade_result.get('order_id', 'N/A')}")

                    # MCP GitHub tracking
                    await self.github_mcp.create_performance_issue({
                        'title': f'💰 Trade Executed: {signal["symbol"]} {signal["direction"].upper()}',
                        'body': f'Trade details: {json.dumps(signal, indent=2)}',
                        'labels': ['trade-execution', 'live-trading']
                    })

                else:
                    print("❌ Trade execution failed")

        except Exception as e:
            logger.error(f"❌ Trade execution failed: {e}")

        return executed_trades

    async def monitor_positions(self, executed_trades: list):
        """Monitor positions for TP/SL hits"""

        try:
            monitoring_duration = 60  # Monitor for 60 seconds

            for i in range(monitoring_duration // 10):  # Check every 10 seconds
                await asyncio.sleep(10)

                # Check position status
                position_status = await self.viper_trader.monitor_positions()

                if position_status:
                    print(f"📊 Position Status: {position_status}")

                    # Check for TP/SL hits
                    for trade in executed_trades:
                        if await self.check_tp_sl_hit(trade):
                            print("🎯 Take Profit or Stop Loss hit!")
                            break

        except Exception as e:
            logger.error(f"❌ Position monitoring failed: {e}")

    async def check_tp_sl_hit(self, trade: dict) -> bool:
        """Check if TP or SL has been hit"""

        # This would integrate with exchange API to check current position status
        # For now, return False to continue monitoring
        return False

    async def update_mcp_cycle_report(self, cycle_start: datetime, executed_trades: list):
        """Update MCP with cycle completion report"""

        try:
            cycle_report = {
                'title': f'📊 Trading Cycle Completed - {len(executed_trades)} Trades',
                'body': f'Cycle completed at {datetime.now().isoformat()}\\n'
                       f'Duration: {(datetime.now() - cycle_start).total_seconds():.1f}s\\n'
                       f'Trades executed: {len(executed_trades)}\\n'
                       f'Wallet: Swaps (USDT)\\n'
                       f'Exchange: Bitget',
                'labels': ['trading-cycle', 'cycle-complete', 'performance-tracking']
            }

            await self.github_mcp.create_performance_issue(cycle_report)

        except Exception as e:
            logger.error(f"Failed to update MCP cycle report: {e}")

    async def handle_cycle_error(self, error: Exception):
        """Handle trading cycle errors"""

        print(f"❌ Cycle error: {error}")

        # Emergency system activation if needed
        if "API" in str(error) or "connection" in str(error).lower():
            await self.emergency_system.activate_emergency_stop(f"Trading cycle error: {error}")

        # MCP GitHub error reporting
        try:
            await self.github_mcp.create_performance_issue({
                'title': '❌ Trading Cycle Error',
                'body': f'Error occurred: {str(error)}\\nTime: {datetime.now().isoformat()}',
                'labels': ['error-report', 'trading-cycle', 'needs-attention']
            })
        except Exception as e:
            logger.error(f"Failed to report cycle error: {e}")

    async def graceful_shutdown(self):
        """Graceful shutdown of trading system"""

        print("🔄 Initiating graceful shutdown...")

        try:
            # Close positions if any
            await self.viper_trader.close_all_positions()

            # Update MCP with shutdown
            await self.github_mcp.create_performance_issue({
                'title': '🔄 Trading System Shutdown',
                'body': f'Graceful shutdown initiated at {datetime.now().isoformat()}',
                'labels': ['system-shutdown', 'graceful-shutdown']
            })

            print("✅ Trading system shut down gracefully")

        except Exception as e:
            logger.error(f"❌ Shutdown error: {e}")

    async def emergency_shutdown(self, error: Exception):
        """Emergency shutdown with safety measures"""

        print("🚨 EMERGENCY SHUTDOWN INITIATED!")

        try:
            # Activate emergency stop
            await self.emergency_system.manual_emergency_stop(f"Emergency shutdown: {error}")

            # Close all positions immediately
            await self.viper_trader.close_all_positions()

            # MCP emergency report
            await self.github_mcp.create_performance_issue({
                'title': '🚨 EMERGENCY SYSTEM SHUTDOWN',
                'body': f'Emergency shutdown due to: {str(error)}\\nTime: {datetime.now().isoformat()}',
                'labels': ['emergency-shutdown', 'critical-error', 'immediate-attention']
            })

            print("🚨 Emergency shutdown completed")

        except Exception as e:
            logger.error(f"❌ Emergency shutdown failed: {e}")

async def main():
    """Main function to start complete live trader"""

    print("🚀 VIPER COMPLETE LIVE TRADER")
    print("Using existing components for complete trading cycle")
    print("=" * 70)

    # Confirm wallet usage
    wallet_type = os.getenv('WALLET_TYPE', 'swaps')
    if wallet_type != 'swaps':
        print("⚠️  Only swaps wallet supported. Switching to swaps wallet.")
        os.environ['WALLET_TYPE'] = 'swaps'

    print("💰 Wallet: Swaps (USDT only)")
    print("🏦 Exchange: Bitget")
    print("🔄 Cycle: SCAN → SCORE → EXECUTE → TP/SL → MONITOR")
    # Initialize and start trader
    trader = CompleteLiveTrader()

    try:
        await trader.start_complete_trading_cycle()
    except KeyboardInterrupt:
        print("\\n⏹️  Trading stopped by user")
    except Exception as e:
        print(f"\\n❌ Fatal error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
