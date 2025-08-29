#!/usr/bin/env python3
"""
🚀 RUN MASSIVE BACKTEST - 50 Pairs × 200 Configs
Launcher for comprehensive backtesting operation

This launcher provides:
✅ Easy execution of massive backtesting
✅ Real-time progress monitoring
✅ Resource management and optimization
✅ GitHub MCP integration
✅ Comprehensive results analysis
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime
from pathlib import Path
import argparse
import psutil

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import massive backtest components
from massive_backtest_orchestrator import (
    MassiveBacktestOrchestrator,
    create_massive_backtest_task,
    run_massive_backtest_operation,
    get_massive_backtest_status
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - MASSIVE_LAUNCHER - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MassiveBacktestLauncher:
    """Launcher for massive backtesting operations"""

    def __init__(self):
        self.orchestrator = None
        self.system_info = self._get_system_info()

    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for resource planning"""
        try:
            memory = psutil.virtual_memory()
            cpu_count = psutil.cpu_count()
            cpu_logical = psutil.cpu_count(logical=True)

            return {
                'total_memory_gb': memory.total / (1024**3),
                'available_memory_gb': memory.available / (1024**3),
                'cpu_cores': cpu_count,
                'cpu_logical': cpu_logical,
                'memory_usage_pct': memory.percent,
                'cpu_usage_pct': psutil.cpu_percent(interval=1)
            }

        except Exception as e:
            logger.warning(f"⚠️ Could not get system info: {e}")
            return {}

    async def validate_system_requirements(self) -> bool:
        """Validate system requirements for massive backtesting"""
        try:
            print("🔍 VALIDATING SYSTEM REQUIREMENTS")
            print("=" * 40)

            # Check memory
            min_memory_gb = 8.0
            available_memory = self.system_info.get('available_memory_gb', 0)

            print(f"💾 Memory: {available_memory:.1f}GB available")
            if available_memory < min_memory_gb:
                print(f"❌ Insufficient memory: {min_memory_gb}GB required")
                return False
            print("✅ Memory requirements met")

            # Check CPU cores
            cpu_cores = self.system_info.get('cpu_cores', 1)
            min_cores = 4

            print(f"🖥️ CPU Cores: {cpu_cores}")
            if cpu_cores < min_cores:
                print(f"⚠️ Low CPU cores: {min_cores} recommended")
            else:
                print("✅ CPU requirements met")

            # Check disk space
            disk = psutil.disk_usage('/')
            free_gb = disk.free / (1024**3)

            print(f"💽 Free Disk Space: {free_gb:.1f}GB")
            if free_gb < 50:
                print("⚠️ Low disk space: Consider freeing up space")
            else:
                print("✅ Disk space sufficient")

            return True

        except Exception as e:
            logger.error(f"❌ System validation failed: {e}")
            return False

    async def estimate_execution_time(self) -> Dict[str, Any]:
        """Estimate execution time for massive backtest"""
        try:
            # Load configuration
            config_path = project_root / "massive_backtest_config.json"
            with open(config_path, 'r') as f:
                config_data = json.load(f)

            massive_config = config_data['massive_backtest_configuration']

            # Calculate combinations
            pairs = len(massive_config['trading_pairs'])
            timeframes = len(massive_config['timeframes'])

            # Count configurations
            variations = massive_config['configuration_variations']
            ma_configs = len(variations['moving_average_configs'])
            atr_configs = len(variations['atr_configs'])
            risk_configs = len(variations['risk_configs'])
            trend_configs = len(variations['trend_configs'])
            entry_filters = len(variations['entry_filters'])

            total_configs = ma_configs * atr_configs * risk_configs * trend_configs * entry_filters
            total_combinations = pairs * timeframes * total_configs

            # Estimate time per backtest (seconds)
            avg_time_per_backtest = 30  # Conservative estimate
            total_seconds = total_combinations * avg_time_per_backtest

            # Adjust for parallel processing
            max_concurrent = massive_config['processing_strategy']['max_concurrent_pairs']
            effective_seconds = total_seconds / max_concurrent

            # Convert to hours
            total_hours = total_seconds / 3600
            effective_hours = effective_seconds / 3600

            return {
                'total_combinations': total_combinations,
                'total_configs': total_configs,
                'trading_pairs': pairs,
                'timeframes': timeframes,
                'max_concurrent_pairs': max_concurrent,
                'estimated_total_hours': total_hours,
                'estimated_effective_hours': effective_hours,
                'estimated_days': effective_hours / 24,
                'avg_time_per_backtest': avg_time_per_backtest
            }

        except Exception as e:
            logger.error(f"❌ Time estimation failed: {e}")
            return {}

    async def initialize_orchestrator(self) -> bool:
        """Initialize the massive backtest orchestrator"""
        try:
            print("🔧 INITIALIZING ORCHESTRATOR...")
            self.orchestrator = MassiveBacktestOrchestrator()
            await self.orchestrator.initialize_optimizer()
            print("✅ Orchestrator initialized successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Orchestrator initialization failed: {e}")
            return False

    async def run_massive_backtest(self) -> int:
        """Run the complete massive backtesting operation"""
        try:
            print("🚀 MASSIVE BACKTEST LAUNCHER")
            print("=" * 60)

            # System validation
            if not await self.validate_system_requirements():
                print("❌ System requirements not met")
                return 1

            # Time estimation
            time_estimate = await self.estimate_execution_time()
            if time_estimate:
                print("
⏱️ EXECUTION ESTIMATE:"                print(f"   Total Combinations: {time_estimate['total_combinations']:,}")
                print(f"   Trading Pairs: {time_estimate['trading_pairs']}")
                print(f"   Configurations: {time_estimate['total_configs']}")
                print(f"   Timeframes: {time_estimate['timeframes']}")
                print(f"   Max Concurrent: {time_estimate['max_concurrent_pairs']}")
                print(f"   Estimated Time: {time_estimate['estimated_effective_hours']:.1f} hours ({time_estimate['estimated_days']:.1f} days)")

                # Confirm execution
                if time_estimate['estimated_effective_hours'] > 24:
                    confirm = input(f"\n🚨 This will take {time_estimate['estimated_days']:.1f} days. Continue? (yes/no): ").strip().lower()
                    if confirm != 'yes':
                        print("❌ Operation cancelled")
                        return 0

            # Initialize orchestrator
            if not await self.initialize_orchestrator():
                return 1

            # Create task
            print("\n📋 Creating massive backtest task...")
            task_id = await create_massive_backtest_task()

            if not task_id:
                print("❌ Failed to create backtest task")
                return 1

            print(f"✅ Task Created: {task_id}")

            # Execute massive backtest
            print("\n🚀 EXECUTING MASSIVE BACKTEST...")
            print("Progress will be logged to massive_backtest.log")
            print("Use Ctrl+C to interrupt (results will be saved)")
            print("=" * 60)

            results = await run_massive_backtest_operation(task_id)

            # Display results
            await self.display_results(results)

            return 0 if results.get('status') != 'failed' else 1

        except KeyboardInterrupt:
            print("\n🛑 Operation interrupted by user")
            await self.handle_interrupt()
            return 0
        except Exception as e:
            logger.error(f"❌ Massive backtest failed: {e}")
            print(f"❌ ERROR: {e}")
            return 1

    async def display_results(self, results: Dict[str, Any]):
        """Display comprehensive results"""
        try:
            print("\n" + "=" * 80)
            print("📊 MASSIVE BACKTEST RESULTS")
            print("=" * 80)

            if results.get('status') == 'failed':
                print(f"❌ Operation Failed: {results.get('error', 'Unknown error')}")
                return

            # Summary stats
            summary = results
            overall_stats = summary.get('overall_stats', {})

            print("📈 OVERALL PERFORMANCE:")
            print(f"   Total Results: {summary.get('total_results', 0):,}")
            print(f"   Failed Tasks: {summary.get('total_failed', 0)}")
            print(f"   Average Win Rate: {overall_stats.get('avg_win_rate', 0):.1f}%")
            print(f"   Median Win Rate: {overall_stats.get('median_win_rate', 0):.1f}%")
            print(f"   Best Win Rate: {overall_stats.get('best_win_rate', 0):.1f}%")
            print(f"   Average Total P&L: ${overall_stats.get('avg_total_pnl', 0):.2f}")
            print(f"   Best Total P&L: ${overall_stats.get('best_total_pnl', 0):.2f}")

            # Success analysis
            success_analysis = summary.get('success_analysis', {})
            print("
🎯 SUCCESS ANALYSIS:"            print(f"   Profitable Configurations: {success_analysis.get('profitable_configs', 0)}")
            print(f"   High Win Rate Configs (≥60%): {success_analysis.get('high_win_rate_configs', 0)}")
            print(f"   High Sharpe Configs (≥1.0): {success_analysis.get('high_sharpe_configs', 0)}")
            print(f"   Elite Configurations: {success_analysis.get('elite_configs', 0)}")
            print(f"   Overall Success Rate: {success_analysis.get('success_rate_pct', 0):.1f}%")

            # Best performers
            await self.display_best_performers(summary)

            # Pair analysis
            await self.display_pair_analysis(summary)

            # Recommendations
            await self.display_recommendations(summary)

            print("
📄 DETAILED RESULTS SAVED:"            print("   - massive_backtest_results_*.json (comprehensive results)")
            print("   - massive_backtest_results_*_summary.csv (summary CSV)")
            print("   - massive_backtest_results_*_top_performers.json (top configs)")

        except Exception as e:
            logger.error(f"❌ Results display failed: {e}")

    async def display_best_performers(self, summary: Dict[str, Any]):
        """Display best performing configurations"""
        try:
            print("
🏆 TOP PERFORMERS:"            print("-" * 40)

            # Best by Sharpe Ratio
            best_sharpe = summary.get('best_by_sharpe_ratio', [])
            if best_sharpe:
                top = best_sharpe[0]
                print("🎯 Best Sharpe Ratio:"                print(f"   Symbol: {top.get('symbol', 'N/A')} {top.get('timeframe', 'N/A')}")
                print(f"   Sharpe Ratio: {top.get('sharpe_ratio', 0):.2f}")
                print(f"   Win Rate: {top.get('win_rate', 0):.1f}%")
                print(f"   Total P&L: ${top.get('total_pnl', 0):.2f}")

            # Best by Win Rate
            best_win_rate = summary.get('best_by_win_rate', [])
            if best_win_rate:
                top = best_win_rate[0]
                print("
💯 Best Win Rate:"                print(f"   Symbol: {top.get('symbol', 'N/A')} {top.get('timeframe', 'N/A')}")
                print(f"   Win Rate: {top.get('win_rate', 0):.1f}%")
                print(f"   Sharpe Ratio: {top.get('sharpe_ratio', 0):.2f}")
                print(f"   Total P&L: ${top.get('total_pnl', 0):.2f}")

            # Best by Total P&L
            best_pnl = summary.get('best_by_total_pnl', [])
            if best_pnl:
                top = best_pnl[0]
                print("
💰 Best Total P&L:"                print(f"   Symbol: {top.get('symbol', 'N/A')} {top.get('timeframe', 'N/A')}")
                print(f"   Total P&L: ${top.get('total_pnl', 0):.2f}")
                print(f"   Win Rate: {top.get('win_rate', 0):.1f}%")
                print(f"   Sharpe Ratio: {top.get('sharpe_ratio', 0):.2f}")

        except Exception as e:
            logger.error(f"❌ Best performers display failed: {e}")

    async def display_pair_analysis(self, summary: Dict[str, Any]):
        """Display pair performance analysis"""
        try:
            pair_performance = summary.get('pair_performance', {})
            if not pair_performance:
                return

            print("
📊 PAIR PERFORMANCE ANALYSIS:"            print("-" * 40)

            # Sort by profitability
            sorted_pairs = sorted(pair_performance.items(),
                                key=lambda x: x[1]['profitability_pct'], reverse=True)

            for symbol, stats in sorted_pairs[:10]:  # Top 10
                print(f"   {symbol}:")
                print(f"     Tests: {stats['total_tests']}")
                print(f"     Profitable: {stats['profitable_tests']} ({stats['profitability_pct']:.1f}%)")
                print(f"     Avg Win Rate: {stats['avg_win_rate']:.1f}%")
                print(f"     Best Sharpe: {stats['best_sharpe']:.2f}")

        except Exception as e:
            logger.error(f"❌ Pair analysis display failed: {e}")

    async def display_recommendations(self, summary: Dict[str, Any]):
        """Display optimization recommendations"""
        try:
            print("
💡 RECOMMENDATIONS:"            print("-" * 40)

            overall_stats = summary.get('overall_stats', {})
            success_analysis = summary.get('success_analysis', {})

            # Win rate recommendations
            avg_win_rate = overall_stats.get('avg_win_rate', 0)
            if avg_win_rate > 65:
                print("✅ Excellent average win rate - maintain current strategy framework")
            elif avg_win_rate > 50:
                print("🟡 Good win rate - focus on parameter optimization")
            else:
                print("🔴 Poor win rate - significant strategy revision needed")

            # Profitability recommendations
            success_rate = success_analysis.get('success_rate_pct', 0)
            if success_rate > 60:
                print("✅ High profitability rate - strategy is fundamentally sound")
            elif success_rate > 40:
                print("🟡 Moderate profitability - focus on risk management")
            else:
                print("🔴 Low profitability - review entry/exit criteria")

            # Sharpe ratio recommendations
            avg_sharpe = overall_stats.get('avg_sharpe_ratio', 0)
            if avg_sharpe > 1.0:
                print("✅ Strong risk-adjusted returns - excellent performance")
            elif avg_sharpe > 0.5:
                print("🟡 Moderate risk-adjusted returns - acceptable performance")
            else:
                print("🔴 Poor risk-adjusted returns - improve return consistency")

        except Exception as e:
            logger.error(f"❌ Recommendations display failed: {e}")

    async def handle_interrupt(self):
        """Handle user interruption gracefully"""
        try:
            print("\n💾 Saving current progress...")
            if self.orchestrator:
                # Save current state
                status = {
                    'interrupted_at': datetime.now().isoformat(),
                    'completed_tasks': self.orchestrator.processing_stats.completed_tasks,
                    'failed_tasks': self.orchestrator.processing_stats.failed_tasks,
                    'message': 'Operation interrupted by user'
                }

                with open('massive_backtest_interrupt_status.json', 'w') as f:
                    json.dump(status, f, indent=2, default=str)

            print("✅ Progress saved. You can resume later.")
            print("📄 Check massive_backtest_interrupt_status.json for details")

        except Exception as e:
            logger.error(f"❌ Interrupt handling failed: {e}")

def main():
    """Main launcher function"""
    parser = argparse.ArgumentParser(description='Massive Backtest Launcher - 50 Pairs × 200 Configs')
    parser.add_argument('--mode', choices=['run', 'estimate', 'status'],
                       default='run', help='Operation mode')
    parser.add_argument('--config', type=str,
                       help='Path to custom configuration file')

    args = parser.parse_args()

    print("🚀 MASSIVE BACKTEST LAUNCHER - 50 Pairs × 200 Configs")
    print("=" * 70)

    if args.config:
        print(f"Config: {args.config}")

    async def run_launcher():
        launcher = MassiveBacktestLauncher()

        if args.mode == 'estimate':
            # Show time estimation
            estimate = await launcher.estimate_execution_time()
            if estimate:
                print("⏱️ EXECUTION ESTIMATE:")
                print(f"   Total Combinations: {estimate['total_combinations']:,}")
                print(f"   Estimated Time: {estimate['estimated_effective_hours']:.1f} hours")
                print(f"   Estimated Days: {estimate['estimated_days']:.1f} days")
            return 0

        elif args.mode == 'status':
            # Show current status
            status = await get_massive_backtest_status()
            print(f"Status: {status.get('status', 'unknown')}")
            if status.get('message'):
                print(f"Message: {status['message']}")
            return 0

        elif args.mode == 'run':
            # Run massive backtest
            return await launcher.run_massive_backtest()

    try:
        exit_code = asyncio.run(run_launcher())
        return exit_code
    except KeyboardInterrupt:
        print("\n🛑 Operation cancelled")
        return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
