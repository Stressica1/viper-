#!/usr/bin/env python3
"""
🚀 RUN BACKTESTING OPTIMIZER
Launcher script for MCP-powered backtesting and entry signal optimization

This script provides:
✅ Easy execution of backtesting tasks
✅ Entry signal optimization for avoiding initial drawdowns
✅ Comprehensive analysis reporting
✅ GitHub MCP integration for results tracking
✅ Automated parameter optimization
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime
from pathlib import Path
import argparse

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import backtesting components
from mcp_backtesting_optimizer import MCPBacktestingOptimizer, create_backtesting_task, run_backtesting_analysis, get_entry_signal_recommendations

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - BACKTEST_LAUNCHER - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BacktestingLauncher:
    """Launcher for MCP Backtesting Optimizer"""

    def __init__(self):
        self.optimizer = None
        self.results = {}

    async def initialize_optimizer(self):
        """Initialize the backtesting optimizer"""
        try:
            self.optimizer = MCPBacktestingOptimizer()
            logger.info("✅ Backtesting optimizer initialized")
            return True
        except Exception as e:
            logger.error(f"❌ Optimizer initialization failed: {e}")
            return False

    async def run_quick_analysis(self, symbols: List[str] = None, timeframes: List[str] = None, days: int = 30):
        """Run quick entry signal analysis"""
        try:
            if not symbols:
                symbols = ['BTCUSDT', 'ETHUSDT']
            if not timeframes:
                timeframes = ['1h', '4h']

            print("🔍 RUNNING QUICK ENTRY SIGNAL ANALYSIS")
            print("=" * 50)

            for symbol in symbols:
                for timeframe in timeframes:
                    print(f"\n📊 Analyzing {symbol} {timeframe}...")

                    # Get entry signal recommendations
                    recs = await get_entry_signal_recommendations(symbol, timeframe)

                    if recs.get('success_rate'):
                        success_rate = recs['success_rate']
                        avg_drawdown = recs['avg_drawdown']
                        max_drawdown = recs['max_drawdown']
                        immediate_loss = recs['immediate_loss_rate']

                        print(f"   ✅ Success Rate: {success_rate:.1f}%")
                        print(f"   📉 Avg Drawdown: {avg_drawdown:.3f}")
                        print(f"   📉 Max Drawdown: {max_drawdown:.3f}")
                        print(f"   ⚠️ Immediate Loss Rate: {immediate_loss:.1f}%")

                        # Store results
                        self.results[f"{symbol}_{timeframe}"] = recs

                        # Provide recommendations
                        if success_rate > 70:
                            print("   🟢 EXCELLENT: Entry signals performing well")
                        elif success_rate > 50:
                            print("   🟡 GOOD: Entry signals acceptable")
                        else:
                            print("   🔴 POOR: Entry signals need improvement")
                    else:
                        print(f"   ❌ No data available for {symbol} {timeframe}")

            print("\n📋 QUICK ANALYSIS SUMMARY")
            print("=" * 30)

            if self.results:
                success_rates = [r['success_rate'] for r in self.results.values()]
                avg_success = sum(success_rates) / len(success_rates)
                best_performer = max(self.results.items(), key=lambda x: x[1]['success_rate'])

                print(f"Average Success Rate: {avg_success:.1f}%")
                print(f"Best Performer: {best_performer[0]} ({best_performer[1]['success_rate']:.1f}%)")

                if avg_success > 65:
                    print("🟢 Overall: Entry signals are performing well")
                elif avg_success > 45:
                    print("🟡 Overall: Entry signals need some optimization")
                else:
                    print("🔴 Overall: Entry signals require significant improvement")
            else:
                print("❌ No analysis results available")

        except Exception as e:
            logger.error(f"❌ Quick analysis failed: {e}")

    async def run_comprehensive_backtest(self, symbols: List[str] = None, timeframes: List[str] = None, days: int = 90):
        """Run comprehensive backtesting analysis"""
        try:
            if not symbols:
                symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']
            if not timeframes:
                timeframes = ['1h', '4h']

            print("🚀 RUNNING COMPREHENSIVE BACKTESTING")
            print("=" * 50)
            print(f"Symbols: {', '.join(symbols)}")
            print(f"Timeframes: {', '.join(timeframes)}")
            print(f"Historical Days: {days}")
            print("=" * 50)

            # Create backtesting task
            task_config = {
                'symbols': symbols,
                'timeframes': timeframes,
                'days': days,
                'optimization_focus': 'entry_signals'
            }

            task_id = await create_backtesting_task(task_config)
            print(f"📋 Task Created: {task_id}")

            # Run comprehensive analysis
            start_time = datetime.now()
            results = await run_backtesting_analysis(symbols, timeframes, days)
            end_time = datetime.now()

            print("\n📊 COMPREHENSIVE ANALYSIS RESULTS")
            print("=" * 40)
            print(f"Status: {results['status']}")
            print(f"Duration: {(end_time - start_time).total_seconds():.1f} seconds")
            print(f"Symbols Analyzed: {results['symbols_analyzed']}")
            print(f"Timeframes Analyzed: {results['timeframes_analyzed']}")
            print(f"Entry Analyses: {results['entry_analyses']}")
            print(f"Backtest Results: {results['backtest_results']}")

            if results['status'] == 'completed':
                print("✅ Analysis completed successfully!")
                print("📄 Check the generated report files for detailed results")
            else:
                print(f"❌ Analysis failed: {results.get('error', 'Unknown error')}")

        except Exception as e:
            logger.error(f"❌ Comprehensive backtest failed: {e}")

    async def optimize_entry_signals(self, symbol: str, timeframe: str):
        """Optimize entry signals for a specific symbol/timeframe"""
        try:
            print(f"🎯 OPTIMIZING ENTRY SIGNALS FOR {symbol} {timeframe}")
            print("=" * 60)

            # Get current entry signal performance
            recs = await get_entry_signal_recommendations(symbol, timeframe)

            if recs.get('success_rate'):
                print("📊 CURRENT PERFORMANCE:")
                print(f"   Success Rate: {recs['success_rate']:.1f}%")
                print(f"   Avg Drawdown: {recs['avg_drawdown']:.3f}")
                print(f"   Max Drawdown: {recs['max_drawdown']:.3f}")
                print(f"   Immediate Loss Rate: {recs['immediate_loss_rate']:.1f}%")

                print("\n🔧 OPTIMIZATION RECOMMENDATIONS:")
                recommendations = recs.get('recommendations', {})

                if recommendations:
                    if recommendations.get('min_confidence_threshold'):
                        print(f"   🎯 Min Confidence Threshold: {recommendations['min_confidence_threshold']}")
                    if recommendations.get('max_allowed_drawdown'):
                        print(f"   📉 Max Allowed Drawdown: {recommendations['max_allowed_drawdown']}")
                    if recommendations.get('min_time_to_profit'):
                        print(f"   ⏱️ Min Time to Profit: {recommendations['min_time_to_profit']} hours")
                else:
                    print("   📋 No specific recommendations available")

                print("
💡 GENERAL IMPROVEMENT SUGGESTIONS:"                if recs['success_rate'] < 50:
                    print("   • Consider stricter entry filters")
                    print("   • Increase minimum confidence threshold")
                    print("   • Add additional technical confirmations")
                elif recs['avg_drawdown'] < -0.05:
                    print("   • Implement entry price validation")
                    print("   • Consider reducing position sizes")
                    print("   • Add immediate stop-loss protection")
                elif recs['immediate_loss_rate'] > 30:
                    print("   • Optimize entry timing")
                    print("   • Add market condition filters")
                    print("   • Consider avoiding certain market hours")

            else:
                print("❌ No entry signal data available for optimization")

        except Exception as e:
            logger.error(f"❌ Entry signal optimization failed: {e}")

    async def generate_optimization_report(self):
        """Generate comprehensive optimization report"""
        try:
            print("📄 GENERATING OPTIMIZATION REPORT")
            print("=" * 40)

            if not self.results:
                print("❌ No analysis results available")
                return

            # Create report
            report = {
                'report_type': 'backtesting_optimization',
                'generated_at': datetime.now().isoformat(),
                'analysis_results': self.results,
                'summary': {
                    'total_analyses': len(self.results),
                    'avg_success_rate': sum(r['success_rate'] for r in self.results.values()) / len(self.results),
                    'best_performer': max(self.results.items(), key=lambda x: x[1]['success_rate'])[0],
                    'worst_performer': min(self.results.items(), key=lambda x: x[1]['success_rate'])[0]
                },
                'recommendations': []
            }

            # Generate recommendations
            success_rates = [r['success_rate'] for r in self.results.values()]
            avg_success = sum(success_rates) / len(success_rates)

            if avg_success > 70:
                report['recommendations'].append("Entry signals are performing excellently - maintain current strategy")
            elif avg_success > 50:
                report['recommendations'].append("Entry signals are acceptable but could be improved")
                report['recommendations'].append("Consider implementing additional filters")
            else:
                report['recommendations'].append("Entry signals need significant improvement")
                report['recommendations'].append("Review and optimize entry criteria")
                report['recommendations'].append("Consider alternative entry strategies")

            # Save report
            report_filename = f"backtesting_optimization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)

            print(f"✅ Report saved: {report_filename}")
            print("📊 SUMMARY:")
            print(f"   Total Analyses: {report['summary']['total_analyses']}")
            print(f"   Average Success Rate: {report['summary']['avg_success_rate']:.1f}%")
            print(f"   Best Performer: {report['summary']['best_performer']}")
            print(f"   Recommendations: {len(report['recommendations'])}")

        except Exception as e:
            logger.error(f"❌ Report generation failed: {e}")

def main():
    """Main launcher function"""
    parser = argparse.ArgumentParser(description='MCP Backtesting Optimizer Launcher')
    parser.add_argument('--mode', choices=['quick', 'comprehensive', 'optimize', 'report'],
                       default='quick', help='Analysis mode')
    parser.add_argument('--symbols', nargs='+',
                       default=['BTCUSDT', 'ETHUSDT'],
                       help='Trading symbols to analyze')
    parser.add_argument('--timeframes', nargs='+',
                       default=['1h', '4h'],
                       help='Timeframes to analyze')
    parser.add_argument('--days', type=int, default=30,
                       help='Historical days to analyze')
    parser.add_argument('--symbol', type=str,
                       help='Single symbol for optimization')
    parser.add_argument('--timeframe', type=str,
                       help='Single timeframe for optimization')

    args = parser.parse_args()

    print("🎯 MCP BACKTESTING OPTIMIZER LAUNCHER")
    print("=" * 50)
    print(f"Mode: {args.mode}")
    print(f"Symbols: {', '.join(args.symbols)}")
    print(f"Timeframes: {', '.join(args.timeframes)}")
    print(f"Days: {args.days}")
    print("=" * 50)

    async def run_launcher():
        launcher = BacktestingLauncher()

        try:
            if args.mode == 'quick':
                await launcher.run_quick_analysis(args.symbols, args.timeframes, args.days)
            elif args.mode == 'comprehensive':
                await launcher.run_comprehensive_backtest(args.symbols, args.timeframes, args.days)
            elif args.mode == 'optimize':
                if args.symbol and args.timeframe:
                    await launcher.optimize_entry_signals(args.symbol, args.timeframe)
                else:
                    print("❌ Please specify --symbol and --timeframe for optimization mode")
                    return 1
            elif args.mode == 'report':
                await launcher.generate_optimization_report()
            else:
                print("❌ Invalid mode specified")
                return 1

            print("✅ Backtesting optimizer completed successfully")
            return 0

        except Exception as e:
            logger.error(f"❌ Launcher error: {e}")
            return 1

    try:
        exit_code = asyncio.run(run_launcher())
        return exit_code
    except KeyboardInterrupt:
        print("\n🛑 Operation cancelled by user")
        return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
