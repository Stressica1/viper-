#!/usr/bin/env python3
"""
💰 $30 PORTFOLIO LIVE TRADING DEMO - VIPER Real-Time Balance Tracking
======================================================================

Demonstration of complete live trading system with $30 portfolio and real-time balance tracking.

Features:
- $30 portfolio with appropriate risk management
- Real-time balance updates every 60 seconds
- 5 high-performance trading strategies
- Risk limits scaled for small portfolio
- Live trading simulation with monitoring
- GitHub MCP integration ready

Author: VIPER Development Team
Version: 1.0.0
Date: 2025-01-29
"""

import os
import sys
import time
import threading
from datetime import datetime
from pathlib import Path

# Import our components
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from live_trading_manager import LiveTradingManager
    from strategy_metrics_dashboard import StrategyMetricsDashboard
    from github_mcp_trading_tasks import GitHubMCPTradingTasks
except ImportError:
    # Fallback for direct execution
    import live_trading_manager as ltm
    import strategy_metrics_dashboard as smd
    import github_mcp_trading_tasks as gmt
    LiveTradingManager = ltm.LiveTradingManager
    StrategyMetricsDashboard = smd.StrategyMetricsDashboard
    GitHubMCPTradingTasks = gmt.GitHubMCPTradingTasks

class ThirtyDollarPortfolioDemo:
    """Demonstration of $30 portfolio live trading system"""

    def __init__(self):
        self.trading_manager = None
        self.strategy_dashboard = None
        self.github_mcp = None
        self.is_running = False

    def initialize_demo(self):
        """Initialize the demo system"""
        print("🚀 VIPER $30 PORTFOLIO LIVE TRADING DEMO")
        print("=" * 60)

        try:
            print("🔧 Initializing demo components...")

            # Initialize strategy dashboard
            print("  📊 Loading strategy metrics dashboard...")
            self.strategy_dashboard = StrategyMetricsDashboard()

            # Initialize GitHub MCP
            print("  🐙 Initializing GitHub MCP integration...")
            self.github_mcp = GitHubMCPTradingTasks()

            # Initialize live trading manager with $30 config
            print("  💰 Setting up live trading manager...")
            config = {
                'max_positions': 5,  # Reduced for $30 portfolio
                'max_positions_per_strategy': 1,
                'risk_per_trade': 0.015,  # 1.5% per trade for safety
                'max_daily_loss': 1.50,  # $1.50 max daily loss
                'max_daily_trades': 10,  # Reduced trading frequency
                'min_leverage': 34.0,
                'monitoring_interval': 30,  # 30 second monitoring
                'alert_thresholds': {
                    'high_drawdown': -3.0,  # 10% drawdown
                    'low_win_rate': 55.0,   # 55% minimum win rate
                    'high_volatility': 12.0  # 12% volatility threshold
                }
            }
            self.trading_manager = LiveTradingManager(config)

            print("✅ Demo system initialized successfully")
            return True

        except Exception as e:
            print(f"❌ Demo initialization failed: {e}")
            return False

    def display_portfolio_overview(self):
        """Display comprehensive portfolio overview"""
        print("\n💰 $30 PORTFOLIO OVERVIEW")
        print("=" * 40)

        if self.strategy_dashboard:
            portfolio_summary = self.strategy_dashboard.display_portfolio_summary()
            print(portfolio_summary)

        print("\n📊 STRATEGY ALLOCATIONS (for $30 portfolio):")
        print("-" * 50)

        if self.strategy_dashboard:
            for strategy in self.strategy_dashboard.strategies.values():
                allocation = (strategy.weight / 100) * 30.0
                print(f"  {strategy.strategy_name[:20]:<20} | {strategy.weight:>5.1f}% | ${allocation:>6.2f}")

        print(f"\n💡 TOTAL ALLOCATION: $30.00")
        print("💡 RISK BUDGET: $1.50 max daily loss (5.0%)")
        print("💡 POSITION LIMIT: $0.50 max per position (1.67%)")

    def display_strategy_performance(self):
        """Display strategy performance table"""
        print("\n📈 STRATEGY PERFORMANCE METRICS")
        print("=" * 50)

        if self.strategy_dashboard:
            table = self.strategy_dashboard.display_strategy_table()
            print(table)

    def simulate_real_time_updates(self, duration_seconds: int = 300):
        """Simulate real-time balance updates"""
        print(f"\n🔄 SIMULATING REAL-TIME BALANCE UPDATES ({duration_seconds} seconds)")
        print("=" * 60)

        start_time = time.time()
        update_count = 0

        while time.time() - start_time < duration_seconds and self.is_running:
            try:
                if self.trading_manager:
                    # Simulate balance updates
                    current_balance = self.trading_manager.get_real_time_balance()
                    timestamp = datetime.now().strftime('%H:%M:%S')

                    # Add small random variation
                    import random
                    variation = random.uniform(-0.05, 0.05)  # ±$0.05
                    new_balance = max(25.0, min(35.0, current_balance + variation))  # Keep within $25-$35 range

                    # Update balance (this would normally come from exchange API)
                    self.trading_manager.real_time_balance = new_balance
                    self.trading_manager.last_balance_update = datetime.now()

                    update_count += 1

                    if update_count % 10 == 0:  # Show every 10th update
                        pnl = new_balance - 30.0
                        pnl_percent = (pnl / 30.0) * 100
                        print(f"[{timestamp}] Balance: ${new_balance:.2f} | P&L: ${pnl:+.2f} ({pnl_percent:+.2f}%)")

                time.sleep(1)  # Update every second

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error in real-time updates: {e}")
                time.sleep(1)

        print(f"\n✅ Completed {update_count} balance updates over {duration_seconds} seconds")

    def demonstrate_risk_management(self):
        """Demonstrate risk management features"""
        print("\n🛡️ RISK MANAGEMENT DEMONSTRATION")
        print("=" * 40)

        print("📋 Risk Limits for $30 Portfolio:")
        print(f"  • Maximum Daily Loss: $1.50 (5.0% of portfolio)")
        print(f"  • Maximum Position Loss: $0.50 (1.67% of portfolio)")
        print(f"  • Circuit Breaker Threshold: -$3.00 (10.0% loss)")
        print(f"  • Emergency Stop Threshold: -$4.50 (15.0% loss)")
        print(f"  • Maximum Active Positions: 5")
        print(f"  • Risk per Trade: 1.5%")

        print("\n🔍 Risk Monitoring Features:")
        print("  ✅ Real-time P&L tracking")
        print("  ✅ Position size validation")
        print("  ✅ Daily loss limits")
        print("  ✅ Automatic emergency stops")
        print("  ✅ Strategy correlation monitoring")
        print("  ✅ Volatility-based position sizing")

    def run_interactive_demo(self):
        """Run interactive demo session"""
        self.is_running = True

        print("\n🎮 INTERACTIVE $30 PORTFOLIO DEMO")
        print("=" * 50)
        print("Commands: status, strategies, balance, risk, simulate, stop")
        print("Type 'stop' to exit")

        # Start background balance updates
        update_thread = threading.Thread(
            target=self.simulate_real_time_updates,
            args=(120,),  # 2 minutes of simulation
            daemon=True
        )
        update_thread.start()

        try:
            while self.is_running:
                command = input("\nVIPER-DEMO> ").strip().lower()

                if command == 'stop':
                    break
                elif command == 'status':
                    self.display_portfolio_overview()
                elif command == 'strategies':
                    self.display_strategy_performance()
                elif command == 'balance':
                    if self.trading_manager:
                        balance = self.trading_manager.get_real_time_balance()
                        print(f"💰 Current Balance: ${balance:.2f}")
                        print(f"📊 Last Update: {self.trading_manager.last_balance_update.strftime('%H:%M:%S')}")
                elif command == 'risk':
                    self.demonstrate_risk_management()
                elif command == 'simulate':
                    print("🔄 Starting 30-second balance simulation...")
                    self.simulate_real_time_updates(30)
                elif command in ['help', 'h', '?']:
                    print("Available commands:")
                    print("  status     - Show portfolio overview")
                    print("  strategies - Show strategy performance")
                    print("  balance    - Show current balance")
                    print("  risk       - Show risk management")
                    print("  simulate   - Run balance simulation")
                    print("  stop       - Exit demo")
                else:
                    print(f"Unknown command: {command} (type 'help' for commands)")

        except KeyboardInterrupt:
            print("\n⚠️  Demo interrupted by user")
        finally:
            self.is_running = False
            print("✅ Demo completed")

def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='$30 Portfolio Live Trading Demo')
    parser.add_argument('--overview', action='store_true', help='Show portfolio overview')
    parser.add_argument('--strategies', action='store_true', help='Show strategy performance')
    parser.add_argument('--risk', action='store_true', help='Show risk management')
    parser.add_argument('--simulate', type=int, help='Simulate real-time updates for N seconds')
    parser.add_argument('--interactive', '-i', action='store_true', help='Run interactive demo')

    args = parser.parse_args()

    # Create demo
    demo = ThirtyDollarPortfolioDemo()

    # Initialize
    if not demo.initialize_demo():
        sys.exit(1)

    # Handle commands
    if args.overview:
        demo.display_portfolio_overview()
    elif args.strategies:
        demo.display_strategy_performance()
    elif args.risk:
        demo.demonstrate_risk_management()
    elif args.simulate:
        demo.simulate_real_time_updates(args.simulate)
    elif args.interactive:
        demo.run_interactive_demo()
    else:
        # Default: show overview and strategies
        demo.display_portfolio_overview()
        demo.display_strategy_performance()
        demo.demonstrate_risk_management()

        print("\n💡 TIP: Run with --interactive for live demo experience")
        print("💡 TIP: Use --simulate 60 to see real-time balance updates")

if __name__ == '__main__':
    main()
