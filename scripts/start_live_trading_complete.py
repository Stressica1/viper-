#!/usr/bin/env python3
"""
🚀 COMPLETE LIVE TRADING SYSTEM - VIPER Full Automation
======================================================

Complete live trading system with strategy monitoring and GitHub MCP integration.

Features:
- Start live trading with 34x leverage requirement
- Real-time strategy performance monitoring
- GitHub MCP task automation
- Risk management and position control
- Performance analytics and reporting
- Emergency stop and circuit breaker systems

Author: VIPER Development Team
Version: 1.0.0
Date: 2025-01-29
"""

import os
import sys
import json
import time
import signal
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Import our components
from .live_trading_manager import LiveTradingManager
from .strategy_metrics_dashboard import StrategyMetricsDashboard
from .github_mcp_trading_tasks import GitHubMCPTradingTasks

@dataclass
class SystemStatus:
    """Overall system status"""
    live_trading_active: bool = False
    monitoring_active: bool = False
    github_mcp_connected: bool = False
    strategies_loaded: int = 0
    last_update: str = ""

class CompleteLiveTradingSystem:
    """Complete live trading system with all components integrated"""

    def __init__(self):
        self.status = SystemStatus()
        self.trading_manager: Optional[LiveTradingManager] = None
        self.strategy_dashboard: Optional[StrategyMetricsDashboard] = None
        self.github_mcp: Optional[GitHubMCPTradingTasks] = None

        # System configuration
        self.config = self._load_system_config()

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)


    def _load_system_config(self) -> Dict[str, Any]:
        """Load system configuration"""
        return {
            'min_leverage': 34.0,
            'risk_per_trade': 0.02,
            'max_daily_loss': 100.0,
            'max_positions': 15,
            'monitoring_interval': 30,
            'github_enabled': True,
            'auto_restart': True,
            'emergency_stop_enabled': True
        }

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print(f"\n⚠️  Received signal {signum}, initiating graceful shutdown...")
        self.stop_system()

    def initialize_system(self) -> bool:
        """Initialize all system components"""
        try:

            # Initialize strategy dashboard
            print("  📊 Initializing strategy metrics dashboard...")
            self.strategy_dashboard = StrategyMetricsDashboard()
            self.status.strategies_loaded = len(self.strategy_dashboard.strategies)

            # Initialize GitHub MCP
            if self.config['github_enabled']:
                print("  🐙 Initializing GitHub MCP integration...")
                self.github_mcp = GitHubMCPTradingTasks()
                self.status.github_mcp_connected = bool(self.github_mcp.github_token)

            # Initialize live trading manager
            self.trading_manager = LiveTradingManager(self.config)

            return True

        except Exception as e:
            return False

    def start_live_trading(self) -> bool:
        """Start the complete live trading system"""
        if not self.initialize_system():
            return False

        try:

            # Start live trading
            if self.trading_manager and self.trading_manager.start_live_trading():
                self.status.live_trading_active = True
            else:
                return False

            # Start strategy monitoring
            if self.strategy_dashboard:
                self.strategy_dashboard.start_monitoring()
                self.status.monitoring_active = True

            # Create initial GitHub tasks
            if self.github_mcp and self.status.github_mcp_connected:
                self._create_startup_tasks()

            self.status.last_update = datetime.now().isoformat()

            print("\n🎉 VIPER Live Trading System Started Successfully!")

            # Display initial status
            self.display_system_status()

            return True

        except Exception as e:
            print(f"❌ Failed to start live trading system: {e}")
            return False

    def stop_system(self) -> bool:
        """Stop the complete trading system"""
        try:
            print("\n⏹️  Stopping VIPER Live Trading System...")

            # Stop live trading
            if self.trading_manager and self.status.live_trading_active:
                self.trading_manager.stop_live_trading("System shutdown")
                self.status.live_trading_active = False

            # Stop monitoring
            if self.strategy_dashboard and self.status.monitoring_active:
                self.strategy_dashboard.stop_monitoring()
                self.status.monitoring_active = False

            # Create shutdown tasks
            if self.github_mcp and self.status.github_mcp_connected:
                self._create_shutdown_tasks()

            return True

        except Exception as e:
            return False

    def _create_startup_tasks(self):
        """Create initial GitHub tasks for system startup"""
        try:
            # Daily performance report
            dashboard_data = {
                'portfolio_summary': {
                    'total_strategies': len(self.strategy_dashboard.strategies),
                    'active_strategies': len([s for s in self.strategy_dashboard.strategies.values() if s.status == 'active']),
                    'total_portfolio_value': 100000.0,
                    'daily_pnl': 0.0,
                    'total_pnl': 0.0,
                    'portfolio_return': 0.0
                },
                'strategy_table': "System startup - monitoring active"
            }

            self.github_mcp.create_daily_performance_report(dashboard_data)

            # System startup task
            self.github_mcp.create_live_trading_task(
                "System Startup",
                {
                    'timestamp': datetime.now().isoformat(),
                    'strategies_loaded': self.status.strategies_loaded,
                    'min_leverage': self.config['min_leverage'],
                    'risk_per_trade': self.config['risk_per_trade'],
                    'max_positions': self.config['max_positions']
                }
            )


        except Exception as e:

    def _create_shutdown_tasks(self):
        """Create GitHub tasks for system shutdown"""
        try:
            # Get final trading status
            trading_status = self.trading_manager.get_trading_status() if self.trading_manager else {}

            self.github_mcp.create_live_trading_task(
                "System Shutdown",
                {
                    'timestamp': datetime.now().isoformat(),
                    'shutdown_reason': 'Graceful shutdown',
                    'final_balance': trading_status.get('current_balance', 0),
                    'total_pnl': trading_status.get('total_pnl', 0),
                    'session_duration': 'System operational'
                }
            )


        except Exception as e:

    def display_system_status(self):
        """Display comprehensive system status"""

        # Trading status
        trading_status = self.trading_manager.get_trading_status() if self.trading_manager else {}
        print(f"  Status: {'✅ Active' if self.status.live_trading_active else '❌ Inactive'}")
        print(f"  Positions: {trading_status.get('active_positions', 0)}")
        print(f"  Total P&L: ${trading_status.get('total_pnl', 0):,.2f}")

        # Strategy monitoring
        print(f"  Status: {'✅ Active' if self.status.monitoring_active else '❌ Inactive'}")
        print(f"  Strategies Loaded: {self.status.strategies_loaded}")

        # GitHub MCP
        print(f"  Status: {'✅ Connected' if self.status.github_mcp_connected else '❌ Disconnected'}")
        if self.github_mcp:
            open_tasks = len(self.github_mcp.get_open_tasks())

        # Portfolio summary
        if self.strategy_dashboard:
            portfolio = self.strategy_dashboard.portfolio_metrics
            if portfolio:
                print(f"  Total Value: ${portfolio.total_portfolio_value:,.2f}")
                print(f"  Daily P&L: ${portfolio.daily_pnl:,.2f}")
                print(f"  Portfolio Return: {portfolio.portfolio_return:.1f}%")
                print(f"  Sharpe Ratio: {portfolio.portfolio_sharpe:.2f}")

        print(f"\n🕒 Last Update: {self.status.last_update}")

    def run_interactive_mode(self):
        """Run in interactive mode with command interface"""
        print("Commands: status, strategies, trading, github, stop, help")

        while True:
            try:
                command = input("\nVIPER> ").strip().lower()

                if command == 'stop':
                    break
                elif command == 'status':
                    self.display_system_status()
                elif command == 'strategies':
                    if self.strategy_dashboard:
                        print(self.strategy_dashboard.display_strategy_table())
                elif command == 'trading':
                    if self.trading_manager:
                        print(self.trading_manager.generate_trading_report())
                elif command == 'github':
                    if self.github_mcp:
                        tasks = self.github_mcp.get_open_tasks()
                        if tasks:
                            for task in tasks[:10]:  # Show first 10
                        else:
                elif command == 'help':
                else:

            except KeyboardInterrupt:
                break
            except Exception as e:

        self.stop_system()

    def run_automated_mode(self, duration_hours: int = None):
        """Run in automated mode for specified duration"""
        if duration_hours:
            end_time = datetime.now() + timedelta(hours=duration_hours)
            print(f"Will stop at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

        try:
            while True:
                # Periodic status updates
                time.sleep(300)  # 5 minutes
                self.display_system_status()

                # Check if duration exceeded
                if duration_hours and datetime.now() >= end_time:
                    print(f"\n⏰ Duration of {duration_hours} hours completed")
                    break

        except KeyboardInterrupt:
        finally:
            self.stop_system()

def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='VIPER Complete Live Trading System')
    parser.add_argument('--start', action='store_true', help='Start the complete system')
    parser.add_argument('--stop', action='store_true', help='Stop the system')
    parser.add_argument('--interactive', '-i', action='store_true', help='Run in interactive mode')
    parser.add_argument('--automated', '-a', type=int, help='Run in automated mode for N hours')
    parser.add_argument('--status', action='store_true', help='Show system status')
    parser.add_argument('--no-github', action='store_true', help='Disable GitHub MCP integration')

    args = parser.parse_args()

    # Create system
    system = CompleteLiveTradingSystem()

    # Configure GitHub
    if args.no_github:
        system.config['github_enabled'] = False

    if args.start:
        if system.start_live_trading():
            if args.interactive:
                system.run_interactive_mode()
            elif args.automated:
                system.run_automated_mode(args.automated)
            else:
                print("💡 System started. Use --interactive for command interface")
                print("   or --automated N for automated operation")
                try:
                    while True:
                        time.sleep(60)
                        # Could add periodic tasks here
                except KeyboardInterrupt:
                    system.stop_system()
        else:
            sys.exit(1)

    elif args.stop:
        if system.stop_system():
        else:

    elif args.status:
        if system.initialize_system():
            system.display_system_status()
        else:
            print("❌ Failed to initialize system for status check")

    else:
        print("  Start interactive: python scripts/start_live_trading_complete.py --start --interactive")
        print("  Start automated:   python scripts/start_live_trading_complete.py --start --automated 24")
        print("  Show status:       python scripts/start_live_trading_complete.py --status")
        print("  Stop system:       python scripts/start_live_trading_complete.py --stop")

if __name__ == '__main__':
    main()
