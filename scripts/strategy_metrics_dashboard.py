#!/usr/bin/env python3
"""
# Chart STRATEGY METRICS DASHBOARD - VIPER Trading Performance Analytics
================================================================

Comprehensive trading strategy analytics and performance dashboard.

Features:
    pass
- Real-time strategy performance monitoring
- Win rates, Sharpe ratios, drawdowns, and returns
- Strategy weight allocation and optimization
- Risk-adjusted performance metrics
- Interactive dashboard with filtering and sorting
- Export capabilities for reporting

Author: VIPER Development Team
Version: 1.0.0
Date: 2025-01-29
"""

import os
import sys
import json
import time
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np
import pandas as pd
# from tabulate import tabulate  # Optional dependency

# Try to import visualization libraries"""
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

@dataclass
class StrategyMetrics:
    """Comprehensive strategy performance metrics"""
    strategy_name: str
    strategy_type: str  # 'momentum', 'mean_reversion', 'breakout', 'scalping', etc.
    status: str  # 'active', 'inactive', 'backtesting'
    weight: float  # Portfolio weight (0-100%)
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    volatility: float
    alpha: float
    beta: float
    information_ratio: float
    last_updated: str
    pairs_traded: List[str]
    timeframes: List[str]

@dataclass"""
class PortfolioMetrics:
    """Overall portfolio performance metrics"""
    total_strategies: int
    active_strategies: int
    total_portfolio_value: float
    daily_pnl: float
    total_pnl: float
    portfolio_return: float
    portfolio_sharpe: float
    portfolio_drawdown: float
    risk_adjusted_return: float
    diversification_ratio: float
    last_updated: str"""

class StrategyMetricsDashboard:
    """Comprehensive strategy metrics dashboard""""""

    def __init__(self):
        self.strategies: Dict[str, StrategyMetrics] = {}
        self.portfolio_metrics: Optional[PortfolioMetrics] = None
        self.historical_data: Dict[str, List[Dict]] = defaultdict(list)
        self.update_interval = 300  # 5 minutes
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None

        # Create reports directory
        self.reports_dir = Path("reports")
        self.reports_dir.mkdir(exist_ok=True)

        # Initialize sample strategies (in production, this would come from database/config)
        self._initialize_sample_strategies()

    def _initialize_sample_strategies(self):
        """Initialize sample trading strategies with realistic metrics"""
        strategies_data = [
            {
                'strategy_name': 'VIPER Momentum Scalper',
                'strategy_type': 'momentum',
                'status': 'active',
                'weight': 25.0,
                'total_trades': 1247,
                'winning_trades': 823,
                'losing_trades': 424,
                'win_rate': 66.0,
                'avg_win': 1.8,
                'avg_loss': -1.2,
                'profit_factor': 2.1,
                'total_return': 15.7,
                'annualized_return': 89.2,
                'sharpe_ratio': 2.3,
                'sortino_ratio': 3.1,
                'max_drawdown': -8.5,
                'calmar_ratio': 10.5,
                'volatility': 12.3,
                'alpha': 0.15,
                'beta': 0.85,
                'information_ratio': 1.8,
                'pairs_traded': ['BTC/USDT', 'ETH/USDT', 'SOL/USDT'],
                'timeframes': ['15m', '1h']
            },
            {
                'strategy_name': 'VIPER Mean Reversion',
                'strategy_type': 'mean_reversion',
                'status': 'active',
                'weight': 20.0,
                'total_trades': 892,
                'winning_trades': 534,
                'losing_trades': 358,
                'win_rate': 59.9,
                'avg_win': 2.1,
                'avg_loss': -1.5,
                'profit_factor': 1.8,
                'total_return': 12.3,
                'annualized_return': 74.1,
                'sharpe_ratio': 1.9,
                'sortino_ratio': 2.4,
                'max_drawdown': -6.2,
                'calmar_ratio': 12.0,
                'volatility': 10.8,
                'alpha': 0.12,
                'beta': 0.72,
                'information_ratio': 1.5,
                'pairs_traded': ['ADA/USDT', 'DOT/USDT', 'LINK/USDT'],
                'timeframes': ['30m', '2h']
            },
            {
                'strategy_name': 'VIPER Breakout Hunter',
                'strategy_type': 'breakout',
                'status': 'active',
                'weight': 30.0,
                'total_trades': 1563,
                'winning_trades': 987,
                'losing_trades': 576,
                'win_rate': 63.1,
                'avg_win': 2.3,
                'avg_loss': -1.8,
                'profit_factor': 1.9,
                'total_return': 18.9,
                'annualized_return': 95.6,
                'sharpe_ratio': 2.1,
                'sortino_ratio': 2.8,
                'max_drawdown': -9.8,
                'calmar_ratio': 9.8,
                'volatility': 14.2,
                'alpha': 0.18,
                'beta': 0.92,
                'information_ratio': 1.9,
                'pairs_traded': ['AVAX/USDT', 'MATIC/USDT', 'ATOM/USDT'],
                'timeframes': ['1h', '4h']
            },
            {
                'strategy_name': 'VIPER Grid Scalper',
                'strategy_type': 'scalping',
                'status': 'active',
                'weight': 15.0,
                'total_trades': 3241,
                'winning_trades': 2109,
                'losing_trades': 1132,
                'win_rate': 65.1,
                'avg_win': 0.8,
                'avg_loss': -0.6,
                'profit_factor': 2.2,
                'total_return': 8.7,
                'annualized_return': 156.3,
                'sharpe_ratio': 2.8,
                'sortino_ratio': 3.5,
                'max_drawdown': -4.2,
                'calmar_ratio': 37.2,
                'volatility': 8.9,
                'alpha': 0.22,
                'beta': 0.45,
                'information_ratio': 2.2,
                'pairs_traded': ['DOGE/USDT', 'SHIB/USDT', 'PEPE/USDT'],
                'timeframes': ['5m', '15m']
            },
            {
                'strategy_name': 'VIPER Trend Follower',
                'strategy_type': 'trend_following',
                'status': 'active',
                'weight': 10.0,
                'total_trades': 678,
                'winning_trades': 412,
                'losing_trades': 266,
                'win_rate': 60.8,
                'avg_win': 3.1,
                'avg_loss': -2.4,
                'profit_factor': 1.7,
                'total_return': 22.4,
                'annualized_return': 67.8,
                'sharpe_ratio': 1.7,
                'sortino_ratio': 2.1,
                'max_drawdown': -12.1,
                'calmar_ratio': 5.6,
                'volatility': 16.7,
                'alpha': 0.14,
                'beta': 1.05,
                'information_ratio': 1.3,
                'pairs_traded': ['LTC/USDT', 'XRP/USDT', 'TRX/USDT'],
                'timeframes': ['4h', '1d']
            }
        ]

        for strategy_data in strategies_data:
            strategy_data['last_updated'] = datetime.now().isoformat()
            strategy = StrategyMetrics(**strategy_data)
            self.strategies[strategy.strategy_name] = strategy

        # Calculate portfolio metrics
        self._calculate_portfolio_metrics()"""

    def _calculate_portfolio_metrics(self):
        """Calculate overall portfolio performance metrics""""""
        if not self.strategies:
            return

        active_strategies = [s for s in self.strategies.values() if s.status == 'active']

        # Weighted average calculations
        total_weight = sum(s.weight for s in active_strategies)
        weighted_return = sum(s.total_return * (s.weight / total_weight) for s in active_strategies)
        weighted_sharpe = sum(s.sharpe_ratio * (s.weight / total_weight) for s in active_strategies)
        max_drawdown = max(s.max_drawdown for s in active_strategies)

        # Risk metrics
        portfolio_volatility = np.sqrt(sum((s.volatility * s.weight / total_weight) ** 2 for s in active_strategies))
        diversification_ratio = len(active_strategies) / sum(1/s.weight for s in active_strategies)

        # Get live balance from balance service
        try:
            from .live_balance_service import get_live_balance
            live_balance = get_live_balance()
            portfolio_value = live_balance.total_usd_balance
            if portfolio_value <= 0:  # Fallback if no live balance
                portfolio_value = 30.0
        except ImportError:
            portfolio_value = 30.0  # Fallback

        self.portfolio_metrics = PortfolioMetrics()
            total_strategies=len(self.strategies),
            active_strategies=len(active_strategies),
            total_portfolio_value=portfolio_value,  # Live portfolio value
            daily_pnl=0.0,  # Start with zero daily P&L
            total_pnl=0.0,  # Start with zero total P&L
            portfolio_return=weighted_return,
            portfolio_sharpe=weighted_sharpe,
            portfolio_drawdown=max_drawdown,
            risk_adjusted_return=weighted_return / abs(max_drawdown) if max_drawdown != 0 else 0,
            diversification_ratio=diversification_ratio,
            last_updated=datetime.now().isoformat()
(        )

    def display_strategy_table(self, sort_by: str = 'sharpe_ratio', ascending: bool = False,)
(                             filter_status: str = 'active') -> str:
                                 pass
        """Display comprehensive strategy performance table"""

        # Filter strategies
        filtered_strategies = [
            s for s in self.strategies.values()"""
            if filter_status == 'all' or s.status == filter_status:
        ]

        if not filtered_strategies:
            return "No strategies found matching criteria"

        # Sort strategies
        if sort_by == 'sharpe_ratio':
            filtered_strategies.sort(key=lambda x: x.sharpe_ratio, reverse=not ascending)
        elif sort_by == 'win_rate':
            filtered_strategies.sort(key=lambda x: x.win_rate, reverse=not ascending)
        elif sort_by == 'total_return':
            filtered_strategies.sort(key=lambda x: x.total_return, reverse=not ascending)
        elif sort_by == 'weight':
            filtered_strategies.sort(key=lambda x: x.weight, reverse=not ascending)

        # Prepare table data
        table_data = []
        for strategy in filtered_strategies:
            table_data.append([)
                strategy.strategy_name[:25],  # Truncate long names
                strategy.strategy_type,
                f"{strategy.weight:.1f}%",
                strategy.total_trades,
                f"{strategy.win_rate:.1f}%",
                f"{strategy.sharpe_ratio:.2f}",
                f"{strategy.total_return:.1f}%",
                f"{strategy.max_drawdown:.1f}%",
                f"{strategy.profit_factor:.2f}",
                f"{strategy.volatility:.1f}%",
                strategy.status
(            ])

        headers = [
            'Strategy Name', 'Type', 'Weight', 'Trades',
            'Win Rate', 'Sharpe', 'Return', 'Max DD', 'Profit Factor', 'Volatility', 'Status'
        ]

        table = self._format_table_simple(headers, table_data)

        return table

    def display_portfolio_summary(self) -> str:
        """Display portfolio-level summary""""""
        if not self.portfolio_metrics:
            return "Portfolio metrics not available"

        pm = self.portfolio_metrics

        # Get live balance status
        balance_status = "🔴 DISCONNECTED"
        exchange_name = "UNKNOWN"
        try:
            from .live_balance_service import get_live_balance
            live_balance = get_live_balance()
            if live_balance.status == 'connected':
                balance_status = f"🟢 LIVE ({live_balance.exchange.upper()})"
                exchange_name = live_balance.exchange.upper()
            elif live_balance.status == 'connecting':
                balance_status = "🟡 CONNECTING"
            else:
                balance_status = "🔴 DISCONNECTED"
        except ImportError:
            balance_status = "⚪ OFFLINE"

        # Calculate dynamic risk limits based on live balance
        portfolio_value = pm.total_portfolio_value
        max_position_loss = portfolio_value * 0.0167  # 1.67%
        max_daily_loss = portfolio_value * 0.05       # 5.0%
        circuit_breaker = -portfolio_value * 0.10      # 10.0%
        emergency_stop = -portfolio_value * 0.15       # 15.0%

        summary = f"""
# Target PORTFOLIO SUMMARY - LIVE BALANCE TRACKING
{'='*50}
# Chart Total Strategies: {pm.total_strategies}
# Check Active Strategies: {pm.active_strategies}
💰 Portfolio Value: ${pm.total_portfolio_value:,.2f}
📈 Daily P&L: ${pm.daily_pnl:,.2f}
💹 Total P&L: ${pm.total_pnl:,.2f} ({pm.total_pnl/pm.total_portfolio_value*100:.2f}%)
# Chart Portfolio Return: {pm.portfolio_return:.1f}%
⚡ Sharpe Ratio: {pm.portfolio_sharpe:.2f}
📉 Max Drawdown: {pm.portfolio_drawdown:.1f}%
🎚️ Risk-Adjusted Return: {pm.risk_adjusted_return:.2f}
🔄 Diversification Ratio: {pm.diversification_ratio:.2f}
🕒 Last Updated: {pm.last_updated}

🔗 BALANCE STATUS: {balance_status}

# Idea DYNAMIC RISK LIMITS (based on live ${portfolio_value:.0f} balance)
   • Max Single Position Loss: ${max_position_loss:.2f} ({max_position_loss/portfolio_value*100:.1f}%)
   • Max Daily Loss: ${max_daily_loss:.2f} ({max_daily_loss/portfolio_value*100:.1f}%)
   • Circuit Breaker: ${circuit_breaker:.2f} ({circuit_breaker/portfolio_value*100:.1f}%)
   • Emergency Stop: ${emergency_stop:.2f} ({emergency_stop/portfolio_value*100:.1f}%)
        """

        return summary.strip()"""

    def display_strategy_details(self, strategy_name: str) -> str:
        """Display detailed information for a specific strategy""""""
        if strategy_name not in self.strategies:
            return f"Strategy '{strategy_name}' not found"

        strategy = self.strategies[strategy_name]

        details = f"""
# Search STRATEGY DETAILS: {strategy.strategy_name}
{'='*60}

# Chart PERFORMANCE METRICS
  Win Rate: {strategy.win_rate:.1f}%
  Total Trades: {strategy.total_trades}
  Winning Trades: {strategy.winning_trades}
  Losing Trades: {strategy.losing_trades}
  Profit Factor: {strategy.profit_factor:.2f}
  Avg Win: ${strategy.avg_win:.2f}
  Avg Loss: ${strategy.avg_loss:.2f}

📈 RETURNS & RISK
  Total Return: {strategy.total_return:.1f}%
  Annualized Return: {strategy.annualized_return:.1f}%
  Sharpe Ratio: {strategy.sharpe_ratio:.2f}
  Sortino Ratio: {strategy.sortino_ratio:.2f}
  Max Drawdown: {strategy.max_drawdown:.1f}%
  Calmar Ratio: {strategy.calmar_ratio:.2f}
  Volatility: {strategy.volatility:.1f}%

🔗 MARKET DATA
  Pairs Traded: {', '.join(strategy.pairs_traded)}
  Timeframes: {', '.join(strategy.timeframes)}
  Strategy Type: {strategy.strategy_type}
  Portfolio Weight: {strategy.weight:.1f}%
  Status: {strategy.status}
  Last Updated: {strategy.last_updated}

# Chart RISK METRICS
  Alpha: {strategy.alpha:.3f}
  Beta: {strategy.beta:.3f}
  Information Ratio: {strategy.information_ratio:.2f}
        """

        return details.strip()"""

    def _format_table_simple(self, headers: List[str], data: List[List[Any]]) -> str:
        """Format table data as a simple text table""""""
        if not data:
            return "No data available"

        # Calculate column widths
        col_widths = []
        for i, header in enumerate(headers):
            max_width = len(header)
            for row in data:
                if i < len(row):
                    max_width = max(max_width, len(str(row[i])))
            col_widths.append(max_width)

        # Create separator line
        separator = "+" + "+".join("-" * (width + 2) for width in col_widths) + "+"

        # Format header
        header_line = "|" + "|".join(f" {headers[i]:<{col_widths[i]}} " for i in range(len(headers))) + "|"

        # Format data rows
        data_lines = []
        for row in data:
            data_line = "|" + "|".join(f" {str(row[i]) if i < len(row) else '':<{col_widths[i]}} " for i in range(len(headers))) + "|"
            data_lines.append(data_line)

        # Combine all lines
        table_lines = [separator, header_line, separator] + data_lines + [separator]

        return "\n".join(table_lines)

    def export_to_csv(self, filename: str = None) -> str:
        """Export strategy metrics to CSV""""""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"strategy_metrics_{timestamp}.csv"

        filepath = self.reports_dir / filename

        # Prepare data for CSV
        csv_data = []
        for strategy in self.strategies.values():
            row = {
                'strategy_name': strategy.strategy_name,
                'strategy_type': strategy.strategy_type,
                'status': strategy.status,
                'weight': strategy.weight,
                'total_trades': strategy.total_trades,
                'win_rate': strategy.win_rate,
                'sharpe_ratio': strategy.sharpe_ratio,
                'total_return': strategy.total_return,
                'max_drawdown': strategy.max_drawdown,
                'profit_factor': strategy.profit_factor,
                'volatility': strategy.volatility,
                'last_updated': strategy.last_updated
            }
            csv_data.append(row)

        # Convert to DataFrame and save
        df = pd.DataFrame(csv_data)
        df.to_csv(filepath, index=False)

        return str(filepath)

    def generate_performance_report(self) -> str:
        """Generate comprehensive performance report"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        report = f"""
# Chart VIPER TRADING STRATEGY PERFORMANCE REPORT
{'='*70}
Generated: {timestamp}

{self.display_portfolio_summary()}

{self.display_strategy_table()}

# Target RECOMMENDATIONS
{'='*30}
• Monitor strategies with Sharpe ratio < 1.5 for potential optimization
• Rebalance portfolio weights quarterly based on performance
• Consider increasing allocation to high-performing strategies
• Review and update risk management parameters regularly

📋 METHODOLOGY
{'='*20}
• Win Rate: Percentage of profitable trades
• Sharpe Ratio: Risk-adjusted return measure (target: >1.5)
• Profit Factor: Gross profit / Gross loss (target: >1.5)
• Max Drawdown: Largest peak-to-valley decline (target: <10%)
• Calmar Ratio: Annual return / Max drawdown (target: >5)

# Warning  DISCLAIMER
{'='*15}
This report contains simulated performance data for demonstration purposes.
Actual trading results may vary significantly due to market conditions,
execution quality, and other factors. Past performance does not guarantee
future results. Always perform your own due diligence and risk assessment.
        """

        # Save report
        report_path = self.reports_dir / f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        return report

    def start_monitoring(self):
        """Start real-time monitoring thread""""""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.is_monitoring = False"""
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:"""
            try:
                # Update strategy metrics
                self._update_live_metrics()

                # Check for alerts
                self._check_alerts()

                time.sleep(self.update_interval)

            except Exception as e:
                time.sleep(60)  # Wait before retrying

    def _update_live_metrics(self):
        """Update live strategy metrics (placeholder for real implementation)"""
        # In production, this would fetch real-time data from trading API
        for strategy in self.strategies.values()""":
            if strategy.status == 'active':
                # Simulate small metric updates
                strategy.last_updated = datetime.now().isoformat()

    def _check_alerts(self):
        """Check for performance alerts"""
        alerts = []

        for strategy in self.strategies.values()""":
            if strategy.status == 'active':
                # Check for concerning metrics
                if strategy.sharpe_ratio < 1.0:
                    alerts.append(f"# Warning  Low Sharpe ratio for {strategy.strategy_name}: {strategy.sharpe_ratio:.2f}")

                if strategy.max_drawdown < -15:  # More than 15% drawdown
                    alerts.append(f"🚨 High drawdown for {strategy.strategy_name}: {strategy.max_drawdown:.1f}%")

                if strategy.win_rate < 50:
                    alerts.append(f"# Warning  Low win rate for {strategy.strategy_name}: {strategy.win_rate:.1f}%")

        if alerts:
            for alert in alerts:
                pass

def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='VIPER Strategy Metrics Dashboard')
    parser.add_argument('--list', '-l', action='store_true', help='List all strategies')
    parser.add_argument('--details', '-d', help='Show details for specific strategy')
    parser.add_argument('--sort', '-s', choices=['sharpe', 'win_rate', 'return', 'weight'],""")
(                       default='sharpe', help='Sort strategies by metric')
    parser.add_argument('--filter', '-f', choices=['active', 'inactive', 'all'],)
(                       default='active', help='Filter strategies by status')
    parser.add_argument('--export', '-e', help='Export to CSV file')
    parser.add_argument('--report', '-r', action='store_true', help='Generate performance report')
    parser.add_argument('--monitor', '-m', action='store_true', help='Start real-time monitoring')
    parser.add_argument('--portfolio', '-p', action='store_true', help='Show portfolio summary')

    args = parser.parse_args()

    # Create dashboard
    dashboard = StrategyMetricsDashboard()

    # Handle commands
    if args.portfolio:
        pass

    if args.list:
        table = dashboard.display_strategy_table()
            sort_by=args.sort,
            filter_status=args.filter
(        )

    if args.details:
        details = dashboard.display_strategy_details(args.details)

    if args.export:
        csv_path = dashboard.export_to_csv(args.export)

    if args.report:
        report = dashboard.generate_performance_report()

    if args.monitor:
        dashboard.start_monitoring()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            dashboard.stop_monitoring()

    # Default action: show portfolio and strategy table
    if not any([args.list, args.details, args.export, args.report, args.monitor, args.portfolio]):
if __name__ == '__main__':
    main()
