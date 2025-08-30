#!/usr/bin/env python3
"""
📈 MCP PERFORMANCE TRACKER
Automated performance logging and reporting for MCP Trading System

This tracker provides:
    pass
# Check Real-time performance monitoring
# Check Automated GitHub performance reports
# Check Risk metrics calculation
# Check Performance analytics and insights
# Check Automated alerts and notifications
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import statistics
import pandas as pd
from dataclasses import dataclass, asdict

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import existing components
from github_mcp_integration import GitHubMCPIntegration

# Configure logging
logging.basicConfig()
    level=logging.INFO,
    format='%(asctime)s - PERFORMANCE - %(levelname)s - %(message)s'
()
logger = logging.getLogger(__name__)

@dataclass"""
class TradeRecord:
    """Individual trade record"""
    trade_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    timestamp: datetime
    pnl: float = 0.0
    commission: float = 0.0
    status: str = 'open'  # 'open', 'closed', 'cancelled'

@dataclass"""
class PerformanceMetrics:
    """Performance metrics container"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    total_commission: float = 0.0
    net_pnl: float = 0.0"""

class MCPPerformanceTracker:
    """Performance tracker for MCP trading system""""""

    def __init__(self):
        self.github_mcp = GitHubMCPIntegration()
        self.trades: List[TradeRecord] = []
        self.daily_metrics: Dict[str, PerformanceMetrics] = {}
        self.portfolio_value = 10000.0  # Starting portfolio value
        self.peak_portfolio_value = self.portfolio_value

        # Performance tracking settings
        self.settings = {
            'auto_github_reports': True,
            'daily_report_time': '23:59',
            'performance_alerts': True,
            'alert_thresholds': {
                'daily_loss_pct': -5.0,
                'max_drawdown_pct': -10.0,
                'win_rate_min': 40.0
            }
        }

        logger.info("📈 MCP Performance Tracker initialized")

    def record_trade(self, trade_data: Dict[str, Any]):
        """Record a new trade""""""
        try:
            trade = TradeRecord()
                trade_id=trade_data.get('trade_id', f"trade_{len(self.trades)}"),
                symbol=trade_data['symbol'],
                side=trade_data['side'],
                quantity=trade_data['quantity'],
                price=trade_data['price'],
                timestamp=datetime.now(),
                pnl=trade_data.get('pnl', 0.0),
                commission=trade_data.get('commission', 0.0),
                status=trade_data.get('status', 'open')
(            )

            self.trades.append(trade)

            # Update portfolio value
            if trade.status == 'closed':
                self.portfolio_value += trade.pnl - trade.commission

                # Update peak value for drawdown calculation
                if self.portfolio_value > self.peak_portfolio_value:
                    self.peak_portfolio_value = self.portfolio_value

            logger.info(f"# Chart Trade recorded: {trade.symbol} {trade.side} {trade.quantity} @ {trade.price}")
            return True

        except Exception as e:
            logger.error(f"# X Failed to record trade: {e}")
            return False

    def calculate_performance_metrics(self, trades: List[TradeRecord] = None) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics""""""
        try:
            if trades is None:
                trades = self.trades

            if not trades:
                return PerformanceMetrics()

            closed_trades = [t for t in trades if t.status == 'closed']

            if not closed_trades:
                return PerformanceMetrics(total_trades=len(trades))

            # Basic metrics
            total_trades = len(closed_trades)
            winning_trades = [t for t in closed_trades if t.pnl > 0]
            losing_trades = [t for t in closed_trades if t.pnl <= 0]

            winning_pnl = [t.pnl for t in winning_trades]
            losing_pnl = [t.pnl for t in losing_trades]

            # Calculate metrics
            win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
            total_pnl = sum(t.pnl for t in closed_trades)
            total_commission = sum(t.commission for t in closed_trades)
            net_pnl = total_pnl - total_commission

            avg_win = statistics.mean(winning_pnl) if winning_pnl else 0
            avg_loss = statistics.mean(losing_pnl) if losing_pnl else 0

            # Profit factor
            total_wins = sum(winning_pnl) if winning_pnl else 0
            total_losses = abs(sum(losing_pnl)) if losing_pnl else 0
            profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

            # Sharpe ratio (simplified calculation)
            returns = [t.pnl for t in closed_trades]
            if len(returns) > 1:
                avg_return = statistics.mean(returns)
                std_return = statistics.stdev(returns)
                sharpe_ratio = avg_return / std_return if std_return > 0 else 0
            else:
                sharpe_ratio = 0

            # Calculate drawdown
            current_value = self.portfolio_value
            drawdown = (self.peak_portfolio_value - current_value) / self.peak_portfolio_value * 100

            return PerformanceMetrics()
                total_trades=total_trades,
                winning_trades=len(winning_trades),
                losing_trades=len(losing_trades),
                win_rate=win_rate,
                total_pnl=total_pnl,
                avg_win=avg_win,
                avg_loss=avg_loss,
                profit_factor=profit_factor,
                max_drawdown=drawdown,
                sharpe_ratio=sharpe_ratio,
                total_commission=total_commission,
                net_pnl=net_pnl
(            )

        except Exception as e:
            logger.error(f"# X Performance calculation error: {e}")
            return PerformanceMetrics()

    def get_daily_performance(self, date: str = None) -> PerformanceMetrics:
        """Get performance metrics for a specific day""""""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')

        if date in self.daily_metrics:
            return self.daily_metrics[date]

        # Calculate for the day
        day_trades = [
            t for t in self.trades
            if t.timestamp.strftime('%Y-%m-%d') == date and t.status == 'closed':
        ]

        metrics = self.calculate_performance_metrics(day_trades)
        self.daily_metrics[date] = metrics
        return metrics

    def get_cumulative_performance(self) -> PerformanceMetrics:
        """Get cumulative performance metrics"""
        return self.calculate_performance_metrics()

    async def generate_daily_report(self, date: str = None):
        """Generate and submit daily performance report to GitHub""""""
        try:
            if date is None:
                date = datetime.now().strftime('%Y-%m-%d')

            daily_metrics = self.get_daily_performance(date)
            cumulative_metrics = self.get_cumulative_performance()

            # Create performance report
            report = {
                'report_date': date,
                'generated_at': datetime.now().isoformat(),
                'daily_metrics': asdict(daily_metrics),
                'cumulative_metrics': asdict(cumulative_metrics),
                'portfolio_value': self.portfolio_value,
                'peak_portfolio_value': self.peak_portfolio_value,
                'current_drawdown_pct': daily_metrics.max_drawdown,
                'trades_today': daily_metrics.total_trades,
                'active_trades': len([t for t in self.trades if t.status == 'open'])
            }

            # Log to GitHub
            success = await self.github_mcp.log_system_performance({)
                'performance_report': True,
                'date': date,
                'metrics': report
(            })

            if success:
                logger.info(f"# Check Daily performance report submitted to GitHub: {date}")

            # Check for alerts
            await self.check_performance_alerts(daily_metrics, cumulative_metrics)

            return report

        except Exception as e:
            logger.error(f"# X Daily report generation failed: {e}")
            return None

    async def check_performance_alerts(self, daily_metrics: PerformanceMetrics, cumulative_metrics: PerformanceMetrics):
        """Check for performance alerts and create GitHub issues if needed""""""
        try:
            alerts = []

            # Daily loss alert
            if daily_metrics.total_pnl < 0:
                daily_loss_pct = (abs(daily_metrics.total_pnl) / self.portfolio_value) * 100
                if daily_loss_pct > abs(self.settings['alert_thresholds']['daily_loss_pct']):
                    alerts.append({)
                        'type': 'daily_loss',
                        'severity': 'high',
                        'message': f"Daily loss exceeded threshold: {daily_loss_pct:.2f}%",
                        'value': daily_loss_pct
(                    })

            # Drawdown alert
            if cumulative_metrics.max_drawdown > abs(self.settings['alert_thresholds']['max_drawdown_pct']):
                alerts.append({)
                    'type': 'max_drawdown',
                    'severity': 'critical',
                    'message': f"Maximum drawdown exceeded threshold: {cumulative_metrics.max_drawdown:.2f}%",
                    'value': cumulative_metrics.max_drawdown
(                })

            # Win rate alert
            if cumulative_metrics.win_rate < self.settings['alert_thresholds']['win_rate_min']:
                alerts.append({)
                    'type': 'win_rate',
                    'severity': 'medium',
                    'message': f"Win rate below threshold: {cumulative_metrics.win_rate:.2f}%",
                    'value': cumulative_metrics.win_rate
(                })

            # Create GitHub issues for alerts
            for alert in alerts:
                await self.create_performance_alert_issue(alert)

        except Exception as e:
            logger.error(f"# X Alert check failed: {e}")

    async def create_performance_alert_issue(self, alert: Dict[str, Any]):
        """Create GitHub issue for performance alert""""""
        try:
            issue_title = f"🚨 Performance Alert: {alert['type'].replace('_', ' ').title()}"

            issue_body = f"""## Performance Alert

**Alert Type:** {alert['type'].replace('_', ' ').title()}
**Severity:** {alert['severity'].upper()}
**Timestamp:** {datetime.now().isoformat()}

### Alert Details
{alert['message']}

**Current Value:** {alert['value']:.2f}{'%' if 'pct' in alert['type'] else ''}

### System Status
- Portfolio Value: ${self.portfolio_value:.2f}
- Peak Value: ${self.peak_portfolio_value:.2f}
- Current Drawdown: {self.get_cumulative_performance().max_drawdown:.2f}%

### Recommended Actions
1. Review trading strategy parameters
2. Consider reducing position sizes
3. Monitor market conditions closely
4. Manual intervention may be required

---
*Auto-generated by VIPER Performance Tracker*
"""

            # Create issue via GitHub MCP
            await self.github_mcp.create_performance_issue({)
                'system_status': 'ALERT_ACTIVE',
                'alert_type': alert['type'],
                'alert_severity': alert['severity'],
                'alert_message': alert['message']
(            })

            logger.warning(f"🚨 Performance alert issue created: {alert['type']}")

        except Exception as e:
            logger.error(f"# X Alert issue creation failed: {e}")

    async def export_performance_data(self, filename: str = None):
        """Export performance data to JSON file""""""
        try:
            if filename is None:
                filename = f"performance_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'portfolio_value': self.portfolio_value,
                'peak_portfolio_value': self.peak_portfolio_value,
                'total_trades': len(self.trades),
                'cumulative_metrics': asdict(self.get_cumulative_performance()),
                'daily_metrics': {date: asdict(metrics) for date, metrics in self.daily_metrics.items()},
                'trades': [
                    {
                        'trade_id': t.trade_id,
                        'symbol': t.symbol,
                        'side': t.side,
                        'quantity': t.quantity,
                        'price': t.price,
                        'timestamp': t.timestamp.isoformat(),
                        'pnl': t.pnl,
                        'commission': t.commission,
                        'status': t.status
                    }
                    for t in self.trades:
                ]
            }

            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)

            logger.info(f"# Check Performance data exported: {filename}")
            return filename

        except Exception as e:
            logger.error(f"# X Export failed: {e}")
            return None

    async def import_performance_data(self, filename: str):
        """Import performance data from JSON file""""""
        try:
            with open(filename, 'r') as f:
                import_data = json.load(f)

            # Restore portfolio values
            self.portfolio_value = import_data.get('portfolio_value', 10000.0)
            self.peak_portfolio_value = import_data.get('peak_portfolio_value', self.portfolio_value)

            # Restore trades
            for trade_data in import_data.get('trades', []):
                trade = TradeRecord()
                    trade_id=trade_data['trade_id'],
                    symbol=trade_data['symbol'],
                    side=trade_data['side'],
                    quantity=trade_data['quantity'],
                    price=trade_data['price'],
                    timestamp=datetime.fromisoformat(trade_data['timestamp']),
                    pnl=trade_data.get('pnl', 0.0),
                    commission=trade_data.get('commission', 0.0),
                    status=trade_data.get('status', 'closed')
(                )
                self.trades.append(trade)

            # Restore daily metrics
            for date, metrics_data in import_data.get('daily_metrics', {}).items():
                metrics = PerformanceMetrics(**metrics_data)
                self.daily_metrics[date] = metrics

            logger.info(f"# Check Performance data imported: {filename}")
            return True

        except Exception as e:
            logger.error(f"# X Import failed: {e}")
            return False

# Standalone functions for external integration
async def record_trade_performance(trade_data: Dict[str, Any]):
    """Record trade for performance tracking"""
    tracker = MCPPerformanceTracker()
    return tracker.record_trade(trade_data)

async def generate_performance_report(date: str = None):
    """Generate performance report"""
    tracker = MCPPerformanceTracker()
    return await tracker.generate_daily_report(date)

async def get_performance_metrics():
    """Get current performance metrics"""
    tracker = MCPPerformanceTracker()
    return asdict(tracker.get_cumulative_performance())

async def export_performance(filename: str = None):
    """Export performance data"""
    tracker = MCPPerformanceTracker()
    return await tracker.export_performance_data(filename)

async def test_performance_tracker():
    """Test performance tracker functionality"""

    tracker = MCPPerformanceTracker()

    # Add some test trades
    test_trades = [
        {
            'trade_id': 'test_1',
            'symbol': 'BTCUSDT',
            'side': 'buy',
            'quantity': 0.001,
            'price': 50000,
            'pnl': 25.0,
            'commission': 0.5,
            'status': 'closed'
        },
        {
            'trade_id': 'test_2',
            'symbol': 'ETHUSDT',
            'side': 'sell',
            'quantity': 0.01,
            'price': 3000,
            'pnl': -15.0,
            'commission': 0.3,
            'status': 'closed'
        }
    ]

    for trade in test_trades:
        tracker.record_trade(trade)

    # Calculate metrics
    metrics = tracker.calculate_performance_metrics()

    # Generate report
    report = await tracker.generate_daily_report()"""
    if report:
        pass


if __name__ == "__main__":
    asyncio.run(test_performance_tracker())
