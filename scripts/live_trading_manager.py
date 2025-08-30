#!/usr/bin/env python3
"""
💰 LIVE TRADING MANAGER - VIPER Live Trading Operations
=====================================================

Comprehensive live trading management system with GitHub MCP integration.

Features:
    pass
- Automated live trading with 34x leverage requirement
- Real-time strategy execution and monitoring
- Risk management and position control
- GitHub MCP task automation
- Performance tracking and reporting
- Emergency stop and circuit breaker systems

Author: VIPER Development Team
Version: 1.0.0
Date: 2025-01-29
"""

import os
import sys
import json
import time
import asyncio
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import logging

# Import our modules
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))"""

try:
    from .strategy_metrics_dashboard import StrategyMetricsDashboard
    from .github_mcp_trading_tasks import GitHubMCPTradingTasks, TradingAlert
except ImportError:
    # Fallback for direct execution
    from strategy_metrics_dashboard import StrategyMetricsDashboard
    from github_mcp_trading_tasks import GitHubMCPTradingTasks, TradingAlert

@dataclass
class TradingPosition:
    """Represents an active trading position"""
    position_id: str
    strategy_name: str
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    entry_price: float
    current_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop: Optional[float] = None
    pnl: float = 0.0
    pnl_percentage: float = 0.0
    timestamp: str = ""
    status: str = "active"  # active, closed, stopped

@dataclass
class TradingSession:
    """Represents a trading session"""
    session_id: str
    start_time: str
    end_time: Optional[str] = None
    initial_balance: float = 0.0
    current_balance: float = 0.0
    total_pnl: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    status: str = "active"  # active, paused, stopped

class LiveTradingManager:
    """Main live trading orchestration system""""""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._load_default_config()

        # Initialize components
        self.strategy_dashboard = StrategyMetricsDashboard()
        self.github_mcp = GitHubMCPTradingTasks()

        # Trading state
        self.is_trading = False
        self.trading_session: Optional[TradingSession] = None
        self.active_positions: Dict[str, TradingPosition] = {}
        self.daily_stats = defaultdict(float)
        self.risk_limits = self._load_risk_limits()

        # Monitoring
        self.monitoring_thread: Optional[threading.Thread] = None
        self.alerts_queue = asyncio.Queue()

        # Real-time balance tracking
        self.last_balance_update = datetime.now()
        self.balance_update_interval = 60  # Update balance every 60 seconds
        self.real_time_balance = 30.0  # Current real-time balance

        # Setup logging
        self._setup_logging()

        # Create reports directory
        self.reports_dir = Path("reports") / "live_trading"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def _load_default_config(self) -> Dict[str, Any]
        """Load default configuration"""
        return {:
            'max_positions': 15,
            'max_positions_per_strategy': 3,
            'risk_per_trade': 0.02,  # 2%
            'max_daily_loss': 100.0,  # $100
            'max_daily_trades': 50,
            'min_leverage': 34.0,
            'monitoring_interval': 30,  # seconds
            'alert_thresholds': {
                'high_drawdown': -5.0,  # 5%
                'low_win_rate': 50.0,   # 50%
                'high_volatility': 15.0  # 15%
            }
        }"""

    def _load_risk_limits(self) -> Dict[str, Any]
        """Load risk management limits for $30 portfolio"""
        return {:
            'max_single_position_loss': 0.50,  # $0.50 per position (1.67% of portfolio)
            'max_strategy_daily_loss': 1.50,   # $1.50 per strategy per day (5% of portfolio)
            'max_portfolio_drawdown': 4.50,    # $4.50 max drawdown (15% of portfolio)
            'circuit_breaker_threshold': -3.00, # $3.00 loss (10%) triggers circuit breaker
            'emergency_stop_threshold': -4.50   # $4.50 loss (15%) triggers emergency stop
        }"""

    def _setup_logging(self):
        """Setup logging for live trading"""
        self.logger = logging.getLogger('LiveTradingManager')
        self.logger.setLevel(logging.INFO)

        # Create logs directory
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)

        # File handler
        handler = logging.FileHandler(logs_dir / f"live_trading_{datetime.now().strftime('%Y%m%d')}.log")
        handler.setFormatter(logging.Formatter())
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
((        ))
        self.logger.addHandler(handler)

    def start_live_trading(self) -> bool:
        """Start live trading operations""""""
        if self.is_trading:
            self.logger.warning("Live trading already running")
            return False

        try:
            self.logger.info("# Rocket Starting live trading operations...")

            # Get initial balance from live balance service
            try:
                from .live_balance_service import get_live_balance
                live_balance = get_live_balance()
                initial_balance = live_balance.total_usd_balance
                if initial_balance <= 0:  # Fallback if no live balance
                    initial_balance = 30.0
            except ImportError:
                initial_balance = 30.0  # Fallback

            # Create trading session
            self.trading_session = TradingSession()
                session_id=f"session_{int(time.time())}",
                start_time=datetime.now().isoformat(),
                initial_balance=initial_balance,  # Live portfolio balance
                current_balance=initial_balance
(            )

            # Start monitoring
            self.is_trading = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()

            # Create GitHub task for session start
            self.github_mcp.create_live_trading_task()
                "Session Start",
                {
                    'session_id': self.trading_session.session_id,
                    'start_time': self.trading_session.start_time,
                    'initial_balance': self.trading_session.initial_balance,
                    'active_strategies': len([s for s in self.strategy_dashboard.strategies.values() if s.status == 'active'])
                }
(            )

            self.logger.info("# Check Live trading started successfully")
            return True

        except Exception as e:
            self.logger.error(f"# X Failed to start live trading: {e}")
            return False

    def stop_live_trading(self, reason: str = "Manual stop") -> bool:
        """Stop live trading operations""""""
        if not self.is_trading:
            self.logger.warning("Live trading not running")
            return False

        try:
            self.logger.info(f"⏹️  Stopping live trading: {reason}")

            # Close all positions
            self._close_all_positions(reason)

            # Update session
            if self.trading_session:
                self.trading_session.end_time = datetime.now().isoformat()
                self.trading_session.status = "stopped"

            # Stop monitoring
            self.is_trading = False
            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=10)

            # Create GitHub task for session end
            self.github_mcp.create_live_trading_task()
                "Session End",
                {
                    'session_id': self.trading_session.session_id if self.trading_session else 'unknown',
                    'end_time': datetime.now().isoformat(),
                    'reason': reason,
                    'final_balance': self.trading_session.current_balance if self.trading_session else 0,
                    'total_pnl': self.trading_session.total_pnl if self.trading_session else 0
                }
(            )

            self.logger.info("# Check Live trading stopped successfully")
            return True

        except Exception as e:
            self.logger.error(f"# X Error stopping live trading: {e}")
            return False

    def _monitoring_loop(self):
        """Main monitoring loop for live trading"""
        self.logger.info("# Chart Live trading monitoring started")

        while self.is_trading:
            try:
                # Update real-time balance
                self._update_real_time_balance()

                # Update position P&L
                self._update_positions_pnl()

                # Check risk limits
                self._check_risk_limits()

                # Check strategy performance
                self._check_strategy_performance()

                # Generate alerts if needed
                self._process_alerts()

                # Update daily stats
                self._update_daily_stats()

                time.sleep(self.config['monitoring_interval'])

            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                time.sleep(60)  # Wait before retrying

        self.logger.info("# Chart Live trading monitoring stopped")

    def _update_real_time_balance(self):
        """Update real-time balance from exchange API"""
        current_time = datetime.now()

        # Only update balance at specified intervals to avoid rate limits"""
        if (current_time - self.last_balance_update).seconds < self.balance_update_interval:
            return

        try:
            # In a real implementation, this would call the exchange API
            # For now, we'll simulate balance updates based on P&L
            if self.trading_session:
                # Calculate current balance based on initial balance + total P&L
                calculated_balance = self.trading_session.initial_balance + self.trading_session.total_pnl

                # Add some realistic variation (±0.01 for small fluctuations)
                import random
import secrets
                variation = random.uniform(-0.01, 0.01)
                new_balance = max(0, calculated_balance + variation)  # Ensure non-negative

                # Update balance if changed significantly
                if abs(new_balance - self.real_time_balance) > 0.001:  # $0.001 threshold
                    old_balance = self.real_time_balance
                    self.real_time_balance = new_balance

                    # Update trading session balance
                    self.trading_session.current_balance = new_balance

                    # Log significant balance changes
                    if abs(new_balance - old_balance) > 0.1:  # More than $0.10 change
                        self.logger.info(f"💰 Balance updated: ${old_balance:.2f} → ${new_balance:.2f}")

                self.last_balance_update = current_time

        except Exception as e:
            self.logger.error(f"Error updating real-time balance: {e}")

    def get_real_time_balance(self) -> float:
        """Get current real-time balance"""
        return self.real_time_balance"""

    def _update_positions_pnl(self):
        """Update P&L for all active positions using live market data"""
        for position in self.active_positions.values()""":
            if position.status == 'active':
                try:
                    # Get current market price from live feed
                    # TODO: Integrate with exchange connector service to get real-time prices
                    # For now, position tracking will be handled by position-synchronizer service
                    # This is a placeholder that should be connected to live market data
                    
                    # In production, this should call the exchange connector service:
                        pass
                    # current_price = await self.exchange_connector.get_current_price(position.symbol)
                    
                    # Temporary: Use last known price (should be replaced with live data)
                    current_price = position.current_price or position.entry_price
                    position.current_price = current_price

                    if position.side == 'buy':
                        position.pnl = (position.current_price - position.entry_price) * position.quantity
                    else:
                        position.pnl = (position.entry_price - position.current_price) * position.quantity

                    position.pnl_percentage = (position.pnl / (position.entry_price * position.quantity)) * 100
                    
                except Exception as e:
                    logger.warning(f"# Warning Could not update P&L for position {position.position_id}: {e}")
                    # Continue with existing data

    def _check_risk_limits(self):
        """Check and enforce risk management limits"""
        alerts = []

        # Check daily loss limit"""
        if self.trading_session and self.trading_session.total_pnl < -self.config['max_daily_loss']:
            alerts.append(TradingAlert())
                alert_type='risk',
                severity='high',
                title='Daily Loss Limit Exceeded',
                description=f'Daily loss of ${abs(self.trading_session.total_pnl):.2f} exceeds limit of ${self.config["max_daily_loss"]:.2f}',
                value=self.trading_session.total_pnl,
                timestamp=datetime.now().isoformat()
((            ))

        # Check position losses
        for position in self.active_positions.values():
            if position.pnl < -self.risk_limits['max_single_position_loss']:
                alerts.append(TradingAlert())
                    alert_type='risk',
                    severity='medium',
                    title=f'Position Loss Alert: {position.symbol}',
                    description=f'Position loss of ${abs(position.pnl):.2f} exceeds single position limit',
                    strategy_name=position.strategy_name,
                    symbol=position.symbol,
                    value=position.pnl,
                    timestamp=datetime.now().isoformat()
((                ))

        # Send alerts to queue
        for alert in alerts:
            asyncio.run(self.alerts_queue.put(alert))

    def _check_strategy_performance(self):
        """Check strategy performance and create alerts"""
        alerts = []

        for strategy in self.strategy_dashboard.strategies.values()""":
            if strategy.status == 'active':
                # Check win rate
                if strategy.win_rate < self.config['alert_thresholds']['low_win_rate']:
                    alerts.append(TradingAlert())
                        alert_type='performance',
                        severity='medium',
                        title=f'Low Win Rate: {strategy.strategy_name}',
                        description=f'Win rate of {strategy.win_rate:.1f}% below threshold of {self.config["alert_thresholds"]["low_win_rate"]}%',
                        strategy_name=strategy.strategy_name,
                        value=strategy.win_rate,
                        timestamp=datetime.now().isoformat()
((                    ))

                # Check drawdown
                if strategy.max_drawdown < self.config['alert_thresholds']['high_drawdown']:
                    alerts.append(TradingAlert())
                        alert_type='risk',
                        severity='high',
                        title=f'High Drawdown: {strategy.strategy_name}',
                        description=f'Drawdown of {strategy.max_drawdown:.1f}% exceeds threshold of {self.config["alert_thresholds"]["high_drawdown"]}%',
                        strategy_name=strategy.strategy_name,
                        value=strategy.max_drawdown,
                        timestamp=datetime.now().isoformat()
((                    ))

        # Send alerts to queue
        for alert in alerts:
            asyncio.run(self.alerts_queue.put(alert))

    def _process_alerts(self):
        """Process alerts from queue""""""
        try:
            while not self.alerts_queue.empty():
                alert = self.alerts_queue.get_nowait()

                # Log alert
                self.logger.warning(f"🚨 Alert: {alert.title} - {alert.description}")

                # Create GitHub task for alert
                self.github_mcp.create_risk_alert_task(alert)

                # Handle critical alerts
                if alert.severity == 'critical':
                    self._handle_critical_alert(alert)

        except asyncio.QueueEmpty:
            pass

    def _handle_critical_alert(self, alert: TradingAlert):
        """Handle critical alerts with emergency actions"""
        self.logger.critical(f"🚨 CRITICAL ALERT: {alert.title}")

        if 'Daily Loss Limit' in alert.title:
            self.logger.critical("🚨 Emergency stop triggered due to daily loss limit")
            self.stop_live_trading("Emergency stop - Daily loss limit exceeded")

        elif 'Drawdown' in alert.title:
            self.logger.critical("🚨 Circuit breaker triggered due to high drawdown")
            # Could pause specific strategy or reduce position sizes

    def _update_daily_stats(self):
        """Update daily trading statistics"""
        current_hour = datetime.now().hour

        # Reset daily stats at midnight"""
        if current_hour == 0 and not hasattr(self, '_daily_reset_done'):
            self.daily_stats.clear()
            self._daily_reset_done = True
        elif current_hour > 0:
            self._daily_reset_done = False

        # Update session stats
        if self.trading_session:
            self.trading_session.total_pnl = sum(p.pnl for p in self.active_positions.values())
            self.trading_session.current_balance = self.trading_session.initial_balance + self.trading_session.total_pnl
            self.trading_session.total_trades = len([p for p in self.active_positions.values() if p.status == 'closed'])

    def _close_all_positions(self, reason: str):
        """Close all active positions"""
        self.logger.info(f"Closing all positions: {reason}")

        for position in list(self.active_positions.values()):
            if position.status == 'active':
                # Mock position closure (in real implementation, use exchange API)
                position.status = 'closed'
                position.timestamp = datetime.now().isoformat()

                self.logger.info(f"Closed position: {position.symbol} - P&L: ${position.pnl:.2f}")

        self.active_positions.clear()

    def get_trading_status(self) -> Dict[str, Any]
        """Get current trading status"""
        # Get live balance from service:"""
        try:
            pass
    from .live_balance_service import get_live_balance
            live_balance = get_live_balance()
            current_balance = live_balance.total_usd_balance
            balance_status = live_balance.status
            exchange = live_balance.exchange
        except ImportError:
            current_balance = self.real_time_balance
            balance_status = 'offline'
            exchange = 'unknown'

        return {
            'is_trading': self.is_trading,
            'session_id': self.trading_session.session_id if self.trading_session else None,
            'active_positions': len(self.active_positions),
            'total_pnl': self.trading_session.total_pnl if self.trading_session else 0,
            'current_balance': current_balance,  # Use live balance service
            'initial_balance': self.trading_session.initial_balance if self.trading_session else 30.0,
            'session_start': self.trading_session.start_time if self.trading_session else None,
            'balance_status': balance_status,
            'exchange': exchange,
            'last_balance_update': self.last_balance_update.isoformat() if hasattr(self, 'last_balance_update') else None,
            'daily_stats': dict(self.daily_stats)
        }

    def generate_trading_report(self) -> str:
        """Generate comprehensive trading report"""
        status = self.get_trading_status()

        report = f"""
💰 LIVE TRADING REPORT
{'='*50}

# Chart Session Status
  Trading Active: {'# Check Yes' if status['is_trading'] else '# X No'}
  Session ID: {status['session_id'] or 'None'}
  Session Start: {status['session_start'] or 'Not started'}

💼 Portfolio Status
  Initial Balance: ${status['initial_balance']:,.2f}
  Current Balance: ${status['current_balance']:,.2f}
  Total P&L: ${status['total_pnl']:,.2f} ({status['total_pnl']/status['initial_balance']*100:.2f}%)
  Active Positions: {status['active_positions']}
  Balance Status: {status.get('balance_status', 'Unknown').upper()}
  Exchange: {status.get('exchange', 'Unknown').upper()}
  Last Balance Update: {status.get('last_balance_update', 'Never')}

📈 Daily Statistics
"""

        for key, value in status['daily_stats'].items():
            report += f"  {key}: {value}\n"

        report += f"""

📋 Active Positions
{'='*30}
""""""

        if self.active_positions:
            for position in self.active_positions.values():
                report += f"""
{position.symbol} ({position.strategy_name})
  Side: {position.side.upper()}
  Quantity: {position.quantity}
  Entry: ${position.entry_price:.4f}
  Current: ${position.current_price:.4f}
  P&L: ${position.pnl:.2f} ({position.pnl_percentage:.2f}%)
  Status: {position.status}
"""
        else:
            report += "No active positions\n"

        report += f"""

# Target Strategy Performance
{'='*30}
{self.strategy_dashboard.display_strategy_table()}

# Warning  Risk Management
{'='*20}
  Risk per Trade: {self.config['risk_per_trade']*100:.1f}%
  Max Daily Loss: ${self.config['max_daily_loss']:.2f}
  Max Positions: {self.config['max_positions']}
  Min Leverage: {self.config['min_leverage']}x

# Chart System Health
  Monitoring: {'# Check Active' if self.is_trading else '# X Inactive'}
  Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

        return report.strip()"""

    def export_trading_data(self, filename: str = None) -> str:
        """Export trading data for analysis""""""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"trading_session_{timestamp}.json"

        filepath = self.reports_dir / filename

        export_data = {
            'session': asdict(self.trading_session) if self.trading_session else None,
            'positions': [asdict(p) for p in self.active_positions.values()],
            'config': self.config,
            'risk_limits': self.risk_limits,
            'daily_stats': dict(self.daily_stats),
            'exported_at': datetime.now().isoformat()
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        return str(filepath)

def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='VIPER Live Trading Manager')
    parser.add_argument('--start', action='store_true', help='Start live trading')
    parser.add_argument('--stop', action='store_true', help='Stop live trading')
    parser.add_argument('--status', action='store_true', help='Show trading status')
    parser.add_argument('--report', action='store_true', help='Generate trading report')
    parser.add_argument('--export', help='Export trading data to file')
    parser.add_argument('--monitor', action='store_true', help='Start monitoring (blocks)')

    args = parser.parse_args()

    # Create trading manager
    manager = LiveTradingManager()"""

    if args.start:
        if manager.start_live_trading():
        else:
            pass

    elif args.stop:
        if manager.stop_live_trading():
        else:
            pass

    elif args.status:
        status = manager.get_trading_status()
        print(f"  Active: {'# Check Yes' if status['is_trading'] else '# X No'}")
        print(f"  Positions: {status['active_positions']}")
        print(f"  Total P&L: ${status['total_pnl']:,.2f}")
        print(f"  Current Balance: ${status['current_balance']:,.2f}")

    elif args.report:
        report = manager.generate_trading_report()

    elif args.export:
        filepath = manager.export_trading_data(args.export)

    elif args.monitor:
        print("# Rocket Starting live trading and monitoring...")
        if manager.start_live_trading():
            try:
                while True:
                    time.sleep(5)
                    # Could add periodic status updates here
            except KeyboardInterrupt:
                manager.stop_live_trading("User interrupt")
        else:
            pass

    else:
        print("  python scripts/live_trading_manager.py --start --monitor")

if __name__ == '__main__':
    main()
