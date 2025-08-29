#!/usr/bin/env python3
"""
📊 VIPER Live Trading Monitor
Real-time monitoring dashboard for live trading operations
"""

import os
import time
import json
import requests
import curses
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LiveTradingMonitor:
    """Real-time monitoring dashboard for live trading"""

    def __init__(self):
        self.api_server_url = "http://localhost:8000"
        self.risk_manager_url = "http://localhost:8002"
        self.order_lifecycle_url = "http://localhost:8013"
        self.exchange_connector_url = "http://localhost:8005"

        self.monitoring_active = False
        self.system_data = {}
        self.update_interval = 5  # seconds

        # Performance tracking
        self.start_time = datetime.now()
        self.initial_balance = 0.0
        self.peak_balance = 0.0

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            status = {
                'timestamp': datetime.now().isoformat(),
                'services': {},
                'trading': {},
                'performance': {},
                'risk': {}
            }

            # Service health
            services = {
                'API Server': f"{self.api_server_url}/health",
                'Risk Manager': f"{self.risk_manager_url}/health",
                'Order Lifecycle': f"{self.order_lifecycle_url}/health",
                'Exchange Connector': f"{self.exchange_connector_url}/health"
            }

            for name, url in services.items():
                try:
                    response = requests.get(url, timeout=3)
                    status['services'][name] = '✅' if response.status_code == 200 else '❌'
                except Exception:
                    status['services'][name] = '❌'

            # Trading data
            try:
                positions_response = requests.get(f"{self.risk_manager_url}/api/tp-sl-tsl/positions", timeout=3)
                if positions_response.status_code == 200:
                    positions_data = positions_response.json()
                    status['trading']['active_positions'] = len(positions_data.get('positions', {}))
                    status['trading']['total_exposure'] = positions_data.get('total_exposure', 0)
            except Exception:
                status['trading']['active_positions'] = 0
                status['trading']['total_exposure'] = 0

            # Account balance
            try:
                balance_response = requests.get(f"{self.exchange_connector_url}/api/balance", timeout=3)
                if balance_response.status_code == 200:
                    balance_data = balance_response.json()
                    current_balance = balance_data.get('free', 0)

                    if self.initial_balance == 0:
                        self.initial_balance = current_balance

                    self.peak_balance = max(self.peak_balance, current_balance)

                    status['performance']['current_balance'] = current_balance
                    status['performance']['daily_pnl'] = current_balance - self.initial_balance
                    status['performance']['peak_balance'] = self.peak_balance
                    status['performance']['drawdown'] = (self.peak_balance - current_balance) / max(self.peak_balance, 1) * 100
            except Exception:
                status['performance']['current_balance'] = 0
                status['performance']['daily_pnl'] = 0

            # Risk metrics
            try:
                risk_config = requests.get(f"{self.risk_manager_url}/api/tp-sl-tsl/config", timeout=3)
                if risk_config.status_code == 200:
                    config_data = risk_config.json()
                    status['risk']['max_positions'] = config_data.get('max_positions', 15)
                    status['risk']['risk_per_trade'] = config_data.get('risk_per_trade', 0.02)
                    status['risk']['daily_loss_limit'] = config_data.get('daily_loss_limit', 0.03)
            except Exception:
                status['risk']['max_positions'] = 15
                status['risk']['risk_per_trade'] = 0.02
                status['risk']['daily_loss_limit'] = 0.03

            return status

        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {}

    def draw_dashboard(self, stdscr, status: Dict[str, Any]):
        """Draw the monitoring dashboard"""
        stdscr.clear()

        # Colors
        curses.start_color()
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)  # Healthy
        curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)    # Error
        curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK) # Warning
        curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)   # Info

        height, width = stdscr.getmaxyx()

        # Header
        header = "🚀 VIPER LIVE TRADING MONITOR"
        stdscr.addstr(0, (width - len(header)) // 2, header, curses.A_BOLD)

        # Timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        stdscr.addstr(1, 0, f"Time: {timestamp}")

        # Runtime
        runtime = datetime.now() - self.start_time
        stdscr.addstr(1, width - 25, f"Runtime: {str(runtime).split('.')[0]}")

        # Service Status
        stdscr.addstr(3, 0, "🏥 SERVICE STATUS:", curses.A_BOLD)
        y_pos = 4
        for service, status in status.get('services', {}).items():
            color = curses.color_pair(1) if status == '✅' else curses.color_pair(2)
            stdscr.addstr(y_pos, 2, f"{service}: {status}", color)
            y_pos += 1

        # Trading Status
        y_pos += 1
        stdscr.addstr(y_pos, 0, "📊 TRADING STATUS:", curses.A_BOLD)
        y_pos += 1
        trading = status.get('trading', {})
        stdscr.addstr(y_pos, 2, f"Active Positions: {trading.get('active_positions', 0)}", curses.color_pair(4))
        y_pos += 1
        exposure = trading.get('total_exposure', 0)
        stdscr.addstr(y_pos, 2, f"Total Exposure: ${exposure:.2f}", curses.color_pair(4))

        # Performance
        y_pos += 2
        stdscr.addstr(y_pos, 0, "💰 PERFORMANCE:", curses.A_BOLD)
        y_pos += 1
        perf = status.get('performance', {})
        balance = perf.get('current_balance', 0)
        pnl = perf.get('daily_pnl', 0)
        drawdown = perf.get('drawdown', 0)

        stdscr.addstr(y_pos, 2, f"Balance: ${balance:.2f}", curses.color_pair(4))
        y_pos += 1
        pnl_color = curses.color_pair(1) if pnl >= 0 else curses.color_pair(2)
        stdscr.addstr(y_pos, 2, f"Daily P&L: ${pnl:.2f}", pnl_color)
        y_pos += 1
        dd_color = curses.color_pair(1) if drawdown < 2 else curses.color_pair(3) if drawdown < 5 else curses.color_pair(2)
        stdscr.addstr(y_pos, 2, f"Drawdown: {drawdown:.2f}%", dd_color)

        # Risk Metrics
        y_pos += 2
        stdscr.addstr(y_pos, 0, "⚖️ RISK METRICS:", curses.A_BOLD)
        y_pos += 1
        risk = status.get('risk', {})
        stdscr.addstr(y_pos, 2, f"Max Positions: {risk.get('max_positions', 15)}", curses.color_pair(4))
        y_pos += 1
        stdscr.addstr(y_pos, 2, f"Risk per Trade: {risk.get('risk_per_trade', 0.02)*100:.1f}%", curses.color_pair(4))
        y_pos += 1
        stdscr.addstr(y_pos, 2, f"Daily Loss Limit: {risk.get('daily_loss_limit', 0.03)*100:.1f}%", curses.color_pair(4))

        # Footer with controls
        footer_y = height - 3
        stdscr.addstr(footer_y, 0, "=" * width, curses.A_DIM)
        stdscr.addstr(footer_y + 1, 0, "Controls: 'q' to quit | 'r' to refresh | Auto-refresh every 5 seconds", curses.A_DIM)
        stdscr.addstr(footer_y + 2, 0, "Emergency: Ctrl+C to stop trading | 'docker compose down' to stop all services", curses.color_pair(2))

        stdscr.refresh()

    def monitor_loop(self, stdscr):
        """Main monitoring loop"""
        self.monitoring_active = True

        while self.monitoring_active:
            try:
                # Get system status
                status = self.get_system_status()

                # Draw dashboard
                self.draw_dashboard(stdscr, status)

                # Handle user input (non-blocking)
                stdscr.timeout(100)  # 100ms timeout
                try:
                    key = stdscr.getch()
                    if key == ord('q'):
                        self.monitoring_active = False
                        break
                    elif key == ord('r'):
                        # Force refresh
                        continue
                except Exception:
                    pass

                # Wait for next update
                time.sleep(self.update_interval)

            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(1)

    def start_monitoring(self):
        """Start the monitoring dashboard"""

        try:
            curses.wrapper(self.monitor_loop)
        except KeyboardInterrupt:
            pass
        finally:

    def get_system_summary(self) -> Dict[str, Any]:
        """Get a summary of system status for logging"""
        status = self.get_system_status()

        summary = {
            'timestamp': datetime.now().isoformat(),
            'services_healthy': sum(1 for s in status.get('services', {}).values() if s == '✅'),
            'total_services': len(status.get('services', {})),
            'active_positions': status.get('trading', {}).get('active_positions', 0),
            'current_balance': status.get('performance', {}).get('current_balance', 0),
            'daily_pnl': status.get('performance', {}).get('daily_pnl', 0),
            'drawdown': status.get('performance', {}).get('drawdown', 0)
        }

        return summary

def print_system_summary():
    """Print a one-time system summary"""
    monitor = LiveTradingMonitor()
    summary = monitor.get_system_summary()

    print(f"Services: {summary['services_healthy']}/{summary['total_services']} healthy")
    print(f"Active Positions: {summary['active_positions']}")
    print(f"Current Balance: ${summary['current_balance']:.2f}")

def main():
    """Main entry point"""
    if len(os.sys.argv) > 1 and os.sys.argv[1] == "--summary":
        print_system_summary()
    else:
        monitor = LiveTradingMonitor()
        monitor.start_monitoring()

if __name__ == "__main__":
    main()
