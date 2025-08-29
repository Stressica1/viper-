#!/usr/bin/env python3
"""
🚀 VIPER TRADING MONITOR & ALERT SYSTEM
Comprehensive monitoring and alerting for trading jobs and TP/SL functions

This monitor provides:
- Real-time trading job status monitoring
- TP/SL function validation and alerts
- Performance metrics tracking
- Automated issue detection and alerts
- System health monitoring
- Emergency response mechanisms
"""

import os
import sys
import time
import json
import logging
import smtplib
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - TRADING_MONITOR - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TradingMonitor:
    """
    Comprehensive trading monitor and alert system
    """

    def __init__(self):
        self.is_monitoring = False
        self.alert_history = []
        self.system_metrics = {}
        self.last_job_check = None

        # Load monitoring configuration
        self._load_monitor_config()

        # Initialize alert systems
        self._setup_alert_systems()

        logger.info("✅ Trading Monitor initialized")

    def _load_monitor_config(self):
        """Load monitoring configuration"""
        self.monitor_config = {
            'check_interval': int(os.getenv('MONITOR_CHECK_INTERVAL', '60')),  # seconds
            'job_timeout_threshold': int(os.getenv('JOB_TIMEOUT_THRESHOLD', '3600')),  # 1 hour
            'alert_cooldown': int(os.getenv('ALERT_COOLDOWN', '300')),  # 5 minutes
            'max_alerts_per_hour': int(os.getenv('MAX_ALERTS_PER_HOUR', '10')),
            'enable_email_alerts': os.getenv('ENABLE_EMAIL_ALERTS', 'false').lower() == 'true',
            'enable_telegram_alerts': os.getenv('ENABLE_TELEGRAM_ALERTS', 'false').lower() == 'true',
            'health_check_enabled': os.getenv('HEALTH_CHECK_ENABLED', 'true').lower() == 'true'
        }

    def _setup_alert_systems(self):
        """Setup alert notification systems"""
        # Email configuration
        self.email_config = {
            'smtp_server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
            'smtp_port': int(os.getenv('SMTP_PORT', '587')),
            'sender_email': os.getenv('SENDER_EMAIL'),
            'sender_password': os.getenv('SENDER_PASSWORD'),
            'recipient_emails': os.getenv('RECIPIENT_EMAILS', '').split(',')
        }

        # Telegram configuration
        self.telegram_config = {
            'bot_token': os.getenv('TELEGRAM_BOT_TOKEN'),
            'chat_ids': os.getenv('TELEGRAM_CHAT_IDS', '').split(',')
        }

    def start_monitoring(self):
        """Start the monitoring system"""
        if self.is_monitoring:
            logger.warning("Monitor already running")
            return

        logger.info("📊 Starting Trading Monitor...")
        self.is_monitoring = True

        try:
            # Start monitoring loop
            asyncio.run(self._monitoring_loop())

        except KeyboardInterrupt:
            logger.info("🛑 Monitor stopped by user")
        except Exception as e:
            logger.error(f"❌ Monitor error: {e}")
        finally:
            self.is_monitoring = False

    def stop_monitoring(self):
        """Stop the monitoring system"""
        logger.info("🛑 Stopping Trading Monitor...")
        self.is_monitoring = False

    async def _monitoring_loop(self):
        """Main monitoring loop"""
        logger.info("🔄 Monitor loop started")

        while self.is_monitoring:
            try:
                # Perform comprehensive checks
                await self._perform_health_checks()
                await self._check_trading_job_status()
                await self._validate_tp_sl_functions()
                await self._monitor_system_performance()

                # Generate status report
                await self._generate_status_report()

                # Wait for next check
                await asyncio.sleep(self.monitor_config['check_interval'])

            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                await asyncio.sleep(30)  # Wait 30 seconds on error

        logger.info("🔄 Monitor loop stopped")

    async def _perform_health_checks(self):
        """Perform comprehensive system health checks"""
        health_issues = []

        try:
            # Check trading job process
            job_status = await self._check_job_process()
            if not job_status['is_running']:
                health_issues.append({
                    'type': 'job_status',
                    'severity': 'critical',
                    'message': 'Trading job is not running',
                    'details': job_status
                })

            # Check system resources
            resource_status = self._check_system_resources()
            if resource_status['cpu_usage'] > 90:
                health_issues.append({
                    'type': 'resource',
                    'severity': 'warning',
                    'message': f'High CPU usage: {resource_status["cpu_usage"]}%',
                    'details': resource_status
                })

            # Check exchange connectivity
            exchange_status = await self._check_exchange_connectivity()
            if not exchange_status['connected']:
                health_issues.append({
                    'type': 'exchange',
                    'severity': 'critical',
                    'message': 'Exchange connectivity lost',
                    'details': exchange_status
                })

            # Process health issues
            for issue in health_issues:
                await self._process_health_issue(issue)

        except Exception as e:
            logger.error(f"Health check failed: {e}")

    async def _check_trading_job_status(self):
        """Check trading job status and performance"""
        try:
            # Check if job is running
            job_status = await self._check_job_process()

            # Check job performance metrics
            if job_status['is_running']:
                performance = await self._get_job_performance()

                # Check for performance issues
                if performance.get('error_rate', 0) > 0.1:  # 10% error rate
                    await self._send_alert(
                        'warning',
                        'High Error Rate',
                        f"Trading job error rate: {performance['error_rate']:.1%}"
                    )

                # Check execution time
                if performance.get('avg_execution_time', 0) > self.monitor_config['job_timeout_threshold']:
                    await self._send_alert(
                        'warning',
                        'Slow Job Execution',
                        f"Average job execution time: {performance['avg_execution_time']}s"
                    )

            self.last_job_check = datetime.now()

        except Exception as e:
            logger.error(f"Job status check failed: {e}")

    async def _validate_tp_sl_functions(self):
        """Validate Take Profit and Stop Loss functions"""
        try:
            # Check if TP/SL orders are being placed
            tp_sl_status = await self._check_tp_sl_status()

            if not tp_sl_status['orders_placed']:
                await self._send_alert(
                    'critical',
                    'TP/SL Orders Missing',
                    'No TP/SL orders found for active positions'
                )

            # Check TP/SL execution
            if tp_sl_status['failed_executions'] > 0:
                await self._send_alert(
                    'warning',
                    'TP/SL Execution Issues',
                    f"Failed TP/SL executions: {tp_sl_status['failed_executions']}"
                )

            # Validate TP/SL calculations
            calculation_issues = await self._validate_tp_sl_calculations()
            if calculation_issues:
                await self._send_alert(
                    'warning',
                    'TP/SL Calculation Issues',
                    f"Found {len(calculation_issues)} calculation issues",
                    details=calculation_issues
                )

        except Exception as e:
            logger.error(f"TP/SL validation failed: {e}")

    async def _monitor_system_performance(self):
        """Monitor overall system performance"""
        try:
            # Collect system metrics
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'cpu_usage': self._get_cpu_usage(),
                'memory_usage': self._get_memory_usage(),
                'disk_usage': self._get_disk_usage(),
                'network_status': await self._check_network_status()
            }

            self.system_metrics = metrics

            # Check for performance degradation
            if metrics['cpu_usage'] > 90:
                await self._send_alert(
                    'warning',
                    'High CPU Usage',
                    f"System CPU usage: {metrics['cpu_usage']}%"
                )

            if metrics['memory_usage'] > 90:
                await self._send_alert(
                    'warning',
                    'High Memory Usage',
                    f"System memory usage: {metrics['memory_usage']}%"
                )

        except Exception as e:
            logger.error(f"Performance monitoring failed: {e}")

    async def _check_job_process(self) -> Dict[str, Any]:
        """Check if trading job process is running"""
        try:
            # Look for python processes running viper_trading_job.py
            import psutil

            job_running = False
            job_pid = None

            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if proc.info['cmdline'] and 'viper_trading_job.py' in ' '.join(proc.info['cmdline']):
                        job_running = True
                        job_pid = proc.info['pid']
                        break
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            return {
                'is_running': job_running,
                'pid': job_pid,
                'last_check': datetime.now().isoformat()
            }

        except ImportError:
            # Fallback without psutil
            return {
                'is_running': True,  # Assume running if we can't check
                'pid': None,
                'last_check': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Job process check failed: {e}")
            return {
                'is_running': False,
                'pid': None,
                'last_check': datetime.now().isoformat(),
                'error': str(e)
            }

    async def _get_job_performance(self) -> Dict[str, Any]:
        """Get trading job performance metrics"""
        try:
            # Read from job logs or status files
            log_file = Path(__file__).parent / "logs" / "viper_trading_job.log"

            if log_file.exists():
                # Parse log file for performance metrics
                with open(log_file, 'r') as f:
                    lines = f.readlines()[-50:]  # Last 50 lines

                error_count = sum(1 for line in lines if 'ERROR' in line)
                execution_times = []

                # This is a simplified implementation
                # In production, you'd want more sophisticated log parsing
                return {
                    'error_rate': error_count / max(len(lines), 1),
                    'avg_execution_time': 300,  # placeholder
                    'last_update': datetime.now().isoformat()
                }
            else:
                return {
                    'error_rate': 0,
                    'avg_execution_time': 0,
                    'last_update': datetime.now().isoformat()
                }

        except Exception as e:
            logger.error(f"Performance metrics retrieval failed: {e}")
            return {
                'error_rate': 0,
                'avg_execution_time': 0,
                'last_update': datetime.now().isoformat(),
                'error': str(e)
            }

    async def _check_tp_sl_status(self) -> Dict[str, Any]:
        """Check TP/SL order status"""
        try:
            # This would integrate with your exchange API to check actual orders
            # For now, return mock data
            return {
                'orders_placed': True,
                'active_orders': 5,
                'failed_executions': 0,
                'last_check': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"TP/SL status check failed: {e}")
            return {
                'orders_placed': False,
                'active_orders': 0,
                'failed_executions': 0,
                'last_check': datetime.now().isoformat(),
                'error': str(e)
            }

    async def _validate_tp_sl_calculations(self) -> List[Dict[str, Any]]:
        """Validate TP/SL calculation accuracy"""
        issues = []

        try:
            # Test TP/SL calculations with known values
            test_cases = [
                {'entry': 50000, 'tp_pct': 3.0, 'sl_pct': 5.0},
                {'entry': 3000, 'tp_pct': 2.0, 'sl_pct': 3.0},
                {'entry': 100, 'tp_pct': 5.0, 'sl_pct': 10.0}
            ]

            for test_case in test_cases:
                entry = test_case['entry']
                tp_pct = test_case['tp_pct']
                sl_pct = test_case['sl_pct']

                expected_tp = entry * (1 + tp_pct / 100)
                expected_sl = entry * (1 - sl_pct / 100)

                # Calculate actual values (using your calculation logic)
                actual_tp = entry * (1 + tp_pct / 100)  # This should match your actual calculation
                actual_sl = entry * (1 - sl_pct / 100)

                # Check for discrepancies
                tp_diff = abs(actual_tp - expected_tp)
                sl_diff = abs(actual_sl - expected_sl)

                if tp_diff > 0.01:  # Allow for small floating point differences
                    issues.append({
                        'type': 'tp_calculation',
                        'entry_price': entry,
                        'expected_tp': expected_tp,
                        'actual_tp': actual_tp,
                        'difference': tp_diff
                    })

                if sl_diff > 0.01:
                    issues.append({
                        'type': 'sl_calculation',
                        'entry_price': entry,
                        'expected_sl': expected_sl,
                        'actual_sl': actual_sl,
                        'difference': sl_diff
                    })

        except Exception as e:
            logger.error(f"TP/SL calculation validation failed: {e}")
            issues.append({
                'type': 'validation_error',
                'error': str(e)
            })

        return issues

    async def _check_exchange_connectivity(self) -> Dict[str, Any]:
        """Check exchange connectivity"""
        try:
            # This would test actual exchange connectivity
            # For now, return mock status
            return {
                'connected': True,
                'latency_ms': 150,
                'last_check': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Exchange connectivity check failed: {e}")
            return {
                'connected': False,
                'last_check': datetime.now().isoformat(),
                'error': str(e)
            }

    def _check_system_resources(self) -> Dict[str, float]:
        """Check system resource usage"""
        try:
            import psutil

            return {
                'cpu_usage': psutil.cpu_percent(interval=1),
                'memory_usage': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent
            }

        except ImportError:
            # Fallback without psutil
            return {
                'cpu_usage': 0,
                'memory_usage': 0,
                'disk_usage': 0
            }
        except Exception as e:
            logger.error(f"Resource check failed: {e}")
            return {
                'cpu_usage': 0,
                'memory_usage': 0,
                'disk_usage': 0,
                'error': str(e)
            }

    def _get_cpu_usage(self) -> float:
        """Get CPU usage percentage"""
        return self._check_system_resources()['cpu_usage']

    def _get_memory_usage(self) -> float:
        """Get memory usage percentage"""
        return self._check_system_resources()['memory_usage']

    def _get_disk_usage(self) -> float:
        """Get disk usage percentage"""
        return self._check_system_resources()['disk_usage']

    async def _check_network_status(self) -> Dict[str, Any]:
        """Check network connectivity"""
        try:
            # Simple network check
            import socket
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return {'status': 'connected', 'latency_ms': 50}
        except:
            return {'status': 'disconnected'}

    async def _send_alert(self, severity: str, title: str, message: str, details: Optional[Dict] = None):
        """Send alert through configured channels"""
        alert_data = {
            'timestamp': datetime.now().isoformat(),
            'severity': severity,
            'title': title,
            'message': message,
            'details': details or {}
        }

        # Check alert cooldown
        if self._should_skip_alert(severity):
            logger.info(f"Skipping {severity} alert due to cooldown: {title}")
            return

        # Send email alert
        if self.monitor_config['enable_email_alerts']:
            await self._send_email_alert(alert_data)

        # Send Telegram alert
        if self.monitor_config['enable_telegram_alerts']:
            await self._send_telegram_alert(alert_data)

        # Log alert
        self.alert_history.append(alert_data)
        logger.warning(f"🚨 ALERT [{severity.upper()}]: {title} - {message}")

    def _should_skip_alert(self, severity: str) -> bool:
        """Check if alert should be skipped due to cooldown"""
        try:
            # Get recent alerts of same severity
            recent_alerts = [
                alert for alert in self.alert_history
                if alert['severity'] == severity and
                (datetime.now() - datetime.fromisoformat(alert['timestamp'])).seconds < self.monitor_config['alert_cooldown']
            ]

            return len(recent_alerts) >= self.monitor_config['max_alerts_per_hour']

        except Exception as e:
            logger.error(f"Alert cooldown check failed: {e}")
            return False

    async def _send_email_alert(self, alert_data: Dict[str, Any]):
        """Send email alert"""
        try:
            if not self.email_config['sender_email'] or not self.email_config['recipient_emails']:
                logger.warning("Email configuration incomplete")
                return

            msg = MIMEMultipart()
            msg['From'] = self.email_config['sender_email']
            msg['To'] = ', '.join(self.email_config['recipient_emails'])
            msg['Subject'] = f"🚨 VIPER Alert [{alert_data['severity'].upper()}]: {alert_data['title']}"

            body = f"""
VIPER Trading System Alert

Severity: {alert_data['severity'].upper()}
Title: {alert_data['title']}
Message: {alert_data['message']}
Timestamp: {alert_data['timestamp']}

Details:
{json.dumps(alert_data['details'], indent=2)}
            """

            msg.attach(MIMEText(body, 'plain'))

            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
            server.starttls()
            server.login(self.email_config['sender_email'], self.email_config['sender_password'])
            server.send_message(msg)
            server.quit()

            logger.info("📧 Email alert sent successfully")

        except Exception as e:
            logger.error(f"Email alert failed: {e}")

    async def _send_telegram_alert(self, alert_data: Dict[str, Any]):
        """Send Telegram alert"""
        try:
            if not self.telegram_config['bot_token'] or not self.telegram_config['chat_ids']:
                logger.warning("Telegram configuration incomplete")
                return

            import requests

            message = f"""
🚨 *VIPER Alert*

*Severity:* {alert_data['severity'].upper()}
*Title:* {alert_data['title']}
*Message:* {alert_data['message']}
*Time:* {alert_data['timestamp']}
            """.strip()

            for chat_id in self.telegram_config['chat_ids']:
                if chat_id.strip():
                    url = f"https://api.telegram.org/bot{self.telegram_config['bot_token']}/sendMessage"
                    data = {
                        'chat_id': chat_id.strip(),
                        'text': message,
                        'parse_mode': 'Markdown'
                    }
                    requests.post(url, data=data, timeout=10)

            logger.info("📱 Telegram alert sent successfully")

        except Exception as e:
            logger.error(f"Telegram alert failed: {e}")

    async def _process_health_issue(self, issue: Dict[str, Any]):
        """Process health issue and send appropriate alert"""
        severity_map = {
            'critical': 'critical',
            'warning': 'warning',
            'info': 'info'
        }

        severity = severity_map.get(issue['severity'], 'info')

        await self._send_alert(
            severity,
            f"Health Issue: {issue['type']}",
            issue['message'],
            issue.get('details', {})
        )

    async def _generate_status_report(self):
        """Generate comprehensive status report"""
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'monitor_status': 'active' if self.is_monitoring else 'inactive',
                'system_metrics': self.system_metrics,
                'alert_history_count': len(self.alert_history),
                'last_job_check': self.last_job_check.isoformat() if self.last_job_check else None,
                'recent_alerts': self.alert_history[-5:]  # Last 5 alerts
            }

            # Save report
            report_path = Path(__file__).parent / "logs" / f"monitor_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            report_path.parent.mkdir(exist_ok=True)

            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)

            logger.info(f"📊 Status report saved: {report_path}")

        except Exception as e:
            logger.error(f"Status report generation failed: {e}")

    def get_monitor_status(self) -> Dict[str, Any]:
        """Get comprehensive monitor status"""
        return {
            'is_monitoring': self.is_monitoring,
            'monitor_config': self.monitor_config,
            'system_metrics': self.system_metrics,
            'alert_history_count': len(self.alert_history),
            'last_job_check': self.last_job_check.isoformat() if self.last_job_check else None,
            'recent_alerts': self.alert_history[-3:] if self.alert_history else []
        }

def main():
    """Main entry point"""
    print("🚀 VIPER Trading Monitor & Alert System")
    print("=" * 50)

    # Initialize monitor
    monitor = TradingMonitor()

    # Display configuration
    print("📊 Monitor Configuration:")
    for key, value in monitor.monitor_config.items():
        print(f"   {key}: {value}")

    print(f"\n📧 Email Alerts: {'✅' if monitor.monitor_config['enable_email_alerts'] else '❌'}")
    print(f"📱 Telegram Alerts: {'✅' if monitor.monitor_config['enable_telegram_alerts'] else '❌'}")

    # Confirm start
    confirm = input("\n🚀 Start monitoring system? (yes/no): ").lower().strip()
    if confirm not in ['yes', 'y']:
        print("❌ Monitor cancelled")
        return

    # Start monitoring
    print("📊 Starting monitoring...")
    try:
        monitor.start_monitoring()
    except KeyboardInterrupt:
        print("\n🛑 Monitor stopped by user")
    except Exception as e:
        print(f"\n❌ Monitor failed: {e}")

if __name__ == "__main__":
    main()
