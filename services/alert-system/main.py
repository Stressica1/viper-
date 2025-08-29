#!/usr/bin/env python3
"""
# Rocket VIPER Trading Bot - Alert System
Automated notification and alert management service

Features:
- Real-time trading signal alerts
- Risk management notifications
- System health monitoring alerts
- Multi-channel notification support
- Configurable alert rules and thresholds
"""

import os
import json
import logging
import asyncio
import threading
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import redis
import requests
from enum import Enum

# Load environment variables
REDIS_URL = os.getenv('REDIS_URL', 'redis://redis:6379')
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
SERVICE_NAME = os.getenv('SERVICE_NAME', 'alert-system')

# Email configuration
SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
SMTP_PORT = int(os.getenv('SMTP_PORT', '587'))
SMTP_USERNAME = os.getenv('SMTP_USERNAME', '')
SMTP_PASSWORD = os.getenv('SMTP_PASSWORD', '')
FROM_EMAIL = os.getenv('FROM_EMAIL', 'viper@tradingbot.com')
TO_EMAILS = os.getenv('TO_EMAILS', '').split(',')

# Telegram configuration
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_IDS = os.getenv('TELEGRAM_CHAT_IDS', '').split(',')

# Configure logging
log_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AlertType(Enum):
    TRADING_SIGNAL = "TRADING_SIGNAL"
    RISK_ALERT = "RISK_ALERT"
    SYSTEM_ERROR = "SYSTEM_ERROR"
    SYSTEM_WARNING = "SYSTEM_WARNING"
    PERFORMANCE_ALERT = "PERFORMANCE_ALERT"
    POSITION_UPDATE = "POSITION_UPDATE"

class AlertSeverity(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class AlertSystem:
    """Automated alert and notification service"""

    def __init__(self):
        self.redis_client = None
        self.is_running = False
        self.alert_history = []
        self.alert_rules = {}
        self.notification_channels = []

        # Alert thresholds
        self.risk_thresholds = {
            'max_daily_loss': float(os.getenv('MAX_DAILY_LOSS', '0.03')),
            'max_position_size': float(os.getenv('MAX_POSITION_SIZE', '0.1')),
            'max_open_positions': int(os.getenv('MAX_OPEN_POSITIONS', '15'))
        }

        # Cooldown periods (seconds)
        self.alert_cooldowns = {
            AlertType.TRADING_SIGNAL: int(os.getenv('SIGNAL_COOLDOWN', '300')),
            AlertType.RISK_ALERT: int(os.getenv('RISK_COOLDOWN', '60')),
            AlertType.SYSTEM_ERROR: int(os.getenv('ERROR_COOLDOWN', '300')),
            AlertType.SYSTEM_WARNING: int(os.getenv('WARNING_COOLDOWN', '600'))
        }

        self.last_alert_time = {}

    def connect_redis(self):
        """Connect to Redis"""
        try:
            self.redis_client = redis.Redis.from_url(REDIS_URL)
            self.redis_client.ping()
            logger.info("# Check Connected to Redis")
        except Exception as e:
            logger.error(f"# X Failed to connect to Redis: {e}")
            raise

    def setup_notification_channels(self):
        """Setup available notification channels"""
        channels = []

        # Email channel
        if SMTP_USERNAME and SMTP_PASSWORD and TO_EMAILS:
            channels.append('email')
            logger.info("📧 Email notifications enabled")

        # Telegram channel
        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_IDS:
            channels.append('telegram')
            logger.info("📱 Telegram notifications enabled")

        self.notification_channels = channels
        logger.info(f"🔔 Notification channels: {channels}")

    def send_email_alert(self, alert: Dict):
        """Send alert via email"""
        try:
            if not self.notification_channels or 'email' not in self.notification_channels:
                return

            msg = MIMEMultipart()
            msg['From'] = FROM_EMAIL
            msg['To'] = ', '.join(TO_EMAILS)
            msg['Subject'] = f"VIPER Alert: {alert['type']} - {alert['severity']}"

            body = f"""
VIPER Trading Bot Alert

Type: {alert['type']}
Severity: {alert['severity']}
Time: {alert['timestamp']}

Message: {alert['message']}

Details:
{json.dumps(alert.get('data', {}), indent=2)}

---
VIPER Trading Bot Alert System
"""

            msg.attach(MIMEText(body, 'plain'))

            server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.sendmail(FROM_EMAIL, TO_EMAILS, msg.as_string())
            server.quit()

            logger.info(f"📧 Email alert sent for {alert['type']}")

        except Exception as e:
            logger.error(f"# X Failed to send email alert: {e}")

    def send_telegram_alert(self, alert: Dict):
        """Send alert via Telegram"""
        try:
            if not self.notification_channels or 'telegram' not in self.notification_channels:
                return

            message = f"""
🚨 VIPER Alert

Type: {alert['type']}
Severity: {alert['severity']}
Time: {alert['timestamp']}

{alert['message']}

Details: {json.dumps(alert.get('data', {}), indent=2)}
"""

            for chat_id in TELEGRAM_CHAT_IDS:
                if chat_id.strip():
                    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
                    data = {
                        'chat_id': chat_id.strip(),
                        'text': message,
                        'parse_mode': 'HTML'
                    }

                    response = requests.post(url, data=data, timeout=10)
                    if response.status_code == 200:
                        logger.info(f"📱 Telegram alert sent to {chat_id}")
                    else:
                        logger.error(f"# X Failed to send Telegram alert: {response.text}")

        except Exception as e:
            logger.error(f"# X Failed to send Telegram alert: {e}")

    def send_alert(self, alert_type: AlertType, severity: AlertSeverity, message: str, data: Optional[Dict] = None):
        """Send alert through all configured channels"""
        try:
            # Check cooldown period
            current_time = datetime.now()
            last_alert = self.last_alert_time.get(alert_type, datetime.min)
            cooldown = self.alert_cooldowns.get(alert_type, 300)

            if (current_time - last_alert).seconds < cooldown:
                return

            alert = {
                'type': alert_type.value,
                'severity': severity.value,
                'message': message,
                'data': data or {},
                'timestamp': current_time.isoformat(),
                'service': SERVICE_NAME
            }

            # Send through all channels
            self.send_email_alert(alert)
            self.send_telegram_alert(alert)

            # Store alert in Redis
            self.redis_client.publish('alerts', json.dumps(alert))

            # Add to history
            self.alert_history.append(alert)
            if len(self.alert_history) > 1000:  # Keep last 1000 alerts
                self.alert_history = self.alert_history[-1000:]

            # Update last alert time
            self.last_alert_time[alert_type] = current_time

            logger.info(f"🚨 Alert sent: {alert_type.value} - {severity.value}")

        except Exception as e:
            logger.error(f"# X Failed to send alert: {e}")

    def process_trading_signals(self, signal_data: Dict):
        """Process trading signals and generate alerts"""
        try:
            symbol = signal_data.get('symbol', 'UNKNOWN')
            signal_type = signal_data.get('type', 'UNKNOWN')
            viper_score = signal_data.get('viper_score', 0)
            confidence = signal_data.get('confidence', 0)

            message = f"Trading Signal: {signal_type} {symbol} (VIPER: {viper_score:.1f}, Confidence: {confidence:.1f}%)"

            severity = AlertSeverity.HIGH if confidence > 90 else AlertSeverity.MEDIUM
            self.send_alert(AlertType.TRADING_SIGNAL, severity, message, signal_data)

        except Exception as e:
            logger.error(f"# X Failed to process trading signal: {e}")

    def process_risk_alerts(self, risk_data: Dict):
        """Process risk management alerts"""
        try:
            alert_type = risk_data.get('alert_type', 'unknown')
            current_value = risk_data.get('current_value', 0)
            threshold = risk_data.get('threshold', 0)

            if alert_type == 'daily_loss_limit':
                message = f"Daily loss limit exceeded: ${current_value:.2f}"
                severity = AlertSeverity.CRITICAL
            elif alert_type == 'position_limit':
                message = f"Position limit exceeded: {current_value}/{self.risk_thresholds['max_open_positions']}"
                severity = AlertSeverity.HIGH
            elif alert_type == 'capital_utilization':
                message = f"Capital utilization high: {current_value:.1f}%"
                severity = AlertSeverity.MEDIUM
            else:
                message = f"Risk alert: {alert_type} - Current: {current_value}, Threshold: {threshold}"
                severity = AlertSeverity.MEDIUM

            self.send_alert(AlertType.RISK_ALERT, severity, message, risk_data)

        except Exception as e:
            logger.error(f"# X Failed to process risk alert: {e}")

    def process_system_status(self, status_data: Dict):
        """Process system status updates and generate health alerts"""
        try:
            service = status_data.get('service', 'unknown')
            status = status_data.get('status', 'unknown')

            if status == 'error':
                message = f"Service {service} reported error: {status_data.get('message', 'No details')}"
                severity = AlertSeverity.CRITICAL
                self.send_alert(AlertType.SYSTEM_ERROR, severity, message, status_data)

            elif status == 'warning':
                message = f"Service {service} warning: {status_data.get('message', 'No details')}"
                severity = AlertSeverity.MEDIUM
                self.send_alert(AlertType.SYSTEM_WARNING, severity, message, status_data)

        except Exception as e:
            logger.error(f"# X Failed to process system status: {e}")

    def monitor_system_health(self):
        """Monitor overall system health and generate alerts"""
        while self.is_running:
            try:
                # Check Redis connectivity
                if not self.redis_client.ping():
                    self.send_alert(
                        AlertType.SYSTEM_ERROR,
                        AlertSeverity.CRITICAL,
                        "Redis connection lost",
                        {'service': 'redis', 'status': 'disconnected'}
                    )

                # Check service status via Redis
                service_status = self.redis_client.get('service_status')
                if service_status:
                    status_data = json.loads(service_status)
                    self.process_system_status(status_data)

                asyncio.run(asyncio.sleep(30))  # Check every 30 seconds

            except Exception as e:
                logger.error(f"# X Health monitoring error: {e}")
                asyncio.run(asyncio.sleep(60))

    def subscribe_to_alerts(self):
        """Subscribe to various alert channels"""
        try:
            pubsub = self.redis_client.pubsub()

            # Subscribe to alert channels
            channels = [
                'trading_signals',
                'risk_alerts',
                'system_status',
                'service_status'
            ]

            pubsub.subscribe(*channels)
            logger.info(f"📡 Subscribed to alert channels: {channels}")

            # Process messages
            for message in pubsub.listen():
                if not self.is_running:
                    break

                if message['type'] == 'message':
                    try:
                        data = json.loads(message['data'])
                        channel = message['channel'].decode('utf-8')

                        if channel == 'trading_signals':
                            self.process_trading_signals(data)
                        elif channel == 'risk_alerts':
                            self.process_risk_alerts(data)
                        elif channel in ['system_status', 'service_status']:
                            self.process_system_status(data)

                    except json.JSONDecodeError as e:
                        logger.error(f"# X Failed to decode alert message: {e}")

        except Exception as e:
            logger.error(f"# X Error in alert subscription: {e}")

    def start_background_monitoring(self):
        """Start background monitoring threads"""
        def run_alert_processor():
            self.subscribe_to_alerts()

        def run_health_monitor():
            asyncio.run(self.monitor_system_health())

        # Start alert processor
        alert_thread = threading.Thread(target=run_alert_processor, daemon=True)
        alert_thread.start()

        # Start health monitor
        health_thread = threading.Thread(target=run_health_monitor, daemon=True)
        health_thread.start()

        logger.info("🔔 Alert system monitoring started")

    def start(self):
        """Start the alert system"""
        try:
            logger.info("# Rocket Starting Alert System...")

            # Connect to Redis
            self.connect_redis()

            # Setup notification channels
            self.setup_notification_channels()

            # Send startup alert
            self.send_alert(
                AlertType.SYSTEM_WARNING,
                AlertSeverity.LOW,
                "VIPER Alert System started successfully",
                {'startup_time': datetime.now().isoformat()}
            )

            # Start monitoring
            self.is_running = True
            self.start_background_monitoring()

            # Keep main thread alive
            while self.is_running:
                asyncio.run(asyncio.sleep(60))

        except KeyboardInterrupt:
            logger.info("⏹️ Stopping Alert System...")
            self.stop()
        except Exception as e:
            logger.error(f"# X Alert System error: {e}")
            self.stop()

    def stop(self):
        """Stop the alert system"""
        self.is_running = False
        logger.info("# Check Alert System stopped")

def create_app():
    """Create FastAPI application for health checks and API"""
    from fastapi import FastAPI

    app = FastAPI(title="Alert System", version="1.0.0")

    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "service": "alert-system"}

    @app.get("/alerts")
    async def get_alerts():
        return {"alerts": alert_system.alert_history[-50:], "count": len(alert_system.alert_history)}

    @app.get("/config")
    async def get_config():
        return {
            "notification_channels": alert_system.notification_channels,
            "risk_thresholds": alert_system.risk_thresholds,
            "alert_cooldowns": {k.value: v for k, v in alert_system.alert_cooldowns.items()}
        }

    @app.post("/test-alert")
    async def test_alert():
        alert_system.send_alert(
            AlertType.SYSTEM_WARNING,
            AlertSeverity.LOW,
            "Test alert from API",
            {'test': True, 'timestamp': datetime.now().isoformat()}
        )
        return {"message": "Test alert sent"}

    return app

if __name__ == "__main__":
    # Check if running as API server or alert system
    if os.getenv('API_MODE', 'false').lower() == 'true':
        import uvicorn
        app = create_app()
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        # Run as alert system
        alert_system = AlertSystem()
        alert_system.start()
