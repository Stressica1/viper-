#!/usr/bin/env python3
"""
🚀 VIPER Trading Bot - Centralized Logger
Unified logging service for all microservices

Features:
- Log aggregation from all services
- Structured logging with correlation IDs
- Real-time log streaming and monitoring
- Log filtering and search capabilities
- Log retention and archival
- Integration with ELK stack
- Alerting on log patterns
"""

import os
import json
import logging
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import deque
import redis
import requests
from fastapi import FastAPI, WebSocket, Query
from fastapi.responses import HTMLResponse
import uvicorn

# Load environment variables
REDIS_URL = os.getenv('REDIS_URL', 'redis://redis:6379')
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
SERVICE_NAME = os.getenv('SERVICE_NAME', 'centralized-logger')
ELASTICSEARCH_URL = os.getenv('ELASTICSEARCH_URL', 'http://elasticsearch:9200')

# Configure logging
log_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LogEntry:
    """Structured log entry"""

    def __init__(self, service: str, level: str, message: str,
                 correlation_id: str = None, data: Dict = None):
        self.timestamp = datetime.now().isoformat()
        self.service = service
        self.level = level.upper()
        self.message = message
        self.correlation_id = correlation_id or self._generate_correlation_id()
        self.data = data or {}
        self.trace_id = self.data.get('trace_id', self.correlation_id)

    def _generate_correlation_id(self) -> str:
        """Generate unique correlation ID"""
        import uuid
        return str(uuid.uuid4())[:8]

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'timestamp': self.timestamp,
            'service': self.service,
            'level': self.level,
            'message': self.message,
            'correlation_id': self.correlation_id,
            'trace_id': self.trace_id,
            'data': self.data
        }

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict())

class LogAggregator:
    """Log aggregation and processing service"""

    def __init__(self):
        self.redis_client = None
        self.log_buffer = deque(maxlen=10000)  # Keep last 10k logs in memory
        self.service_logs = {}  # Logs per service
        self.correlation_logs = {}  # Logs by correlation ID
        self.alert_rules = {}
        self.is_running = False

        # Log retention settings
        self.retention_days = int(os.getenv('LOG_RETENTION_DAYS', '30'))
        self.max_logs_per_service = int(os.getenv('MAX_LOGS_PER_SERVICE', '1000'))

    def connect_redis(self):
        """Connect to Redis"""
        try:
            self.redis_client = redis.Redis.from_url(REDIS_URL)
            self.redis_client.ping()
            logger.info("✅ Connected to Redis")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Redis: {e}")
            raise

    def setup_alert_rules(self):
        """Setup default alert rules"""
        self.alert_rules = {
            'error_spike': {
                'condition': lambda logs: len([l for l in logs if l.level == 'ERROR']) > 10,
                'message': 'High error rate detected',
                'severity': 'HIGH'
            },
            'service_down': {
                'condition': lambda logs: len(logs) == 0,  # No logs from service
                'message': 'Service appears to be down',
                'severity': 'CRITICAL'
            },
            'memory_warning': {
                'condition': lambda logs: any('memory' in l.message.lower() and 'high' in l.message.lower() for l in logs),
                'message': 'Memory usage warning detected',
                'severity': 'MEDIUM'
            }
        }

    def process_log_entry(self, log_data: Dict):
        """Process incoming log entry"""
        try:
            # Create LogEntry object
            log_entry = LogEntry(
                service=log_data.get('service', 'unknown'),
                level=log_data.get('level', 'INFO'),
                message=log_data.get('message', ''),
                correlation_id=log_data.get('correlation_id'),
                data=log_data.get('data', {})
            )

            # Add to buffers
            self.log_buffer.append(log_entry)

            # Organize by service
            service = log_entry.service
            if service not in self.service_logs:
                self.service_logs[service] = deque(maxlen=self.max_logs_per_service)
            self.service_logs[service].append(log_entry)

            # Organize by correlation ID
            corr_id = log_entry.correlation_id
            if corr_id not in self.correlation_logs:
                self.correlation_logs[corr_id] = []
            self.correlation_logs[corr_id].append(log_entry)

            # Clean old correlation logs (keep last 24h)
            cutoff_time = datetime.now() - timedelta(hours=24)
            for corr_id in list(self.correlation_logs.keys()):
                self.correlation_logs[corr_id] = [
                    log for log in self.correlation_logs[corr_id]
                    if datetime.fromisoformat(log.timestamp.replace('Z', '+00:00')) > cutoff_time
                ]
                if not self.correlation_logs[corr_id]:
                    del self.correlation_logs[corr_id]

            # Send to Elasticsearch
            self.send_to_elasticsearch(log_entry)

            # Check alert rules
            self.check_alerts(service)

            # Publish to Redis for real-time subscribers
            self.redis_client.publish('logs:realtime', log_entry.to_json())
            self.redis_client.publish(f'logs:{service}', log_entry.to_json())

            logger.debug(f"📝 Processed log from {service}: {log_entry.level} - {log_entry.message[:50]}...")

        except Exception as e:
            logger.error(f"❌ Error processing log entry: {e}")

    def send_to_elasticsearch(self, log_entry: LogEntry):
        """Send log entry to Elasticsearch"""
        try:
            if not ELASTICSEARCH_URL:
                return

            # Prepare document
            doc = {
                'timestamp': log_entry.timestamp,
                'service': log_entry.service,
                'level': log_entry.level,
                'message': log_entry.message,
                'correlation_id': log_entry.correlation_id,
                'trace_id': log_entry.trace_id,
                'data': log_entry.data
            }

            # Index document
            index_name = f"viper-logs-{datetime.now().strftime('%Y-%m-%d')}"
            response = requests.post(
                f"{ELASTICSEARCH_URL}/{index_name}/_doc",
                json=doc,
                timeout=5
            )

            if response.status_code not in [200, 201]:
                logger.warning(f"⚠️ Failed to index log in Elasticsearch: {response.text}")

        except Exception as e:
            logger.debug(f"Debug: Elasticsearch not available: {e}")

    def check_alerts(self, service: str):
        """Check alert rules for a service"""
        try:
            if service not in self.service_logs:
                return

            recent_logs = list(self.service_logs[service])

            for rule_name, rule in self.alert_rules.items():
                if rule['condition'](recent_logs):
                    alert = {
                        'rule': rule_name,
                        'service': service,
                        'severity': rule['severity'],
                        'message': rule['message'],
                        'timestamp': datetime.now().isoformat(),
                        'log_count': len(recent_logs)
                    }

                    # Publish alert
                    self.redis_client.publish('log_alerts', json.dumps(alert))
                    logger.warning(f"🚨 Log alert: {rule_name} for {service}")

        except Exception as e:
            logger.error(f"❌ Error checking alerts: {e}")

    def subscribe_to_logs(self):
        """Subscribe to log streams from all services"""
        try:
            pubsub = self.redis_client.pubsub()

            # Subscribe to all log channels
            channels = [
                'logs:api-server',
                'logs:ultra-backtester',
                'logs:risk-manager',
                'logs:data-manager',
                'logs:strategy-optimizer',
                'logs:exchange-connector',
                'logs:monitoring-service',
                'logs:live-trading-engine',
                'logs:credential-vault',
                'logs:market-data-streamer',
                'logs:signal-processor',
                'logs:alert-system',
                'logs:order-lifecycle-manager',
                'logs:position-synchronizer'
            ]

            pubsub.subscribe(*channels)
            logger.info(f"📡 Subscribed to {len(channels)} log channels")

            # Process log messages
            for message in pubsub.listen():
                if not self.is_running:
                    break

                if message['type'] == 'message':
                    try:
                        log_data = json.loads(message['data'])
                        self.process_log_entry(log_data)
                    except json.JSONDecodeError as e:
                        logger.error(f"❌ Failed to decode log message: {e}")

        except Exception as e:
            logger.error(f"❌ Error in log subscription: {e}")

    def start_background_processing(self):
        """Start background log processing"""
        def run_log_processor():
            self.subscribe_to_logs()

        thread = threading.Thread(target=run_log_processor, daemon=True)
        thread.start()
        logger.info("🎯 Log processing started")

    def get_logs_by_service(self, service: str, limit: int = 100) -> List[Dict]:
        """Get logs for a specific service"""
        if service not in self.service_logs:
            return []

        logs = list(self.service_logs[service])
        return [log.to_dict() for log in logs[-limit:]]

    def get_logs_by_correlation(self, correlation_id: str) -> List[Dict]:
        """Get logs for a specific correlation ID"""
        if correlation_id not in self.correlation_logs:
            return []

        logs = self.correlation_logs[correlation_id]
        return [log.to_dict() for log in logs]

    def search_logs(self, query: str, limit: int = 100) -> List[Dict]:
        """Search logs by message content"""
        matching_logs = []

        for log in self.log_buffer:
            if query.lower() in log.message.lower():
                matching_logs.append(log.to_dict())
                if len(matching_logs) >= limit:
                    break

        return matching_logs

    def get_log_statistics(self) -> Dict[str, Any]:
        """Get log statistics"""
        stats = {
            'total_logs': len(self.log_buffer),
            'services': len(self.service_logs),
            'correlations': len(self.correlation_logs),
            'timestamp': datetime.now().isoformat()
        }

        # Service breakdown
        service_counts = {}
        for service, logs in self.service_logs.items():
            service_counts[service] = len(logs)

        stats['service_breakdown'] = service_counts

        # Error rate by service
        error_rates = {}
        for service, logs in self.service_logs.items():
            errors = sum(1 for log in logs if log.level == 'ERROR')
            error_rates[service] = errors / len(logs) if logs else 0

        stats['error_rates'] = error_rates

        return stats

    def start(self):
        """Start the log aggregator"""
        try:
            logger.info("🚀 Starting Centralized Logger...")

            # Connect to Redis
            self.connect_redis()

            # Setup alert rules
            self.setup_alert_rules()

            # Start processing
            self.is_running = True
            self.start_background_processing()

            # Keep main thread alive
            while self.is_running:
                # Publish periodic statistics
                stats = self.get_log_statistics()
                self.redis_client.publish('log_stats', json.dumps(stats))
                asyncio.run(asyncio.sleep(60))  # Update every minute

        except KeyboardInterrupt:
            logger.info("⏹️ Stopping Centralized Logger...")
            self.stop()
        except Exception as e:
            logger.error(f"❌ Centralized Logger error: {e}")
            self.stop()

    def stop(self):
        """Stop the log aggregator"""
        self.is_running = False
        logger.info("✅ Centralized Logger stopped")

# Global log aggregator instance
log_aggregator = LogAggregator()

def create_app():
    """Create FastAPI application for log management"""
    app = FastAPI(title="Centralized Logger", version="1.0.0")

    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "service": "centralized-logger"}

    @app.get("/logs/{service}")
    async def get_service_logs(service: str, limit: int = Query(100, ge=1, le=1000)):
        logs = log_aggregator.get_logs_by_service(service, limit)
        return {"service": service, "logs": logs, "count": len(logs)}

    @app.get("/logs/correlation/{correlation_id}")
    async def get_correlation_logs(correlation_id: str):
        logs = log_aggregator.get_logs_by_correlation(correlation_id)
        return {"correlation_id": correlation_id, "logs": logs, "count": len(logs)}

    @app.get("/logs/search")
    async def search_logs(query: str, limit: int = Query(100, ge=1, le=1000)):
        logs = log_aggregator.search_logs(query, limit)
        return {"query": query, "logs": logs, "count": len(logs)}

    @app.get("/stats")
    async def get_statistics():
        stats = log_aggregator.get_log_statistics()
        return {"statistics": stats}

    @app.get("/services")
    async def get_services():
        services = list(log_aggregator.service_logs.keys())
        return {"services": services, "count": len(services)}

    @app.websocket("/ws/logs")
    async def websocket_logs(websocket: WebSocket):
        await websocket.accept()

        try:
            # Subscribe to real-time logs
            pubsub = log_aggregator.redis_client.pubsub()
            pubsub.subscribe('logs:realtime')

            for message in pubsub.listen():
                if message['type'] == 'message':
                    await websocket.send_text(message['data'].decode())

        except Exception as e:
            logger.error(f"WebSocket error: {e}")

    return app

if __name__ == "__main__":
    # Check if running as API server or logger
    if os.getenv('API_MODE', 'false').lower() == 'true':
        import uvicorn
        app = create_app()
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        # Run as log aggregator
        log_aggregator.start()
