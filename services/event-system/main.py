#!/usr/bin/env python3
"""
# Rocket VIPER Trading Bot - Event System Service
Redis-based pub/sub event system for real-time communication between services

Features:
- Centralized event routing and processing
- Real-time signal propagation
- Service health monitoring
- Event logging and analytics
- Dead letter queue handling
- Circuit breaker pattern integration
"""

import os
import json
import time
import logging
import asyncio
import threading
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
import uvicorn
import redis
from collections import defaultdict

# Load environment variables
REDIS_URL = os.getenv('REDIS_URL', 'redis://redis:6379')
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
SERVICE_NAME = os.getenv('SERVICE_NAME', 'event-system')

# Event System Configuration
EVENT_RETENTION_SECONDS = int(os.getenv('EVENT_RETENTION_SECONDS', '3600'))  # 1 hour
MAX_EVENTS_PER_CHANNEL = int(os.getenv('MAX_EVENTS_PER_CHANNEL', '1000'))
HEALTH_CHECK_INTERVAL = int(os.getenv('HEALTH_CHECK_INTERVAL', '30'))  # seconds
DEAD_LETTER_TTL = int(os.getenv('DEAD_LETTER_TTL', '86400'))  # 24 hours

# Configure logging
log_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EventSystemService:
    """Centralized event system for service communication"""

    def __init__(self):
        self.redis_client = None
        self.is_running = False
        self.event_counts = defaultdict(int)
        self.service_health = {}
        self.event_processors = {}
        self.dead_letter_queue = []

        # Service URLs for health checks
        self.service_urls = {
            'market-data-manager': os.getenv('MARKET_DATA_MANAGER_URL', 'http://market-data-manager:8003'),
            'viper-scoring-service': os.getenv('VIPER_SCORING_SERVICE_URL', 'http://viper-scoring-service:8009'),
            'live-trading-engine': os.getenv('LIVE_TRADING_ENGINE_URL', 'http://live-trading-engine:8007'),
            'risk-manager': os.getenv('RISK_MANAGER_URL', 'http://risk-manager:8002'),
            'signal-processor': os.getenv('SIGNAL_PROCESSOR_URL', 'http://signal-processor:8000')
        }

        logger.info("📡 Event System Service initialized")

    def initialize_redis(self) -> bool:
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
            self.redis_client.ping()
            logger.info("# Check Redis connection established")
            return True
        except Exception as e:
            logger.error(f"# X Failed to connect to Redis: {e}")
            return False

    def publish_event(self, channel: str, event_data: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Publish event to Redis channel with optional TTL"""
        try:
            event = {
                'id': f"{channel}_{int(time.time() * 1000000)}",
                'channel': channel,
                'data': event_data,
                'timestamp': datetime.now().isoformat(),
                'ttl': ttl or EVENT_RETENTION_SECONDS
            }

            # Publish to channel
            self.redis_client.publish(channel, json.dumps(event))

            # Store in channel-specific list (with size limit)
            list_key = f"events:{channel}"
            self.redis_client.lpush(list_key, json.dumps(event))
            self.redis_client.ltrim(list_key, 0, MAX_EVENTS_PER_CHANNEL - 1)
            self.redis_client.expire(list_key, EVENT_RETENTION_SECONDS)

            # Update event counts
            self.event_counts[channel] += 1

            logger.debug(f"📡 Published event to {channel}: {event['id']}")
            return True

        except Exception as e:
            logger.error(f"# X Failed to publish event to {channel}: {e}")
            return False

    def get_channel_events(self, channel: str, limit: int = 100) -> List[Dict]:
        """Get recent events from a channel"""
        try:
            list_key = f"events:{channel}"
            events_data = self.redis_client.lrange(list_key, 0, limit - 1)

            events = []
            for event_data in events_data:
                try:
                    events.append(json.loads(event_data))
                except json.JSONDecodeError:
                    continue

            return events

        except Exception as e:
            logger.error(f"# X Failed to get events from {channel}: {e}")
            return []

    def register_event_processor(self, channel: str, processor_func):
        """Register an event processor for a specific channel"""
        self.event_processors[channel] = processor_func
        logger.info(f"# Tool Registered processor for channel: {channel}")

    def process_trading_signals(self, event_data: Dict):
        """Process trading signals and route to appropriate services"""
        try:
            signal = event_data.get('data', {})
            symbol = signal.get('symbol', '')
            signal_type = signal.get('type', '')

            if not symbol or not signal_type:
                return

            logger.info(f"# Target Processing {signal_type} signal for {symbol}")

            # Route to risk manager for validation
            risk_event = {
                'type': 'signal_validation_request',
                'signal': signal,
                'symbol': symbol,
                'timestamp': datetime.now().isoformat()
            }
            self.publish_event('risk_validation', risk_event)

            # Route to trading engine if validation passes
            trade_event = {
                'type': 'trading_signal',
                'signal': signal,
                'symbol': symbol,
                'timestamp': datetime.now().isoformat()
            }
            self.publish_event('trading_signals', trade_event)

            # Log signal for analytics
            self.publish_event('signal_log', {
                'signal': signal,
                'processed_at': datetime.now().isoformat(),
                'source': 'viper_scoring_service'
            })

        except Exception as e:
            logger.error(f"# X Error processing trading signal: {e}")

    def process_risk_alerts(self, event_data: Dict):
        """Process risk alerts and take appropriate actions"""
        try:
            alert = event_data.get('data', {})
            alert_type = alert.get('type', '')

            logger.warning(f"🚨 Risk Alert: {alert_type}")

            # Route critical alerts to trading engine for immediate action
            if alert_type in ['daily_loss_limit_breached', 'auto_stop_triggered']:
                emergency_event = {
                    'type': 'emergency_stop',
                    'alert': alert,
                    'timestamp': datetime.now().isoformat()
                }
                self.publish_event('trading_emergency', emergency_event)

            # Log all risk alerts
            self.publish_event('risk_log', alert)

        except Exception as e:
            logger.error(f"# X Error processing risk alert: {e}")

    def process_service_status(self, event_data: Dict):
        """Process service status updates"""
        try:
            status = event_data.get('data', {})
            service_name = status.get('service', '')

            if service_name:
                self.service_health[service_name] = {
                    'status': status,
                    'last_update': datetime.now().isoformat()
                }

                logger.debug(f"# Chart Updated status for {service_name}")

        except Exception as e:
            logger.error(f"# X Error processing service status: {e}")

    async def check_service_health(self):
        """Check health of all registered services"""
        while self.is_running:
            try:
                for service_name, url in self.service_urls.items():
                    try:
                        # Use asyncio-compatible HTTP client
                        # This is a simplified check - in production you'd want more robust health checking
                        response = self.redis_client.get(f"service_status:{service_name}")
                        if response:
                            status_data = json.loads(response)
                            self.service_health[service_name] = {
                                'status': status_data,
                                'last_update': datetime.now().isoformat(),
                                'health': 'healthy'
                            }
                        else:
                            self.service_health[service_name] = {
                                'status': {},
                                'last_update': datetime.now().isoformat(),
                                'health': 'unknown'
                            }

                    except Exception as e:
                        logger.error(f"# X Health check failed for {service_name}: {e}")
                        self.service_health[service_name] = {
                            'status': {},
                            'last_update': datetime.now().isoformat(),
                            'health': 'unhealthy',
                            'error': str(e)
                        }

                # Publish consolidated health status
                health_summary = {
                    'services': self.service_health,
                    'overall_health': self.get_overall_health(),
                    'timestamp': datetime.now().isoformat()
                }
                self.publish_event('system_health', health_summary)

                await asyncio.sleep(HEALTH_CHECK_INTERVAL)

            except Exception as e:
                logger.error(f"# X Error in health check loop: {e}")
                await asyncio.sleep(5)

    def get_overall_health(self) -> str:
        """Determine overall system health"""
        if not self.service_health:
            return 'unknown'

        unhealthy_count = sum(1 for service in self.service_health.values()
                            if service.get('health') == 'unhealthy'):

        if unhealthy_count == 0:
            return 'healthy'
        elif unhealthy_count < len(self.service_health) / 2:
            return 'degraded'
        else:
            return 'critical'

    def subscribe_to_events(self):
        """Subscribe to all relevant event channels"""
        try:
            pubsub = self.redis_client.pubsub()

            # Core channels to monitor
            channels = [
                'trading_signals',
                'risk_alerts',
                'service_status',
                'market_data:all',
                'system_events'
            ]

            pubsub.subscribe(*channels)
            logger.info(f"📡 Subscribed to channels: {channels}")

            # Register event processors
            self.register_event_processor('trading_signals', self.process_trading_signals)
            self.register_event_processor('risk_alerts', self.process_risk_alerts)
            self.register_event_processor('service_status', self.process_service_status)

            # Process messages
            for message in pubsub.listen():
                if not self.is_running:
                    break

                if message['type'] == 'message':
                    try:
                        channel = message['channel']
                        event_data = json.loads(message['data'])

                        # Route to appropriate processor
                        if channel in self.event_processors:
                            self.event_processors[channel](event_data)
                        else:
                            logger.debug(f"📨 Received event on {channel}: {event_data.get('id', 'unknown')}")

                    except json.JSONDecodeError as e:
                        logger.error(f"# X Failed to decode message: {e}")
                    except Exception as e:
                        logger.error(f"# X Error processing message: {e}")

        except Exception as e:
            logger.error(f"# X Error in event subscription: {e}")

    def start_background_tasks(self):
        """Start background event processing tasks"""
        # Start event subscription
        subscription_thread = threading.Thread(target=self.subscribe_to_events, daemon=True)
        subscription_thread.start()

        # Start health monitoring
        health_thread = threading.Thread(target=lambda: asyncio.run(self.check_service_health()), daemon=True)
        health_thread.start()

        logger.info("# Target Event processing started in background")

    def start(self):
        """Start the event system service"""
        try:
            logger.info("# Rocket Starting Event System Service...")

            # Connect to Redis
            if not self.initialize_redis():
                raise Exception("Failed to connect to Redis")

            # Start background tasks
            self.is_running = True
            self.start_background_tasks()

            # Keep main thread alive
            while self.is_running:
                # Publish periodic statistics
                stats = {
                    'service': 'event-system',
                    'event_counts': dict(self.event_counts),
                    'service_health': self.service_health,
                    'uptime': time.time(),
                    'timestamp': datetime.now().isoformat()
                }

                self.redis_client.setex('service_status:event-system', 60, json.dumps(stats))
                self.publish_event('system_stats', stats)

                asyncio.run(asyncio.sleep(60))  # Update every minute

        except KeyboardInterrupt:
            logger.info("⏹️ Stopping Event System Service...")
            self.stop()
        except Exception as e:
            logger.error(f"# X Event System Service error: {e}")
            self.stop()

    def stop(self):
        """Stop the event system service"""
        self.is_running = False
        logger.info("# Check Event System Service stopped")

# FastAPI application
app = FastAPI(
    title="VIPER Event System",
    version="1.0.0",
    description="Centralized event routing and communication service"
)

event_service = EventSystemService()

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    if not event_service.initialize_redis():
        logger.error("# X Failed to initialize Redis")
        return

    # Start background tasks
    asyncio.create_task(asyncio.to_thread(event_service.start_background_tasks))
    logger.info("# Check Event System Service started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean shutdown"""
    event_service.stop()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        return {
            "status": "healthy",
            "service": "event-system",
            "redis_connected": event_service.redis_client is not None,
            "processing_active": event_service.is_running,
            "channels_monitored": len(event_service.event_processors),
            "services_tracked": len(event_service.service_health)
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "service": "event-system",
                "error": str(e)
            }
        )

@app.post("/api/events/publish")
async def publish_event(request: Request):
    """Publish a custom event"""
    try:
        data = await request.json()
        channel = data.get('channel', '')
        event_data = data.get('data', {})
        ttl = data.get('ttl')

        if not channel or not event_data:
            raise HTTPException(status_code=400, detail="Channel and data are required")

        success = event_service.publish_event(channel, event_data, ttl)
        if success:
            return {"status": "published", "channel": channel}
        else:
            raise HTTPException(status_code=500, detail="Failed to publish event")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Publish failed: {e}")

@app.get("/api/events/{channel}")
async def get_channel_events(channel: str, limit: int = Query(100, ge=1, le=1000)):
    """Get recent events from a channel"""
    try:
        events = event_service.get_channel_events(channel, limit)
        return {
            "channel": channel,
            "events": events,
            "count": len(events)
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Unable to get events: {e}")

@app.get("/api/health/services")
async def get_service_health():
    """Get health status of all services"""
    try:
        return {
            "services": event_service.service_health,
            "overall_health": event_service.get_overall_health(),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Unable to get health status: {e}")

@app.get("/api/stats")
async def get_event_stats():
    """Get event system statistics"""
    try:
        return {
            "event_counts": dict(event_service.event_counts),
            "total_events": sum(event_service.event_counts.values()),
            "channels_active": len(event_service.event_counts),
            "service_health": event_service.service_health,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Unable to get stats: {e}")

@app.delete("/api/events/{channel}")
async def clear_channel_events(channel: str):
    """Clear all events from a channel"""
    try:
        list_key = f"events:{channel}"
        event_service.redis_client.delete(list_key)

        # Reset counter
        if channel in event_service.event_counts:
            event_service.event_counts[channel] = 0

        return {"status": "cleared", "channel": channel}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Unable to clear events: {e}")

@app.post("/api/alerts/send")
async def send_system_alert(request: Request):
    """Send a system-wide alert"""
    try:
        data = await request.json()
        alert_type = data.get('type', 'general')
        message = data.get('message', '')
        severity = data.get('severity', 'info')

        if not message:
            raise HTTPException(status_code=400, detail="Message is required")

        alert_event = {
            'type': alert_type,
            'message': message,
            'severity': severity,
            'timestamp': datetime.now().isoformat(),
            'source': 'event_system'
        }

        event_service.publish_event('system_alerts', alert_event)

        return {"status": "sent", "alert": alert_event}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Alert failed: {e}")

if __name__ == "__main__":
    port = int(os.getenv("EVENT_SYSTEM_PORT", 8010))
    logger.info(f"Starting VIPER Event System on port {port}")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=os.getenv("DEBUG_MODE", "false").lower() == "true",
        log_level="info"
    )
